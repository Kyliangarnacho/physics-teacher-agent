"""Streamlit 与 Generation Job 后台系统集成测试（无真实 API）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import streamlit as st
from PIL import Image
from streamlit.testing.v1 import AppTest

from src.conversation import enqueue_conversation_turn, execute_generation_job
from src.storage import (
    claim_generation_job,
    create_conversation,
    fail_generation_job,
    get_agent_runs,
    get_conversation,
    get_generation_job,
    initialize_database,
    list_generation_jobs,
    list_messages,
    mark_running_interrupted,
    rename_conversation,
    retry_generation_job,
)
from src.vision.schemas import ImageQuestionExtraction


def _agent_result(answer: str = "后台回答") -> dict:
    return {
        "answer": answer,
        "analysis": {"teaching_mode": "solve"},
        "route": {
            "teaching_mode": "solve",
            "use_rag": False,
            "use_tools": False,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": {
            "status": "completed",
            "total_model_requests": 2,
            "total_duration_ms": 10.0,
            "run_id": "background-app-test",
            "steps": [],
        },
    }


class FakeTaskManager:
    instances: list["FakeTaskManager"] = []

    def __init__(self, *, db_path=None, execute_func=None) -> None:
        self.db_path = db_path
        self.submitted: list[str] = []
        self.scheduled: set[str] = set()
        self.recover_calls = 0
        self.shutdown_calls = 0
        type(self).instances.append(self)

    def recover(self):
        self.recover_calls += 1
        return {
            "already_recovered": self.recover_calls > 1,
            "interrupted_count": 0,
            "pending_count": 0,
            "submitted_job_ids": [],
        }

    def submit(self, job_id: str) -> bool:
        if job_id in self.scheduled:
            return False
        self.scheduled.add(job_id)
        self.submitted.append(job_id)
        return True

    def is_scheduled(self, job_id: str) -> bool:
        return job_id in self.scheduled

    def retry(self, job_id: str):
        job = retry_generation_job(job_id, path=self.db_path)
        return {"generation_job": job, "scheduled": self.submit(job_id)}

    def shutdown(self) -> None:
        self.shutdown_calls += 1


def _png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def _vision_result() -> dict:
    extraction = ImageQuestionExtraction(
        status="complete",
        image_type="circuit",
        extracted_text="电压为 12 V，电阻为 6 Ω。",
        visual_elements=["电源", "电阻"],
        relationships=["电源与电阻构成闭合回路"],
        values_and_units=["12 V", "6 Ω"],
        formulas=[],
        student_work=[],
        uncertain_items=[],
        suggested_user_question="求电流。",
        needs_ocr=False,
    )
    return {
        "prepared_image": {
            "image_hash": "a" * 64,
            "mime_type": "image/png",
            "width": 24,
            "height": 16,
            "byte_size": 80,
            "resized": False,
        },
        "extraction": extraction,
        "ocr_used": False,
        "vision_run_id": "vision-safe",
        "vision_step_traces": [],
        "model_requests": 1,
    }


class AppBackgroundJobTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()
        FakeTaskManager.instances.clear()
        self._manager_patch = patch(
            "src.tasks.GenerationTaskManager",
            FakeTaskManager,
        )
        self._manager_patch.start()
        self.addCleanup(self._manager_patch.stop)
        self.addCleanup(st.cache_resource.clear)
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "app_background.db")
        os.environ["PHYSICS_AGENT_DB_PATH"] = self.db_path
        self.addCleanup(os.environ.pop, "PHYSICS_AGENT_DB_PATH", None)
        initialize_database(self.db_path)

    def run_app(self, conversation_id: str | None = None) -> AppTest:
        app = AppTest.from_file("app.py")
        if conversation_id:
            app.session_state["current_conversation_id"] = conversation_id
        app.run()
        return app

    @staticmethod
    def button_by_key(app: AppTest, key: str):
        return next(button for button in app.button if button.key == key)

    def _enqueue(self, title: str, question: str = "题目") -> tuple[dict, dict]:
        conversation = create_conversation(title, path=self.db_path)
        enqueued = enqueue_conversation_turn(
            conversation["id"],
            question,
            question,
            db_path=self.db_path,
        )
        return conversation, enqueued

    def test_submit_persists_user_immediately_without_sync_agent_call(self) -> None:
        with patch("src.conversation.service._default_run_agent") as agent:
            app = self.run_app()
            app.chat_input[0].set_value("为什么金属更凉？").run()

        conversation_id = app.session_state["current_conversation_id"]
        messages = list_messages(conversation_id, path=self.db_path)
        jobs = list_generation_jobs(conversation_id, path=self.db_path)
        self.assertEqual([message["role"] for message in messages], ["user"])
        self.assertEqual(jobs[0]["status"], "pending")
        self.assertEqual(FakeTaskManager.instances[-1].submitted, [jobs[0]["id"]])
        self.assertEqual(agent.call_count, 0)
        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("为什么金属更凉", rendered)

    def test_pending_running_failed_interrupted_and_completed_display(self) -> None:
        pending_conv, _pending = self._enqueue("pending")
        pending_app = self.run_app(pending_conv["id"])
        self.assertTrue(any("等待生成" in str(item.value) for item in pending_app.info))

        running_conv, running = self._enqueue("running")
        claim_generation_job(running["generation_job_id"], path=self.db_path)
        running_app = self.run_app(running_conv["id"])
        self.assertTrue(any("正在生成" in str(item.value) for item in running_app.info))

        failed_conv, failed = self._enqueue("failed")
        claim_generation_job(failed["generation_job_id"], path=self.db_path)
        fail_generation_job(
            failed["generation_job_id"],
            "agent_error",
            "安全错误。",
            path=self.db_path,
        )
        failed_app = self.run_app(failed_conv["id"])
        self.assertTrue(any("生成失败" in str(item.value) for item in failed_app.error))
        self.assertTrue(
            any(button.key == f"retry_job_{failed['generation_job_id']}" for button in failed_app.button)
        )

        interrupted_conv, interrupted = self._enqueue("interrupted")
        claim_generation_job(interrupted["generation_job_id"], path=self.db_path)
        mark_running_interrupted(path=self.db_path)
        interrupted_app = self.run_app(interrupted_conv["id"])
        self.assertTrue(
            any("服务中断" in str(item.value) for item in interrupted_app.warning)
        )
        self.assertTrue(
            any(
                button.key == f"retry_job_{interrupted['generation_job_id']}"
                for button in interrupted_app.button
            )
        )
        self.assertFalse(interrupted_app.chat_input[0].disabled)

        completed_conv, completed = self._enqueue("completed")
        execute_generation_job(
            completed["generation_job_id"],
            db_path=self.db_path,
            agent_func=lambda question, **kwargs: _agent_result("完成回答"),
        )
        completed_app = self.run_app(completed_conv["id"])
        self.assertFalse(any("等待生成" in str(item.value) for item in completed_app.info))
        self.assertFalse(
            any(button.key == f"retry_job_{completed['generation_job_id']}" for button in completed_app.button)
        )
        rendered = "\n".join(str(item.value) for item in completed_app.markdown)
        self.assertIn("完成回答", rendered)

    def test_retry_reuses_job_and_does_not_insert_user(self) -> None:
        conversation, enqueued = self._enqueue("重试")
        claim_generation_job(enqueued["generation_job_id"], path=self.db_path)
        fail_generation_job(
            enqueued["generation_job_id"],
            "agent_error",
            "安全错误。",
            path=self.db_path,
        )
        app = self.run_app(conversation["id"])

        self.button_by_key(app, f"retry_job_{enqueued['generation_job_id']}").click().run()

        job = get_generation_job(enqueued["generation_job_id"], path=self.db_path)
        messages = list_messages(conversation["id"], path=self.db_path)
        self.assertEqual(job["status"], "pending")
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["id"], enqueued["user_message_id"])
        self.assertIn(enqueued["generation_job_id"], FakeTaskManager.instances[-1].submitted)

    def test_polling_is_read_only_and_completed_stops_polling(self) -> None:
        conversation, enqueued = self._enqueue("轮询")
        app = self.run_app(conversation["id"])
        manager = FakeTaskManager.instances[-1]
        before_jobs = list_generation_jobs(conversation["id"], path=self.db_path)

        app.run()

        self.assertEqual(manager.submitted, [])
        self.assertEqual(
            list_generation_jobs(conversation["id"], path=self.db_path),
            before_jobs,
        )
        self.assertEqual(len(list_messages(conversation["id"], path=self.db_path)), 1)

        execute_generation_job(
            enqueued["generation_job_id"],
            db_path=self.db_path,
            agent_func=lambda question, **kwargs: _agent_result(),
        )
        app.run()
        self.assertNotIn(
            f"polling_generation_job_{conversation['id']}",
            app.session_state,
        )

    def test_running_a_allows_switch_and_enqueue_in_b_with_isolated_status(self) -> None:
        conversation_a, enqueued_a = self._enqueue("A", "A 的问题")
        conversation_b = create_conversation("B", path=self.db_path)
        app = self.run_app(conversation_a["id"])
        self.assertTrue(any("等待生成" in str(item.value) for item in app.info))

        self.button_by_key(app, f"select_conv_{conversation_b['id']}").click().run()
        self.assertEqual(app.session_state["current_conversation_id"], conversation_b["id"])
        self.assertFalse(any("等待生成" in str(item.value) for item in app.info))
        app.chat_input[0].set_value("B 的问题").run()

        jobs_a = list_generation_jobs(conversation_a["id"], path=self.db_path)
        jobs_b = list_generation_jobs(conversation_b["id"], path=self.db_path)
        self.assertEqual(jobs_a[0]["id"], enqueued_a["generation_job_id"])
        self.assertEqual(jobs_a[0]["status"], "pending")
        self.assertEqual(jobs_b[0]["status"], "pending")
        self.assertEqual(
            [message["model_content"] for message in list_messages(conversation_b["id"], path=self.db_path)],
            ["B 的问题"],
        )

    def test_active_job_blocks_delete_and_clear_but_not_rename(self) -> None:
        conversation, _enqueued = self._enqueue("活动会话", "不能丢失")
        app = self.run_app(conversation["id"])

        self.button_by_key(app, f"delete_conv_{conversation['id']}").click().run()
        self.assertIsNotNone(get_conversation(conversation["id"], path=self.db_path))
        self.assertTrue(any("不能删除" in str(item.value) for item in app.warning))

        self.button_by_key(app, "clear_conversation").click().run()
        self.assertEqual(len(list_messages(conversation["id"], path=self.db_path)), 1)
        self.assertTrue(any("不能清空" in str(item.value) for item in app.warning))

        rename_conversation(conversation["id"], "允许重命名", path=self.db_path)
        app.run()
        self.assertEqual(get_conversation(conversation["id"], path=self.db_path)["title"], "允许重命名")

    def test_learning_analysis_exists_only_after_completed_assistant(self) -> None:
        pending_conv, pending = self._enqueue("学习分析 pending")
        pending_app = self.run_app(pending_conv["id"])
        self.assertFalse(any(button.key.startswith("analyze_memory_") for button in pending_app.button))

        execute_generation_job(
            pending["generation_job_id"],
            db_path=self.db_path,
            agent_func=lambda question, **kwargs: _agent_result(),
        )
        pending_app.run()
        self.assertTrue(any(button.key.startswith("analyze_memory_") for button in pending_app.button))
        self.assertEqual(len(get_agent_runs(pending_conv["id"], path=self.db_path)), 1)

    def test_confirmed_image_context_is_enqueued_without_raw_image_data(self) -> None:
        app = AppTest.from_file("app.py")
        app.session_state["pending_chat_submission"] = {
            "text": "求图中电流",
            "pasted_images": [],
            "attached_images": [
                {
                    "filename": "circuit.png",
                    "mime_type": "image/png",
                    "bytes": _png_bytes(),
                }
            ],
        }
        with patch(
            "src.vision.batch.analyze_uploaded_image",
            return_value=_vision_result(),
        ):
            app.run()

        conversation_id = app.session_state["current_conversation_id"]
        job = list_generation_jobs(conversation_id, path=self.db_path)[0]
        self.assertTrue(job["payload"]["image_context_available"])
        self.assertIn("电压为 12 V", job["payload"]["image_context"])
        serialized = json.dumps(job, ensure_ascii=False)
        self.assertNotIn("data:image", serialized)
        self.assertNotIn("base64", serialized.lower())
        self.assertNotIn("bytes", serialized)


if __name__ == "__main__":
    unittest.main()
