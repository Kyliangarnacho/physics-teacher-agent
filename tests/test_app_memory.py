"""Stage 10 长期记忆页面交互 AppTest 测试（无真实 API）。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
import streamlit as st

from src.memory import normalize_memory_content
from src.memory.retrieval import retrieve_relevant_memories
from src.storage import (
    create_conversation,
    initialize_database,
    insert_or_merge_memory,
    list_memories,
)
from tests.app_task_manager import ImmediateGenerationTaskManager


def fake_agent_result(
    answer: str = "测试回答",
    status: str = "completed",
    tool_error: bool = False,
) -> dict:
    return {
        "answer": answer,
        "analysis": {},
        "route": {
            "teaching_mode": "solve",
            "use_rag": False,
            "use_tools": False,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": (
            [{"name": "calculate_ohms_law", "status": "error", "error": "参数错误"}]
            if tool_error
            else []
        ),
        "tool_model_requests": 0,
        "trace": {
            "status": status,
            "total_model_requests": 2,
            "total_duration_ms": 10.0,
            "run_id": "run-fixed",
            "steps": [],
        },
    }


def extractor_payload(
    memory_type: str = "weakness",
    topic: str = "机械能守恒",
    content: str = "误认为物体下落机械能一定减小",
) -> dict:
    return {
        "memory_type": memory_type,
        "topic": topic,
        "content": content,
        "normalized_content": content,
        "confidence": 0.8,
        "evidence_summary": "单次出现",
    }


def extractor_fake(payloads):
    def fake(user_question, assistant_answer, explicit_user_feedback, conversation_history):
        return json.dumps({"candidates": payloads}, ensure_ascii=False)

    return fake


class AppMemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        st.cache_resource.clear()
        self._manager_patch = patch(
            "src.tasks.GenerationTaskManager",
            ImmediateGenerationTaskManager,
        )
        self._manager_patch.start()
        self.addCleanup(self._manager_patch.stop)
        self.addCleanup(st.cache_resource.clear)
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        os.environ["PHYSICS_AGENT_DB_PATH"] = self.db_path
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        os.environ.pop("PHYSICS_AGENT_DB_PATH", None)

    def run_app(self, *, current_conversation_id=None) -> AppTest:
        app = AppTest.from_file("app.py")
        if current_conversation_id is not None:
            app.session_state["current_conversation_id"] = current_conversation_id
        app.run()
        return app

    def button_by_key(self, app: AppTest, key: str):
        return next(item for item in app.button if item.key == key)

    def has_button(self, app: AppTest, key_prefix: str) -> bool:
        return any(item.key.startswith(key_prefix) for item in app.button)

    def agent_patch(self, result=None, results=None):
        if results is None:
            results = [result if result is not None else fake_agent_result()]
        calls = []

        def fake_agent(question: str, **kwargs):
            calls.append(question)
            if len(results) > 1:
                return results.pop(0)
            return results[0]

        self._agent_calls = calls
        return patch(
            "src.conversation.service._default_run_agent",
            side_effect=fake_agent,
        )

    def submit_turn(self, app: AppTest, text: str) -> None:
        app.chat_input[0].set_value(text).run()

    def extract_turn(self, app: AppTest, payloads) -> None:
        with patch(
            "src.memory.extractor._call_extractor_model",
            side_effect=extractor_fake(payloads),
        ):
            analyze = next(
                item
                for item in app.button
                if item.key.startswith("analyze_memory_")
            )
            analyze.click().run()
            for _ in range(20):
                if self.has_button(app, "confirm_memory_"):
                    break
                app.run()

    def test_normal_answer_shows_analyze_button(self) -> None:
        with self.agent_patch():
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")

        self.assertTrue(self.has_button(app, "analyze_memory_"))
        self.assertEqual(len(app.exception), 0)

    def test_blocked_answer_has_no_analyze_button(self) -> None:
        with self.agent_patch(
            result=fake_agent_result(
                answer="请补充题图，或完整描述图中的物体、连接关系和已知信息。",
                status="blocked",
            )
        ):
            app = self.run_app()
            self.submit_turn(app, "图中电压表读数是多少？")

        self.assertFalse(self.has_button(app, "analyze_memory_"))

    def test_failed_run_has_no_analyze_button(self) -> None:
        with self.agent_patch(result=fake_agent_result(status="failed")):
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")

        self.assertFalse(self.has_button(app, "analyze_memory_"))

    def test_tool_error_answer_has_no_analyze_button(self) -> None:
        with self.agent_patch(
            result=fake_agent_result(
                answer="工具参数校验失败，请检查必填字段。",
                tool_error=True,
            )
        ):
            app = self.run_app()
            self.submit_turn(app, "电压 12V 求电流。")

        self.assertFalse(self.has_button(app, "analyze_memory_"))

    def test_no_click_does_not_call_extractor(self) -> None:
        with (
            self.agent_patch(),
            patch(
                "src.memory.extractor._call_extractor_model"
            ) as extractor,
        ):
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")

        self.assertEqual(extractor.call_count, 0)

    def test_single_click_calls_extractor_once(self) -> None:
        with (
            self.agent_patch(),
            patch(
                "src.memory.extractor._call_extractor_model",
                side_effect=extractor_fake([extractor_payload()]),
            ) as extractor,
        ):
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")
            analyze = next(
                item
                for item in app.button
                if item.key.startswith("analyze_memory_")
            )
            analyze.click().run()
            app.run()

        self.assertEqual(extractor.call_count, 1)

    def test_extraction_does_not_write_database(self) -> None:
        with self.agent_patch():
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")
            self.extract_turn(app, [extractor_payload()])

        self.assertEqual(list_memories(), [])

    def test_confirm_writes_memory_with_sources(self) -> None:
        with self.agent_patch():
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")
            self.extract_turn(app, [extractor_payload()])
            conversation_id = app.session_state["current_conversation_id"]

            self.button_by_key(app, self._confirm_key(app)).click().run()

        memories = list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["confirmed"], 1)
        self.assertEqual(memories[0]["active"], 1)
        self.assertEqual(memories[0]["source_conversation_id"], conversation_id)
        self.assertTrue(
            any("已保存" in str(caption.value) for caption in app.caption)
        )

    def _confirm_key(self, app: AppTest) -> str:
        return next(
            item.key
            for item in app.button
            if item.key.startswith("confirm_memory_")
        )

    def test_ignore_does_not_write_memory(self) -> None:
        with self.agent_patch():
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")
            self.extract_turn(app, [extractor_payload()])

            ignore_key = next(
                item.key
                for item in app.button
                if item.key.startswith("ignore_memory_")
            )
            self.button_by_key(app, ignore_key).click().run()

        self.assertEqual(list_memories(), [])
        entry = next(iter(app.session_state["memory_candidates"].values()))
        self.assertEqual(entry["statuses"], ["ignored"])

    def test_rerun_does_not_reconfirm(self) -> None:
        with self.agent_patch():
            app = self.run_app()
            self.submit_turn(app, "机械能怎样变化？")
            self.extract_turn(app, [extractor_payload()])
            self.button_by_key(app, self._confirm_key(app)).click().run()
            app.run()
            app.run()

        memories = list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["evidence_count"], 1)

    def test_repeated_confirm_across_turns_increments_evidence_count(self) -> None:
        with self.agent_patch(
            results=[
                fake_agent_result(answer="第一答"),
                fake_agent_result(answer="第二答"),
            ]
        ):
            app = self.run_app()
            self.submit_turn(app, "第一问")
            self.extract_turn(app, [extractor_payload()])
            self.button_by_key(app, self._confirm_key(app)).click().run()
            self.submit_turn(app, "第二问")
            self.extract_turn(app, [extractor_payload()])
            self.button_by_key(app, self._confirm_key(app)).click().run()

        memories = list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["evidence_count"], 2)

    def test_candidates_isolated_and_cleared_on_switch(self) -> None:
        conv_a = create_conversation("A", path=self.db_path)
        conv_b = create_conversation("B", path=self.db_path)
        with self.agent_patch():
            app = self.run_app(current_conversation_id=conv_a["id"])
            self.submit_turn(app, "机械能怎样变化？")
            self.extract_turn(app, [extractor_payload()])
            self.assertIn(
                "memory_candidates",
                app.session_state,
            )

            self.button_by_key(app, f"select_conv_{conv_b['id']}").click().run()

        self.assertEqual(app.session_state["memory_candidates"], {})
        self.assertFalse(self.has_button(app, "confirm_memory_"))

    def test_learning_archive_shows_only_confirmed(self) -> None:
        confirmed = insert_or_merge_memory(
            {
                "memory_type": "weakness",
                "topic": "已确认主题",
                "content": "已确认内容",
                "normalized_content": "秘密规范化内容",
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
            },
            path=self.db_path,
        )
        insert_or_merge_memory(
            {
                "memory_type": "misconception",
                "topic": "未确认主题",
                "content": "未确认内容",
                "normalized_content": "未确认",
                "confidence": 0.9,
                "confirmed": 0,
                "active": 1,
            },
            path=self.db_path,
        )

        app = self.run_app()

        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("已确认主题", rendered)
        self.assertIn("已确认内容", rendered)
        self.assertIn("证据次数：1", rendered)
        self.assertNotIn("未确认主题", rendered)
        self.assertNotIn("秘密规范化内容", rendered)
        self.assertTrue(
            any(
                item.key == f"memory_deactivate_{confirmed['id']}"
                for item in app.button
            )
        )

    def test_deactivate_stops_recall(self) -> None:
        stored = insert_or_merge_memory(
            {
                "memory_type": "misconception",
                "topic": "机械能守恒条件",
                "content": "误认为下落机械能一定减小",
                "normalized_content": normalize_memory_content(
                    "误认为下落机械能一定减小"
                ),
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
            },
            path=self.db_path,
        )

        app = self.run_app()
        self.button_by_key(
            app,
            f"memory_deactivate_{stored['id']}",
        ).click().run()

        self.assertEqual(
            retrieve_relevant_memories(
                "物体下落机械能怎样变化？",
                db_path=self.db_path,
            ),
            [],
        )

    def test_delete_memory_affects_only_target(self) -> None:
        first = insert_or_merge_memory(
            {
                "memory_type": "weakness",
                "topic": "第一个",
                "content": "内容一",
                "normalized_content": "内容一",
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
            },
            path=self.db_path,
        )
        second = insert_or_merge_memory(
            {
                "memory_type": "preference",
                "topic": "第二个",
                "content": "内容二",
                "normalized_content": "内容二",
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
            },
            path=self.db_path,
        )

        app = self.run_app()
        self.button_by_key(
            app,
            f"memory_delete_{first['id']}",
        ).click().run()

        memories = list_memories()
        self.assertEqual(len(memories), 1)
        self.assertEqual(memories[0]["id"], second["id"])

    def test_page_hides_normalized_and_sensitive_fields(self) -> None:
        insert_or_merge_memory(
            {
                "memory_type": "weakness",
                "topic": "安全主题",
                "content": "安全内容",
                "normalized_content": "隐藏规范化内容",
                "confidence": 0.9,
                "confirmed": 1,
                "active": 1,
            },
            path=self.db_path,
        )

        app = self.run_app()

        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertNotIn("隐藏规范化内容", rendered)
        self.assertNotIn("data:image", rendered)
        self.assertNotIn("base64", rendered.lower())
        self.assertNotIn("sk-", rendered)


if __name__ == "__main__":
    unittest.main()
