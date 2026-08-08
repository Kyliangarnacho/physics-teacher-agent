"""Stage 10 Streamlit SQLite 会话接入页面测试（无真实 API）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from streamlit.testing.v1 import AppTest

from src.storage import (
    create_conversation,
    get_agent_runs,
    get_conversation,
    get_conversation_state,
    initialize_database,
    insert_agent_run,
    insert_message,
    list_conversations,
    list_messages,
)
from src.vision.schemas import ImageQuestionExtraction


def fake_agent_result(
    answer: str = "测试回答",
    status: str = "completed",
    teaching_mode: str = "solve",
) -> dict:
    return {
        "answer": answer,
        "analysis": {"teaching_mode": teaching_mode},
        "route": {
            "teaching_mode": teaching_mode,
            "use_rag": False,
            "use_tools": False,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": {
            "status": status,
            "total_model_requests": 2,
            "total_duration_ms": 10.0,
            "rag_searches": 0,
            "tool_executions": 0,
            "run_id": "run-fixed",
            "steps": [],
        },
    }


def png_bytes() -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def image_item(name: str, data: bytes) -> dict[str, object]:
    return {"filename": name, "mime_type": "image/png", "bytes": data}


def service_result(name: str = "image") -> dict:
    extraction = ImageQuestionExtraction(
        status="complete",
        image_type="circuit",
        extracted_text=f"{name}：电压为 12 V，电阻为 6 Ω。",
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
        "vision_run_id": f"vision-{name}",
        "vision_step_traces": [],
        "model_requests": 1,
    }


class AppConversationTests(unittest.TestCase):
    def setUp(self) -> None:
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

    def patch_agent(self, calls=None, error=None, result=None):
        def fake_agent(question: str, **kwargs):
            if calls is not None:
                calls.append((question, kwargs))
            if error is not None:
                raise error
            return result if result is not None else fake_agent_result()

        return patch(
            "src.conversation.service._default_run_agent",
            side_effect=fake_agent,
        )

    def submit(self, app: AppTest, text: str) -> None:
        app.chat_input[0].set_value(text).run()

    def test_first_startup_creates_db_and_first_conversation(self) -> None:
        app = self.run_app()

        self.assertEqual(len(app.exception), 0)
        self.assertIn("current_conversation_id", app.session_state)
        conversations = list_conversations()
        self.assertEqual(len(conversations), 1)
        self.assertEqual(
            conversations[0]["id"],
            app.session_state["current_conversation_id"],
        )
        self.assertEqual(list_messages(conversations[0]["id"]), [])

    def test_conversation_list_sorted_and_new_conversation(self) -> None:
        first = create_conversation("第一个", path=self.db_path)
        second = create_conversation("第二个", path=self.db_path)
        app = self.run_app(current_conversation_id=first["id"])

        cached_titles = [
            str(item.get("title", ""))
            for item in app.session_state["conversation_cache"]
        ]
        self.assertLess(
            cached_titles.index("第二个"),
            cached_titles.index("第一个"),
        )

        self.button_by_key(app, "new_conversation").click().run()

        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(list_conversations()), 3)
        self.assertEqual(
            app.session_state["current_conversation_id"],
            list_conversations()[0]["id"],
        )

    def test_switch_conversation_shows_its_own_history(self) -> None:
        conv_a = create_conversation("A", path=self.db_path)
        conv_b = create_conversation("B", path=self.db_path)
        with self.patch_agent():
            app = self.run_app(current_conversation_id=conv_a["id"])
            self.submit(app, "A 的问题")
        self.assertEqual(len(list_messages(conv_a["id"], path=self.db_path)), 2)

        self.button_by_key(app, f"select_conv_{conv_b['id']}").click().run()

        self.assertEqual(app.session_state["current_conversation_id"], conv_b["id"])
        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertNotIn("A 的问题", rendered)
        self.assertEqual(len(list_messages(conv_b["id"], path=self.db_path)), 0)

    def test_conversation_title_editor_replaces_old_rename_controls(self) -> None:
        conv = create_conversation("旧标题", path=self.db_path)
        app = self.run_app(current_conversation_id=conv["id"])

        self.assertFalse(
            any(item.key == f"rename_btn_{conv['id']}" for item in app.button)
        )
        self.assertFalse(
            any(
                item.key == f"rename_input_{conv['id']}"
                for item in app.text_input
            )
        )
        self.assertEqual(len(app.exception), 0)

    def test_delete_current_conversation_selects_remaining(self) -> None:
        conv_a = create_conversation("A", path=self.db_path)
        conv_b = create_conversation("B", path=self.db_path)
        with self.patch_agent():
            app = self.run_app(current_conversation_id=conv_a["id"])
            self.submit(app, "A 的问题")

        self.button_by_key(app, f"delete_conv_{conv_a['id']}").click().run()

        self.assertEqual(app.session_state["current_conversation_id"], conv_b["id"])
        self.assertIsNone(get_conversation(conv_a["id"], path=self.db_path))
        self.assertEqual(len(list_messages(conv_b["id"], path=self.db_path)), 0)
        self.assertEqual(len(app.exception), 0)

    def test_history_restored_on_new_session(self) -> None:
        with self.patch_agent():
            app_one = self.run_app()
            self.submit(app_one, "会持久化的问题")
        conversation_id = app_one.session_state["current_conversation_id"]

        app_two = self.run_app(current_conversation_id=conversation_id)

        rendered = "\n".join(str(item.value) for item in app_two.markdown)
        self.assertIn("会持久化的问题", rendered)
        self.assertIn("测试回答", rendered)
        self.assertEqual(len(app_two.exception), 0)

    def test_page_uses_display_content_not_model_content(self) -> None:
        conv = create_conversation("展示", path=self.db_path)
        insert_message(
            {
                "conversation_id": conv["id"],
                "role": "user",
                "display_content": "显示版问题",
                "model_content": "模型版问题",
            },
            path=self.db_path,
        )
        insert_message(
            {
                "conversation_id": conv["id"],
                "role": "assistant",
                "display_content": "显示版回答",
                "model_content": "模型版回答",
            },
            path=self.db_path,
        )

        app = self.run_app(current_conversation_id=conv["id"])

        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("显示版问题", rendered)
        self.assertIn("显示版回答", rendered)
        self.assertNotIn("模型版问题", rendered)
        self.assertNotIn("模型版回答", rendered)

    def test_user_and_assistant_saved_exactly_once(self) -> None:
        with self.patch_agent():
            app = self.run_app()
            self.submit(app, "不要重复保存")

        conversation_id = app.session_state["current_conversation_id"]
        messages = list_messages(conversation_id, path=self.db_path)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(len(app.exception), 0)

    def test_service_error_keeps_user_visible_and_shows_safe_error(self) -> None:
        with self.patch_agent(error=RuntimeError("boom")):
            app = self.run_app()
            self.submit(app, "会出错的问题")

        conversation_id = app.session_state["current_conversation_id"]
        messages = list_messages(conversation_id, path=self.db_path)
        self.assertEqual([message["role"] for message in messages], ["user"])
        errors = [str(item.value) for item in app.error]
        self.assertTrue(any("回答生成失败" in value for value in errors))
        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("会出错的问题", rendered)
        self.assertEqual(len(app.exception), 0)

    def test_switch_conversation_clears_pending_image_state(self) -> None:
        conv_a = create_conversation("A", path=self.db_path)
        conv_b = create_conversation("B", path=self.db_path)
        app = self.run_app(current_conversation_id=conv_a["id"])
        app.session_state["pending_submission"] = {
            "question": "待确认",
            "batch": {"batch_id": "b1", "images": []},
        }
        app.session_state["paste_image_bridge"] = {"images": []}

        self.button_by_key(app, f"select_conv_{conv_b['id']}").click().run()

        self.assertEqual(app.session_state["current_conversation_id"], conv_b["id"])
        self.assertNotIn("pending_submission", app.session_state)
        self.assertNotIn("paste_image_bridge", app.session_state)
        self.assertNotIn("pending_chat_submission", app.session_state)

    def test_image_context_passed_to_service_and_no_bytes_in_db(self) -> None:
        calls = []
        app = AppTest.from_file("app.py")
        app.session_state["pending_chat_submission"] = {
            "text": "请解答",
            "pasted_images": [],
            "attached_images": [image_item("safe.png", png_bytes())],
        }
        with (
            patch(
                "src.vision.batch.analyze_uploaded_image",
                return_value=service_result(name="safe"),
            ),
            self.patch_agent(calls=calls),
        ):
            app.run()

        self.assertEqual(len(calls), 1)
        self.assertIn("safe.png", calls[0][1].get("image_context", ""))
        conversation_id = app.session_state["current_conversation_id"]
        messages = list_messages(conversation_id, path=self.db_path)
        serialized = str(messages)
        self.assertNotIn("data_url", serialized)
        self.assertNotIn("base64", serialized.lower())
        self.assertNotIn("已确认图片上下文", serialized)
        self.assertNotIn("bytes", serialized)
        user_message = messages[0]
        self.assertEqual(user_message["image_metadata_json"]["image_count"], 1)
        self.assertIn("safe.png", user_message["image_metadata_json"]["image_filenames"])

    def test_clear_current_conversation_does_not_affect_others(self) -> None:
        conv_a = create_conversation("A", path=self.db_path)
        conv_b = create_conversation("B", path=self.db_path)
        with self.patch_agent():
            app = self.run_app(current_conversation_id=conv_a["id"])
            self.submit(app, "A 的问题")
            self.button_by_key(app, f"select_conv_{conv_b['id']}").click().run()
            self.submit(app, "B 的问题")
            self.button_by_key(app, f"select_conv_{conv_a['id']}").click().run()

        self.assertEqual(len(list_messages(conv_a["id"], path=self.db_path)), 2)
        self.button_by_key(app, "clear_conversation").click().run()

        self.assertEqual(list_messages(conv_a["id"], path=self.db_path), [])
        self.assertEqual(len(list_messages(conv_b["id"], path=self.db_path)), 2)
        self.assertIsNotNone(get_conversation(conv_a["id"], path=self.db_path))
        self.assertEqual(len(app.exception), 0)

    def test_old_assistant_details_restored_from_agent_run(self) -> None:
        conv = create_conversation("恢复", path=self.db_path)
        user_message = insert_message(
            {
                "id": "user-1",
                "conversation_id": conv["id"],
                "role": "user",
                "display_content": "题目",
                "model_content": "题目",
            },
            path=self.db_path,
        )
        assistant_message = insert_message(
            {
                "id": "assistant-1",
                "conversation_id": conv["id"],
                "role": "assistant",
                "display_content": "回答",
                "model_content": "回答",
            },
            path=self.db_path,
        )
        insert_agent_run(
            {
                "conversation_id": conv["id"],
                "user_message_id": user_message["id"],
                "assistant_message_id": assistant_message["id"],
                "status": "completed_with_fallback",
                "teaching_mode": "solve",
                "use_rag": True,
                "use_tools": True,
                "total_model_requests": 3,
                "total_duration_ms": 12.5,
                "sources_json": [
                    {
                        "id": "KB-ELEC-001",
                        "topic": "欧姆定律",
                        "source": "课后总结",
                    }
                ],
                "tool_records_json": [
                    {
                        "tool_call_id": "call-1",
                        "name": "calculate_ohms_law",
                        "status": "success",
                        "arguments": {"voltage_v": 12, "resistance_ohm": 6},
                        "result": {"display_value": "2", "unit": "A"},
                    }
                ],
                "analysis_json": {
                    "physics_topic": "电学",
                    "question_type": "计算题",
                    "calculation_required": True,
                    "short_reason": "需要计算电流。",
                },
                "trace_json": {
                    "run_id": "run-restore",
                    "status": "completed_with_fallback",
                    "total_model_requests": 3,
                    "total_duration_ms": 12.5,
                    "rag_searches": 1,
                    "tool_executions": 1,
                    "analysis_fallback": True,
                    "steps": [
                        {
                            "name": "analyzer",
                            "status": "success",
                            "attempts": 1,
                            "duration_ms": 1,
                            "model_requests": 1,
                        },
                        {
                            "name": "tool_selection",
                            "status": "success",
                            "attempts": 1,
                            "duration_ms": 1,
                            "model_requests": 1,
                        },
                        {
                            "name": "tool_result_answer",
                            "status": "success",
                            "attempts": 1,
                            "duration_ms": 1,
                            "model_requests": 1,
                        },
                    ],
                },
            },
            path=self.db_path,
        )

        app = self.run_app(current_conversation_id=conv["id"])

        self.assertEqual(len(app.exception), 0)
        status_labels = [element.label for element in app.status]
        self.assertIn("Agent 运行轨迹", status_labels)
        self.assertIn("本地计算工具记录", status_labels)
        self.assertIn("本地知识库参考来源", status_labels)
        rendered = "\n".join(str(item.value) for item in app.markdown)
        self.assertIn("run-restore", rendered)
        self.assertIn("calculate_ohms_law", rendered)
        self.assertIn("2 A", rendered)
        self.assertIn("电学", rendered)
        self.assertIn("需要计算电流", rendered)
        captions = "\n".join(str(item.value) for item in app.caption)
        self.assertIn("Tool Client 内部模型请求数：2", captions)


if __name__ == "__main__":
    unittest.main()
