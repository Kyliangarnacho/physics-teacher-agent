"""Streamlit Agent 运行轨迹展示测试（SQLite 恢复，无 API）。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from src.storage import (
    create_conversation,
    initialize_database,
    insert_agent_run,
    insert_message,
)


class AppTraceDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        os.environ["PHYSICS_AGENT_DB_PATH"] = self.db_path
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        os.environ.pop("PHYSICS_AGENT_DB_PATH", None)

    def seed_assistant(self, trace: dict | None) -> str:
        conversation = create_conversation("轨迹", path=self.db_path)
        user_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "题目",
                "model_content": "题目",
            },
            path=self.db_path,
        )
        assistant_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "assistant",
                "display_content": "测试回答",
                "model_content": "测试回答",
            },
            path=self.db_path,
        )
        insert_agent_run(
            {
                "conversation_id": conversation["id"],
                "user_message_id": user_message["id"],
                "assistant_message_id": assistant_message["id"],
                "status": "completed_with_fallback",
                "teaching_mode": "solve",
                "use_rag": False,
                "use_tools": False,
                "total_model_requests": 3,
                "total_duration_ms": 125.5,
                "trace_json": trace,
            },
            path=self.db_path,
        )
        return conversation["id"]

    def run_app(self, conversation_id: str) -> AppTest:
        app = AppTest.from_file("app.py")
        app.session_state["current_conversation_id"] = conversation_id
        app.run()
        return app

    def status_labels(self, app: AppTest) -> list[str]:
        return [element.label for element in app.status]

    def visible_text(self, app: AppTest) -> str:
        values: list[str] = []
        for collection_name in ("markdown", "caption", "error", "info"):
            for element in getattr(app, collection_name):
                values.append(str(element.value))
        return "\n".join(values)

    def test_trace_summary_steps_and_safe_error_are_visible(self) -> None:
        trace = {
            "run_id": "run-stage08-test",
            "status": "completed_with_fallback",
            "total_duration_ms": 125.5,
            "total_model_requests": 3,
            "rag_searches": 1,
            "tool_executions": 1,
            "analysis_fallback": True,
            "steps": [
                {
                    "name": "analyzer",
                    "status": "error",
                    "attempts": 1,
                    "duration_ms": 25.5,
                    "model_requests": 1,
                    "error_type": "analyzer_parse",
                    "error_message": "问题分析结果不是有效 JSON。",
                    "metadata": {"hidden": "不应展示"},
                },
                {
                    "name": "final_answer",
                    "status": "success",
                    "attempts": 1,
                    "duration_ms": 100,
                    "model_requests": 1,
                    "error_type": None,
                    "error_message": None,
                    "metadata": {},
                },
            ],
        }

        app = self.run_app(self.seed_assistant(trace))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Agent 运行轨迹", self.status_labels(app))
        rendered = self.visible_text(app)
        for expected in (
            "run-stage08-test",
            "completed_with_fallback",
            "total_model_requests：** `3`",
            "rag_searches：** `1`",
            "tool_executions：** `1`",
            "analyzer_parse",
            "问题分析结果不是有效 JSON。",
            "final_answer",
        ):
            self.assertIn(expected, rendered)
        self.assertNotIn("不应展示", rendered)

    def test_old_message_without_trace_has_no_trace_expander(self) -> None:
        app = self.run_app(self.seed_assistant(None))

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Agent 运行轨迹", self.status_labels(app))


if __name__ == "__main__":
    unittest.main()
