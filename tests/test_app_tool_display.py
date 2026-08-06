"""Streamlit 页面本地工具记录与决策展示测试（SQLite 恢复）。"""

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
from src.tools.registry import ToolExecutionRecord, ToolExecutionStatus


class AppToolDisplayTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "physics_teacher.db")
        os.environ["PHYSICS_AGENT_DB_PATH"] = self.db_path
        initialize_database(self.db_path)

    def tearDown(self) -> None:
        os.environ.pop("PHYSICS_AGENT_DB_PATH", None)

    def seed_assistant(
        self,
        tool_records,
        *,
        tool_model_requests: int = 2,
        use_tools: bool = True,
    ) -> str:
        conversation = create_conversation("展示", path=self.db_path)
        user_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "求电流",
                "model_content": "求电流",
            },
            path=self.db_path,
        )
        assistant_message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "assistant",
                "display_content": "根据欧姆定律，电流为 2 A。",
                "model_content": "根据欧姆定律，电流为 2 A。",
            },
            path=self.db_path,
        )
        insert_agent_run(
            {
                "conversation_id": conversation["id"],
                "user_message_id": user_message["id"],
                "assistant_message_id": assistant_message["id"],
                "status": "completed",
                "teaching_mode": "solve",
                "use_rag": False,
                "use_tools": use_tools,
                "total_model_requests": tool_model_requests,
                "total_duration_ms": 10.0,
                "tool_records_json": tool_records,
                "trace_json": {
                    "run_id": "run-tool",
                    "status": "completed",
                    "total_model_requests": tool_model_requests,
                    "total_duration_ms": 10.0,
                    "steps": [],
                },
            },
            path=self.db_path,
        )
        return conversation["id"]

    def run_app(self, conversation_id: str) -> AppTest:
        app = AppTest.from_file("app.py")
        app.session_state["current_conversation_id"] = conversation_id
        app.run()
        return app

    def _status_labels(self, app: AppTest) -> list[str]:
        return [element.label for element in app.status]

    def _visible_text(self, app: AppTest) -> str:
        values: list[str] = []
        for collection_name in ("markdown", "caption", "error", "info"):
            for element in getattr(app, collection_name):
                values.append(str(element.value))
        return "\n".join(values)

    def test_dict_tool_record_and_route_decision_are_visible(self) -> None:
        record = {
            "tool_call_id": "call-1",
            "name": "calculate_ohms_law",
            "arguments": {"voltage_v": 12, "resistance_ohm": 6},
            "normalized_fields": ["voltage_v", "resistance_ohm"],
            "status": "success",
            "result": {
                "formula": "I = U / R",
                "raw_result": "2",
                "display_value": "2",
                "unit": "A",
            },
            "error": None,
        }

        app = self.run_app(self.seed_assistant([record]))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("本地计算工具记录", self._status_labels(app))
        rendered = self._visible_text(app)
        self.assertIn("calculate_ohms_law", rendered)
        self.assertIn("success", rendered)
        self.assertIn("2 A", rendered)
        self.assertIn("Tool Client 内部模型请求数：2", rendered)
        self.assertIn("error：** 无", rendered)
        self.assertIn("Router 启用本地工具：** 是", rendered)

    def test_pydantic_tool_record_is_supported(self) -> None:
        record = ToolExecutionRecord(
            tool_call_id="call-2",
            name="calculate_ohms_law",
            arguments={"voltage_v": 12, "resistance_ohm": 6},
            normalized_fields=[],
            status=ToolExecutionStatus.SUCCESS,
            result={
                "formula": "I = U / R",
                "raw_result": "2",
                "display_value": "2",
                "unit": "A",
            },
            error=None,
        )

        app = self.run_app(
            self.seed_assistant([record.model_dump(mode="json")])
        )

        self.assertEqual(len(app.exception), 0)
        self.assertIn("本地计算工具记录", self._status_labels(app))
        self.assertIn("2 A", self._visible_text(app))

    def test_empty_tool_records_do_not_render_tool_expander(self) -> None:
        app = self.run_app(
            self.seed_assistant([], tool_model_requests=0)
        )

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("本地计算工具记录", self._status_labels(app))
        self.assertNotIn("Tool Client 内部模型请求数", self._visible_text(app))


if __name__ == "__main__":
    unittest.main()
