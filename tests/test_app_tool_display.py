"""Streamlit 页面本地工具记录展示测试。"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest

from src.tools.registry import ToolExecutionRecord, ToolExecutionStatus


def _assistant_message(tool_records: list[object]) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "根据欧姆定律，电流为 2 A。",
        "sources": [],
        "analysis": {
            "teaching_mode": "solve",
            "physics_topic": "欧姆定律",
            "question_type": "计算题",
            "calculation_required": True,
            "short_reason": "需要根据电压和电阻计算电流。",
        },
        "route": {
            "teaching_mode": "solve",
            "use_rag": False,
            "use_tools": True,
            "should_answer": True,
        },
        "analysis_fallback": False,
        "tool_records": tool_records,
        "tool_model_requests": 2,
    }


def _run_app(message: dict[str, object]) -> AppTest:
    app = AppTest.from_file("app.py")
    app.session_state["messages"] = [message]
    app.run()
    return app


def _status_labels(app: AppTest) -> list[str]:
    """Streamlit 1.60 在 AppTest 中以 status 节点表示 expander。"""
    return [element.label for element in app.status]


def _visible_text(app: AppTest) -> str:
    values: list[str] = []
    for collection_name in ("markdown", "caption", "error", "info"):
        for element in getattr(app, collection_name):
            values.append(str(element.value))
    return "\n".join(values)


class TestAppToolDisplay(unittest.TestCase):
    def test_dict_tool_record_and_agent_decision_are_visible(self) -> None:
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

        app = _run_app(_assistant_message([record]))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("本地计算工具记录", _status_labels(app))
        rendered = _visible_text(app)
        self.assertIn("calculate_ohms_law", rendered)
        self.assertIn("success", rendered)
        self.assertIn("2 A", rendered)
        self.assertIn("Tool Client 内部模型请求数：2", rendered)
        self.assertIn("error：** 无", rendered)
        self.assertIn("Analyzer 判断需要计算：** 是", rendered)
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

        app = _run_app(_assistant_message([record]))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("本地计算工具记录", _status_labels(app))
        self.assertIn("2 A", _visible_text(app))

    def test_empty_tool_records_do_not_render_tool_expander(self) -> None:
        message = _assistant_message([])
        message["tool_model_requests"] = 0

        app = _run_app(message)

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("本地计算工具记录", _status_labels(app))
        self.assertNotIn("Tool Client 内部模型请求数", _visible_text(app))


if __name__ == "__main__":
    unittest.main()
