"""Streamlit Agent 运行轨迹展示的无 API 测试。"""

from __future__ import annotations

import unittest

from streamlit.testing.v1 import AppTest


def run_app(message: dict[str, object]) -> AppTest:
    app = AppTest.from_file("app.py")
    app.session_state["messages"] = [message]
    app.run()
    return app


def status_labels(app: AppTest) -> list[str]:
    return [element.label for element in app.status]


def visible_text(app: AppTest) -> str:
    values: list[str] = []
    for collection_name in ("markdown", "caption", "error", "info"):
        for element in getattr(app, collection_name):
            values.append(str(element.value))
    return "\n".join(values)


def assistant_message(trace: dict | None) -> dict[str, object]:
    return {
        "role": "assistant",
        "content": "测试回答",
        "sources": [],
        "analysis": {},
        "route": {},
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": trace,
    }


class AppTraceDisplayTests(unittest.TestCase):
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

        app = run_app(assistant_message(trace))

        self.assertEqual(len(app.exception), 0)
        self.assertIn("Agent 运行轨迹", status_labels(app))
        rendered = visible_text(app)
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
        message = assistant_message(None)
        del message["trace"]

        app = run_app(message)

        self.assertEqual(len(app.exception), 0)
        self.assertNotIn("Agent 运行轨迹", status_labels(app))


if __name__ == "__main__":
    unittest.main()
