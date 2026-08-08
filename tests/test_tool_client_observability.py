"""Tool Client 内部步骤 Trace、有限重试与幂等边界测试。"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.tool_client import answer_with_tools
from src.tools.registry import execute_tool_call
from tests.test_tool_client import (
    FakeCompletion,
    completion_response,
    tool_call,
)


def trace_by_name(result: dict, name: str) -> dict:
    return next(trace for trace in result["step_traces"] if trace["name"] == name)


def successful_selection():
    return completion_response(tool_calls=[tool_call()])


def successful_answer():
    return completion_response(
        tool_calls=None,
        content="根据欧姆定律，电流为 2 A。",
    )


class ToolClientObservabilityTests(unittest.TestCase):
    def test_normal_path_has_three_ordered_success_steps(self) -> None:
        result = answer_with_tools(
            "求电流。",
            completion_func=FakeCompletion(
                [successful_selection(), successful_answer()]
            ),
        )

        self.assertEqual(
            [trace["name"] for trace in result["step_traces"]],
            ["tool_selection", "tool_execution", "tool_result_answer"],
        )
        self.assertEqual(
            [trace["status"] for trace in result["step_traces"]],
            ["success", "success", "success"],
        )
        self.assertEqual(
            [trace["model_requests"] for trace in result["step_traces"]],
            [1, 0, 1],
        )
        self.assertEqual(result["model_requests"], 2)

    def test_no_tool_needed_uses_traced_model_only_answer(self) -> None:
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[], content="direct answer"),
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(
            [trace["name"] for trace in result["step_traces"]],
            ["tool_selection", "tool_execution", "tool_result_answer"],
        )
        self.assertEqual(trace_by_name(result, "tool_selection")["status"], "success")
        self.assertEqual(
            trace_by_name(result, "tool_selection")["metadata"]["decision"],
            "no_tool_needed",
        )
        self.assertEqual(trace_by_name(result, "tool_execution")["status"], "skipped")
        final_trace = trace_by_name(result, "tool_result_answer")
        self.assertEqual(final_trace["status"], "skipped")
        self.assertEqual(final_trace["metadata"]["answer_strategy"], "model_only")
        self.assertEqual(result["model_requests"], 1)

    def test_tool_selection_first_failure_then_success_is_retried(self) -> None:
        fake = FakeCompletion(
            [
                TimeoutError("temporary selection failure"),
                successful_selection(),
                successful_answer(),
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        selection = trace_by_name(result, "tool_selection")
        self.assertEqual(selection["status"], "retry_success")
        self.assertEqual(selection["attempts"], 2)
        self.assertEqual(selection["model_requests"], 2)
        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(len(fake.calls), 3)

    def test_tool_result_retry_reuses_single_local_execution(self) -> None:
        fake = FakeCompletion(
            [
                successful_selection(),
                TimeoutError("temporary result failure"),
                successful_answer(),
            ]
        )

        with patch(
            "src.tool_client.execute_tool_call",
            wraps=execute_tool_call,
        ) as mocked_execute:
            result = answer_with_tools("求电流。", completion_func=fake)

        result_trace = trace_by_name(result, "tool_result_answer")
        self.assertEqual(result_trace["status"], "retry_success")
        self.assertEqual(result_trace["attempts"], 2)
        self.assertEqual(result_trace["model_requests"], 2)
        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(result["tool_records"][0]["result"]["display_value"], "2")
        mocked_execute.assert_called_once()

        first_result_messages = fake.calls[1]["messages"]
        retried_result_messages = fake.calls[2]["messages"]
        self.assertEqual(first_result_messages, retried_result_messages)
        self.assertEqual(
            first_result_messages[-1]["tool_call_id"],
            "call-ohms",
        )

    def test_selection_api_failure_records_two_attempts(self) -> None:
        result = answer_with_tools(
            "求电流。",
            completion_func=FakeCompletion(
                [TimeoutError("first"), TimeoutError("second")]
            ),
        )

        selection = trace_by_name(result, "tool_selection")
        self.assertEqual(selection["status"], "error")
        self.assertEqual(selection["attempts"], 2)
        self.assertEqual(selection["model_requests"], 2)
        self.assertEqual(selection["error_type"], "tool_selection_api")
        self.assertEqual(trace_by_name(result, "tool_execution")["status"], "skipped")

    def test_final_result_api_failure_does_not_repeat_tool(self) -> None:
        fake = FakeCompletion(
            [
                successful_selection(),
                TimeoutError("first"),
                TimeoutError("second"),
            ]
        )

        with patch(
            "src.tool_client.execute_tool_call",
            wraps=execute_tool_call,
        ) as mocked_execute:
            result = answer_with_tools("求电流。", completion_func=fake)

        result_trace = trace_by_name(result, "tool_result_answer")
        mocked_execute.assert_called_once()
        self.assertEqual(result_trace["status"], "error")
        self.assertEqual(result_trace["attempts"], 2)
        self.assertEqual(result_trace["model_requests"], 2)
        self.assertEqual(result_trace["error_type"], "tool_result_api")
        self.assertEqual(result["model_requests"], 3)

    def test_empty_final_answer_is_classified_without_retry(self) -> None:
        fake = FakeCompletion(
            [
                successful_selection(),
                completion_response(tool_calls=None, content="   "),
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        result_trace = trace_by_name(result, "tool_result_answer")
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(result_trace["status"], "error")
        self.assertEqual(result_trace["attempts"], 1)
        self.assertEqual(result_trace["model_requests"], 1)
        self.assertEqual(result_trace["error_type"], "empty_answer")


if __name__ == "__main__":
    unittest.main()
