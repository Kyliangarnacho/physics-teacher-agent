"""Agent Trace 数据模型的纯本地单元测试。"""

from __future__ import annotations

import json
import unittest

from pydantic import ValidationError

from src.schemas import (
    AgentRunTrace,
    RunStatus,
    StepStatus,
    StepTrace,
    TeachingMode,
)


def valid_step_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "name": "analyzer",
        "status": "success",
        "attempts": 1,
        "duration_ms": 12.5,
        "model_requests": 1,
    }
    data.update(overrides)
    return data


def valid_run_data(**overrides: object) -> dict[str, object]:
    data: dict[str, object] = {
        "run_id": "run-001",
        "case_id": "case-001",
        "started_at": "2026-08-03T08:00:00+08:00",
        "finished_at": "2026-08-03T08:00:01+08:00",
        "total_duration_ms": 1000.0,
        "status": "completed",
        "total_model_requests": 2,
        "rag_searches": 0,
        "tool_executions": 0,
        "analysis_fallback": False,
        "teaching_mode": "explain",
        "use_rag": False,
        "use_tools": False,
    }
    data.update(overrides)
    return data


class TraceEnumTests(unittest.TestCase):
    def test_status_enums_have_expected_values(self) -> None:
        self.assertEqual(
            [status.value for status in RunStatus],
            ["completed", "completed_with_fallback", "blocked", "failed"],
        )
        self.assertEqual(
            [status.value for status in StepStatus],
            ["success", "retry_success", "skipped", "error"],
        )


class StepTraceTests(unittest.TestCase):
    def test_accepts_normal_success_step(self) -> None:
        trace = StepTrace(**valid_step_data())

        self.assertEqual(trace.status, StepStatus.SUCCESS)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 1)

    def test_accepts_retry_success_with_two_attempts(self) -> None:
        trace = StepTrace(
            **valid_step_data(status="retry_success", attempts=2)
        )

        self.assertEqual(trace.status, StepStatus.RETRY_SUCCESS)
        self.assertEqual(trace.attempts, 2)

    def test_accepts_error_with_both_error_fields(self) -> None:
        trace = StepTrace(
            **valid_step_data(
                status="error",
                error_type="network_error",
                error_message="分析请求失败。",
            )
        )

        self.assertEqual(trace.status, StepStatus.ERROR)
        self.assertEqual(trace.error_type, "network_error")

    def test_accepts_skipped_with_zero_attempts_and_requests(self) -> None:
        trace = StepTrace(
            **valid_step_data(
                status="skipped",
                attempts=0,
                model_requests=0,
            )
        )

        self.assertEqual(trace.status, StepStatus.SKIPPED)

    def test_rejects_attempt_counts_incompatible_with_status(self) -> None:
        invalid_cases = (
            {"status": "success", "attempts": 0},
            {"status": "error", "attempts": 0,
             "error_type": "api_error", "error_message": "失败。"},
            {"status": "retry_success", "attempts": 1},
            {"status": "retry_success", "attempts": 3},
            {"status": "skipped", "attempts": 1, "model_requests": 0},
            {"status": "skipped", "attempts": 0, "model_requests": 1},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValidationError):
                    StepTrace(**valid_step_data(**overrides))

    def test_rejects_negative_counts_and_duration(self) -> None:
        for field_name in ("attempts", "duration_ms", "model_requests"):
            with self.subTest(field_name=field_name):
                with self.assertRaises(ValidationError):
                    StepTrace(**valid_step_data(**{field_name: -1}))

    def test_error_requires_both_error_fields(self) -> None:
        invalid_cases = (
            {"status": "error", "error_type": "api_error"},
            {"status": "error", "error_message": "请求失败。"},
            {"status": "error"},
        )
        for overrides in invalid_cases:
            with self.subTest(overrides=overrides):
                with self.assertRaises(ValidationError):
                    StepTrace(**valid_step_data(**overrides))

    def test_non_error_status_rejects_final_error_fields(self) -> None:
        with self.assertRaises(ValidationError):
            StepTrace(
                **valid_step_data(
                    error_type="unexpected",
                    error_message="不应存在。",
                )
            )

    def test_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            StepTrace(**valid_step_data(unexpected=True))

    def test_metadata_defaults_are_not_shared(self) -> None:
        first = StepTrace(**valid_step_data())
        second = StepTrace(**valid_step_data(name="router"))

        first.metadata["route"] = "tool"

        self.assertEqual(second.metadata, {})


class AgentRunTraceTests(unittest.TestCase):
    def test_steps_defaults_are_not_shared(self) -> None:
        first = AgentRunTrace(**valid_run_data())
        second = AgentRunTrace(**valid_run_data(run_id="run-002"))

        first.steps.append(StepTrace(**valid_step_data()))

        self.assertEqual(second.steps, [])

    def test_model_dump_is_json_serializable(self) -> None:
        trace = AgentRunTrace(
            **valid_run_data(
                status="completed_with_fallback",
                analysis_fallback=True,
                teaching_mode="solve",
                steps=[StepTrace(**valid_step_data())],
            )
        )

        dumped = trace.model_dump(mode="json")

        self.assertEqual(dumped["status"], "completed_with_fallback")
        self.assertEqual(dumped["teaching_mode"], TeachingMode.SOLVE.value)
        self.assertEqual(dumped["steps"][0]["status"], "success")
        self.assertIsInstance(json.dumps(dumped, ensure_ascii=False), str)

    def test_rejects_extra_fields(self) -> None:
        with self.assertRaises(ValidationError):
            AgentRunTrace(**valid_run_data(unexpected=True))


if __name__ == "__main__":
    unittest.main()
