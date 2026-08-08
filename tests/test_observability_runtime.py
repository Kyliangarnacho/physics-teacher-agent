"""RunTraceBuilder 的纯本地运行时测试。"""

from __future__ import annotations

import json
import unittest
from datetime import datetime, timedelta, timezone

from src.observability import (
    ErrorType,
    RunTraceBuilder,
    StepTimer,
    create_skipped_step,
)
from src.schemas import RunStatus, StepStatus, StepTrace, TeachingMode


START = datetime(2026, 8, 3, 8, 0, 0, tzinfo=timezone.utc)
FINISH = START + timedelta(seconds=1)


def step(name: str, model_requests: int = 0) -> StepTrace:
    return StepTrace(
        name=name,
        status="success",
        attempts=1,
        duration_ms=10,
        model_requests=model_requests,
    )


def finish_builder(
    builder: RunTraceBuilder,
    **overrides: object,
):
    arguments: dict[str, object] = {
        "status": RunStatus.COMPLETED,
        "analysis_fallback": False,
        "teaching_mode": TeachingMode.SOLVE,
        "use_rag": False,
        "use_tools": False,
        "rag_searches": 0,
        "tool_executions": 0,
    }
    arguments.update(overrides)
    return builder.finish(**arguments)


def deterministic_builder(
    *,
    run_id: str = "run-fixed",
    case_id: str | None = None,
    monotonic_values: tuple[float, float] = (10.0, 11.0),
) -> RunTraceBuilder:
    utc_values = iter((START, FINISH))
    clock_values = iter(monotonic_values)
    return RunTraceBuilder(
        case_id=case_id,
        run_id=run_id,
        utc_now_func=lambda: next(utc_values),
        monotonic_func=lambda: next(clock_values),
    )


class RunTraceBuilderTests(unittest.TestCase):
    def test_generated_run_ids_are_non_empty_and_unique(self) -> None:
        first = finish_builder(RunTraceBuilder())
        second = finish_builder(RunTraceBuilder())

        self.assertTrue(first.run_id)
        self.assertTrue(second.run_id)
        self.assertNotEqual(first.run_id, second.run_id)

    def test_timestamps_are_parseable_timezone_aware_iso_strings(self) -> None:
        trace = finish_builder(deterministic_builder())

        started = datetime.fromisoformat(trace.started_at)
        finished = datetime.fromisoformat(trace.finished_at)

        self.assertIsNotNone(started.utcoffset())
        self.assertIsNotNone(finished.utcoffset())
        self.assertEqual(started.utcoffset(), timedelta(0))
        self.assertEqual(finished.utcoffset(), timedelta(0))

    def test_steps_keep_insertion_order(self) -> None:
        builder = deterministic_builder()
        builder.add_step(step("analyzer", 1))
        builder.add_step(step("router"))
        builder.add_step(step("answer", 1))

        trace = finish_builder(builder)

        self.assertEqual(
            [item.name for item in trace.steps],
            ["analyzer", "router", "answer"],
        )

    def test_total_model_requests_are_summed_from_steps(self) -> None:
        builder = deterministic_builder()
        builder.add_step(step("analyzer", 1))
        builder.add_step(step("retrieval", 0))
        builder.add_step(step("tool_client", 2))

        trace = finish_builder(builder)

        self.assertEqual(trace.total_model_requests, 3)

    def test_total_duration_uses_monotonic_clock(self) -> None:
        trace = finish_builder(
            deterministic_builder(monotonic_values=(20.0, 20.25))
        )

        self.assertEqual(trace.total_duration_ms, 250.0)

    def test_case_id_and_route_fields_enter_trace(self) -> None:
        trace = finish_builder(
            deterministic_builder(case_id="case-008"),
            status=RunStatus.COMPLETED_WITH_FALLBACK,
            analysis_fallback=True,
            teaching_mode=TeachingMode.DIAGNOSE,
            use_rag=True,
            use_tools=True,
            rag_searches=1,
            tool_executions=1,
        )

        self.assertEqual(trace.case_id, "case-008")
        self.assertEqual(trace.status, RunStatus.COMPLETED_WITH_FALLBACK)
        self.assertTrue(trace.analysis_fallback)
        self.assertEqual(trace.teaching_mode, TeachingMode.DIAGNOSE)
        self.assertTrue(trace.use_rag)
        self.assertTrue(trace.use_tools)
        self.assertEqual(trace.rag_searches, 1)
        self.assertEqual(trace.tool_executions, 1)

    def test_add_step_is_forbidden_after_finish(self) -> None:
        builder = deterministic_builder()
        finish_builder(builder)

        with self.assertRaisesRegex(RuntimeError, "不能继续添加步骤"):
            builder.add_step(step("late"))

    def test_finish_is_forbidden_twice(self) -> None:
        builder = deterministic_builder()
        finish_builder(builder)

        with self.assertRaisesRegex(RuntimeError, "不能重复 finish"):
            finish_builder(builder)

    def test_builders_do_not_share_steps(self) -> None:
        first = deterministic_builder(run_id="run-first")
        second = deterministic_builder(run_id="run-second")
        first.add_step(step("analyzer", 1))

        first_trace = finish_builder(first)
        second_trace = finish_builder(second)

        self.assertEqual(len(first_trace.steps), 1)
        self.assertEqual(second_trace.steps, [])

    def test_result_is_json_serializable(self) -> None:
        builder = deterministic_builder()
        builder.add_step(step("analyzer", 1))

        dumped = finish_builder(builder).model_dump(mode="json")

        self.assertIsInstance(json.dumps(dumped, ensure_ascii=False), str)


class StepTimerTests(unittest.TestCase):
    def test_error_type_contains_all_required_values(self) -> None:
        self.assertEqual(
            {item.value for item in ErrorType},
            {
                "analyzer_api",
                "analyzer_parse",
                "analyzer_schema",
                "router_error",
                "retrieval_error",
                "tool_selection_api",
                "tool_protocol",
                "tool_validation",
                "tool_repair_api",
                "tool_execution",
                "tool_result_api",
                "final_answer_api",
                "empty_answer",
                "unknown",
            },
        )

    def test_timer_calculates_duration_in_milliseconds(self) -> None:
        values = iter((5.0, 5.125))
        timer = StepTimer("analyzer", clock=lambda: next(values))

        trace = timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=1,
        )

        self.assertEqual(trace.duration_ms, 125.0)

    def test_timer_creates_success_trace(self) -> None:
        values = iter((2.0, 2.01))
        timer = StepTimer("router", clock=lambda: next(values))

        trace = timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
            metadata={"use_tools": True},
        )

        self.assertEqual(trace.name, "router")
        self.assertEqual(trace.status, StepStatus.SUCCESS)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 0)
        self.assertEqual(trace.metadata, {"use_tools": True})

    def test_timer_creates_error_trace_with_error_type_value(self) -> None:
        values = iter((3.0, 3.02))
        timer = StepTimer("tool", clock=lambda: next(values))

        trace = timer.finish(
            status=StepStatus.ERROR,
            attempts=1,
            model_requests=0,
            error_type=ErrorType.TOOL_VALIDATION,
            error_message="工具参数校验失败",
        )

        self.assertEqual(trace.error_type, "tool_validation")
        self.assertEqual(trace.error_message, "工具参数校验失败")

    def test_timer_copies_metadata(self) -> None:
        values = iter((4.0, 4.01))
        metadata = {"route": {"use_rag": False}}
        timer = StepTimer("answer", clock=lambda: next(values))
        trace = timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=1,
            metadata=metadata,
        )

        metadata["route"]["use_rag"] = True

        self.assertEqual(trace.metadata, {"route": {"use_rag": False}})

    def test_timer_cannot_finish_twice(self) -> None:
        values = iter((6.0, 6.01))
        timer = StepTimer("answer", clock=lambda: next(values))
        timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=1,
        )

        with self.assertRaisesRegex(RuntimeError, "不能重复 finish"):
            timer.finish(
                status=StepStatus.SUCCESS,
                attempts=1,
                model_requests=1,
            )

    def test_clock_going_backwards_never_creates_negative_duration(self) -> None:
        values = iter((10.0, 9.5))
        timer = StepTimer("retrieval", clock=lambda: next(values))

        trace = timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
        )

        self.assertEqual(trace.duration_ms, 0.0)

    def test_create_skipped_step_has_zero_cost_fields(self) -> None:
        trace = create_skipped_step("retrieval", {"reason": "RAG disabled"})

        self.assertEqual(trace.status, StepStatus.SKIPPED)
        self.assertEqual(trace.attempts, 0)
        self.assertEqual(trace.duration_ms, 0.0)
        self.assertEqual(trace.model_requests, 0)
        self.assertEqual(trace.metadata, {"reason": "RAG disabled"})

    def test_skipped_steps_do_not_share_metadata(self) -> None:
        first = create_skipped_step("retrieval")
        second = create_skipped_step("tool")

        first.metadata["reason"] = "disabled"

        self.assertEqual(second.metadata, {})

    def test_step_timer_result_is_json_serializable(self) -> None:
        values = iter((7.0, 7.01))
        timer = StepTimer("analyzer", clock=lambda: next(values))
        trace = timer.finish(
            status=StepStatus.ERROR,
            attempts=1,
            model_requests=1,
            error_type=ErrorType.ANALYZER_API,
            error_message="分析服务调用失败",
            metadata={"fallback": True},
        )

        dumped = trace.model_dump(mode="json")

        self.assertIsInstance(json.dumps(dumped, ensure_ascii=False), str)


if __name__ == "__main__":
    unittest.main()
