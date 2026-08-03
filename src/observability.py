"""Agent 运行 Trace 的独立构建基础工具。"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timezone
from enum import Enum
from time import monotonic
from uuid import uuid4

from src.schemas import (
    AgentRunTrace,
    RunStatus,
    StepStatus,
    StepTrace,
    TeachingMode,
)


UtcNowFunc = Callable[[], datetime]
MonotonicFunc = Callable[[], float]


class ErrorType(str, Enum):
    """Stable error categories used by observability records."""

    ANALYZER_API = "analyzer_api"
    ANALYZER_PARSE = "analyzer_parse"
    ANALYZER_SCHEMA = "analyzer_schema"
    ROUTER_ERROR = "router_error"
    RETRIEVAL_ERROR = "retrieval_error"
    TOOL_SELECTION_API = "tool_selection_api"
    TOOL_PROTOCOL = "tool_protocol"
    TOOL_VALIDATION = "tool_validation"
    TOOL_EXECUTION = "tool_execution"
    TOOL_RESULT_API = "tool_result_api"
    FINAL_ANSWER_API = "final_answer_api"
    EMPTY_ANSWER = "empty_answer"
    UNKNOWN = "unknown"


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_iso(value: datetime) -> str:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("UTC 时间函数必须返回带时区的 datetime。")
    return value.astimezone(timezone.utc).isoformat()


class StepTimer:
    """Measure one step and create a validated ``StepTrace`` once."""

    def __init__(
        self,
        name: str,
        *,
        clock: MonotonicFunc | None = None,
    ) -> None:
        self._name = name
        self._clock = clock if clock is not None else monotonic
        self._started = self._clock()
        self._finished = False

    def finish(
        self,
        *,
        status: StepStatus,
        attempts: int,
        model_requests: int,
        error_type: ErrorType | str | None = None,
        error_message: str | None = None,
        metadata: dict | None = None,
    ) -> StepTrace:
        """Finish timing and return the existing validated trace model."""
        if self._finished:
            raise RuntimeError("StepTimer 已完成，不能重复 finish。")

        duration_ms = max(0.0, (self._clock() - self._started) * 1000)
        stored_error_type = (
            error_type.value if isinstance(error_type, ErrorType) else error_type
        )
        trace = StepTrace(
            name=self._name,
            status=status,
            attempts=attempts,
            duration_ms=duration_ms,
            model_requests=model_requests,
            error_type=stored_error_type,
            error_message=error_message,
            metadata=deepcopy(metadata) if metadata is not None else {},
        )
        self._finished = True
        return trace


def create_skipped_step(
    name: str,
    metadata: dict | None = None,
) -> StepTrace:
    """Create a zero-cost skipped step without starting a timer."""
    return StepTrace(
        name=name,
        status=StepStatus.SKIPPED,
        attempts=0,
        duration_ms=0,
        model_requests=0,
        metadata=deepcopy(metadata) if metadata is not None else {},
    )


class RunTraceBuilder:
    """按顺序收集步骤并一次性生成 AgentRunTrace。"""

    def __init__(
        self,
        case_id: str | None = None,
        *,
        run_id: str | None = None,
        utc_now_func: UtcNowFunc | None = None,
        monotonic_func: MonotonicFunc | None = None,
    ) -> None:
        self._run_id = run_id if run_id is not None else uuid4().hex
        self._case_id = case_id
        self._utc_now = utc_now_func if utc_now_func is not None else _utc_now
        self._monotonic = (
            monotonic_func if monotonic_func is not None else monotonic
        )
        self._started_at = _utc_iso(self._utc_now())
        self._monotonic_started = self._monotonic()
        self._steps: list[StepTrace] = []
        self._finished = False

    def add_step(self, step: StepTrace) -> None:
        """按调用顺序保存一个已经校验的步骤记录。"""
        if self._finished:
            raise RuntimeError("Trace 已完成，不能继续添加步骤。")
        if not isinstance(step, StepTrace):
            raise TypeError("step 必须是 StepTrace。")
        self._steps.append(step)

    def finish(
        self,
        *,
        status: RunStatus,
        analysis_fallback: bool,
        teaching_mode: TeachingMode | None,
        use_rag: bool,
        use_tools: bool,
        rag_searches: int,
        tool_executions: int,
    ) -> AgentRunTrace:
        """汇总步骤计数、耗时和路由字段，并封闭当前 Builder。"""
        if self._finished:
            raise RuntimeError("Trace 已完成，不能重复 finish。")

        finished_at = _utc_iso(self._utc_now())
        total_duration_ms = (
            self._monotonic() - self._monotonic_started
        ) * 1000
        trace = AgentRunTrace(
            run_id=self._run_id,
            case_id=self._case_id,
            started_at=self._started_at,
            finished_at=finished_at,
            total_duration_ms=total_duration_ms,
            status=status,
            total_model_requests=sum(
                step.model_requests for step in self._steps
            ),
            rag_searches=rag_searches,
            tool_executions=tool_executions,
            analysis_fallback=analysis_fallback,
            teaching_mode=teaching_mode,
            use_rag=use_rag,
            use_tools=use_tools,
            steps=list(self._steps),
        )
        self._finished = True
        return trace
