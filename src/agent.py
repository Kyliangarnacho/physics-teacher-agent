"""初中物理教师 Agent 的统一分析、路由与回答编排入口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.analyzer import analyze_question_with_trace
from src.model_client import answer_question
from src.observability import RunTraceBuilder, StepTimer, create_skipped_step
from src.prompts import MODE_INSTRUCTIONS
from src.rag import retrieve_rag_context
from src.router import route_question
from src.schemas import RunStatus, StepStatus, StepTrace
from src.tool_client import answer_with_tools


_RECOVERABLE_IMAGE_TOOL_ERRORS = {
    "tool_selection_api",
    "tool_protocol",
    "tool_validation",
}

_MULTI_IMAGE_SEPARATE_MARKERS = ("分别", "依次", "各自", "每张")


def _requires_multiple_independent_answers(
    question: str,
    image_context: str | None,
) -> bool:
    """Detect an explicit request to answer multiple image problems separately."""
    if not isinstance(image_context, str):
        return False
    return (
        image_context.count("【图片 ") > 1
        and any(marker in question for marker in _MULTI_IMAGE_SEPARATE_MARKERS)
    )


def _recoverable_image_tool_error(tool_result: dict[str, Any]) -> str | None:
    """Return a safe recoverable Tool Client error type, if present."""
    traces = tool_result.get("step_traces")
    if not isinstance(traces, list):
        return None
    for trace in traces:
        if not isinstance(trace, dict):
            continue
        error_type = trace.get("error_type")
        if error_type in _RECOVERABLE_IMAGE_TOOL_ERRORS:
            return str(error_type)
    return None


def _tool_traces_from_result(tool_result: dict[str, Any]) -> list[StepTrace]:
    """Load Tool Client traces while keeping older injected fakes compatible."""
    trace_data = tool_result.get("step_traces")
    if trace_data is not None:
        return [StepTrace.model_validate(item) for item in trace_data]

    model_requests = tool_result["model_requests"]
    result_requests = max(0, model_requests - 1)
    result_status = (
        StepStatus.RETRY_SUCCESS
        if result_requests == 2
        else StepStatus.SUCCESS
    )
    return [
        StepTrace(
            name="tool_selection",
            status=StepStatus.SUCCESS,
            attempts=1,
            duration_ms=0,
            model_requests=min(model_requests, 1),
        ),
        StepTrace(
            name="tool_execution",
            status=StepStatus.SUCCESS,
            attempts=1,
            duration_ms=0,
            model_requests=0,
            metadata={"tool_record_count": len(tool_result["tool_records"])},
        ),
        StepTrace(
            name="tool_result_answer",
            status=result_status,
            attempts=2 if result_status is StepStatus.RETRY_SUCCESS else 1,
            duration_ms=0,
            model_requests=result_requests,
        ),
    ]


def run_teacher_agent(
    question: str,
    mode_override: str = "auto",
    rag_policy: str = "auto",
    analyzer_func: Callable[[str], str] | None = None,
    retriever: Any = None,
    answer_func: Callable[..., str] | None = None,
    tool_answer_func: Callable[..., dict[str, Any]] | None = None,
    case_id: str | None = None,
    image_context: str | None = None,
    image_context_available: bool = False,
) -> dict[str, Any]:
    """分析、路由并回答一道初中物理问题。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")
    if not isinstance(image_context_available, bool):
        raise ValueError("image_context_available 必须是布尔值。")
    if image_context_available and (
        not isinstance(image_context, str) or not image_context.strip()
    ):
        raise ValueError("图片上下文已标记为可用时，image_context 不能为空。")

    normalized_question = question.strip()
    active_question = normalized_question
    if image_context_available:
        active_question = (
            f"用户要求：\n{normalized_question}\n\n"
            f"已确认图片上下文：\n{image_context.strip()}"
        )
    trace_builder = RunTraceBuilder(case_id=case_id)
    analysis, analysis_fallback, analyzer_trace = analyze_question_with_trace(
        active_question,
        analyze_func=analyzer_func,
    )
    trace_builder.add_step(analyzer_trace)

    router_timer = StepTimer("router")
    route = route_question(
        analysis,
        mode_override=mode_override,
        rag_policy=rag_policy,
        image_context_available=image_context_available,
    )
    trace_builder.add_step(
        router_timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
            metadata={
                "teaching_mode": route.teaching_mode.value,
                "use_rag": route.use_rag,
                "use_tools": route.use_tools,
                "should_answer": route.should_answer,
            },
        )
    )

    if not route.should_answer:
        trace_builder.add_step(
            create_skipped_step(
                "rag_retrieval",
                {"reason": "route_blocked"},
            )
        )
        trace = trace_builder.finish(
            status=RunStatus.BLOCKED,
            analysis_fallback=analysis_fallback,
            teaching_mode=route.teaching_mode,
            use_rag=route.use_rag,
            use_tools=route.use_tools,
            rag_searches=0,
            tool_executions=0,
        )
        return {
            "answer": route.user_message or "",
            "analysis": analysis.model_dump(mode="json"),
            "route": route.model_dump(mode="json"),
            "sources": [],
            "analysis_fallback": analysis_fallback,
            "tool_records": [],
            "tool_model_requests": 0,
            "trace": trace.model_dump(mode="json"),
        }

    mode_instruction = MODE_INSTRUCTIONS[route.teaching_mode.value]
    active_answer_func = (
        answer_func if answer_func is not None else answer_question
    )

    context = None
    sources: list[dict[str, Any]] = []
    rag_searches = 0
    if route.use_rag:
        retrieval_timer = StepTimer("rag_retrieval")
        retrieval = retrieve_rag_context(
            active_question,
            top_k=3,
            retriever=retriever,
        )
        context = retrieval["context"]
        sources = retrieval["sources"]
        rag_searches = 1
        trace_builder.add_step(
            retrieval_timer.finish(
                status=StepStatus.SUCCESS,
                attempts=1,
                model_requests=0,
                metadata={"source_count": len(sources)},
            )
        )
    else:
        trace_builder.add_step(
            create_skipped_step(
                "rag_retrieval",
                {"reason": "rag_disabled"},
            )
        )

    if route.use_tools:
        separate_multi_image_questions = (
            image_context_available
            and _requires_multiple_independent_answers(
                normalized_question,
                image_context,
            )
        )
        if separate_multi_image_questions:
            fallback_reason = "no_single_tool_for_multiple_image_questions"
            for step_name in (
                "tool_selection",
                "tool_execution",
                "tool_result_answer",
            ):
                trace_builder.add_step(
                    create_skipped_step(step_name, {"reason": fallback_reason})
                )
            tool_records = []
            tool_model_requests = 0
            tool_fallback = True
            fallback_timer = StepTimer("final_answer")
            answer = active_answer_func(
                active_question,
                context=context,
                mode_instruction=mode_instruction,
            )
            trace_builder.add_step(
                fallback_timer.finish(
                    status=StepStatus.SUCCESS,
                    attempts=1,
                    model_requests=1,
                    metadata={
                        "tool_fallback": True,
                        "reason": fallback_reason,
                    },
                )
            )
        else:
            active_tool_answer_func = (
                tool_answer_func
                if tool_answer_func is not None
                else answer_with_tools
            )
            tool_result = active_tool_answer_func(
                active_question,
                context=context,
                mode_instruction=mode_instruction,
            )
            answer = tool_result["answer"]
            tool_records = tool_result["tool_records"]
            tool_model_requests = tool_result["model_requests"]
            for tool_trace in _tool_traces_from_result(tool_result):
                trace_builder.add_step(tool_trace)
            recoverable_tool_error = _recoverable_image_tool_error(tool_result)
            tool_fallback = bool(
                image_context_available and recoverable_tool_error is not None
            )
            if tool_fallback:
                fallback_timer = StepTimer("final_answer")
                answer = active_answer_func(
                    active_question,
                    context=context,
                    mode_instruction=mode_instruction,
                )
                trace_builder.add_step(
                    fallback_timer.finish(
                        status=StepStatus.SUCCESS,
                        attempts=1,
                        model_requests=1,
                        metadata={
                            "tool_fallback": True,
                            "reason": recoverable_tool_error,
                        },
                    )
                )
    else:
        tool_fallback = False
        answer_timer = StepTimer("final_answer")
        answer = active_answer_func(
            active_question,
            context=context,
            mode_instruction=mode_instruction,
        )
        tool_records = []
        tool_model_requests = 0
        trace_builder.add_step(
            answer_timer.finish(
                status=StepStatus.SUCCESS,
                attempts=1,
                model_requests=1,
            )
        )

    run_status = (
        RunStatus.COMPLETED_WITH_FALLBACK
        if analysis_fallback or tool_fallback
        else RunStatus.COMPLETED
    )
    trace = trace_builder.finish(
        status=run_status,
        analysis_fallback=analysis_fallback,
        teaching_mode=route.teaching_mode,
        use_rag=route.use_rag,
        use_tools=route.use_tools,
        rag_searches=rag_searches,
        tool_executions=len(tool_records),
    )

    return {
        "answer": answer,
        "analysis": analysis.model_dump(mode="json"),
        "route": route.model_dump(mode="json"),
        "sources": sources,
        "analysis_fallback": analysis_fallback,
        "tool_records": tool_records,
        "tool_model_requests": tool_model_requests,
        "trace": trace.model_dump(mode="json"),
    }
