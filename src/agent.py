"""初中物理教师 Agent 的统一分析、路由与回答编排入口。"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

from src.analyzer import analyze_question_with_trace
from src.model_client import answer_question, validate_conversation_history
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


def _invoke_with_context(
    func: Callable[..., Any],
    question: str,
    *,
    context: str | None,
    mode_instruction: str | None,
    teaching_state_context: str | None,
    learning_memory_context: str | None,
    conversation_history: list[dict[str, str]] | None,
) -> Any:
    """以兼容方式调用回答/工具函数。

    支持新上下文的函数会收到 teaching_state_context / conversation_history；
    旧签名（question/context/mode_instruction）的函数保持 Stage 09 原调用方式。
    """
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return func(question, context=context, mode_instruction=mode_instruction)
    parameters = signature.parameters
    has_var_keyword = any(
        param.kind is inspect.Parameter.VAR_KEYWORD
        for param in parameters.values()
    )
    supported = {
        name
        for name, param in parameters.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    kwargs: dict[str, object] = {}
    if has_var_keyword or "teaching_state_context" in supported:
        kwargs["teaching_state_context"] = teaching_state_context
    if has_var_keyword or "learning_memory_context" in supported:
        kwargs["learning_memory_context"] = learning_memory_context
    if has_var_keyword or "conversation_history" in supported:
        kwargs["conversation_history"] = conversation_history
    return func(
        question,
        context=context,
        mode_instruction=mode_instruction,
        **kwargs,
    )


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
    conversation_history: list[dict[str, str]] | None = None,
    teaching_state_context: str | None = None,
    learning_memory_context: str | None = None,
) -> dict[str, Any]:
    """分析、路由并回答一道初中物理问题。

    conversation_history：最近的完整轮次（user/assistant），按旧到新传入，
    会注入 Analyzer、普通最终回答与 Tool Client；传入结构不会被修改。
    teaching_state_context：已确认的安全教学状态文本，只注入一次。
    两者默认均为 None，保持 Stage 09 行为兼容。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")
    if not isinstance(image_context_available, bool):
        raise ValueError("image_context_available 必须是布尔值。")
    validate_conversation_history(conversation_history)
    if teaching_state_context is not None and not isinstance(
        teaching_state_context,
        str,
    ):
        raise ValueError("teaching_state_context 必须是字符串。")
    if learning_memory_context is not None and not isinstance(
        learning_memory_context,
        str,
    ):
        raise ValueError("learning_memory_context 必须是字符串。")
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
        teaching_state_context=teaching_state_context,
        learning_memory_context=learning_memory_context,
        conversation_history=conversation_history,
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
            answer = _invoke_with_context(
                active_answer_func,
                active_question,
                context=context,
                mode_instruction=mode_instruction,
                teaching_state_context=teaching_state_context,
                learning_memory_context=learning_memory_context,
                conversation_history=conversation_history,
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
            tool_result = _invoke_with_context(
                active_tool_answer_func,
                active_question,
                context=context,
                mode_instruction=mode_instruction,
                teaching_state_context=teaching_state_context,
                learning_memory_context=learning_memory_context,
                conversation_history=conversation_history,
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
                answer = _invoke_with_context(
                    active_answer_func,
                    active_question,
                    context=context,
                    mode_instruction=mode_instruction,
                    teaching_state_context=teaching_state_context,
                    learning_memory_context=learning_memory_context,
                    conversation_history=conversation_history,
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
        answer = _invoke_with_context(
            active_answer_func,
            active_question,
            context=context,
            mode_instruction=mode_instruction,
            teaching_state_context=teaching_state_context,
            learning_memory_context=learning_memory_context,
            conversation_history=conversation_history,
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
