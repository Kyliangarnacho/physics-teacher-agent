from __future__ import annotations

import inspect
import json
from collections.abc import Callable

from openai import OpenAI
from pydantic import ValidationError

from src.config import load_qwen_config
from src.model_client import build_conversation_messages
from src.observability import ErrorType, StepTimer
from src.prompts import QUESTION_ANALYZER_SYSTEM_PROMPT
from src.retry import run_with_one_retry
from src.schemas import QuestionAnalysis, StepStatus, StepTrace, TeachingMode


AnalyzeFunc = Callable[[str], str]
ShouldRetryFunc = Callable[[Exception], bool]


def _safe_default_analysis() -> QuestionAnalysis:
    return QuestionAnalysis(
        teaching_mode=TeachingMode.SOLVE,
        physics_topic="综合",
        question_type="未知",
        needs_rag=False,
        missing_conditions=False,
        image_required=False,
        student_work_provided=False,
        short_reason="问题分析失败，按普通完整解题处理。",
        calculation_required=False,
    )


def _invoke_analyze_func(
    analyze_func: AnalyzeFunc,
    question: str,
    *,
    teaching_state_context: str | None,
    learning_memory_context: str | None,
    conversation_history: list[dict[str, str]] | None,
) -> str:
    """以兼容方式调用注入的分析函数。

    支持新上下文的函数会收到 teaching_state_context / conversation_history；
    旧签名（只有 question）的函数保持 Stage 09 原调用方式。
    """
    try:
        signature = inspect.signature(analyze_func)
    except (TypeError, ValueError):
        return analyze_func(question)
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
    return analyze_func(question, **kwargs)


def _call_analyzer_model(
    question: str,
    *,
    teaching_state_context: str | None = None,
    learning_memory_context: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> str:
    api_key, base_url, model = load_qwen_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = build_conversation_messages(
        QUESTION_ANALYZER_SYSTEM_PROMPT,
        question,
        teaching_state_context=teaching_state_context,
        learning_memory_context=learning_memory_context,
        conversation_history=conversation_history,
    )
    response = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    content = response.choices[0].message.content
    if not content:
        raise ValueError("问题分析模型返回了空内容。")
    return content


def analyze_question(
    question: str,
    analyze_func: AnalyzeFunc | None = None,
    *,
    teaching_state_context: str | None = None,
    learning_memory_context: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[QuestionAnalysis, bool]:
    analysis, analysis_fallback, _ = analyze_question_with_trace(
        question,
        analyze_func=analyze_func,
        teaching_state_context=teaching_state_context,
        learning_memory_context=learning_memory_context,
        conversation_history=conversation_history,
    )
    return analysis, analysis_fallback


def analyze_question_with_trace(
    question: str,
    analyze_func: AnalyzeFunc | None = None,
    should_retry: ShouldRetryFunc | None = None,
    *,
    teaching_state_context: str | None = None,
    learning_memory_context: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> tuple[QuestionAnalysis, bool, StepTrace]:
    """分析问题，并返回与本次 Analyzer 执行对应的步骤记录。

    可选注入教学状态与最近历史，让跟进问题在路由阶段也能正确判断。
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")

    normalized_question = question.strip()
    timer = StepTimer("analyzer")
    retry_predicate = should_retry if should_retry is not None else lambda error: True

    def call_model() -> str:
        return _invoke_analyze_func(
            analyze_func,
            normalized_question,
            teaching_state_context=teaching_state_context,
            learning_memory_context=learning_memory_context,
            conversation_history=conversation_history,
        ) if analyze_func is not None else _call_analyzer_model(
            normalized_question,
            teaching_state_context=teaching_state_context,
            learning_memory_context=learning_memory_context,
            conversation_history=conversation_history,
        )

    outcome = run_with_one_retry(
        call_model,
        retry_predicate,
        delay_seconds=0.0,
    )
    if not outcome.succeeded:
        fallback = _safe_default_analysis()
        trace = timer.finish(
            status=StepStatus.ERROR,
            attempts=outcome.attempts,
            model_requests=outcome.attempts,
            error_type=ErrorType.ANALYZER_API,
            error_message="问题分析模型调用失败。",
            metadata={"fallback": True},
        )
        return fallback, True, trace

    try:
        raw_result = outcome.value
        payload = json.loads(raw_result)
    except (json.JSONDecodeError, TypeError):
        fallback = _safe_default_analysis()
        trace = timer.finish(
            status=StepStatus.ERROR,
            attempts=outcome.attempts,
            model_requests=outcome.attempts,
            error_type=ErrorType.ANALYZER_PARSE,
            error_message="问题分析结果不是有效 JSON。",
            metadata={"fallback": True},
        )
        return fallback, True, trace

    try:
        analysis = QuestionAnalysis.model_validate(payload)
    except ValidationError:
        fallback = _safe_default_analysis()
        trace = timer.finish(
            status=StepStatus.ERROR,
            attempts=outcome.attempts,
            model_requests=outcome.attempts,
            error_type=ErrorType.ANALYZER_SCHEMA,
            error_message="问题分析结果不符合字段要求。",
            metadata={"fallback": True},
        )
        return fallback, True, trace

    status = (
        StepStatus.SUCCESS
        if outcome.attempts == 1
        else StepStatus.RETRY_SUCCESS
    )
    trace = timer.finish(
        status=status,
        attempts=outcome.attempts,
        model_requests=outcome.attempts,
        metadata={"fallback": False},
    )
    return analysis, False, trace
