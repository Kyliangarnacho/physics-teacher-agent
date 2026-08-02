"""Stage 06 的统一教师 Agent 编排入口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.analyzer import analyze_question
from src.model_client import answer_question
from src.prompts import MODE_INSTRUCTIONS
from src.rag import retrieve_rag_context
from src.router import route_question
from src.tool_client import answer_with_tools


def run_teacher_agent(
    question: str,
    mode_override: str = "auto",
    rag_policy: str = "auto",
    analyzer_func: Callable[[str], str] | None = None,
    retriever: Any = None,
    answer_func: Callable[..., str] | None = None,
    tool_answer_func: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """分析、路由并回答一道初中物理问题。"""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")

    normalized_question = question.strip()
    analysis, analysis_fallback = analyze_question(
        normalized_question,
        analyze_func=analyzer_func,
    )
    route = route_question(
        analysis,
        mode_override=mode_override,
        rag_policy=rag_policy,
    )

    if not route.should_answer:
        return {
            "answer": route.user_message or "",
            "analysis": analysis.model_dump(mode="json"),
            "route": route.model_dump(mode="json"),
            "sources": [],
            "analysis_fallback": analysis_fallback,
            "tool_records": [],
            "tool_model_requests": 0,
        }

    mode_instruction = MODE_INSTRUCTIONS[route.teaching_mode.value]
    active_answer_func = (
        answer_func if answer_func is not None else answer_question
    )

    context = None
    sources: list[dict[str, Any]] = []
    if route.use_rag:
        retrieval = retrieve_rag_context(
            normalized_question,
            top_k=3,
            retriever=retriever,
        )
        context = retrieval["context"]
        sources = retrieval["sources"]

    if route.use_tools:
        active_tool_answer_func = (
            tool_answer_func if tool_answer_func is not None else answer_with_tools
        )
        tool_result = active_tool_answer_func(
            normalized_question,
            context=context,
            mode_instruction=mode_instruction,
        )
        answer = tool_result["answer"]
        tool_records = tool_result["tool_records"]
        tool_model_requests = tool_result["model_requests"]
    else:
        answer = active_answer_func(
            normalized_question,
            context=context,
            mode_instruction=mode_instruction,
        )
        tool_records = []
        tool_model_requests = 0

    return {
        "answer": answer,
        "analysis": analysis.model_dump(mode="json"),
        "route": route.model_dump(mode="json"),
        "sources": sources,
        "analysis_fallback": analysis_fallback,
        "tool_records": tool_records,
        "tool_model_requests": tool_model_requests,
    }
