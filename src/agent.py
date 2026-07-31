"""Stage 06 的统一教师 Agent 编排入口。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.analyzer import analyze_question
from src.model_client import answer_question
from src.prompts import MODE_INSTRUCTIONS
from src.rag import answer_with_rag
from src.router import route_question


def run_teacher_agent(
    question: str,
    mode_override: str = "auto",
    rag_policy: str = "auto",
    analyzer_func: Callable[[str], str] | None = None,
    retriever: Any = None,
    answer_func: Callable[..., str] | None = None,
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
        }

    mode_instruction = MODE_INSTRUCTIONS[route.teaching_mode.value]
    active_answer_func = (
        answer_func if answer_func is not None else answer_question
    )

    if route.use_rag:
        def answer_with_mode(
            rag_question: str,
            context: str | None = None,
        ) -> str:
            return active_answer_func(
                rag_question,
                context=context,
                mode_instruction=mode_instruction,
            )

        rag_result = answer_with_rag(
            normalized_question,
            top_k=3,
            retriever=retriever,
            answer_func=answer_with_mode,
        )
        answer = rag_result["answer"]
        sources = rag_result["sources"]
    else:
        answer = active_answer_func(
            normalized_question,
            context=None,
            mode_instruction=mode_instruction,
        )
        sources = []

    return {
        "answer": answer,
        "analysis": analysis.model_dump(mode="json"),
        "route": route.model_dump(mode="json"),
        "sources": sources,
        "analysis_fallback": analysis_fallback,
    }
