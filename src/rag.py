"""Stage 05 最小 RAG 编排层。"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.model_client import answer_question
from src.retriever import KnowledgeRetriever


def build_context(cards: list[dict[str, Any]]) -> str:
    """将检索卡片整理为供模型参考的文本。"""
    sections = []
    for card in cards:
        sections.append(
            "\n".join(
                [
                    f"[{card['id']}]",
                    f"主题：{card['topic']}",
                    f"来源：{card['source']}",
                    f"正文：{card['content']}",
                ]
            )
        )
    return "\n\n".join(sections)


def retrieve_rag_context(
    question: str,
    top_k: int = 3,
    retriever: Any = None,
) -> dict[str, Any]:
    """Retrieve one RAG context and its reduced, ordered source metadata."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("问题不能为空，请提供一道初中物理问题。")

    active_retriever = retriever if retriever is not None else KnowledgeRetriever()
    cards = active_retriever.search(question, top_k=top_k)
    context = build_context(cards) if cards else None
    sources = [
        {
            "id": card["id"],
            "topic": card["topic"],
            "source": card["source"],
            "score": card["score"],
        }
        for card in cards
    ]
    return {"context": context, "sources": sources}


def answer_with_rag(
    question: str,
    top_k: int = 3,
    retriever: Any = None,
    answer_func: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """检索相关知识、调用模型并返回回答与精简来源。"""
    active_answer_func = answer_func if answer_func is not None else answer_question
    retrieval = retrieve_rag_context(question, top_k=top_k, retriever=retriever)
    answer = active_answer_func(question, context=retrieval["context"])
    return {"answer": answer, "sources": retrieval["sources"]}
