"""Conversation-scoped retrieval of covered old user-to-assistant turns."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import jieba
from pydantic import ValidationError
from rank_bm25 import BM25Okapi

from src.context.schemas import (
    ConversationSummary,
    HistoryRetrievalResult,
    RetrievedHistoryTurn,
)
from src.conversation.schemas import StoredMessage
from src.storage.database import DatabasePath
from src.storage.repositories import (
    get_conversation_summary,
    list_generation_jobs,
    list_messages,
)


DEFAULT_HISTORY_TOP_K = 2
DEFAULT_HISTORY_MIN_SCORE_RATIO = 0.3
# Fixture calibration: explicit old-topic matches score at least 1.867, while
# weak generic overlap scores at most 1.503. This is an application relevance
# floor, not a model-token or context-window limit.
DEFAULT_HISTORY_MIN_ABSOLUTE_SCORE = 1.7
RETRIEVED_HISTORY_CAUTION = (
    "Retrieved conversation history only restores what was discussed before. "
    "Old assistant content may be wrong; current problem conditions and reliable "
    "physics knowledge or RAG take precedence."
)


_HISTORY_STOPWORDS = {
    "一个",
    "为什么",
    "什么",
    "之前",
    "可以",
    "回答",
    "如何",
    "怎么",
    "我们",
    "是不是",
    "这个",
    "问题",
    "那个",
    # Conversation-control and low-information words must not make an old
    # physics turn look relevant.  The real long-conversation calibration that
    # motivated this list had a buoyancy query retrieving a seat-belt turn via
    # words such as “受到 / 解释 / 大 / 小” plus option labels “a / b”.
    "的",
    "了",
    "是",
    "为",
    "在",
    "中",
    "下",
    "时",
    "只",
    "这",
    "那",
    "我",
    "比",
    "说",
    "大",
    "小",
    "情况",
    "同样",
    "受到",
    "物体",
    "两个",
    "一点",
    "解释",
    "计算",
    "结果",
    "告诉",
    "不要",
    "不能",
    "而且",
    "这次",
    "换题",
}


def tokenize_history_text(text: str) -> list[str]:
    """Tokenize Chinese conversation text without any model or remote service."""

    tokens: list[str] = []
    for raw_token in jieba.lcut(text):
        token = raw_token.strip().casefold()
        if (
            token
            and token not in _HISTORY_STOPWORDS
            # Single ASCII letters are normally option labels or units in this
            # project (A/B/C, A, V), not stable old-topic identifiers.
            and not (len(token) == 1 and token.isascii() and token.isalpha())
            and any(character.isalnum() for character in token)
        ):
            tokens.append(token)
    return tokens


def _validate_options(top_k: int, min_score_ratio: float) -> None:
    if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
        raise ValueError("top_k must be a positive integer.")
    if (
        isinstance(min_score_ratio, bool)
        or not isinstance(min_score_ratio, (int, float))
        or not math.isfinite(float(min_score_ratio))
        or not 0 <= float(min_score_ratio) <= 1
    ):
        raise ValueError("min_score_ratio must be a finite number from 0 to 1.")


def _build_query(current_question: str, active_problem_text: str | None) -> str:
    if not isinstance(current_question, str) or not current_question.strip():
        raise ValueError("current_question must be a non-empty string.")
    if active_problem_text is not None and not isinstance(active_problem_text, str):
        raise ValueError("active_problem_text must be a string or None.")
    parts = [current_question.strip()]
    if active_problem_text is not None and active_problem_text.strip():
        parts.append(active_problem_text.strip())
    return "\n".join(parts)


def _safe_covered_candidates(
    messages: Sequence[StoredMessage | Mapping[str, Any]],
    generation_jobs: Sequence[Mapping[str, Any] | Any],
    existing_summary: ConversationSummary | Mapping[str, Any] | None,
) -> list[RetrievedHistoryTurn]:
    # Delay the shared window selector import so rolling-summary services can
    # initialize independently of the context-manager import graph.
    from src.context.rolling_summary import select_unsummarized_bridge_turns

    windows = select_unsummarized_bridge_turns(
        messages,
        generation_jobs,
        existing_summary,
    )
    candidates: list[RetrievedHistoryTurn] = []
    for turn in windows.covered_turns:
        try:
            candidate = RetrievedHistoryTurn(
                user_message_id=turn.user_message_id,
                assistant_message_id=turn.assistant_message_id,
                user_content=turn.user_content,
                assistant_content=turn.assistant_content,
                score=0.0,
            )
        except ValidationError:
            continue
        candidates.append(candidate)
    return candidates


def retrieve_history_from_records(
    messages: Sequence[StoredMessage | Mapping[str, Any]],
    generation_jobs: Sequence[Mapping[str, Any] | Any],
    existing_summary: ConversationSummary | Mapping[str, Any] | None,
    current_question: str,
    *,
    active_problem_text: str | None = None,
    top_k: int = DEFAULT_HISTORY_TOP_K,
    min_score_ratio: float = DEFAULT_HISTORY_MIN_SCORE_RATIO,
) -> HistoryRetrievalResult:
    """Retrieve only covered old turns; Recent and Bridge stay out by design."""

    _validate_options(top_k, min_score_ratio)
    query = _build_query(current_question, active_problem_text)
    candidates = _safe_covered_candidates(
        messages,
        generation_jobs,
        existing_summary,
    )
    query_tokens = tokenize_history_text(query)
    if not candidates or not query_tokens:
        return HistoryRetrievalResult(
            turns=[],
            candidate_count=len(candidates),
            query=query,
            retrieval_used=False,
        )

    user_corpus = [
        tokenize_history_text(candidate.user_content)
        for candidate in candidates
    ]
    corpus = [
        user_tokens + tokenize_history_text(candidate.assistant_content)
        for candidate, user_tokens in zip(candidates, user_corpus, strict=True)
    ]
    query_token_set = set(query_tokens)
    overlapping_indexes = {
        index
        for index, tokens in enumerate(corpus)
        if (
            # A candidate normally needs a topic word in the old user request.
            # Two distinct effective matches anywhere in the turn also allow
            # recall of a fact named only by the old assistant.  This gate is
            # independent of BM25 ranking and prevents generic prose from
            # becoming the best of an unrelated candidate set.
            query_token_set.intersection(user_corpus[index])
            or len(query_token_set.intersection(tokens)) >= 2
        )
    }
    if not overlapping_indexes:
        return HistoryRetrievalResult(
            turns=[],
            candidate_count=len(candidates),
            query=query,
            retrieval_used=True,
        )

    # BM25Okapi yields zero/negative IDF for tiny corpora. Empty calibration
    # documents are never returned, but keep matching terms positively scored.
    calibration_documents = [[] for _ in range(len(corpus) + 1)]
    scores = BM25Okapi(corpus + calibration_documents).get_scores(query_tokens)
    ranked = sorted(
        (
            (index, float(scores[index]))
            for index in overlapping_indexes
            if math.isfinite(float(scores[index])) and float(scores[index]) > 0
        ),
        key=lambda item: (-item[1], -item[0]),
    )
    if not ranked:
        return HistoryRetrievalResult(
            turns=[],
            candidate_count=len(candidates),
            query=query,
            retrieval_used=True,
        )

    highest_score = ranked[0][1]
    if highest_score < DEFAULT_HISTORY_MIN_ABSOLUTE_SCORE:
        return HistoryRetrievalResult(
            turns=[],
            candidate_count=len(candidates),
            query=query,
            retrieval_used=True,
        )

    minimum_score = highest_score * float(min_score_ratio)
    selected = [
        item for item in ranked if item[1] >= minimum_score
    ][:top_k]
    turns = [
        candidates[index].model_copy(update={"score": score})
        for index, score in selected
    ]
    return HistoryRetrievalResult(
        turns=turns,
        candidate_count=len(candidates),
        query=query,
        retrieval_used=True,
    )


def retrieve_conversation_history(
    conversation_id: str,
    current_question: str,
    *,
    active_problem_text: str | None = None,
    top_k: int = DEFAULT_HISTORY_TOP_K,
    min_score_ratio: float = DEFAULT_HISTORY_MIN_SCORE_RATIO,
    path: DatabasePath | None = None,
) -> HistoryRetrievalResult:
    """Load and search one conversation without crossing its repository scope."""

    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string.")
    normalized_id = conversation_id.strip()
    return retrieve_history_from_records(
        list_messages(normalized_id, path=path),
        list_generation_jobs(normalized_id, path=path),
        get_conversation_summary(normalized_id, path=path),
        current_question,
        active_problem_text=active_problem_text,
        top_k=top_k,
        min_score_ratio=min_score_ratio,
    )


def format_retrieved_history(
    result: HistoryRetrievalResult | Mapping[str, Any],
) -> str:
    """Format retrieved turns with an explicit non-authoritative boundary."""

    normalized = (
        result
        if isinstance(result, HistoryRetrievalResult)
        else HistoryRetrievalResult.model_validate(dict(result))
    )
    if not normalized.turns:
        return ""
    sections = [RETRIEVED_HISTORY_CAUTION]
    for index, turn in enumerate(normalized.turns, 1):
        sections.append(
            f"History match {index}:\n"
            f"User: {turn.user_content}\n"
            f"Assistant: {turn.assistant_content}"
        )
    return "\n\n".join(sections)
