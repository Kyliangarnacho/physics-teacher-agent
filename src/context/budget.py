"""Deterministic application-level character budget for context snapshots."""

from __future__ import annotations

from collections.abc import Sequence

from src.context.schemas import (
    ContextBundle,
    ContextTurn,
    HistoryRetrievalResult,
    TrimmedContextComponent,
)


# This is an internal application safety budget, not an advertised Qwen context
# window. It can be changed independently when the future integration is tuned.
DEFAULT_CONTEXT_CHAR_BUDGET = 12_000
MIN_ROLLING_SUMMARY_CHARS = 256
SUMMARY_TRUNCATION_MARKER = "\n[Summary truncated by application budget]"


def estimate_context_chars(
    *,
    current_question: str,
    rolling_summary: str | None,
    bridge_history: Sequence[ContextTurn],
    recent_history: Sequence[ContextTurn],
    retrieved_history: HistoryRetrievalResult,
    teaching_state_context: str | None,
    learning_memory_context: str | None,
) -> int:
    """Estimate model-visible characters without claiming tokenizer precision."""

    total = len(current_question)
    total += len(rolling_summary or "")
    total += sum(
        len(turn.user_content) + len(turn.assistant_content)
        for turn in bridge_history
    )
    total += sum(
        len(turn.user_content) + len(turn.assistant_content)
        for turn in recent_history
    )
    total += sum(
        len(turn.user_content) + len(turn.assistant_content)
        for turn in retrieved_history.turns
    )
    total += len(teaching_state_context or "")
    total += len(learning_memory_context or "")
    return total


def _trim_one_memory_entry(context: str) -> str | None:
    lines = context.splitlines()
    if len(lines) <= 2:
        return None
    return "\n".join(lines[:-1]).strip() or None


def _truncate_summary(summary: str, max_chars: int) -> str:
    if max_chars >= len(summary):
        return summary
    minimum = min(len(summary), MIN_ROLLING_SUMMARY_CHARS)
    target = max(minimum, max_chars)
    if target >= len(summary):
        return summary
    if target <= len(SUMMARY_TRUNCATION_MARKER):
        return summary[:target]
    prefix = summary[: target - len(SUMMARY_TRUNCATION_MARKER)].rstrip()
    return prefix + SUMMARY_TRUNCATION_MARKER


def apply_context_budget(
    *,
    current_question: str,
    rolling_summary: str | None,
    bridge_history: Sequence[ContextTurn],
    recent_history: Sequence[ContextTurn],
    retrieved_history: HistoryRetrievalResult,
    teaching_state_context: str | None,
    learning_memory_context: str | None,
    summary_revision: int | None,
    budget_limit: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    initial_trimmed_components: Sequence[TrimmedContextComponent] = (),
) -> ContextBundle:
    """Apply a simple priority ladder while preserving state, Recent, and Bridge."""

    if not isinstance(current_question, str) or not current_question.strip():
        raise ValueError("current_question must be a non-empty string.")
    if isinstance(budget_limit, bool) or not isinstance(budget_limit, int):
        raise ValueError("budget_limit must be a positive integer.")
    if budget_limit <= 0:
        raise ValueError("budget_limit must be a positive integer.")

    question = current_question.strip()
    selected_summary = rolling_summary
    selected_bridge = list(bridge_history)
    selected_recent = list(recent_history)
    selected_retrieved = retrieved_history.model_copy(
        update={"turns": list(retrieved_history.turns)}
    )
    selected_memory = learning_memory_context
    trimmed = list(dict.fromkeys(initial_trimmed_components))

    def estimate() -> int:
        return estimate_context_chars(
            current_question=question,
            rolling_summary=selected_summary,
            bridge_history=selected_bridge,
            recent_history=selected_recent,
            retrieved_history=selected_retrieved,
            teaching_state_context=teaching_state_context,
            learning_memory_context=selected_memory,
        )

    while estimate() > budget_limit and selected_retrieved.turns:
        selected_retrieved = selected_retrieved.model_copy(
            update={"turns": list(selected_retrieved.turns[:-1])}
        )
        if "retrieved_history" not in trimmed:
            trimmed.append("retrieved_history")

    while estimate() > budget_limit and selected_memory is not None:
        selected_memory = _trim_one_memory_entry(selected_memory)
        if "learning_memory_context" not in trimmed:
            trimmed.append("learning_memory_context")

    if estimate() > budget_limit and selected_summary is not None:
        cost_without_summary = estimate() - len(selected_summary)
        available = max(0, budget_limit - cost_without_summary)
        truncated = _truncate_summary(selected_summary, available)
        if truncated != selected_summary:
            selected_summary = truncated
            if "rolling_summary" not in trimmed:
                trimmed.append("rolling_summary")

    estimated_chars = estimate()
    return ContextBundle(
        rolling_summary=selected_summary,
        bridge_history=selected_bridge,
        recent_history=selected_recent,
        retrieved_history=selected_retrieved,
        teaching_state_context=teaching_state_context,
        learning_memory_context=selected_memory,
        estimated_chars=estimated_chars,
        budget_limit=budget_limit,
        current_question_chars=len(question),
        trimmed_components=trimmed,
        budget_exceeded=estimated_chars > budget_limit,
        summary_revision=summary_revision,
        bridge_turn_count=len(selected_bridge),
        recent_turn_count=len(selected_recent),
        retrieved_turn_count=len(selected_retrieved.turns),
    )
