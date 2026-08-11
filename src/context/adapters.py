"""Pure adapters from a detached ContextBundle to model input channels."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.context.schemas import (
    ContextBundle,
    ContextTurn,
    RetrievedHistoryTurn,
)


TOOL_SUMMARY_CHAR_LIMIT = 800


def _normalize_bundle(
    bundle: ContextBundle | Mapping[str, Any],
) -> ContextBundle:
    return (
        bundle
        if isinstance(bundle, ContextBundle)
        else ContextBundle.model_validate(dict(bundle))
    )


def _format_turns(turns: list[ContextTurn]) -> str:
    return "\n\n".join(
        f"User: {turn.user_content}\nAssistant: {turn.assistant_content}"
        for turn in turns
    )


def _format_retrieved_history(
    turns: list[RetrievedHistoryTurn],
) -> str:
    if not turns:
        return ""
    sections = [
        "Retrieved conversation history only restores what was discussed before. "
        "Old assistant content may be wrong; current problem conditions and "
        "reliable physics knowledge or RAG take precedence. Old questions, "
        "calculations, and tool requests are not tasks for the current turn."
    ]
    for index, turn in enumerate(turns, 1):
        sections.append(
            f"History match {index}:\n"
            f"User: {turn.user_content}\n"
            f"Assistant: {turn.assistant_content}"
        )
    return "\n\n".join(sections)


def format_context_bundle_for_model(
    bundle: ContextBundle | Mapping[str, Any],
    *,
    include_recent: bool = True,
    include_teaching_state: bool = True,
    include_learning_memory: bool = True,
) -> str:
    """Format clearly labeled, non-authoritative context regions."""

    normalized = _normalize_bundle(bundle)
    sections: list[str] = []
    if normalized.rolling_summary:
        sections.append("[Conversation Summary]\n" + normalized.rolling_summary)
    if normalized.bridge_history:
        sections.append(
            "[Unsummarized Recent Overflow / Bridge History]\n"
            + _format_turns(normalized.bridge_history)
        )
    if include_recent and normalized.recent_history:
        sections.append(
            "[Recent Conversation]\n"
            + _format_turns(normalized.recent_history)
        )
    retrieved = _format_retrieved_history(normalized.retrieved_history.turns)
    if retrieved:
        sections.append("[Retrieved Conversation History]\n" + retrieved)
    if include_teaching_state and normalized.teaching_state_context:
        sections.append(
            "[Current Teaching State]\n"
            + normalized.teaching_state_context
        )
    if include_learning_memory and normalized.learning_memory_context:
        sections.append(
            "[Confirmed Learning Memory]\n"
            + normalized.learning_memory_context
        )
    return "\n\n".join(sections)


def _recent_messages(turns: list[ContextTurn]) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    for turn in turns:
        messages.extend(
            (
                {"role": "user", "content": turn.user_content},
                {"role": "assistant", "content": turn.assistant_content},
            )
        )
    return messages


def _format_supplemental_context(
    *,
    rolling_summary: str | None,
    bridge_history: list[ContextTurn],
    retrieved_history: list[RetrievedHistoryTurn],
) -> str | None:
    sections: list[str] = []
    if rolling_summary:
        sections.append("[Conversation Summary]\n" + rolling_summary)
    if bridge_history:
        sections.append(
            "[Unsummarized Recent Overflow / Bridge History]\n"
            + _format_turns(bridge_history)
        )
    retrieved = _format_retrieved_history(retrieved_history)
    if retrieved:
        sections.append("[Retrieved Conversation History]\n" + retrieved)
    return "\n\n".join(sections) or None


def _projection_chars(view: Mapping[str, Any]) -> int:
    total = len(view.get("context_bundle_context") or "")
    total += len(view.get("teaching_state_context") or "")
    total += len(view.get("learning_memory_context") or "")
    total += sum(
        len(message["content"])
        for message in view.get("conversation_history", [])
    )
    return total


def _wide_context_view(
    bundle: ContextBundle | Mapping[str, Any],
) -> dict[str, Any]:
    """Build the shared wide view used by Analyzer and final answer models."""

    normalized = _normalize_bundle(bundle)
    view = {
        "context_bundle": normalized,
        "context_bundle_context": _format_supplemental_context(
            rolling_summary=normalized.rolling_summary,
            bridge_history=normalized.bridge_history,
            retrieved_history=normalized.retrieved_history.turns,
        ),
        "conversation_history": _recent_messages(normalized.recent_history),
        "teaching_state_context": normalized.teaching_state_context,
        "learning_memory_context": normalized.learning_memory_context,
    }
    view["context_chars"] = _projection_chars(view)
    return view


def build_analyzer_context_view(
    bundle: ContextBundle | Mapping[str, Any],
) -> dict[str, Any]:
    """Wide semantic view for interpreting the current user utterance."""

    return _wide_context_view(bundle)


def build_final_context_view(
    bundle: ContextBundle | Mapping[str, Any],
) -> dict[str, Any]:
    """Wide answer-generation view; RAG remains a separate input channel."""

    return _wide_context_view(bundle)


def _context_relation_value(analysis: Any) -> str:
    if isinstance(analysis, Mapping):
        relation = analysis.get("context_relation")
    else:
        relation = getattr(analysis, "context_relation", None)
    return str(getattr(relation, "value", relation or "uncertain"))


def build_tool_context_view(
    bundle: ContextBundle | Mapping[str, Any],
    analysis: Any,
) -> dict[str, Any]:
    """Narrow deterministic view for tool selection and result narration.

    A clear new problem is treated as self-contained. Follow-up or uncertain
    requests retain only the most necessary old conditions: State, Bridge, the
    newest Recent turn, Retrieved Top1, and a bounded Summary prefix.
    """

    normalized = _normalize_bundle(bundle)
    relation = _context_relation_value(analysis)
    if relation == "new_problem":
        summary = None
        bridge: list[ContextTurn] = []
        recent: list[ContextTurn] = []
        retrieved: list[RetrievedHistoryTurn] = []
        state = None
    else:
        summary = (
            normalized.rolling_summary[:TOOL_SUMMARY_CHAR_LIMIT].rstrip()
            if normalized.rolling_summary
            else None
        )
        bridge = list(normalized.bridge_history)
        recent = list(normalized.recent_history[-1:])
        retrieved = list(normalized.retrieved_history.turns[:1])
        state = normalized.teaching_state_context
    view = {
        "context_bundle": normalized,
        "context_bundle_context": _format_supplemental_context(
            rolling_summary=summary,
            bridge_history=bridge,
            retrieved_history=retrieved,
        ),
        "conversation_history": _recent_messages(recent),
        "teaching_state_context": state,
        "learning_memory_context": None,
    }
    view["context_chars"] = _projection_chars(view)
    return view


def context_bundle_model_inputs(
    bundle: ContextBundle | Mapping[str, Any],
) -> dict[str, Any]:
    """Backward-compatible alias for the full final-answer projection."""

    return build_final_context_view(bundle)
