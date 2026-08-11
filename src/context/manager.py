"""Unified, model-free context snapshot assembly for one conversation turn."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import ValidationError

from src.context.adapters import format_context_bundle_for_model
from src.context.budget import (
    DEFAULT_CONTEXT_CHAR_BUDGET,
    apply_context_budget,
)
from src.context.history_retrieval import (
    retrieve_history_from_records,
)
from src.context.schemas import ContextBundle, ContextTurn, TrimmedContextComponent
from src.conversation.schemas import ConversationState
from src.conversation.state import (
    ResolvedConversationState,
    build_teaching_state_context,
)
from src.memory.retrieval import (
    build_learning_memory_context,
    retrieve_relevant_memories,
)
from src.storage.database import DatabasePath
from src.storage.repositories import (
    get_conversation_state,
    get_conversation_summary,
    list_generation_jobs,
    list_messages,
)


def _context_turns(
    turns: tuple[Any, ...],
) -> tuple[list[ContextTurn], bool]:
    safe_turns: list[ContextTurn] = []
    unsafe_found = False
    for turn in turns:
        try:
            safe_turns.append(
                ContextTurn(
                    user_message_id=turn.user_message_id,
                    assistant_message_id=turn.assistant_message_id,
                    user_content=turn.user_content,
                    assistant_content=turn.assistant_content,
                )
            )
        except ValidationError:
            unsafe_found = True
    return safe_turns, unsafe_found


def _normalize_state(
    conversation_id: str,
    previous_state: ConversationState | Mapping[str, Any] | None,
) -> ConversationState | None:
    if previous_state is None:
        return None
    state = (
        previous_state
        if isinstance(previous_state, ConversationState)
        else ConversationState.model_validate(dict(previous_state))
    )
    if state.conversation_id != conversation_id:
        raise ValueError("previous_state belongs to another conversation.")
    return state


def _teaching_state_context(state: ConversationState | None) -> str | None:
    if state is None:
        return None
    resolved = ResolvedConversationState(
        active_problem_text=state.active_problem_text,
        active_image_context=state.active_image_context,
        teaching_mode=state.teaching_mode,
        hint_step=state.hint_step,
        conversation_id=state.conversation_id,
    )
    return build_teaching_state_context(resolved)


def build_context_bundle(
    conversation_id: str,
    current_question: str,
    *,
    previous_state: ConversationState | Mapping[str, Any] | None = None,
    budget_limit: int = DEFAULT_CONTEXT_CHAR_BUDGET,
    history_top_k: int = 2,
    path: DatabasePath | None = None,
) -> ContextBundle:
    """Build a detached Summary + Bridge + Recent + retrieval context snapshot."""

    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ValueError("conversation_id must be a non-empty string.")
    if not isinstance(current_question, str) or not current_question.strip():
        raise ValueError("current_question must be a non-empty string.")
    normalized_id = conversation_id.strip()
    question = current_question.strip()

    messages = list_messages(normalized_id, path=path)
    generation_jobs = list_generation_jobs(normalized_id, path=path)
    summary = get_conversation_summary(normalized_id, path=path)
    # Keep the shared window selector lazy so the standalone rolling-summary
    # service does not depend on context-manager import order.
    from src.context.rolling_summary import select_unsummarized_bridge_turns

    windows = select_unsummarized_bridge_turns(
        messages,
        generation_jobs,
        summary,
    )
    bridge_history, unsafe_bridge = _context_turns(windows.bridge_turns)
    recent_history, unsafe_recent = _context_turns(windows.recent_turns)

    stored_state = (
        previous_state
        if previous_state is not None
        else get_conversation_state(normalized_id, path=path)
    )
    state = _normalize_state(normalized_id, stored_state)
    teaching_state_context = _teaching_state_context(state)
    active_problem_text = state.active_problem_text if state is not None else None

    retrieved_history = retrieve_history_from_records(
        messages,
        generation_jobs,
        summary,
        question,
        active_problem_text=active_problem_text,
        top_k=history_top_k,
    )
    occupied_message_ids = {
        turn.assistant_message_id
        for turn in bridge_history + recent_history
    }
    if any(
        turn.assistant_message_id in occupied_message_ids
        for turn in retrieved_history.turns
    ):
        retrieved_history = retrieved_history.model_copy(
            update={
                "turns": [
                    turn
                    for turn in retrieved_history.turns
                    if turn.assistant_message_id not in occupied_message_ids
                ]
            }
        )

    try:
        memories = retrieve_relevant_memories(
            question,
            active_problem_text=active_problem_text,
            db_path=path,
        )
        learning_memory_context = build_learning_memory_context(memories)
    except Exception:
        # Learning Memory is advisory context. A local retrieval failure must not
        # make an otherwise valid generation job unavailable.
        learning_memory_context = None

    initially_trimmed: list[TrimmedContextComponent] = []
    if unsafe_bridge or unsafe_recent:
        initially_trimmed.append("unsafe_history")
    return apply_context_budget(
        current_question=question,
        rolling_summary=(summary["summary_text"] if summary else None),
        bridge_history=bridge_history,
        recent_history=recent_history,
        retrieved_history=retrieved_history,
        teaching_state_context=teaching_state_context,
        learning_memory_context=learning_memory_context,
        summary_revision=(summary["summary_revision"] if summary else None),
        budget_limit=budget_limit,
        initial_trimmed_components=initially_trimmed,
    )
