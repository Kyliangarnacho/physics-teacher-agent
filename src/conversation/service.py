"""Stage 10 Conversation Service：单轮对话的编排、持久化与状态推进。

一轮流程：校验 → 读取最近历史与旧状态 → 解析状态与上下文 → 短事务保存
user 消息 → 事务外调用 Agent → 短事务原子保存 assistant、agent_run 与状态。
本模块不直接写 SQL，所有持久化都经由 Repository。
"""

from __future__ import annotations

from typing import Any

from src.agent import run_teacher_agent as _default_run_agent
from src.conversation.history import build_recent_history
from src.conversation.schemas import StoredMessage
from src.conversation.state import (
    ResolvedConversationState,
    build_state_update_after_turn,
    build_teaching_state_context,
    resolve_conversation_state,
)
from src.memory.retrieval import (
    build_learning_memory_context,
    retrieve_relevant_memories,
)
from src.storage import (
    RepositoryError,
    finalize_conversation_turn,
    get_conversation,
    get_conversation_state,
    get_recent_messages,
    insert_agent_run,
    insert_message,
)


class ConversationServiceError(Exception):
    """Conversation Service 统一异常，不泄露 SQL、绝对路径或密钥。"""


def _require_non_empty_text(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConversationServiceError(f"{name} 不能为空。")
    return value.strip()


def _build_effective_model_question(
    resolved: ResolvedConversationState,
    model_question: str,
) -> str:
    """跟进问题时保留当前指令并补充原题，让 Agent 明确知道承接的是哪道题。"""
    if resolved.is_follow_up and resolved.active_problem_text:
        return (
            f"当前指令：{model_question}\n\n"
            f"原题：{resolved.active_problem_text}"
        )
    return model_question


def _trace_status(agent_result: dict[str, Any]) -> str:
    trace = agent_result.get("trace")
    if isinstance(trace, dict):
        return str(trace.get("status", ""))
    return ""


def _build_agent_run(
    conversation_id: str,
    user_message_id: str,
    agent_result: dict[str, Any],
) -> dict[str, Any]:
    route = agent_result.get("route") or {}
    trace = agent_result.get("trace") or {}
    return {
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "status": str(trace.get("status") or "completed"),
        "teaching_mode": route.get("teaching_mode"),
        "use_rag": bool(route.get("use_rag", False)),
        "use_tools": bool(route.get("use_tools", False)),
        "total_model_requests": int(trace.get("total_model_requests", 0)),
        "total_duration_ms": float(trace.get("total_duration_ms", 0.0)),
        "sources_json": agent_result.get("sources", []),
        "tool_records_json": agent_result.get("tool_records", []),
        "trace_json": agent_result.get("trace"),
        "analysis_json": agent_result.get("analysis", {}),
    }


def _write_failed_agent_run(
    conversation_id: str,
    user_message_id: str,
    db_path,
) -> None:
    """尽力写入 failed agent_run；失败不影响原始异常。"""
    try:
        insert_agent_run(
            {
                "conversation_id": conversation_id,
                "user_message_id": user_message_id,
                "status": "failed",
                "use_rag": False,
                "use_tools": False,
                "total_model_requests": 0,
                "total_duration_ms": 0.0,
            },
            path=db_path,
        )
    except RepositoryError:
        pass


def run_conversation_turn(
    conversation_id: str,
    display_question: str,
    model_question: str,
    image_metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    confirmed_image_context: str | None = None,
    mode_override: str = "auto",
    rag_policy: str = "auto",
    db_path=None,
    agent_func=None,
    max_history_turns: int = 3,
    max_history_chars: int = 6000,
) -> dict[str, Any]:
    """执行一轮对话并返回原 Agent 结果与持久化标识。

    display_question 用于页面展示，model_question 用于发送给模型与后续历史；
    Agent 异常时保留已保存的 user 消息并记录 failed run，然后抛
    ConversationServiceError；blocked 返回时不推进状态。
    """
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ConversationServiceError("会话 ID 不能为空。")
    display_question = _require_non_empty_text(display_question, "display_question")
    model_question = _require_non_empty_text(model_question, "model_question")

    conversation = get_conversation(conversation_id, path=db_path)
    if conversation is None:
        raise ConversationServiceError("会话不存在，无法继续对话。")

    recent_messages = get_recent_messages(
        conversation_id,
        limit=max_history_turns * 2 + 1,
        path=db_path,
    )
    stored_messages = [
        StoredMessage.model_validate(item) for item in recent_messages
    ]
    history = build_recent_history(
        stored_messages,
        max_turns=max_history_turns,
        max_chars=max_history_chars,
    )
    old_state = get_conversation_state(conversation_id, path=db_path)

    resolved = resolve_conversation_state(
        model_question,
        previous_state=old_state,
        current_image_context=confirmed_image_context,
        mode_override=mode_override,
    )
    teaching_state_context = build_teaching_state_context(resolved)
    try:
        memories = retrieve_relevant_memories(
            model_question,
            active_problem_text=resolved.active_problem_text,
            db_path=db_path,
        )
        memory_context = build_learning_memory_context(memories)
    except Exception:
        memories = []
        memory_context = None
    effective_model_question = _build_effective_model_question(
        resolved,
        model_question,
    )

    user_message = insert_message(
        {
            "conversation_id": conversation_id,
            "role": "user",
            "display_content": display_question,
            "model_content": model_question,
            "image_metadata_json": image_metadata,
        },
        path=db_path,
    )
    user_message_id = user_message["id"]

    agent = agent_func if agent_func is not None else _default_run_agent
    agent_mode_override = (
        resolved.teaching_mode
        if resolved.teaching_mode is not None
        else mode_override
    )
    try:
        agent_result = agent(
            effective_model_question,
            mode_override=agent_mode_override,
            rag_policy=rag_policy,
            image_context=resolved.active_image_context,
            image_context_available=bool(resolved.active_image_context),
            conversation_history=history,
            teaching_state_context=teaching_state_context,
            learning_memory_context=memory_context,
        )
    except Exception as exc:
        _write_failed_agent_run(conversation_id, user_message_id, db_path)
        raise ConversationServiceError("本轮回答生成失败，请稍后重试。") from exc

    trace_status = _trace_status(agent_result)
    succeeded = trace_status not in ("blocked", "failed")
    effective_mode = (
        agent_result.get("route", {}).get("teaching_mode")
        or resolved.teaching_mode
    )
    state_update = build_state_update_after_turn(
        resolved,
        succeeded=succeeded,
        effective_mode=effective_mode,
    )
    if state_update is not None:
        state_update["conversation_id"] = conversation_id

    try:
        finalized = finalize_conversation_turn(
            conversation_id,
            assistant_message={
                "conversation_id": conversation_id,
                "role": "assistant",
                "display_content": agent_result.get("answer", ""),
                "model_content": agent_result.get("answer", ""),
            },
            agent_run=_build_agent_run(
                conversation_id,
                user_message_id,
                agent_result,
            ),
            conversation_state=state_update,
            path=db_path,
        )
    except RepositoryError as exc:
        raise ConversationServiceError("本轮结果保存失败，请稍后重试。") from exc

    return {
        "agent_result": agent_result,
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "assistant_message_id": finalized["assistant_message_id"],
        "resolved_state": resolved,
        "history_turn_count": len(history) // 2,
        "state_context_used": bool(teaching_state_context),
        "memory_count": len(memories),
        "memory_context_used": bool(memory_context),
    }
