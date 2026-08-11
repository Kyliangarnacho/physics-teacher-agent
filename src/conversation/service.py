"""Conversation Service：回答任务入队、执行与同步兼容入口。"""

from __future__ import annotations

from typing import Any

from src.agent import run_teacher_agent as _default_run_agent
from src.context.adapters import context_bundle_model_inputs
from src.conversation.state import (
    ResolvedConversationState,
    build_analyzer_state_context,
    build_state_update_after_turn,
    resolve_conversation_state,
)
from src.storage import (
    RepositoryError,
    claim_generation_job,
    enqueue_generation_job,
    fail_generation_job,
    finalize_generation_job,
    get_conversation,
    get_conversation_state,
    get_generation_job,
)


class ConversationServiceError(Exception):
    """Conversation Service 统一异常，不泄露 SQL、绝对路径或密钥。"""


def build_context_bundle(*args, **kwargs):
    """Lazy proxy that preserves the service-level patch seam."""

    from src.context.manager import build_context_bundle as _build_context_bundle

    return _build_context_bundle(*args, **kwargs)


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


def _mark_job_failed(job_id: str, error_type: str, db_path) -> None:
    """尽力把 running Job 标记失败；失败不遮蔽原始异常。"""
    try:
        fail_generation_job(
            job_id,
            error_type,
            "本轮回答生成失败，请稍后重试。",
            path=db_path,
        )
    except RepositoryError:
        pass


def _memory_context_count(context: str | None) -> int:
    if not context:
        return 0
    return sum(1 for line in context.splitlines() if line.startswith("- ["))


def enqueue_conversation_turn(
    conversation_id: str,
    display_question: str,
    model_question: str,
    image_metadata: dict[str, Any] | list[dict[str, Any]] | None = None,
    confirmed_image_context: str | None = None,
    mode_override: str = "auto",
    rag_policy: str = "auto",
    db_path=None,
    max_history_turns: int = 3,
    max_history_chars: int = 6000,
) -> dict[str, Any]:
    """原子保存 user 消息和 pending Job；不读取上下文、不调用 Agent。"""
    if not isinstance(conversation_id, str) or not conversation_id.strip():
        raise ConversationServiceError("会话 ID 不能为空。")
    if not isinstance(display_question, str):
        raise ConversationServiceError("display_question 必须是字符串。")
    display_question = display_question.strip()
    if not display_question and not image_metadata:
        raise ConversationServiceError("无图片时 display_question 不能为空。")
    model_question = _require_non_empty_text(model_question, "model_question")

    conversation = get_conversation(conversation_id, path=db_path)
    if conversation is None:
        raise ConversationServiceError("会话不存在，无法继续对话。")

    try:
        enqueued = enqueue_generation_job(
            {
                "conversation_id": conversation_id,
                "role": "user",
                "display_content": display_question,
                "model_content": model_question,
                "image_metadata_json": image_metadata,
            },
            {
                "question": model_question,
                "mode_override": mode_override,
                "rag_policy": rag_policy,
                "image_context": confirmed_image_context,
                "image_context_available": bool(confirmed_image_context),
                "max_history_turns": max_history_turns,
                "max_history_chars": max_history_chars,
            },
            path=db_path,
        )
    except RepositoryError as exc:
        raise ConversationServiceError("本轮问题入队失败，请稍后重试。") from exc

    job = enqueued["generation_job"]
    user_message = enqueued["user_message"]
    return {
        "conversation_id": conversation_id,
        "user_message_id": user_message["id"],
        "generation_job_id": job["id"],
        "generation_job": job,
    }


def execute_generation_job(
    job_id: str,
    *,
    db_path=None,
    agent_func=None,
) -> dict[str, Any]:
    """领取 Job，从数据库恢复上下文，调用 Agent 并原子保存成功结果。"""
    job_id = _require_non_empty_text(job_id, "job_id")
    try:
        job = claim_generation_job(job_id, path=db_path)
    except RepositoryError as exc:
        raise ConversationServiceError("回答任务领取失败，请稍后重试。") from exc
    if job is None:
        try:
            existing = get_generation_job(job_id, path=db_path)
        except RepositoryError as exc:
            raise ConversationServiceError("回答任务状态读取失败，请稍后重试。") from exc
        if existing is None:
            raise ConversationServiceError("回答任务不存在。")
        return {
            "conversation_id": existing["conversation_id"],
            "user_message_id": existing["user_message_id"],
            "generation_job_id": job_id,
            "generation_job": existing,
            "skipped": True,
        }

    conversation_id = job["conversation_id"]
    user_message_id = job["user_message_id"]
    payload = job["payload"]
    model_question = payload["question"]
    mode_override = payload["mode_override"]
    rag_policy = payload["rag_policy"]
    confirmed_image_context = payload.get("image_context")

    conversation = get_conversation(conversation_id, path=db_path)
    if conversation is None:
        _mark_job_failed(job_id, "conversation_missing", db_path)
        raise ConversationServiceError("会话不存在，无法执行回答任务。")

    old_state = get_conversation_state(conversation_id, path=db_path)
    try:
        context_bundle = build_context_bundle(
            conversation_id,
            model_question,
            previous_state=old_state,
            path=db_path,
        )
    except Exception as exc:
        _mark_job_failed(job_id, "context_error", db_path)
        raise ConversationServiceError("本轮上下文构建失败，请稍后重试。") from exc
    bundle_inputs = context_bundle_model_inputs(context_bundle)
    history = bundle_inputs["conversation_history"]
    memory_context = bundle_inputs["learning_memory_context"]

    preliminary_resolved = resolve_conversation_state(
        model_question,
        previous_state=old_state,
        current_image_context=confirmed_image_context,
        mode_override=mode_override,
    )
    teaching_state_context = build_analyzer_state_context(old_state)
    effective_model_question = _build_effective_model_question(
        preliminary_resolved,
        model_question,
    )

    agent = agent_func if agent_func is not None else _default_run_agent
    agent_mode_override = (
        preliminary_resolved.teaching_mode
        if preliminary_resolved.teaching_mode is not None
        else mode_override
    )
    try:
        agent_result = agent(
            effective_model_question,
            mode_override=agent_mode_override,
            rag_policy=rag_policy,
            image_context=confirmed_image_context,
            image_context_available=bool(confirmed_image_context),
            conversation_history=history,
            teaching_state_context=teaching_state_context,
            learning_memory_context=memory_context,
            previous_problem_text=(
                old_state.get("active_problem_text")
                if old_state and not preliminary_resolved.is_follow_up
                else None
            ),
            previous_image_context=(
                old_state.get("active_image_context") if old_state else None
            ),
            context_bundle=context_bundle,
        )
    except Exception as exc:
        _mark_job_failed(job_id, "agent_error", db_path)
        raise ConversationServiceError("本轮回答生成失败，请稍后重试。") from exc

    analysis_data = agent_result.get("analysis") or {}
    context_relation = analysis_data.get("context_relation")
    needs_previous_image_context = analysis_data.get(
        "needs_previous_image_context"
    )
    resolved = resolve_conversation_state(
        model_question,
        previous_state=old_state,
        current_image_context=confirmed_image_context,
        mode_override=mode_override,
        context_relation=context_relation,
        needs_previous_image_context=needs_previous_image_context,
    )

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
        finalized = finalize_generation_job(
            job_id,
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
        _mark_job_failed(job_id, "persistence_error", db_path)
        raise ConversationServiceError("本轮结果保存失败，请稍后重试。") from exc

    return {
        "agent_result": agent_result,
        "conversation_id": conversation_id,
        "user_message_id": user_message_id,
        "assistant_message_id": finalized["assistant_message_id"],
        "generation_job_id": job_id,
        "generation_job": finalized["generation_job"],
        "resolved_state": resolved,
        "history_turn_count": len(history) // 2,
        "state_context_used": bool(teaching_state_context),
        "memory_count": _memory_context_count(memory_context),
        "memory_context_used": bool(memory_context),
    }


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
    """同步兼容入口：先入队，再立即执行同一个 Generation Job。"""
    enqueued = enqueue_conversation_turn(
        conversation_id,
        display_question,
        model_question,
        image_metadata=image_metadata,
        confirmed_image_context=confirmed_image_context,
        mode_override=mode_override,
        rag_policy=rag_policy,
        db_path=db_path,
        max_history_turns=max_history_turns,
        max_history_chars=max_history_chars,
    )
    return execute_generation_job(
        enqueued["generation_job_id"],
        db_path=db_path,
        agent_func=agent_func,
    )
