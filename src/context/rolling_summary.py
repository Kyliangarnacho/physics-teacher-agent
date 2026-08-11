"""Deterministic rolling-summary selection and explicit refresh service."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from pydantic import ValidationError

from src.config import load_qwen_config
from src.conversation.schemas import StoredMessage
from src.context.schemas import ConversationSummary, RollingSummaryOutput
from src.retry import run_with_one_retry
from src.storage.database import DatabasePath
from src.storage.repositories import (
    RepositoryError,
    get_conversation_summary,
    list_generation_jobs,
    list_messages,
    upsert_conversation_summary,
)


RECENT_RAW_TURNS = 3
INITIAL_TURN_THRESHOLD = 4
INITIAL_CHAR_THRESHOLD = 6000
ROLLING_TURN_THRESHOLD = 2
ROLLING_CHAR_THRESHOLD = 4000


_SYSTEM_PROMPT = """你负责维护物理辅导对话的滚动摘要。只压缩已有对话，不解题。
请保留未来可能引用的重要明确数值与条件、已确认图片的安全文字关系、重要结论与定义、尚未解决的问题，以及用于辨识旧题的特征。不确定内容必须继续标记为不确定；当前明确条件优先于旧内容。
不要推断用户人格或长期学习弱点，不要把摘要写成 Learning Memory，不要添加原对话没有的事实。不得保存 Trace、Tool 内部协议、API Key、SQL、Base64、Data URL，也不得输出隐藏思维链。
请将旧摘要与本次新增旧轮次合并成简洁摘要，不要生成长篇 Markdown 报告，控制在约 2200 个中文字符以内。
只输出严格 JSON：{"summary":"..."}。"""


@dataclass(frozen=True)
class SummaryTurn:
    """One complete, safe user-to-assistant turn eligible for summarization."""

    user_message_id: str
    user_content: str
    assistant_message_id: str
    assistant_content: str
    confirmed_image_context: str | None = None

    @property
    def char_count(self) -> int:
        return (
            len(self.user_content)
            + len(self.assistant_content)
            + len(self.confirmed_image_context or "")
        )

    def to_prompt_dict(self) -> dict[str, str]:
        payload = {
            "user_message_id": self.user_message_id,
            "user": self.user_content,
            "assistant_message_id": self.assistant_message_id,
            "assistant": self.assistant_content,
        }
        if self.confirmed_image_context:
            payload["confirmed_image_context"] = self.confirmed_image_context
        return payload


@dataclass(frozen=True)
class SummaryRefreshDecision:
    """Structured stale decision plus the exact delta approved for compression."""

    should_refresh: bool
    eligible_turn_count: int
    new_turn_count: int
    eligible_chars: int
    reason: str
    eligible_turns: tuple[SummaryTurn, ...] = field(default=(), repr=False)


@dataclass(frozen=True)
class UnsummarizedBridgeSelection:
    """Old raw turns not yet proven covered by the persisted summary boundary."""

    bridge_turns: tuple[SummaryTurn, ...]
    covered_turns: tuple[SummaryTurn, ...]
    recent_turns: tuple[SummaryTurn, ...]
    old_turn_count: int
    boundary_status: str


def _summary_model(
    summary: ConversationSummary | Mapping[str, Any] | None,
) -> ConversationSummary | None:
    if summary is None or isinstance(summary, ConversationSummary):
        return summary
    return ConversationSummary.model_validate(dict(summary))


def _job_value(job: Mapping[str, Any] | Any, key: str, default: Any = None) -> Any:
    if isinstance(job, Mapping):
        return job.get(key, default)
    return getattr(job, key, default)


def _status_text(value: Any) -> str:
    return str(getattr(value, "value", value))


def _payload_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="python")
    return {}


def _confirmed_image_context(job: Mapping[str, Any] | Any | None) -> str | None:
    if job is None or _status_text(_job_value(job, "status")) != "completed":
        return None
    payload = _payload_mapping(_job_value(job, "payload", {}))
    if payload.get("image_context_available") is not True:
        return None
    value = payload.get("image_context")
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _collect_complete_turns(
    messages: Sequence[StoredMessage | Mapping[str, Any]],
    generation_jobs: Sequence[Mapping[str, Any] | Any],
) -> list[SummaryTurn]:
    normalized = [
        item if isinstance(item, StoredMessage) else StoredMessage.model_validate(item)
        for item in messages
    ]
    jobs_by_user_message: dict[str, Mapping[str, Any] | Any] = {}
    for job in generation_jobs:
        user_message_id = _job_value(job, "user_message_id")
        if isinstance(user_message_id, str) and user_message_id:
            jobs_by_user_message[user_message_id] = job

    turns: list[SummaryTurn] = []
    index = 0
    while index + 1 < len(normalized):
        user_message = normalized[index]
        assistant_message = normalized[index + 1]
        if user_message.role != "user" or assistant_message.role != "assistant":
            index += 1
            continue

        job = jobs_by_user_message.get(user_message.id)
        if job is not None and _status_text(_job_value(job, "status")) != "completed":
            index += 2
            continue

        turns.append(
            SummaryTurn(
                user_message_id=user_message.id,
                user_content=user_message.model_content,
                assistant_message_id=assistant_message.id,
                assistant_content=assistant_message.model_content,
                confirmed_image_context=_confirmed_image_context(job),
            )
        )
        index += 2
    return turns


def select_unsummarized_bridge_turns(
    messages: Sequence[StoredMessage | Mapping[str, Any]],
    generation_jobs: Sequence[Mapping[str, Any] | Any],
    existing_summary: ConversationSummary | Mapping[str, Any] | None = None,
) -> UnsummarizedBridgeSelection:
    """Derive Bridge History from messages, the recent window, and the boundary.

    A missing or unknown boundary cannot prove that any old turn is covered, so
    all old turns remain visible in the bridge. Summary refresh separately fails
    closed for those boundary states instead of resummarizing uncertain history.
    """

    summary = _summary_model(existing_summary)
    complete_turns = _collect_complete_turns(messages, generation_jobs)
    old_turns = (
        complete_turns[:-RECENT_RAW_TURNS]
        if len(complete_turns) > RECENT_RAW_TURNS
        else []
    )
    recent_turns = (
        complete_turns[-RECENT_RAW_TURNS:]
        if complete_turns
        else []
    )
    if summary is None:
        return UnsummarizedBridgeSelection(
            bridge_turns=tuple(old_turns),
            covered_turns=(),
            recent_turns=tuple(recent_turns),
            old_turn_count=len(old_turns),
            boundary_status="no_summary",
        )

    boundary = summary.covered_until_message_id
    if not boundary:
        return UnsummarizedBridgeSelection(
            bridge_turns=tuple(old_turns),
            covered_turns=(),
            recent_turns=tuple(recent_turns),
            old_turn_count=len(old_turns),
            boundary_status="missing",
        )
    boundary_index = next(
        (
            index
            for index, turn in enumerate(complete_turns)
            if turn.assistant_message_id == boundary
        ),
        None,
    )
    if boundary_index is None:
        return UnsummarizedBridgeSelection(
            bridge_turns=tuple(old_turns),
            covered_turns=(),
            recent_turns=tuple(recent_turns),
            old_turn_count=len(old_turns),
            boundary_status="not_found",
        )
    return UnsummarizedBridgeSelection(
        bridge_turns=tuple(
            turn
            for index, turn in enumerate(old_turns)
            if index > boundary_index
        ),
        covered_turns=tuple(
            turn
            for index, turn in enumerate(old_turns)
            if index <= boundary_index
        ),
        recent_turns=tuple(recent_turns),
        old_turn_count=len(old_turns),
        boundary_status="found",
    )


def should_refresh_summary(
    messages: Sequence[StoredMessage | Mapping[str, Any]],
    generation_jobs: Sequence[Mapping[str, Any] | Any],
    existing_summary: ConversationSummary | Mapping[str, Any] | None = None,
) -> SummaryRefreshDecision:
    """Select the new old-turn delta and decide whether it is stale enough."""

    summary = _summary_model(existing_summary)
    bridge = select_unsummarized_bridge_turns(
        messages,
        generation_jobs,
        summary,
    )
    if summary is not None and bridge.boundary_status == "missing":
        return SummaryRefreshDecision(
            should_refresh=False,
            eligible_turn_count=bridge.old_turn_count,
            new_turn_count=0,
            eligible_chars=0,
            reason="covered_boundary_missing",
        )
    if summary is not None and bridge.boundary_status == "not_found":
        return SummaryRefreshDecision(
            should_refresh=False,
            eligible_turn_count=bridge.old_turn_count,
            new_turn_count=0,
            eligible_chars=0,
            reason="covered_boundary_not_found",
        )

    new_turns = list(bridge.bridge_turns)

    new_turn_count = len(new_turns)
    eligible_chars = sum(turn.char_count for turn in new_turns)
    if summary is None:
        should_refresh = (
            new_turn_count >= INITIAL_TURN_THRESHOLD
            or eligible_chars >= INITIAL_CHAR_THRESHOLD
        )
        if new_turn_count >= INITIAL_TURN_THRESHOLD:
            reason = "initial_turn_threshold"
        elif eligible_chars >= INITIAL_CHAR_THRESHOLD:
            reason = "initial_char_threshold"
        else:
            reason = "initial_threshold_not_met"
    else:
        should_refresh = (
            new_turn_count >= ROLLING_TURN_THRESHOLD
            or eligible_chars >= ROLLING_CHAR_THRESHOLD
        )
        if new_turn_count >= ROLLING_TURN_THRESHOLD:
            reason = "rolling_turn_threshold"
        elif eligible_chars >= ROLLING_CHAR_THRESHOLD:
            reason = "rolling_char_threshold"
        else:
            reason = "rolling_threshold_not_met"

    return SummaryRefreshDecision(
        should_refresh=should_refresh,
        eligible_turn_count=bridge.old_turn_count,
        new_turn_count=new_turn_count,
        eligible_chars=eligible_chars,
        reason=reason,
        eligible_turns=tuple(new_turns),
    )


def build_summary_model_messages(
    decision: SummaryRefreshDecision,
    existing_summary: ConversationSummary | Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    """Build the bounded prompt from the prior summary and only the new delta."""

    summary = _summary_model(existing_summary)
    payload = {
        "existing_summary": summary.summary_text if summary else None,
        "newly_eligible_turns": [
            turn.to_prompt_dict() for turn in decision.eligible_turns
        ],
    }
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _default_should_retry(error: Exception) -> bool:
    if isinstance(error, (APIConnectionError, RateLimitError)):
        return True
    if isinstance(error, APIError):
        status_code = getattr(error, "status_code", None)
        return status_code is None or status_code >= 500
    return False


def _safe_failure(
    *,
    decision: SummaryRefreshDecision,
    existing_summary: Mapping[str, Any] | None,
    error_type: str,
    attempts: int,
) -> dict[str, Any]:
    return {
        "status": "failed",
        "decision": decision,
        "summary": existing_summary,
        "error_type": error_type,
        "error_message": "滚动摘要更新失败，旧摘要保持不变。",
        "model_requests": attempts,
    }


def refresh_conversation_summary(
    conversation_id: str,
    *,
    completion_func: Callable[..., Any] | None = None,
    should_retry: Callable[[Exception], bool] | None = None,
    model_name: str | None = None,
    path: DatabasePath | None = None,
) -> dict[str, Any]:
    """Explicitly refresh one conversation summary when deterministic rules allow."""

    messages = list_messages(conversation_id, path=path)
    jobs = list_generation_jobs(conversation_id, path=path)
    existing = get_conversation_summary(conversation_id, path=path)
    decision = should_refresh_summary(messages, jobs, existing)
    if not decision.should_refresh:
        return {
            "status": "skipped",
            "decision": decision,
            "summary": existing,
            "error_type": None,
            "error_message": None,
            "model_requests": 0,
        }

    if completion_func is None:
        try:
            api_key, base_url, configured_model = load_qwen_config()
            client = OpenAI(api_key=api_key, base_url=base_url)
            resolved_model_name = model_name or configured_model
            completion_func = client.chat.completions.create
        except Exception:
            return _safe_failure(
                decision=decision,
                existing_summary=existing,
                error_type="summary_configuration_error",
                attempts=0,
            )
    else:
        resolved_model_name = model_name

    model_messages = build_summary_model_messages(decision, existing)

    def _complete() -> Any:
        return completion_func(
            model=resolved_model_name,
            messages=model_messages,
            stream=False,
            extra_body={"enable_thinking": False},
        )

    outcome = run_with_one_retry(
        _complete,
        should_retry=should_retry or _default_should_retry,
    )
    if outcome.error is not None:
        return _safe_failure(
            decision=decision,
            existing_summary=existing,
            error_type="summary_api_error",
            attempts=outcome.attempts,
        )

    try:
        raw_content = outcome.value.choices[0].message.content
        parsed = json.loads(raw_content)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError):
        return _safe_failure(
            decision=decision,
            existing_summary=existing,
            error_type="summary_parse_error",
            attempts=outcome.attempts,
        )

    try:
        output = RollingSummaryOutput.model_validate(parsed)
    except ValidationError:
        return _safe_failure(
            decision=decision,
            existing_summary=existing,
            error_type="summary_schema_error",
            attempts=outcome.attempts,
        )

    covered_turn_count = (
        (existing["covered_turn_count"] if existing else 0)
        + decision.new_turn_count
    )
    last_turn = decision.eligible_turns[-1]
    try:
        saved = upsert_conversation_summary(
            conversation_id,
            output.summary,
            covered_until_message_id=last_turn.assistant_message_id,
            covered_turn_count=covered_turn_count,
            model_name=resolved_model_name,
            path=path,
        )
    except (RepositoryError, ValidationError):
        return _safe_failure(
            decision=decision,
            existing_summary=existing,
            error_type="summary_persistence_error",
            attempts=outcome.attempts,
        )

    return {
        "status": "updated",
        "decision": decision,
        "summary": saved,
        "error_type": None,
        "error_message": None,
        "model_requests": outcome.attempts,
    }
