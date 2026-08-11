"""Stage 11.3 会话摘要的严格 Pydantic v2 数据合同。"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]
SummaryText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=2200),
]
NonNegativeInt = Annotated[int, Field(ge=0)]
PositiveInt = Annotated[int, Field(ge=1)]


def _validate_summary_text(value: str) -> str:
    lowered = value.casefold()
    if "data:image" in lowered or ";base64," in lowered:
        raise ValueError("摘要不能包含 Data URL。")
    if (
        len(value) >= 16
        and len(value) % 4 == 0
        and re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", value)
    ):
        raise ValueError("摘要不能包含 Base64 内容。")
    unsafe_markers = (
        "api_key",
        "api key",
        "apikey",
        "authorization:",
        "bearer ",
        "sk-",
        "trace_json",
        "tool_records",
        "tool_call_id",
        "function_call",
    )
    if any(marker in lowered for marker in unsafe_markers):
        raise ValueError("Context text contains secret or internal protocol data.")
    return value


class ConversationSummary(BaseModel):
    """conversation_summaries 表的一条严格记录。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    conversation_id: NonEmptyText
    summary_text: SummaryText
    covered_until_message_id: NonEmptyText | None = None
    covered_turn_count: NonNegativeInt
    summary_revision: PositiveInt
    model_name: NonEmptyText | None = None
    created_at: NonEmptyText
    updated_at: NonEmptyText

    @field_validator("summary_text")
    @classmethod
    def _summary_must_be_safe(cls, value: str) -> str:
        return _validate_summary_text(value)


class RollingSummaryOutput(BaseModel):
    """未来 Summary 模型输出的最小结构；本阶段不负责调用模型。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    summary: SummaryText

    @field_validator("summary")
    @classmethod
    def _summary_must_be_safe(cls, value: str) -> str:
        return _validate_summary_text(value)


class RetrievedHistoryTurn(BaseModel):
    """Minimal safe projection of one retrieved conversation turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user_message_id: NonEmptyText
    assistant_message_id: NonEmptyText
    user_content: NonEmptyText
    assistant_content: NonEmptyText
    score: Annotated[float, Field(ge=0, allow_inf_nan=False)]

    @field_validator("user_content", "assistant_content")
    @classmethod
    def _content_must_be_safe(cls, value: str) -> str:
        return _validate_summary_text(value)


class HistoryRetrievalResult(BaseModel):
    """Conversation-scoped BM25 result and deterministic retrieval statistics."""

    model_config = ConfigDict(extra="forbid", strict=True)

    turns: list[RetrievedHistoryTurn]
    candidate_count: NonNegativeInt
    query: NonEmptyText
    retrieval_used: StrictBool

    @field_validator("query")
    @classmethod
    def _query_must_be_safe(cls, value: str) -> str:
        return _validate_summary_text(value)


class ContextTurn(BaseModel):
    """One complete safe user-to-assistant turn in a context snapshot."""

    model_config = ConfigDict(extra="forbid", strict=True)

    user_message_id: NonEmptyText
    assistant_message_id: NonEmptyText
    user_content: NonEmptyText
    assistant_content: NonEmptyText

    @field_validator("user_content", "assistant_content")
    @classmethod
    def _content_must_be_safe(cls, value: str) -> str:
        return _validate_summary_text(value)


TrimmedContextComponent = Literal[
    "retrieved_history",
    "learning_memory_context",
    "rolling_summary",
    "unsafe_history",
]


class ContextBundle(BaseModel):
    """Structured, detached context snapshot for one future model turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    rolling_summary: SummaryText | None = None
    bridge_history: list[ContextTurn]
    recent_history: list[ContextTurn]
    retrieved_history: HistoryRetrievalResult
    teaching_state_context: NonEmptyText | None = None
    learning_memory_context: NonEmptyText | None = None
    estimated_chars: NonNegativeInt
    budget_limit: PositiveInt
    current_question_chars: PositiveInt
    trimmed_components: list[TrimmedContextComponent]
    budget_exceeded: StrictBool
    summary_revision: PositiveInt | None = None
    bridge_turn_count: NonNegativeInt
    recent_turn_count: NonNegativeInt
    retrieved_turn_count: NonNegativeInt

    @field_validator(
        "rolling_summary",
        "teaching_state_context",
        "learning_memory_context",
    )
    @classmethod
    def _optional_context_must_be_safe(cls, value: str | None) -> str | None:
        return _validate_summary_text(value) if value is not None else None

    @model_validator(mode="after")
    def _statistics_must_match_snapshot(self) -> "ContextBundle":
        if (self.rolling_summary is None) != (self.summary_revision is None):
            raise ValueError(
                "rolling_summary and summary_revision must appear together."
            )
        if self.bridge_turn_count != len(self.bridge_history):
            raise ValueError("bridge_turn_count does not match bridge_history.")
        if self.recent_turn_count != len(self.recent_history):
            raise ValueError("recent_turn_count does not match recent_history.")
        if self.retrieved_turn_count != len(self.retrieved_history.turns):
            raise ValueError(
                "retrieved_turn_count does not match retrieved_history.turns."
            )
        if len(self.trimmed_components) != len(set(self.trimmed_components)):
            raise ValueError("trimmed_components must not contain duplicates.")
        if not self.budget_exceeded and self.estimated_chars > self.budget_limit:
            raise ValueError("estimated_chars exceeds budget without metadata.")
        return self
