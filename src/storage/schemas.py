"""后台回答 Generation Job 的严格 Pydantic v2 数据合同。"""

from __future__ import annotations

import re
from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    field_validator,
    model_validator,
)


NonEmptyText = Annotated[str, Field(min_length=1)]


def _validate_safe_text(value: object, field_name: str) -> object:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串。")
    lowered = value.casefold()
    if "data:image" in lowered or ";base64," in lowered:
        raise ValueError(f"{field_name} 不能包含 Data URL。")
    if re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", value):
        raise ValueError(f"{field_name} 不能包含 Base64 内容。")
    return value


class GenerationJobStatus(str, Enum):
    """后台回答 Job 的持久状态。"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


class GenerationJobPayload(BaseModel):
    """后台回答所需的安全、可持久化输入。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    question: NonEmptyText
    mode_override: Literal["auto", "solve", "explain", "hint", "diagnose"] = (
        "auto"
    )
    rag_policy: Literal["auto", "force", "off"] = "auto"
    image_context: str | None = None
    image_context_available: StrictBool = False
    max_history_turns: Annotated[int, Field(ge=0)] = 3
    max_history_chars: Annotated[int, Field(ge=0)] = 6000

    @field_validator("question", mode="before")
    @classmethod
    def _validate_question_content(cls, value: object) -> object:
        return _validate_safe_text(value, "question")

    @field_validator("question")
    @classmethod
    def _question_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question 不能为空。")
        return value.strip()

    @field_validator("image_context", mode="before")
    @classmethod
    def _validate_image_context(cls, value: object) -> object:
        return _validate_safe_text(value, "image_context")

    @model_validator(mode="after")
    def _validate_confirmed_image_context(self) -> "GenerationJobPayload":
        if self.image_context_available:
            if self.image_context is None or not self.image_context.strip():
                raise ValueError("图片上下文可用时 image_context 必须为非空字符串。")
        elif self.image_context is not None:
            raise ValueError("未确认的图片上下文不得写入后台 Job。")
        return self


class GenerationJob(BaseModel):
    """generation_jobs 表对外返回的数据合同。"""

    model_config = ConfigDict(extra="forbid", strict=True)

    id: NonEmptyText
    conversation_id: NonEmptyText
    user_message_id: NonEmptyText
    status: GenerationJobStatus
    payload: GenerationJobPayload
    attempts: Annotated[int, Field(ge=0)] = 0
    error_type: str | None = None
    error_message: str | None = None
    created_at: NonEmptyText
    updated_at: NonEmptyText
    started_at: str | None = None
    finished_at: str | None = None

    @model_validator(mode="after")
    def _validate_status_fields(self) -> "GenerationJob":
        has_any_error = self.error_type is not None or self.error_message is not None
        has_complete_error = bool(self.error_type and self.error_type.strip()) and bool(
            self.error_message and self.error_message.strip()
        )
        if self.status is GenerationJobStatus.PENDING:
            if self.started_at is not None or self.finished_at is not None or has_any_error:
                raise ValueError("pending Job 不能携带运行、完成或错误字段。")
        elif self.status is GenerationJobStatus.RUNNING:
            if self.attempts < 1 or not self.started_at or self.finished_at or has_any_error:
                raise ValueError("running Job 的运行字段不完整。")
        elif self.status is GenerationJobStatus.COMPLETED:
            if self.attempts < 1 or not self.started_at or not self.finished_at or has_any_error:
                raise ValueError("completed Job 的完成字段不完整。")
        elif (
            self.attempts < 1
            or not self.started_at
            or not self.finished_at
            or not has_complete_error
        ):
            raise ValueError("失败或中断 Job 必须包含安全错误摘要。")
        return self
