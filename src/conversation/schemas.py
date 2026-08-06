"""Stage 10 会话领域 Schema（Pydantic v2）。

字段名与 Repository 返回结构保持一致；安全图片元数据统一表示为 list[dict]，
不接受原始图片字节、Base64 或 Data URL。禁止未声明的额外字段。
"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MessageRole = Literal["user", "assistant"]
HintStep = Annotated[int, Field(ge=0)]
MetadataValue = dict[str, object]


def _reject_raw_image_data(value: object) -> None:
    """递归拒绝原始图片字节、Base64 与 Data URL。"""
    if isinstance(value, bytes):
        raise ValueError("不接受原始图片字节。")
    if isinstance(value, str):
        if value.startswith("data:image"):
            raise ValueError("不接受 Data URL。")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in {"base64", "data_url", "data_uri"}:
                raise ValueError("不接受 Base64 或 Data URL 字段。")
            _reject_raw_image_data(key)
            _reject_raw_image_data(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_raw_image_data(item)


def validate_safe_text(value: str | None, field_name: str) -> str | None:
    """校验已确认的安全文本；拒绝 bytes、Data URL 与 Base64 内容。"""
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是安全文本。")
    if value.startswith("data:image"):
        raise ValueError(f"{field_name} 不能包含 Data URL。")
    if re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", value):
        raise ValueError(f"{field_name} 不能包含 Base64 内容。")
    return value


class ConversationRecord(BaseModel):
    """conversations 表的领域记录。"""

    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: str
    updated_at: str
    archived: int = 0


class StoredMessage(BaseModel):
    """匹配 Repository 返回结构的存储消息。

    image_metadata_json 只保存安全图片元数据，统一表示为 list[dict]；
    不接受原始图片字节、Base64 或 Data URL。
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    conversation_id: str
    role: MessageRole
    display_content: str
    model_content: str
    image_metadata_json: list[MetadataValue] = Field(default_factory=list)
    created_at: str

    @field_validator("image_metadata_json", mode="before")
    @classmethod
    def _normalize_image_metadata(cls, value: object) -> list[MetadataValue]:
        if value is None:
            return []
        if isinstance(value, dict):
            _reject_raw_image_data(value)
            return [value]
        if isinstance(value, list):
            _reject_raw_image_data(value)
            for item in value:
                if not isinstance(item, dict):
                    raise ValueError("图片元数据列表项必须是 dict。")
            return value
        raise ValueError("图片元数据必须是 dict 或 list[dict]。")


class ConversationState(BaseModel):
    """conversation_states 表的领域记录。"""

    model_config = ConfigDict(extra="forbid")

    conversation_id: str
    active_problem_text: str | None = None
    active_image_context: str | None = None
    teaching_mode: str | None = None
    hint_step: HintStep = 0
    updated_at: str

    @field_validator("active_image_context", mode="before")
    @classmethod
    def _validate_active_image_context(
        cls,
        value: object,
    ) -> object:
        return validate_safe_text(value, "active_image_context")
