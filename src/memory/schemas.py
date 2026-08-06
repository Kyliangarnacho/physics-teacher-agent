"""Stage 10 长期记忆候选 Schema 与内容规范化。"""

from __future__ import annotations

import re
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)


MemoryType = Literal["weakness", "misconception", "preference"]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
SafeText = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

_SECRET_MARKERS = (
    "api_key",
    "apikey",
    "secret",
    "password",
    "token",
    "bearer",
    "authorization",
    "dashscope",
)


def validate_safe_text(value: str, field_name: str) -> str:
    """校验安全文本：拒绝 bytes、Data URL、Base64 与密钥内容。"""
    if not isinstance(value, str):
        raise ValueError(f"{field_name} 必须是字符串。")
    if value.startswith("data:image"):
        raise ValueError(f"{field_name} 不能包含 Data URL。")
    if re.fullmatch(r"[A-Za-z0-9+/]{16,}={0,2}", value):
        raise ValueError(f"{field_name} 不能包含 Base64 内容。")
    lowered = value.lower()
    if "sk-" in lowered or any(marker in lowered for marker in _SECRET_MARKERS):
        raise ValueError(f"{field_name} 不能包含密钥内容。")
    return value.strip()


def _fullwidth_to_halfwidth(text: str) -> str:
    result: list[str] = []
    for char in text:
        code = ord(char)
        if char == "\u3000":
            result.append(" ")
        elif 0xFF01 <= code <= 0xFF5E:
            result.append(chr(code - 0xFEE0))
        else:
            result.append(char)
    return "".join(result)


def normalize_memory_content(text: str) -> str:
    """规范化记忆内容：安全校验、全角转半角并压缩连续空白。"""
    if not isinstance(text, str):
        raise ValueError("记忆内容必须是字符串。")
    validate_safe_text(text, "记忆内容")
    normalized = _fullwidth_to_halfwidth(text.strip())
    return re.sub(r"\s+", " ", normalized).strip()


class MemoryCandidate(BaseModel):
    """一条待确认的长期记忆候选；只生成候选，不直接写数据库。"""

    model_config = ConfigDict(extra="forbid")

    memory_type: MemoryType
    topic: SafeText
    content: SafeText
    normalized_content: SafeText
    confidence: Confidence
    evidence_summary: SafeText

    @model_validator(mode="after")
    def _validate_text_safety(self) -> "MemoryCandidate":
        for name in ("topic", "content", "normalized_content", "evidence_summary"):
            validate_safe_text(getattr(self, name), name)
        return self
