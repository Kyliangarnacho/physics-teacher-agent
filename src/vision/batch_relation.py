"""Batch-level relation analysis for two or three extracted images."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from enum import Enum
from typing import Any, Annotated, Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from src.config import load_vision_config
from src.retry import run_with_one_retry


class BatchRelation(str, Enum):
    SAME_PROBLEM = "same_problem"
    INDEPENDENT = "independent"
    UNCERTAIN = "uncertain"


class BatchImageRole(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: Annotated[int, Field(ge=1)]
    role: Literal[
        "stem",
        "diagram",
        "options",
        "continuation",
        "subquestion",
        "other",
    ]


class BatchRelationAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    relationship: BatchRelation
    image_roles: list[BatchImageRole] = Field(default_factory=list)
    combined_context: str
    short_reason: Annotated[str, Field(min_length=1, max_length=120)]


class BatchRelationError(ValueError):
    """Safe failure from the batch relation model."""


_PROMPT = """你只负责判断同一次上传的多张物理题图之间的关系，不解题。
优先考虑它们可能属于同一道题，并判断为 same_problem、independent 或 uncertain。
same_problem 时按顺序标注每张图的作用：stem、diagram、options、continuation、subquestion 或 other，
并把题干、图示、选项和小问融合成一段完整、忠实的 combined_context。
只有明确是互不相关的题目时才用 independent；证据不足时用 uncertain，不要猜。
只返回符合 JSON Schema 的 JSON，不输出 Markdown 或额外文字。"""


def _content(response: Any) -> str:
    try:
        value = response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise BatchRelationError("多图关系模型响应结构无效。") from exc
    if not isinstance(value, str) or not value.strip():
        raise BatchRelationError("多图关系模型返回了空内容。")
    stripped = value.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(.*?)\s*```",
        stripped,
        flags=re.IGNORECASE | re.DOTALL,
    )
    return fenced.group(1).strip() if fenced else stripped


def analyze_batch_relation(
    image_summaries: list[dict[str, object]],
    *,
    user_instruction: str = "",
    completion_func: Callable[..., Any] | None = None,
) -> dict[str, object]:
    """Classify and merge already-extracted image summaries."""
    if not isinstance(image_summaries, list) or not 2 <= len(image_summaries) <= 3:
        raise ValueError("多图关系分析只支持 2~3 张图片。")
    if not isinstance(user_instruction, str):
        raise ValueError("user_instruction 必须是字符串。")

    options: dict[str, object] = {}
    if completion_func is None:
        api_key, base_url, model = load_vision_config()
        completion = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create
        options["model"] = model
    else:
        completion = completion_func
    schema = json.dumps(
        BatchRelationAnalysis.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    payload = {
        "images": image_summaries,
        "user_instruction": user_instruction.strip() or None,
    }
    messages = [
        {"role": "system", "content": f"{_PROMPT}\nJSON Schema：{schema}"},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]

    outcome = run_with_one_retry(
        lambda: completion(
            **options,
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=False,
        ),
        should_retry=lambda _error: True,
        delay_seconds=0.0,
    )
    if not outcome.succeeded:
        raise BatchRelationError("多图关系分析调用失败。") from outcome.error
    try:
        raw = json.loads(_content(outcome.value))
    except json.JSONDecodeError as exc:
        raise BatchRelationError("多图关系结果不是合法 JSON。") from exc
    try:
        analysis = BatchRelationAnalysis.model_validate(raw)
    except ValidationError as exc:
        raise BatchRelationError("多图关系结果不符合字段要求。") from exc
    expected = list(range(1, len(image_summaries) + 1))
    actual = [item.index for item in analysis.image_roles]
    if analysis.relationship is BatchRelation.SAME_PROBLEM and actual != expected:
        raise BatchRelationError("同一题的多图作用标注不完整。")
    return {"relation": analysis, "model_requests": outcome.attempts}
