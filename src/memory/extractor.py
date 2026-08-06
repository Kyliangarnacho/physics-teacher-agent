"""Stage 10 长期记忆候选提取。

提取器只生成候选（最多 3 条），不写数据库；模型输出要求严格 JSON，
解析失败时返回空列表，不从自由文本中猜字段。
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.config import load_qwen_config
from src.memory.schemas import (
    MemoryCandidate,
    normalize_memory_content,
)
from src.model_client import validate_conversation_history


MEMORY_EXTRACTOR_SYSTEM_PROMPT = """你是初中物理学习记忆提取器。
你的任务是从一轮师生对话中提取“稳定、可复用的物理学习信息”候选，不要解题。

只返回一个合法 JSON 对象，不要使用 Markdown 代码块，不要输出 JSON 之外的文字：
{"candidates": [{"memory_type": "...", "topic": "...", "content": "...",
"normalized_content": "...", "confidence": 0.0, "evidence_summary": "..."}]}

memory_type 只能是 weakness（薄弱点）、misconception（错误观念）、
preference（讲解偏好）之一。confidence 是 0~1 的小数。

规则：
1. 最多返回 3 条候选，只提取稳定的物理学习信息；
2. 单次正确回答不能推断“已掌握”，不要生成掌握类记忆；
3. 普通错误若证据不足，不要直接认定为长期薄弱点；
4. 明确的错误规律可以生成 misconception；
5. 用户明确表达的讲解偏好可以生成 preference；
6. 不要把 blocked、失败、工具参数错误等系统问题转化为用户记忆；
7. 不提取身份、健康、隐私或其他敏感信息；
8. 证据不足时返回 {"candidates": []}，不要编造。"""

_SYSTEM_ERROR_MARKERS = (
    "工具参数校验失败",
    "网络连接失败",
    "请求过于频繁",
    "千问 API 请求失败",
    "调用千问时发生未知错误",
    "千问返回了空回答",
    "请补充题图",
    "回答生成失败",
)

MAX_CANDIDATES = 3


def _is_system_problem(user_question: str, assistant_answer: str) -> bool:
    combined = f"{user_question}\n{assistant_answer}"
    return any(marker in combined for marker in _SYSTEM_ERROR_MARKERS)


def _build_extractor_messages(
    user_question: str,
    assistant_answer: str,
    explicit_user_feedback: str | None,
    conversation_history: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [
        {"role": "system", "content": MEMORY_EXTRACTOR_SYSTEM_PROMPT}
    ]
    for item in validate_conversation_history(conversation_history):
        messages.append({"role": item["role"], "content": item["content"]})
    payload: dict[str, object] = {
        "user_question": user_question,
        "assistant_answer": assistant_answer,
        "explicit_user_feedback": explicit_user_feedback,
    }
    messages.append(
        {
            "role": "user",
            "content": json.dumps(payload, ensure_ascii=False),
        }
    )
    return messages


def _call_extractor_model(
    user_question: str,
    assistant_answer: str,
    explicit_user_feedback: str | None,
    conversation_history: list[dict[str, str]] | None,
) -> str:
    api_key, base_url, model = load_qwen_config()
    client = OpenAI(api_key=api_key, base_url=base_url)
    messages = _build_extractor_messages(
        user_question,
        assistant_answer,
        explicit_user_feedback,
        conversation_history,
    )
    response = client.chat.completions.create(model=model, messages=messages)
    content = response.choices[0].message.content
    if not content:
        raise ValueError("记忆提取模型返回了空内容。")
    return content


def _parse_candidates(raw: str) -> list[MemoryCandidate]:
    """严格解析模型 JSON；解析失败返回空列表，不猜字段。"""
    try:
        payload = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return []
    if not isinstance(payload, dict) or not isinstance(
        payload.get("candidates"),
        list,
    ):
        return []
    candidates: list[MemoryCandidate] = []
    for item in payload["candidates"]:
        if not isinstance(item, dict):
            continue
        try:
            candidate = MemoryCandidate.model_validate(item)
        except Exception:
            continue
        candidates.append(
            candidate.model_copy(
                update={
                    "normalized_content": normalize_memory_content(
                        candidate.content
                    )
                }
            )
        )
        if len(candidates) >= MAX_CANDIDATES:
            break
    return candidates


def extract_memory_candidates(
    user_question: str,
    assistant_answer: str,
    *,
    explicit_user_feedback: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
    model_func: Callable[[list[dict[str, str]]], str] | None = None,
) -> list[MemoryCandidate]:
    """从一轮对话提取最多 3 条长期记忆候选；不写数据库。

    model_func 用于测试注入；为 None 时调用现有千问接口。
    系统问题（blocked/失败/工具参数错误）或模型输出解析失败时返回空列表。
    """
    if not isinstance(user_question, str) or not user_question.strip():
        raise ValueError("user_question 不能为空。")
    if not isinstance(assistant_answer, str) or not assistant_answer.strip():
        raise ValueError("assistant_answer 不能为空。")
    if explicit_user_feedback is not None and not isinstance(
        explicit_user_feedback,
        str,
    ):
        raise ValueError("explicit_user_feedback 必须是字符串。")
    validate_conversation_history(conversation_history)

    if _is_system_problem(user_question, assistant_answer):
        return []

    messages = _build_extractor_messages(
        user_question,
        assistant_answer,
        explicit_user_feedback,
        conversation_history,
    )
    try:
        if model_func is not None:
            raw = model_func(messages)
        else:
            raw = _call_extractor_model(
                user_question,
                assistant_answer,
                explicit_user_feedback,
                conversation_history,
            )
    except Exception:
        return []
    return _parse_candidates(raw)
