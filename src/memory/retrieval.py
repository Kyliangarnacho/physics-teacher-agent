"""Stage 10 长期记忆检索与模型上下文构建。

只召回 confirmed=1 且 active=1 的记忆；weakness/misconception 必须与当前
问题或 active_problem_text 有明确主题关联（确定性规则，不调用模型）；
preference 可跨主题使用但最多 2 条。检索失败由调用方安全降级。
"""

from __future__ import annotations

import re
from typing import Any

from src.memory.schemas import (
    normalize_memory_content,
    validate_safe_text,
)
from src.storage import list_memories


MAX_PREFERENCES = 2
DEFAULT_MAX_MEMORIES = 5
DEFAULT_MAX_CHARS = 1500

_STOPWORDS = {
    "的",
    "了",
    "是",
    "和",
    "与",
    "在",
    "中",
    "吗",
    "呢",
    "什么",
    "怎么",
    "如何",
    "为什么",
    "请",
    "求",
}


def _keywords(text: str) -> set[str]:
    """简单确定性关键词：英文单词 + 中文双字滑动窗口 + 单字（过滤停用词）。"""
    text = normalize_memory_content(text)
    words = set(re.findall(r"[A-Za-z0-9]+", text))
    chars = [char for char in text if "\u4e00" <= char <= "\u9fff"]
    grams = {
        chars[index] + chars[index + 1]
        for index in range(len(chars) - 1)
    }
    singles = {
        char
        for char in chars
        if char not in _STOPWORDS
    }
    return words | grams | singles


def _relevance_score(
    memory: dict[str, Any],
    query_text: str,
    active_problem_text: str | None,
) -> int:
    """weakness/misconception 的主题相关性得分；0 表示无关联。"""
    combined = query_text
    if active_problem_text:
        combined = f"{query_text}\n{active_problem_text}"
    query_keywords = _keywords(combined)
    topic_keywords = _keywords(str(memory.get("topic", "")))
    content_keywords = _keywords(str(memory.get("content", "")))
    normalized_topic = normalize_memory_content(str(memory.get("topic", "")))

    score = 0
    if normalized_topic and normalized_topic in combined:
        score += 3
    if topic_keywords & query_keywords:
        score += 2
    if content_keywords & query_keywords:
        score += 1
    return score


def _safe_memory(memory: dict[str, Any]) -> bool:
    """记忆必须可安全注入：文本字段为安全字符串。"""
    if memory.get("confirmed") != 1 or memory.get("active") != 1:
        return False
    if memory.get("memory_type") not in ("weakness", "misconception", "preference"):
        return False
    for name in ("topic", "content", "normalized_content"):
        value = memory.get(name)
        if not isinstance(value, str):
            return False
        try:
            validate_safe_text(value, name)
        except ValueError:
            return False
    return True


def retrieve_relevant_memories(
    query: str,
    *,
    active_problem_text: str | None = None,
    db_path=None,
    max_memories: int = DEFAULT_MAX_MEMORIES,
) -> list[dict[str, Any]]:
    """检索最多 max_memories 条可注入的长期记忆。

    preference 跨主题、最多 2 条；weakness/misconception 仅保留主题相关，
    按相关性得分降序、同分按 updated_at 倒序。
    """
    if not isinstance(query, str) or not query.strip():
        raise ValueError("query 不能为空。")
    if not isinstance(max_memories, int) or max_memories <= 0:
        raise ValueError("max_memories 必须是正整数。")

    candidates = [
        memory
        for memory in list_memories(include_inactive=False, path=db_path)
        if _safe_memory(memory)
    ]
    preferences = [
        memory
        for memory in candidates
        if memory["memory_type"] == "preference"
    ]
    topical = [
        memory
        for memory in candidates
        if memory["memory_type"] in ("weakness", "misconception")
    ]

    def preference_key(memory: dict[str, Any]) -> tuple[int, str]:
        return (
            _relevance_score(memory, query, active_problem_text),
            str(memory.get("updated_at", "")),
        )

    def topical_key(memory: dict[str, Any]) -> tuple[int, str]:
        return (
            _relevance_score(memory, query, active_problem_text),
            str(memory.get("updated_at", "")),
        )

    preferences.sort(key=preference_key, reverse=True)
    selected_preferences = preferences[:MAX_PREFERENCES]

    scored_topical = [
        (memory, _relevance_score(memory, query, active_problem_text))
        for memory in topical
    ]
    related_topical = [
        memory
        for memory, score in scored_topical
        if score > 0
    ]
    related_topical.sort(key=topical_key, reverse=True)

    remaining = max(0, max_memories - len(selected_preferences))
    result = related_topical[:remaining] + selected_preferences
    return result[:max_memories]


def build_learning_memory_context(
    memories: list[dict[str, Any]],
    *,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> str | None:
    """把记忆构造成模型可注入的安全文本（只含类型、主题与内容）。

    超出 max_chars 时按整条丢弃多余记忆，不拆开条目。
    """
    if not memories:
        return None
    labels = {
        "weakness": "薄弱点",
        "misconception": "错误观念",
        "preference": "讲解偏好",
    }
    entries: list[str] = []
    for memory in memories:
        memory_type = memory.get("memory_type")
        if memory_type not in labels:
            continue
        topic = str(memory.get("topic", "")).strip()
        content = str(memory.get("content", "")).strip()
        if not topic or not content:
            continue
        try:
            validate_safe_text(topic, "topic")
            validate_safe_text(content, "content")
        except ValueError:
            continue
        entries.append(f"- [{labels[memory_type]}] {topic}：{content}")
    if not entries:
        return None

    header = "以下是长期记忆参考（仅作参考，不要当成新的物理结论）："
    selected: list[str] = []
    total = len(header)
    for entry in entries:
        cost = len(entry) + 1
        if total + cost > max_chars:
            break
        selected.append(entry)
        total += cost
    if not selected:
        return None
    return header + "\n" + "\n".join(selected)
