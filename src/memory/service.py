"""Stage 10 长期记忆确认服务：候选确认后写入 learning_memories。"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from src.memory.schemas import MemoryCandidate
from src.storage import RepositoryError, insert_or_merge_memory


class MemoryServiceError(Exception):
    """记忆服务统一异常，不泄露 SQL、绝对路径或密钥。"""


def confirm_memory_candidate(
    candidate: MemoryCandidate | dict[str, Any],
    *,
    source_conversation_id: str | None = None,
    source_message_id: str | None = None,
    db_path=None,
) -> dict[str, Any]:
    """确认一条候选并写入数据库（confirmed=1、active=1）。

    重复确认同一 (memory_type, topic, normalized_content) 走现有去重合并，
    evidence_count 递增；evidence_summary 属于候选评审元数据，V1 表不持久化。
    """
    try:
        candidate_obj = (
            candidate
            if isinstance(candidate, MemoryCandidate)
            else MemoryCandidate.model_validate(candidate)
        )
    except ValidationError as exc:
        raise MemoryServiceError("记忆候选不合法，无法确认。") from exc
    try:
        return insert_or_merge_memory(
            {
                "memory_type": candidate_obj.memory_type,
                "topic": candidate_obj.topic,
                "content": candidate_obj.content,
                "normalized_content": candidate_obj.normalized_content,
                "confidence": candidate_obj.confidence,
                "source_conversation_id": source_conversation_id,
                "source_message_id": source_message_id,
                "confirmed": 1,
                "active": 1,
            },
            path=db_path,
        )
    except RepositoryError as exc:
        raise MemoryServiceError("记忆保存失败，请稍后重试。") from exc
