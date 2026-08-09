"""Independent background executor for optional learning-memory analysis."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from threading import RLock
from typing import Any

from src.memory.extractor import extract_memory_candidates


MAX_MEMORY_ANALYSIS_WORKERS = 1


class MemoryAnalysisTaskError(ValueError):
    """Safe scheduling/validation error for memory analysis tasks."""


class MemoryAnalysisTaskManager:
    """Run memory extraction separately from generation workers."""

    def __init__(
        self,
        *,
        extractor_func: Callable[..., list[Any]] | None = None,
    ) -> None:
        self._extractor_func = extractor_func or extract_memory_candidates
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_MEMORY_ANALYSIS_WORKERS,
            thread_name_prefix="memory-analysis",
        )
        self._lock = RLock()
        self._tasks: dict[str, dict[str, Any]] = {}
        self._futures: dict[str, Future] = {}
        self._shutdown = False

    @staticmethod
    def make_key(conversation_id: str, assistant_message_id: str) -> str:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise MemoryAnalysisTaskError("conversation_id 不能为空。")
        if not isinstance(assistant_message_id, str) or not assistant_message_id.strip():
            raise MemoryAnalysisTaskError("assistant_message_id 不能为空。")
        return f"{conversation_id.strip()}:{assistant_message_id.strip()}"

    def _run(self, key: str, request: dict[str, Any]) -> None:
        with self._lock:
            self._tasks[key]["status"] = "running"
        try:
            candidates = self._extractor_func(
                request["user_question"],
                request["assistant_answer"],
                conversation_history=request["conversation_history"],
                raise_on_model_error=True,
            )
            serialized = [
                item.model_dump(mode="json")
                if callable(getattr(item, "model_dump", None))
                else dict(item)
                for item in candidates
            ]
        except Exception:
            with self._lock:
                self._tasks[key].update(
                    {
                        "status": "failed",
                        "error": "记忆分析失败，请稍后重试。",
                        "candidates": [],
                    }
                )
            return
        with self._lock:
            self._tasks[key].update(
                {
                    "status": "completed",
                    "error": None,
                    "candidates": serialized,
                }
            )

    def submit(
        self,
        conversation_id: str,
        assistant_message_id: str,
        *,
        user_message_id: str,
        user_question: str,
        assistant_answer: str,
        conversation_history: list[dict[str, str]],
    ) -> bool:
        """Schedule exactly one task for a conversation/assistant pair."""
        key = self.make_key(conversation_id, assistant_message_id)
        if not isinstance(user_message_id, str) or not user_message_id.strip():
            raise MemoryAnalysisTaskError("user_message_id 不能为空。")
        if not isinstance(user_question, str) or not user_question.strip():
            raise MemoryAnalysisTaskError("user_question 不能为空。")
        if not isinstance(assistant_answer, str) or not assistant_answer.strip():
            raise MemoryAnalysisTaskError("assistant_answer 不能为空。")
        request = {
            "user_question": user_question,
            "assistant_answer": assistant_answer,
            "conversation_history": deepcopy(conversation_history),
        }
        with self._lock:
            if self._shutdown:
                raise MemoryAnalysisTaskError("记忆分析任务管理器已关闭。")
            if key in self._tasks:
                return False
            self._tasks[key] = {
                "key": key,
                "conversation_id": conversation_id,
                "assistant_message_id": assistant_message_id,
                "user_message_id": user_message_id,
                "status": "pending",
                "error": None,
                "candidates": [],
            }
            future = self._executor.submit(self._run, key, request)
            self._futures[key] = future
            future.add_done_callback(
                lambda _future, task_key=key: self._discard_future(task_key)
            )
            return True

    def _discard_future(self, key: str) -> None:
        with self._lock:
            self._futures.pop(key, None)

    def get(self, conversation_id: str, assistant_message_id: str) -> dict[str, Any] | None:
        key = self.make_key(conversation_id, assistant_message_id)
        with self._lock:
            task = self._tasks.get(key)
            return deepcopy(task) if task is not None else None

    def is_scheduled(self, conversation_id: str, assistant_message_id: str) -> bool:
        task = self.get(conversation_id, assistant_message_id)
        return bool(task and task.get("status") in {"pending", "running"})

    def shutdown(self) -> None:
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=False)
