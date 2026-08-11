"""Process-local background scheduling for rolling-summary maintenance."""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any

from src.context.rolling_summary import (
    refresh_conversation_summary,
    should_refresh_summary,
)
from src.storage.repositories import (
    get_conversation_summary,
    list_generation_jobs,
    list_messages,
)


MAX_CONTEXT_MAINTENANCE_WORKERS = 1
logger = logging.getLogger(__name__)


class ContextMaintenanceTaskError(ValueError):
    """Safe validation or scheduling error for summary maintenance."""


class ContextMaintenanceTaskManager:
    """Run at most one summary-maintenance task per conversation at a time."""

    def __init__(
        self,
        *,
        db_path=None,
        refresh_func: Callable[..., dict[str, Any]] | None = None,
    ) -> None:
        self._db_path = db_path
        self._refresh_func = refresh_func or refresh_conversation_summary
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_CONTEXT_MAINTENANCE_WORKERS,
            thread_name_prefix="context-maintenance",
        )
        self._lock = RLock()
        self._scheduled_conversation_ids: set[str] = set()
        self._futures: dict[str, Future] = {}
        self._last_results: dict[str, dict[str, Any]] = {}
        self._shutdown = False

    @staticmethod
    def _validate_conversation_id(conversation_id: object) -> str:
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            raise ContextMaintenanceTaskError(
                "conversation_id must be a non-empty string."
            )
        return conversation_id.strip()

    def _run(self, conversation_id: str) -> dict[str, Any]:
        """Recheck persisted state at execution time, then delegate refresh."""

        try:
            decision = self._stale_decision(conversation_id)
        except Exception:
            return {
                "status": "failed",
                "error_type": "summary_stale_check_error",
                "error_message": "Summary maintenance stale check failed.",
            }

        if not decision.should_refresh:
            return {
                "status": "skipped",
                "decision": decision,
                "error_type": None,
                "error_message": None,
            }

        try:
            return self._refresh_func(conversation_id, path=self._db_path)
        except Exception:
            return {
                "status": "failed",
                "decision": decision,
                "error_type": "summary_service_error",
                "error_message": "Summary maintenance failed; existing context is unchanged.",
            }

    def _stale_decision(self, conversation_id: str):
        messages = list_messages(conversation_id, path=self._db_path)
        jobs = list_generation_jobs(conversation_id, path=self._db_path)
        summary = get_conversation_summary(conversation_id, path=self._db_path)
        return should_refresh_summary(messages, jobs, summary)

    @staticmethod
    def _safe_result_metadata(result: object) -> dict[str, Any]:
        """Keep only bounded task diagnostics; never retain context text."""

        if not isinstance(result, dict):
            return {
                "status": "failed",
                "error_type": "summary_worker_result_error",
            }
        metadata: dict[str, Any] = {
            "status": result.get("status"),
            "error_type": result.get("error_type"),
            "model_requests": result.get("model_requests"),
        }
        decision = result.get("decision")
        for name in (
            "should_refresh",
            "eligible_turn_count",
            "new_turn_count",
            "eligible_chars",
            "reason",
        ):
            value = getattr(decision, name, None)
            if value is not None:
                metadata[name] = value
        summary = result.get("summary")
        if isinstance(summary, dict):
            metadata["summary_revision"] = summary.get("summary_revision")
        return metadata

    def _complete_scheduled(
        self,
        conversation_id: str,
        future: Future,
    ) -> None:
        try:
            metadata = self._safe_result_metadata(future.result())
        except Exception:
            metadata = {
                "status": "failed",
                "error_type": "summary_worker_unhandled_error",
            }
        if metadata.get("status") == "failed":
            logger.warning(
                "Summary maintenance failed for conversation %s (%s).",
                conversation_id,
                metadata.get("error_type") or "unknown_error",
            )
        with self._lock:
            self._last_results[conversation_id] = metadata
            self._scheduled_conversation_ids.discard(conversation_id)
            self._futures.pop(conversation_id, None)

    def submit(self, conversation_id: str) -> bool:
        """Schedule a conversation once; duplicates return False while active."""

        normalized_id = self._validate_conversation_id(conversation_id)
        with self._lock:
            if self._shutdown:
                raise ContextMaintenanceTaskError(
                    "Context maintenance task manager is shut down."
                )
            if normalized_id in self._scheduled_conversation_ids:
                return False
            self._scheduled_conversation_ids.add(normalized_id)
            try:
                future = self._executor.submit(self._run, normalized_id)
            except Exception as exc:
                self._scheduled_conversation_ids.discard(normalized_id)
                raise ContextMaintenanceTaskError(
                    "Context maintenance task submission failed."
                ) from exc
            self._futures[normalized_id] = future
            future.add_done_callback(
                lambda completed, scheduled_id=normalized_id: self._complete_scheduled(
                    scheduled_id,
                    completed,
                )
            )
            return True

    def submit_if_stale(self, conversation_id: str) -> bool:
        """Run a light persisted stale check, then schedule only when needed.

        The worker deliberately checks again before refreshing so queued work
        cannot rely on a decision made against older database state.
        """

        normalized_id = self._validate_conversation_id(conversation_id)
        with self._lock:
            if self._shutdown:
                raise ContextMaintenanceTaskError(
                    "Context maintenance task manager is shut down."
                )
            if normalized_id in self._scheduled_conversation_ids:
                return False
        try:
            decision = self._stale_decision(normalized_id)
        except Exception:
            with self._lock:
                self._last_results[normalized_id] = {
                    "status": "failed",
                    "error_type": "summary_stale_check_error",
                }
            logger.warning(
                "Summary maintenance stale check failed for conversation %s.",
                normalized_id,
            )
            return False
        if not decision.should_refresh:
            with self._lock:
                self._last_results[normalized_id] = self._safe_result_metadata(
                    {
                        "status": "skipped",
                        "decision": decision,
                        "error_type": None,
                        "model_requests": 0,
                    }
                )
            return False
        return self.submit(normalized_id)

    @property
    def scheduled_conversation_ids(self) -> frozenset[str]:
        """Return a snapshot of conversations currently queued or running."""

        with self._lock:
            return frozenset(self._scheduled_conversation_ids)

    def is_scheduled(self, conversation_id: str) -> bool:
        normalized_id = self._validate_conversation_id(conversation_id)
        with self._lock:
            return normalized_id in self._scheduled_conversation_ids

    def get_last_result(self, conversation_id: str) -> dict[str, Any] | None:
        """Return safe process-local diagnostics for the last finished attempt."""

        normalized_id = self._validate_conversation_id(conversation_id)
        with self._lock:
            result = self._last_results.get(normalized_id)
            return dict(result) if result is not None else None

    def shutdown(self) -> None:
        """Stop accepting work and wait for already submitted tasks."""

        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=False)
