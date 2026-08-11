"""基于 ThreadPoolExecutor 的轻量 Generation Job 任务管理器。"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from threading import RLock
from typing import Any

from src.conversation.service import execute_generation_job
from src.tasks.context_maintenance import ContextMaintenanceTaskManager
from src.storage import (
    RepositoryError,
    claim_generation_job,
    fail_generation_job,
    get_generation_job,
    list_pending_generation_jobs,
    mark_running_interrupted,
    retry_generation_job,
)


MAX_GENERATION_WORKERS = 2
logger = logging.getLogger(__name__)


class TaskManagerError(Exception):
    """任务管理层安全异常。"""


class GenerationTaskManager:
    """在当前进程内调度 Generation Job，最终防重仍由数据库 claim 保证。"""

    def __init__(
        self,
        *,
        db_path=None,
        execute_func: Callable[..., dict[str, Any]] | None = None,
        context_maintenance_manager: ContextMaintenanceTaskManager | None = None,
    ) -> None:
        self._db_path = db_path
        self._execute_func = execute_func or execute_generation_job
        self._context_maintenance_manager = context_maintenance_manager
        self._owns_context_maintenance_manager = False
        if self._context_maintenance_manager is None and execute_func is None:
            self._context_maintenance_manager = ContextMaintenanceTaskManager(
                db_path=db_path,
            )
            self._owns_context_maintenance_manager = True
        self._executor = ThreadPoolExecutor(
            max_workers=MAX_GENERATION_WORKERS,
            thread_name_prefix="generation-job",
        )
        self._lock = RLock()
        self._scheduled: set[str] = set()
        self._futures: dict[str, Future] = {}
        self._shutdown = False
        self._recovery_started = False
        self._recovered = False

    @staticmethod
    def _validate_job_id(job_id: object) -> str:
        if not isinstance(job_id, str) or not job_id.strip():
            raise TaskManagerError("job_id 必须为非空字符串。")
        return job_id.strip()

    def _run_job(self, job_id: str) -> dict[str, Any]:
        result = self._execute_func(job_id, db_path=self._db_path)
        self._schedule_context_maintenance(result)
        return result

    def _schedule_context_maintenance(self, result: dict[str, Any]) -> None:
        """Best-effort handoff after a newly completed Generation Job."""

        manager = self._context_maintenance_manager
        if manager is None or result.get("skipped"):
            return
        job = result.get("generation_job")
        if not isinstance(job, dict) or job.get("status") != "completed":
            return
        conversation_id = result.get("conversation_id")
        if not isinstance(conversation_id, str) or not conversation_id.strip():
            return
        try:
            # The maintenance worker performs the persisted stale check.  This
            # completion handoff stays O(1) and never waits for SQLite history
            # reads or the Summary API.
            manager.submit(conversation_id)
        except Exception:
            # Summary maintenance is advisory and must never change Generation
            # success or expose background scheduling details to the caller.
            return

    def _mark_unhandled_worker_failure(self, job_id: str) -> None:
        """Best-effort terminalization when a Future escapes service handling."""

        try:
            job = get_generation_job(job_id, path=self._db_path)
            if job is None:
                return
            if job.get("status") == "pending":
                # The worker did start, even if an injected/orchestration
                # callable failed before the normal service claim.
                claimed = claim_generation_job(job_id, path=self._db_path)
                job = claimed or get_generation_job(job_id, path=self._db_path)
            if job is not None and job.get("status") == "running":
                fail_generation_job(
                    job_id,
                    "worker_unhandled_error",
                    "The background generation worker stopped unexpectedly.",
                    path=self._db_path,
                )
        except Exception as exc:
            logger.error(
                "Generation worker failure could not be persisted for job %s (%s).",
                job_id,
                type(exc).__name__,
            )

    def _complete_scheduled(self, job_id: str, future: Future) -> None:
        try:
            future.result()
        except BaseException as exc:
            # Never log exception text: provider errors can contain request
            # details.  Persist only a fixed safe failure for UI/retry flow.
            self._mark_unhandled_worker_failure(job_id)
            logger.error(
                "Generation worker stopped unexpectedly for job %s (%s).",
                job_id,
                type(exc).__name__,
            )
        with self._lock:
            self._scheduled.discard(job_id)
            self._futures.pop(job_id, None)

    def _submit_locked(self, job_id: str) -> bool:
        if self._shutdown:
            raise TaskManagerError("Task Manager 已关闭，不能提交任务。")
        if job_id in self._scheduled:
            return False
        self._scheduled.add(job_id)
        try:
            future = self._executor.submit(self._run_job, job_id)
        except Exception as exc:
            self._scheduled.discard(job_id)
            raise TaskManagerError("后台任务提交失败。") from exc
        self._futures[job_id] = future
        future.add_done_callback(
            lambda completed, scheduled_job_id=job_id: self._complete_scheduled(
                scheduled_job_id,
                completed,
            )
        )
        return True

    def submit(self, job_id: str) -> bool:
        """调度一个 Job；当前进程已调度时返回 False。"""
        normalized_job_id = self._validate_job_id(job_id)
        with self._lock:
            return self._submit_locked(normalized_job_id)

    def is_scheduled(self, job_id: str) -> bool:
        """返回 Job 当前是否仍在本 Manager 的排队或执行集合中。"""
        normalized_job_id = self._validate_job_id(job_id)
        with self._lock:
            return normalized_job_id in self._scheduled

    def recover(self) -> dict[str, Any]:
        """启动恢复：中断遗留 running，并自动调度 pending；实例内幂等。"""
        with self._lock:
            if self._recovered or self._recovery_started:
                return {
                    "already_recovered": True,
                    "interrupted_count": 0,
                    "pending_count": 0,
                    "submitted_job_ids": [],
                }
            if self._shutdown:
                raise TaskManagerError("Task Manager 已关闭，不能执行恢复。")
            self._recovery_started = True
            try:
                interrupted_count = mark_running_interrupted(path=self._db_path)
                pending_jobs = list_pending_generation_jobs(path=self._db_path)
                submitted_job_ids = [
                    job["id"]
                    for job in pending_jobs
                    if self._submit_locked(job["id"])
                ]
            except RepositoryError as exc:
                self._recovery_started = False
                raise TaskManagerError("后台任务恢复失败。") from exc
            except Exception:
                self._recovery_started = False
                raise
            self._recovered = True
            self._recovery_started = False
            return {
                "already_recovered": False,
                "interrupted_count": interrupted_count,
                "pending_count": len(pending_jobs),
                "submitted_job_ids": submitted_job_ids,
            }

    def retry(self, job_id: str) -> dict[str, Any]:
        """人工执行 failed/interrupted → pending，并调度原 Job。"""
        normalized_job_id = self._validate_job_id(job_id)
        with self._lock:
            if self._shutdown:
                raise TaskManagerError("Task Manager 已关闭，不能重试任务。")
            if normalized_job_id in self._scheduled:
                raise TaskManagerError("任务仍在当前进程执行，不能重试。")
            try:
                job = retry_generation_job(
                    normalized_job_id,
                    path=self._db_path,
                )
            except RepositoryError as exc:
                raise TaskManagerError("该任务当前不允许重试。") from exc
            scheduled = self._submit_locked(normalized_job_id)
            return {"generation_job": job, "scheduled": scheduled}

    def shutdown(self) -> None:
        """停止接收新任务，并等待已提交 Worker 结束；可重复调用。"""
        with self._lock:
            if self._shutdown:
                return
            self._shutdown = True
        self._executor.shutdown(wait=True, cancel_futures=False)
        if self._owns_context_maintenance_manager:
            self._context_maintenance_manager.shutdown()
