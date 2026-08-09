"""Streamlit 旧页面测试使用的确定性 Generation Task Manager。"""

from __future__ import annotations

from src.conversation.service import execute_generation_job
from src.storage import retry_generation_job


class ImmediateGenerationTaskManager:
    """在测试线程中立即执行 Job，避免真实线程池越过 mock 生命周期。"""

    def __init__(self, *, db_path=None, **_kwargs) -> None:
        self.db_path = db_path

    def recover(self) -> dict[str, object]:
        return {
            "already_recovered": False,
            "interrupted_count": 0,
            "pending_count": 0,
            "submitted_job_ids": [],
        }

    def submit(self, job_id: str) -> bool:
        execute_generation_job(job_id, db_path=self.db_path)
        return True

    def retry(self, job_id: str) -> dict[str, object]:
        job = retry_generation_job(job_id, path=self.db_path)
        self.submit(job_id)
        return {"generation_job": job, "scheduled": True}

    def is_scheduled(self, _job_id: str) -> bool:
        return False

    def shutdown(self) -> None:
        return None
