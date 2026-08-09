"""GenerationTaskManager 的并发、恢复、Retry 与幂等测试。"""

from __future__ import annotations

import tempfile
import threading
import unittest
from pathlib import Path

from src.conversation import (
    ConversationServiceError,
    enqueue_conversation_turn,
    execute_generation_job,
)
from src.storage import (
    claim_generation_job,
    create_conversation,
    get_agent_runs,
    get_generation_job,
    initialize_database,
    list_messages,
)
from src.tasks import GenerationTaskManager, TaskManagerError


def _agent_result(answer: str = "后台回答") -> dict:
    return {
        "answer": answer,
        "analysis": {"teaching_mode": "solve"},
        "route": {
            "teaching_mode": "solve",
            "use_rag": False,
            "use_tools": False,
            "should_answer": True,
        },
        "sources": [],
        "analysis_fallback": False,
        "tool_records": [],
        "tool_model_requests": 0,
        "trace": {
            "status": "completed",
            "total_model_requests": 2,
            "total_duration_ms": 10.0,
            "rag_searches": 0,
            "tool_executions": 0,
            "run_id": "task-run",
            "steps": [],
        },
    }


class ImmediateAgent:
    def __init__(self, error: Exception | None = None) -> None:
        self.error = error
        self.calls = 0

    def __call__(self, question: str, **kwargs) -> dict:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return _agent_result()


class SlowAgent:
    def __init__(self, expected_parallel: int = 2) -> None:
        self.release = threading.Event()
        self.parallel_ready = threading.Event()
        self.expected_parallel = expected_parallel
        self.calls = 0
        self.active = 0
        self.max_active = 0
        self._lock = threading.Lock()

    def __call__(self, question: str, **kwargs) -> dict:
        with self._lock:
            self.calls += 1
            self.active += 1
            self.max_active = max(self.max_active, self.active)
            if self.active >= self.expected_parallel:
                self.parallel_ready.set()
        if not self.release.wait(timeout=5):
            raise RuntimeError("slow agent test timeout")
        with self._lock:
            self.active -= 1
        return _agent_result()


class GenerationTaskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "task_manager.db")
        initialize_database(self.db_path)

    def _enqueue(self, title: str, question: str = "求平均速度") -> dict:
        conversation = create_conversation(title, path=self.db_path)
        enqueued = enqueue_conversation_turn(
            conversation["id"],
            question,
            question,
            db_path=self.db_path,
        )
        return {"conversation": conversation, **enqueued}

    def _execute_with(self, agent):
        def execute(job_id: str, *, db_path=None):
            return execute_generation_job(
                job_id,
                db_path=db_path,
                agent_func=agent,
            )

        return execute

    def test_two_workers_run_two_conversations_and_limit_parallelism(self) -> None:
        jobs = [self._enqueue(f"会话 {index}") for index in range(3)]
        agent = SlowAgent(expected_parallel=2)
        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )
        try:
            for item in jobs:
                self.assertTrue(manager.submit(item["generation_job_id"]))
            self.assertTrue(agent.parallel_ready.wait(timeout=3))

            statuses = [
                get_generation_job(item["generation_job_id"], path=self.db_path)[
                    "status"
                ]
                for item in jobs
            ]
            self.assertEqual(statuses.count("running"), 2)
            self.assertEqual(statuses.count("pending"), 1)
            self.assertEqual(agent.max_active, 2)
        finally:
            agent.release.set()
            manager.shutdown()

        self.assertEqual(agent.calls, 3)
        self.assertTrue(
            all(
                get_generation_job(item["generation_job_id"], path=self.db_path)[
                    "status"
                ]
                == "completed"
                for item in jobs
            )
        )

    def test_duplicate_submit_runs_same_job_once(self) -> None:
        item = self._enqueue("去重")
        agent = SlowAgent(expected_parallel=1)
        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )
        try:
            self.assertTrue(manager.submit(item["generation_job_id"]))
            self.assertTrue(agent.parallel_ready.wait(timeout=3))
            self.assertTrue(manager.is_scheduled(item["generation_job_id"]))
            self.assertFalse(manager.submit(item["generation_job_id"]))
        finally:
            agent.release.set()
            manager.shutdown()

        self.assertEqual(agent.calls, 1)
        self.assertFalse(manager.is_scheduled(item["generation_job_id"]))

    def test_same_conversation_second_active_job_is_rejected(self) -> None:
        conversation = create_conversation("同会话", path=self.db_path)
        enqueue_conversation_turn(
            conversation["id"],
            "第一题",
            "第一题",
            db_path=self.db_path,
        )

        with self.assertRaises(ConversationServiceError):
            enqueue_conversation_turn(
                conversation["id"],
                "第二题",
                "第二题",
                db_path=self.db_path,
            )

        self.assertEqual(len(list_messages(conversation["id"], path=self.db_path)), 1)

    def test_worker_completion_and_failure_update_job_status(self) -> None:
        success = self._enqueue("成功")
        success_manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(ImmediateAgent()),
        )
        success_manager.submit(success["generation_job_id"])
        success_manager.shutdown()
        self.assertEqual(
            get_generation_job(success["generation_job_id"], path=self.db_path)[
                "status"
            ],
            "completed",
        )

        failed = self._enqueue("失败")
        failed_manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(ImmediateAgent(RuntimeError("boom"))),
        )
        failed_manager.submit(failed["generation_job_id"])
        failed_manager.shutdown()
        failed_job = get_generation_job(failed["generation_job_id"], path=self.db_path)
        self.assertEqual(failed_job["status"], "failed")
        self.assertEqual(failed_job["error_type"], "agent_error")

    def test_recover_submits_pending_once(self) -> None:
        item = self._enqueue("恢复 pending")
        agent = ImmediateAgent()
        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )

        first = manager.recover()
        second = manager.recover()
        manager.shutdown()

        self.assertEqual(first["submitted_job_ids"], [item["generation_job_id"]])
        self.assertTrue(second["already_recovered"])
        self.assertEqual(second["submitted_job_ids"], [])
        self.assertEqual(agent.calls, 1)
        self.assertEqual(
            get_generation_job(item["generation_job_id"], path=self.db_path)[
                "status"
            ],
            "completed",
        )

    def test_recover_marks_running_interrupted_without_auto_retry(self) -> None:
        item = self._enqueue("恢复 running")
        claim_generation_job(item["generation_job_id"], path=self.db_path)
        agent = ImmediateAgent()
        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )

        recovered = manager.recover()
        manager.shutdown()

        job = get_generation_job(item["generation_job_id"], path=self.db_path)
        self.assertEqual(recovered["interrupted_count"], 1)
        self.assertEqual(recovered["submitted_job_ids"], [])
        self.assertEqual(job["status"], "interrupted")
        self.assertEqual(agent.calls, 0)

    def test_interrupted_manual_retry_reuses_job_and_user(self) -> None:
        item = self._enqueue("人工重试")
        claim_generation_job(item["generation_job_id"], path=self.db_path)
        agent = ImmediateAgent()
        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )
        manager.recover()

        retried = manager.retry(item["generation_job_id"])
        manager.shutdown()

        job = get_generation_job(item["generation_job_id"], path=self.db_path)
        messages = list_messages(item["conversation"]["id"], path=self.db_path)
        self.assertTrue(retried["scheduled"])
        self.assertEqual(job["status"], "completed")
        self.assertEqual(job["attempts"], 2)
        self.assertEqual([message["role"] for message in messages], ["user", "assistant"])
        self.assertEqual(job["user_message_id"], item["user_message_id"])

    def test_completed_cannot_retry_and_duplicate_execute_is_safe(self) -> None:
        item = self._enqueue("幂等")
        agent = ImmediateAgent()
        first = execute_generation_job(
            item["generation_job_id"],
            db_path=self.db_path,
            agent_func=agent,
        )
        duplicate = execute_generation_job(
            item["generation_job_id"],
            db_path=self.db_path,
            agent_func=agent,
        )

        self.assertEqual(first["generation_job"]["status"], "completed")
        self.assertTrue(duplicate["skipped"])
        self.assertEqual(agent.calls, 1)
        self.assertEqual(
            [item["role"] for item in list_messages(item["conversation"]["id"], path=self.db_path)],
            ["user", "assistant"],
        )
        self.assertEqual(
            len(get_agent_runs(item["conversation"]["id"], path=self.db_path)),
            1,
        )

        manager = GenerationTaskManager(
            db_path=self.db_path,
            execute_func=self._execute_with(agent),
        )
        with self.assertRaises(TaskManagerError):
            manager.retry(item["generation_job_id"])
        manager.shutdown()


if __name__ == "__main__":
    unittest.main()
