"""Automatic rolling-summary handoff after completed Generation Jobs."""

from __future__ import annotations

import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.tasks import ContextMaintenanceTaskManager, GenerationTaskManager
from src.context.rolling_summary import select_unsummarized_bridge_turns
from src.storage import (
    create_conversation,
    get_conversation_summary,
    initialize_database,
    insert_message,
    list_generation_jobs,
    list_messages,
    upsert_conversation_summary,
)


def completed_result(conversation_id: str) -> dict:
    return {
        "conversation_id": conversation_id,
        "generation_job": {"status": "completed"},
        "agent_result": {
            "trace": {"total_model_requests": 2},
            "answer": "已完成回答",
        },
    }


class RecordingMaintenance:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def submit(self, conversation_id: str) -> bool:
        self.calls.append(conversation_id)
        return True

    def submit_if_stale(self, _conversation_id: str) -> bool:
        raise AssertionError("Generation completion must not run a synchronous stale check")


class SummaryAutoScheduleTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "summary-auto.db")
        initialize_database(self.db_path)

    def _conversation_with_turns(self, turn_count: int, title: str = "auto") -> str:
        conversation = create_conversation(title, path=self.db_path)
        conversation_id = conversation["id"]
        for index in range(1, turn_count + 1):
            insert_message(
                {
                    "id": f"{conversation_id}-u{index}",
                    "conversation_id": conversation_id,
                    "role": "user",
                    "display_content": f"问题 {index}",
                    "model_content": f"问题 {index}",
                },
                path=self.db_path,
            )
            insert_message(
                {
                    "id": f"{conversation_id}-a{index}",
                    "conversation_id": conversation_id,
                    "role": "assistant",
                    "display_content": f"回答 {index}",
                    "model_content": f"回答 {index}",
                },
                path=self.db_path,
            )
        return conversation_id

    @staticmethod
    def _wait_until(predicate, timeout: float = 2.0) -> None:
        deadline = time.monotonic() + timeout
        while not predicate() and time.monotonic() < deadline:
            time.sleep(0.01)
        if not predicate():
            raise AssertionError("timed out waiting for background state")

    def test_completed_generation_enqueues_background_stale_check(self) -> None:
        maintenance = RecordingMaintenance()
        result = completed_result("conversation-completed")
        manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: result,
            context_maintenance_manager=maintenance,
        )

        manager.submit("job-completed")
        manager.shutdown()

        self.assertEqual(maintenance.calls, ["conversation-completed"])
        self.assertEqual(result["agent_result"]["trace"]["total_model_requests"], 2)

    @patch("src.context.rolling_summary.OpenAI")
    @patch("src.context.rolling_summary.load_qwen_config")
    def test_completed_generation_default_summary_service_writes_revision_one(
        self,
        mock_load_qwen_config: Mock,
        mock_openai: Mock,
    ) -> None:
        conversation_id = self._conversation_with_turns(7)
        mock_load_qwen_config.return_value = (
            "test-api-key",
            "https://example.invalid/v1",
            "qwen-test",
        )
        mock_openai.return_value.chat.completions.create.return_value = (
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content='{"summary":"自动维护摘要"}'
                        )
                    )
                ]
            )
        )
        context_manager = ContextMaintenanceTaskManager(db_path=self.db_path)
        generation_manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: completed_result(conversation_id),
            context_maintenance_manager=context_manager,
        )

        generation_manager.submit("job-default-summary-service")
        generation_manager.shutdown()
        context_manager.shutdown()

        summary = get_conversation_summary(conversation_id, path=self.db_path)
        self.assertEqual(summary["summary_revision"], 1)
        self.assertEqual(summary["covered_turn_count"], 4)
        self.assertEqual(
            context_manager.get_last_result(conversation_id)["status"],
            "updated",
        )

    def test_failed_interrupted_and_skipped_do_not_request_maintenance(self) -> None:
        maintenance = RecordingMaintenance()
        results = {
            "failed": {
                "conversation_id": "conversation-a",
                "generation_job": {"status": "failed"},
            },
            "interrupted": {
                "conversation_id": "conversation-b",
                "generation_job": {"status": "interrupted"},
            },
            "skipped": {
                "conversation_id": "conversation-c",
                "generation_job": {"status": "completed"},
                "skipped": True,
            },
        }
        manager = GenerationTaskManager(
            execute_func=lambda job_id, **_kwargs: results[job_id],
            context_maintenance_manager=maintenance,
        )

        for job_id in results:
            manager.submit(job_id)
        manager.shutdown()

        self.assertEqual(maintenance.calls, [])

    def test_generation_does_not_wait_for_summary_worker(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        refresh_started = threading.Event()
        release_refresh = threading.Event()

        def refresh(*_args, **_kwargs):
            refresh_started.set()
            release_refresh.wait(timeout=2.0)
            return {"status": "failed"}

        context_manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        generation_manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: completed_result(conversation_id),
            context_maintenance_manager=context_manager,
        )
        try:
            generation_manager.submit("job-non-blocking")
            self.assertTrue(refresh_started.wait(timeout=1.0))
            self._wait_until(
                lambda: not generation_manager.is_scheduled("job-non-blocking")
            )
            self.assertTrue(context_manager.is_scheduled(conversation_id))
        finally:
            release_refresh.set()
            generation_manager.shutdown()
            context_manager.shutdown()

    def test_generation_does_not_wait_for_stale_check(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        stale_check_started = threading.Event()
        release_stale_check = threading.Event()

        context_manager = ContextMaintenanceTaskManager(db_path=self.db_path)

        def blocking_stale_check(_conversation_id: str):
            stale_check_started.set()
            release_stale_check.wait(timeout=2.0)
            return SimpleNamespace(should_refresh=False)

        context_manager._stale_decision = blocking_stale_check
        generation_manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: completed_result(conversation_id),
            context_maintenance_manager=context_manager,
        )
        try:
            generation_manager.submit("job-non-blocking-stale-check")
            self.assertTrue(stale_check_started.wait(timeout=1.0))
            self._wait_until(
                lambda: not generation_manager.is_scheduled(
                    "job-non-blocking-stale-check"
                )
            )
            self.assertTrue(context_manager.is_scheduled(conversation_id))
        finally:
            release_stale_check.set()
            generation_manager.shutdown()
            context_manager.shutdown()

    def test_same_conversation_deduplicates_maintenance(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        refresh_started = threading.Event()
        release_refresh = threading.Event()
        refresh_calls = 0

        def refresh(*_args, **_kwargs):
            nonlocal refresh_calls
            refresh_calls += 1
            refresh_started.set()
            release_refresh.wait(timeout=2.0)
            return {"status": "failed"}

        context_manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        generation_manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: completed_result(conversation_id),
            context_maintenance_manager=context_manager,
        )
        try:
            generation_manager.submit("job-a")
            self.assertTrue(refresh_started.wait(timeout=1.0))
            generation_manager.submit("job-b")
            self._wait_until(lambda: not generation_manager.is_scheduled("job-b"))
            self.assertEqual(refresh_calls, 1)
        finally:
            release_refresh.set()
            generation_manager.shutdown()
            context_manager.shutdown()

    def test_summary_failure_preserves_answer_old_summary_and_bridge(self) -> None:
        conversation_id = self._conversation_with_turns(9)
        original = upsert_conversation_summary(
            conversation_id,
            "旧摘要",
            covered_until_message_id=f"{conversation_id}-a4",
            covered_turn_count=4,
            model_name="old-model",
            path=self.db_path,
        )

        def failed_refresh(*_args, **_kwargs):
            return {"status": "failed", "error_type": "summary_api_error"}

        context_manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=failed_refresh,
        )
        result = completed_result(conversation_id)
        generation_manager = GenerationTaskManager(
            execute_func=lambda *_args, **_kwargs: result,
            context_maintenance_manager=context_manager,
        )
        generation_manager.submit("job-summary-failure")
        generation_manager.shutdown()
        context_manager.shutdown()

        stored = get_conversation_summary(conversation_id, path=self.db_path)
        bridge = select_unsummarized_bridge_turns(
            list_messages(conversation_id, path=self.db_path),
            list_generation_jobs(conversation_id, path=self.db_path),
            stored,
        )
        self.assertEqual(result["agent_result"]["answer"], "已完成回答")
        self.assertEqual(stored["summary_revision"], original["summary_revision"])
        self.assertEqual(stored["covered_until_message_id"], original["covered_until_message_id"])
        self.assertEqual(
            [turn.user_message_id for turn in bridge.bridge_turns],
            [f"{conversation_id}-u5", f"{conversation_id}-u6"],
        )


if __name__ == "__main__":
    unittest.main()
