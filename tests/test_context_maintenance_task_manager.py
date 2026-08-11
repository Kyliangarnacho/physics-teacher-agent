"""Tests for one-worker rolling-summary background maintenance."""

from __future__ import annotations

import inspect
import tempfile
import threading
import time
import unittest
from pathlib import Path

import src.tasks.context_maintenance as context_maintenance_module
from src.context.rolling_summary import (
    refresh_conversation_summary,
    select_unsummarized_bridge_turns,
)
from src.storage import (
    create_conversation,
    get_conversation_summary,
    initialize_database,
    insert_message,
    list_generation_jobs,
    list_messages,
    upsert_conversation_summary,
)
from src.tasks import ContextMaintenanceTaskManager, GenerationTaskManager


class ContextMaintenanceTaskManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "context-maintenance.db")
        initialize_database(self.db_path)

    def _conversation_with_turns(self, turn_count: int, title: str = "context") -> str:
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

    def _wait_unscheduled(
        self,
        manager: ContextMaintenanceTaskManager,
        conversation_id: str,
        timeout: float = 2.0,
    ) -> None:
        deadline = time.monotonic() + timeout
        while manager.is_scheduled(conversation_id) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertFalse(manager.is_scheduled(conversation_id))

    def test_stale_submit_refreshes_summary_and_cleans_scheduled(self) -> None:
        conversation_id = self._conversation_with_turns(7)

        def refresh(conversation_id: str, *, path=None):
            return refresh_conversation_summary(
                conversation_id,
                completion_func=lambda **_kwargs: _summary_response("后台摘要"),
                model_name="qwen-test",
                path=path,
            )

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        self.assertTrue(manager.submit(conversation_id))
        manager.shutdown()

        summary = get_conversation_summary(conversation_id, path=self.db_path)
        self.assertEqual(summary["summary_text"], "后台摘要")
        self.assertEqual(summary["summary_revision"], 1)
        self.assertFalse(manager.is_scheduled(conversation_id))
        self.assertEqual(manager.scheduled_conversation_ids, frozenset())

    def test_submit_if_stale_schedules_only_stale_conversation(self) -> None:
        fresh_id = self._conversation_with_turns(6, "fresh")
        stale_id = self._conversation_with_turns(7, "stale")
        refresh_calls: list[str] = []

        def refresh(conversation_id: str, *, path=None):
            refresh_calls.append(conversation_id)
            return {"status": "updated"}

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        self.assertFalse(manager.submit_if_stale(fresh_id))
        self.assertTrue(manager.submit_if_stale(stale_id))
        manager.shutdown()

        self.assertEqual(refresh_calls, [stale_id])

    def test_worker_rechecks_and_skips_when_not_stale(self) -> None:
        conversation_id = self._conversation_with_turns(6)
        calls = 0

        def refresh(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return {"status": "updated"}

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        manager.submit(conversation_id)
        manager.shutdown()

        self.assertEqual(calls, 0)
        self.assertFalse(manager.is_scheduled(conversation_id))

    def test_duplicate_submit_runs_once_and_executor_has_one_worker(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        started = threading.Event()
        release = threading.Event()
        calls = 0

        def refresh(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            started.set()
            release.wait(2)
            return {"status": "updated"}

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        try:
            self.assertTrue(manager.submit(conversation_id))
            self.assertTrue(started.wait(1))
            self.assertFalse(manager.submit(conversation_id))
            self.assertIn(conversation_id, manager.scheduled_conversation_ids)
            self.assertEqual(manager._executor._max_workers, 1)
            self.assertIn(
                "context-maintenance", manager._executor._thread_name_prefix
            )
        finally:
            release.set()
            manager.shutdown()

        self.assertEqual(calls, 1)
        self.assertFalse(manager.is_scheduled(conversation_id))

    def test_different_conversations_queue_and_run_serially(self) -> None:
        conversation_a = self._conversation_with_turns(7, "A")
        conversation_b = self._conversation_with_turns(7, "B")
        first_started = threading.Event()
        release = threading.Event()
        calls: list[str] = []
        active = 0
        max_active = 0
        lock = threading.Lock()

        def refresh(conversation_id: str, **_kwargs):
            nonlocal active, max_active
            with lock:
                calls.append(conversation_id)
                active += 1
                max_active = max(max_active, active)
                first_started.set()
            release.wait(2)
            with lock:
                active -= 1
            return {"status": "updated"}

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        try:
            self.assertTrue(manager.submit(conversation_a))
            self.assertTrue(first_started.wait(1))
            self.assertTrue(manager.submit(conversation_b))
            time.sleep(0.05)
            self.assertEqual(calls, [conversation_a])
            self.assertEqual(
                manager.scheduled_conversation_ids,
                frozenset({conversation_a, conversation_b}),
            )
        finally:
            release.set()
            manager.shutdown()

        self.assertEqual(calls, [conversation_a, conversation_b])
        self.assertEqual(max_active, 1)

    def test_queued_worker_observes_summary_written_after_submit(self) -> None:
        conversation_a = self._conversation_with_turns(7, "blocker")
        conversation_b = self._conversation_with_turns(7, "becomes-fresh")
        started = threading.Event()
        release = threading.Event()
        refresh_calls: list[str] = []

        def refresh(conversation_id: str, **_kwargs):
            refresh_calls.append(conversation_id)
            if conversation_id == conversation_a:
                started.set()
                release.wait(2)
            return {"status": "updated"}

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        try:
            manager.submit(conversation_a)
            self.assertTrue(started.wait(1))
            manager.submit(conversation_b)
            upsert_conversation_summary(
                conversation_b,
                "其他执行者已写入摘要",
                covered_until_message_id=f"{conversation_b}-a4",
                covered_turn_count=4,
                model_name="qwen-test",
                path=self.db_path,
            )
        finally:
            release.set()
            manager.shutdown()

        self.assertEqual(refresh_calls, [conversation_a])
        self.assertFalse(manager.is_scheduled(conversation_b))

    def test_api_failure_preserves_summary_boundary_and_bridge(self) -> None:
        conversation_id = self._conversation_with_turns(9)
        original = upsert_conversation_summary(
            conversation_id,
            "旧摘要",
            covered_until_message_id=f"{conversation_id}-a4",
            covered_turn_count=4,
            model_name="qwen-old",
            path=self.db_path,
        )

        def failed_refresh(conversation_id: str, *, path=None):
            def fail(**_kwargs):
                raise RuntimeError("temporary API failure")

            return refresh_conversation_summary(
                conversation_id,
                completion_func=fail,
                should_retry=lambda _error: True,
                model_name="qwen-test",
                path=path,
            )

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=failed_refresh,
        )
        manager.submit(conversation_id)
        manager.shutdown()

        current = get_conversation_summary(conversation_id, path=self.db_path)
        bridge = select_unsummarized_bridge_turns(
            list_messages(conversation_id, path=self.db_path),
            list_generation_jobs(conversation_id, path=self.db_path),
            current,
        )
        self.assertEqual(current, original)
        self.assertEqual(
            [turn.assistant_message_id for turn in bridge.bridge_turns],
            [f"{conversation_id}-a5", f"{conversation_id}-a6"],
        )
        self.assertFalse(manager.is_scheduled(conversation_id))

    def test_service_exception_cleans_scheduled_and_allows_resubmit(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        calls = 0

        def fail(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            raise RuntimeError("safe test failure")

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=fail,
        )
        try:
            self.assertTrue(manager.submit(conversation_id))
            self._wait_unscheduled(manager, conversation_id)
            self.assertTrue(manager.submit(conversation_id))
        finally:
            manager.shutdown()

        self.assertEqual(calls, 2)
        self.assertFalse(manager.is_scheduled(conversation_id))

    def test_failed_future_records_safe_process_local_diagnostics(self) -> None:
        conversation_id = self._conversation_with_turns(7)

        def fail(*_args, **_kwargs):
            return {
                "status": "failed",
                "error_type": "summary_configuration_error",
                "error_message": "must not be retained",
                "model_requests": 0,
            }

        manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=fail,
        )
        with self.assertLogs(
            "src.tasks.context_maintenance",
            level="WARNING",
        ) as captured:
            self.assertTrue(manager.submit_if_stale(conversation_id))
            manager.shutdown()

        result = manager.get_last_result(conversation_id)
        self.assertEqual(
            result,
            {
                "status": "failed",
                "error_type": "summary_configuration_error",
                "model_requests": 0,
            },
        )
        self.assertNotIn("error_message", result)
        self.assertIn("summary_configuration_error", captured.output[0])

    def test_context_executor_is_independent_from_generation_workers(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        generation_ready = threading.Event()
        context_ready = threading.Event()
        release = threading.Event()
        generation_active = 0
        lock = threading.Lock()

        def generation(*_args, **_kwargs):
            nonlocal generation_active
            with lock:
                generation_active += 1
                if generation_active == 2:
                    generation_ready.set()
            release.wait(2)
            return {}

        def refresh(*_args, **_kwargs):
            context_ready.set()
            release.wait(2)
            return {"status": "updated"}

        generation_manager = GenerationTaskManager(execute_func=generation)
        context_manager = ContextMaintenanceTaskManager(
            db_path=self.db_path,
            refresh_func=refresh,
        )
        try:
            generation_manager.submit("generation-1")
            generation_manager.submit("generation-2")
            self.assertTrue(generation_ready.wait(1))
            context_manager.submit(conversation_id)
            self.assertTrue(context_ready.wait(1))
            self.assertIsNot(
                generation_manager._executor,
                context_manager._executor,
            )
            self.assertEqual(generation_manager._executor._max_workers, 2)
            self.assertEqual(context_manager._executor._max_workers, 1)
        finally:
            release.set()
            generation_manager.shutdown()
            context_manager.shutdown()

    def test_module_does_not_import_or_use_streamlit(self) -> None:
        source = inspect.getsource(context_maintenance_module).casefold()
        self.assertNotIn("import streamlit", source)
        self.assertNotIn("from streamlit", source)
        self.assertNotIn("st.session_state", source)


def _summary_response(summary: str):
    import json
    from types import SimpleNamespace

    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps({"summary": summary}, ensure_ascii=False)
                )
            )
        ]
    )


if __name__ == "__main__":
    unittest.main()
