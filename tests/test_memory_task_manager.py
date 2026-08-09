"""Tests for the independent one-worker memory-analysis executor."""

from __future__ import annotations

from threading import Event
import time
import unittest

from src.tasks.manager import GenerationTaskManager
from src.tasks.memory_analysis import MemoryAnalysisTaskManager


class MemoryAnalysisTaskManagerTests(unittest.TestCase):
    def test_submit_is_non_blocking_and_deduplicated(self) -> None:
        release = Event()
        started = Event()

        def slow(*_args, **_kwargs):
            started.set()
            release.wait(2)
            return []

        manager = MemoryAnalysisTaskManager(extractor_func=slow)
        self.addCleanup(manager.shutdown)
        before = time.monotonic()
        submitted = manager.submit(
            "conversation",
            "assistant",
            user_message_id="user",
            user_question="question",
            assistant_answer="answer",
            conversation_history=[],
        )
        elapsed = time.monotonic() - before
        self.assertTrue(submitted)
        self.assertLess(elapsed, 0.5)
        self.assertTrue(started.wait(1))
        self.assertFalse(
            manager.submit(
                "conversation",
                "assistant",
                user_message_id="user",
                user_question="question",
                assistant_answer="answer",
                conversation_history=[],
            )
        )
        release.set()

    def test_generation_and_memory_executors_are_independent(self) -> None:
        generation_started = Event()
        memory_started = Event()
        release = Event()

        def generation(_job_id, **_kwargs):
            generation_started.set()
            release.wait(2)
            return {}

        def memory(*_args, **_kwargs):
            memory_started.set()
            release.wait(2)
            return []

        generation_manager = GenerationTaskManager(execute_func=generation)
        memory_manager = MemoryAnalysisTaskManager(extractor_func=memory)
        self.addCleanup(generation_manager.shutdown)
        self.addCleanup(memory_manager.shutdown)
        generation_manager.submit("generation-job")
        memory_manager.submit(
            "conversation",
            "assistant",
            user_message_id="user",
            user_question="question",
            assistant_answer="answer",
            conversation_history=[],
        )
        self.assertTrue(generation_started.wait(1))
        self.assertTrue(memory_started.wait(1))
        self.assertEqual(memory_manager._executor._max_workers, 1)
        self.assertIn("memory-analysis", memory_manager._executor._thread_name_prefix)
        release.set()

    def test_result_is_isolated_by_conversation_and_assistant(self) -> None:
        manager = MemoryAnalysisTaskManager(extractor_func=lambda *_a, **_k: [])
        self.addCleanup(manager.shutdown)
        manager.submit(
            "a",
            "assistant-1",
            user_message_id="u1",
            user_question="q",
            assistant_answer="a",
            conversation_history=[],
        )
        for _ in range(100):
            task = manager.get("a", "assistant-1")
            if task and task["status"] == "completed":
                break
            time.sleep(0.01)
        self.assertEqual(task["status"], "completed")
        self.assertIsNone(manager.get("b", "assistant-1"))
        self.assertIsNone(manager.get("a", "assistant-2"))

    def test_failure_is_safe_and_does_not_resubmit_duplicate(self) -> None:
        manager = MemoryAnalysisTaskManager(
            extractor_func=lambda *_a, **_k: (_ for _ in ()).throw(
                RuntimeError("secret path C:/private")
            )
        )
        self.addCleanup(manager.shutdown)
        manager.submit(
            "conversation",
            "assistant",
            user_message_id="user",
            user_question="question",
            assistant_answer="answer",
            conversation_history=[],
        )
        for _ in range(100):
            task = manager.get("conversation", "assistant")
            if task and task["status"] == "failed":
                break
            time.sleep(0.01)
        self.assertEqual(task["status"], "failed")
        self.assertNotIn("C:/private", task["error"])
        self.assertFalse(
            manager.submit(
                "conversation",
                "assistant",
                user_message_id="user",
                user_question="question",
                assistant_answer="answer",
                conversation_history=[],
            )
        )


if __name__ == "__main__":
    unittest.main()
