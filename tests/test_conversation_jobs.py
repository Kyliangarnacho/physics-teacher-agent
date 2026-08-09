"""Conversation Service 的 Generation Job 拆分流程测试。"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from src.conversation import (
    ConversationServiceError,
    enqueue_conversation_turn,
    execute_generation_job,
    run_conversation_turn,
)
from src.storage import (
    create_conversation,
    get_agent_runs,
    get_conversation_state,
    get_generation_job,
    initialize_database,
    insert_or_merge_memory,
    list_messages,
    upsert_conversation_state,
)


def _agent_result(answer: str = "教师回答") -> dict:
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
            "run_id": "run-fixed",
            "steps": [],
        },
    }


class RecordingAgent:
    def __init__(self, *, error: Exception | None = None) -> None:
        self.error = error
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> dict:
        self.calls.append({"question": question, **kwargs})
        if self.error is not None:
            raise self.error
        return _agent_result()


class ConversationJobServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "conversation_jobs.db")
        initialize_database(self.db_path)
        self.conversation = create_conversation("Job 测试", path=self.db_path)
        self.conversation_id = self.conversation["id"]

    def _enqueue(self, question: str = "求平均速度", **kwargs):
        return enqueue_conversation_turn(
            self.conversation_id,
            question,
            question,
            db_path=self.db_path,
            **kwargs,
        )

    def test_enqueue_does_not_call_agent(self) -> None:
        with mock.patch("src.conversation.service._default_run_agent") as agent:
            enqueued = self._enqueue()

        agent.assert_not_called()
        self.assertEqual(enqueued["generation_job"]["status"], "pending")
        self.assertEqual(len(list_messages(self.conversation_id, path=self.db_path)), 1)

    def test_user_and_job_are_atomic(self) -> None:
        self._enqueue("第一题")

        with self.assertRaises(ConversationServiceError):
            self._enqueue("第二题")

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual([item["model_content"] for item in messages], ["第一题"])

    def test_current_user_is_not_duplicated_in_history(self) -> None:
        run_conversation_turn(
            self.conversation_id,
            "第一题",
            "第一题",
            db_path=self.db_path,
            agent_func=RecordingAgent(),
        )
        enqueued = self._enqueue("第二题")
        agent = RecordingAgent()

        result = execute_generation_job(
            enqueued["generation_job_id"],
            db_path=self.db_path,
            agent_func=agent,
        )

        self.assertEqual(result["history_turn_count"], 1)
        self.assertEqual(
            agent.calls[0]["conversation_history"],
            [
                {"role": "user", "content": "第一题"},
                {"role": "assistant", "content": "教师回答"},
            ],
        )
        self.assertNotIn("第二题", str(agent.calls[0]["conversation_history"]))

    def test_execute_restores_state_memory_and_confirmed_image_context(self) -> None:
        upsert_conversation_state(
            {
                "conversation_id": self.conversation_id,
                "active_problem_text": "物体下落时机械能怎样变化？",
                "active_image_context": "旧图信息",
                "teaching_mode": "hint",
                "hint_step": 1,
            },
            path=self.db_path,
        )
        insert_or_merge_memory(
            {
                "memory_type": "misconception",
                "topic": "机械能",
                "content": "学生曾误认为物体下落时机械能一定减小。",
                "normalized_content": "学生曾误认为物体下落时机械能一定减小",
                "confidence": 0.9,
            },
            path=self.db_path,
        )
        enqueued = self._enqueue(
            "继续",
            confirmed_image_context="新图显示物体沿光滑斜面下滑。",
        )
        agent = RecordingAgent()

        execute_generation_job(
            enqueued["generation_job_id"],
            db_path=self.db_path,
            agent_func=agent,
        )

        call = agent.calls[0]
        self.assertIn("原题：物体下落时机械能怎样变化？", call["question"])
        self.assertEqual(call["image_context"], "新图显示物体沿光滑斜面下滑。")
        self.assertIn("已完成提示步数：1", call["teaching_state_context"])
        self.assertIn("上一题存在已确认图片上下文：是", call["teaching_state_context"])
        self.assertNotIn("新图显示物体沿光滑斜面下滑", call["teaching_state_context"])
        self.assertEqual(call["previous_image_context"], "旧图信息")
        self.assertIn("机械能", call["learning_memory_context"])

    def test_success_persists_one_assistant_one_run_and_completes_job(self) -> None:
        enqueued = self._enqueue()

        result = execute_generation_job(
            enqueued["generation_job_id"],
            db_path=self.db_path,
            agent_func=RecordingAgent(),
        )

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant"])
        self.assertEqual(len(get_agent_runs(self.conversation_id, path=self.db_path)), 1)
        job = get_generation_job(result["generation_job_id"], path=self.db_path)
        self.assertEqual(job["status"], "completed")

    def test_agent_failure_marks_job_failed_without_assistant(self) -> None:
        enqueued = self._enqueue()

        with self.assertRaises(ConversationServiceError):
            execute_generation_job(
                enqueued["generation_job_id"],
                db_path=self.db_path,
                agent_func=RecordingAgent(error=RuntimeError("private detail")),
            )

        messages = list_messages(self.conversation_id, path=self.db_path)
        self.assertEqual([item["role"] for item in messages], ["user"])
        self.assertEqual(get_agent_runs(self.conversation_id, path=self.db_path), [])
        job = get_generation_job(enqueued["generation_job_id"], path=self.db_path)
        self.assertEqual(job["status"], "failed")
        self.assertEqual(job["error_type"], "agent_error")
        self.assertNotIn("private detail", job["error_message"])

    def test_sync_wrapper_keeps_legacy_result_and_completes_job(self) -> None:
        agent = RecordingAgent()

        result = run_conversation_turn(
            self.conversation_id,
            "题目",
            "题目",
            db_path=self.db_path,
            agent_func=agent,
        )

        for key in (
            "agent_result",
            "conversation_id",
            "user_message_id",
            "assistant_message_id",
            "resolved_state",
            "history_turn_count",
            "state_context_used",
            "memory_count",
            "memory_context_used",
        ):
            self.assertIn(key, result)
        self.assertEqual(len(agent.calls), 1)
        self.assertEqual(result["generation_job"]["status"], "completed")


if __name__ == "__main__":
    unittest.main()
