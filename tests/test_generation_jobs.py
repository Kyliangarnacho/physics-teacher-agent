"""Stage 11.2 generation job Schema 与 Repository 测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from src.storage import (
    GenerationJob,
    GenerationJobPayload,
    GenerationJobStatus,
    RepositoryError,
    claim_generation_job,
    complete_generation_job,
    create_conversation,
    create_generation_job,
    fail_generation_job,
    finalize_generation_job,
    get_active_generation_job,
    get_generation_job,
    initialize_database,
    insert_agent_run,
    insert_message,
    list_generation_jobs,
    list_messages,
    list_pending_generation_jobs,
    mark_running_interrupted,
    retry_generation_job,
)


class GenerationJobSchemaTests(unittest.TestCase):
    def test_payload_defaults_and_json_serialization(self) -> None:
        payload = GenerationJobPayload(question="  求平均速度  ")

        self.assertEqual(payload.question, "求平均速度")
        self.assertEqual(payload.mode_override, "auto")
        self.assertEqual(payload.rag_policy, "auto")
        self.assertFalse(payload.image_context_available)
        self.assertEqual(payload.max_history_turns, 3)
        self.assertEqual(payload.max_history_chars, 6000)
        json.dumps(payload.model_dump(mode="json"), ensure_ascii=False)

    def test_payload_rejects_extra_or_unconfirmed_image_context(self) -> None:
        with self.assertRaises(ValidationError):
            GenerationJobPayload(question="题目", extra_field="no")
        with self.assertRaises(ValidationError):
            GenerationJobPayload(question="题目", image_context="未确认内容")

    def test_payload_rejects_embedded_image_data_and_non_text_values(self) -> None:
        unsafe = "已确认文字\ndata:image/png;base64,AAAA"
        with self.assertRaises(ValidationError):
            GenerationJobPayload(question=unsafe)
        with self.assertRaises(ValidationError):
            GenerationJobPayload(
                question="题目",
                image_context=unsafe,
                image_context_available=True,
            )
        with self.assertRaises(ValidationError):
            GenerationJobPayload(question=b"raw image bytes")

    def test_job_status_contract_accepts_pending_and_rejects_invalid_fields(self) -> None:
        job = GenerationJob(
            id="job-1",
            conversation_id="conv-1",
            user_message_id="msg-1",
            status=GenerationJobStatus.PENDING,
            payload=GenerationJobPayload(question="题目"),
            created_at="2026-08-09T00:00:00+00:00",
            updated_at="2026-08-09T00:00:00+00:00",
        )
        self.assertEqual(job.status, GenerationJobStatus.PENDING)
        invalid = job.model_dump()
        invalid["error_type"] = "unexpected"
        with self.assertRaises(ValidationError):
            GenerationJob.model_validate(invalid)


class GenerationJobRepositoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "jobs.db")
        initialize_database(self.db_path)
        self.conversation, self.user_message = self._create_conversation_with_message(
            "会话一"
        )

    def _create_conversation_with_message(self, title: str):
        conversation = create_conversation(title, path=self.db_path)
        message = insert_message(
            {
                "conversation_id": conversation["id"],
                "role": "user",
                "display_content": "求平均速度",
                "model_content": "求平均速度",
            },
            path=self.db_path,
        )
        return conversation, message

    def _create_job(self, conversation=None, message=None, **overrides):
        conversation = conversation or self.conversation
        message = message or self.user_message
        data = {
            "conversation_id": conversation["id"],
            "user_message_id": message["id"],
            "payload": {
                "question": "一辆车行驶 50 m 用时 10 s，求平均速度。",
                "mode_override": "solve",
                "rag_policy": "off",
            },
        }
        data.update(overrides)
        return create_generation_job(data, path=self.db_path)

    def test_create_get_list_active_and_pending(self) -> None:
        created = self._create_job(id="job-fixed")

        self.assertEqual(created["id"], "job-fixed")
        self.assertEqual(created["status"], "pending")
        self.assertEqual(created["attempts"], 0)
        self.assertEqual(created["payload"]["mode_override"], "solve")
        self.assertEqual(get_generation_job("job-fixed", path=self.db_path), created)
        self.assertEqual(
            get_active_generation_job(self.conversation["id"], path=self.db_path),
            created,
        )
        self.assertEqual(
            [item["id"] for item in list_pending_generation_jobs(path=self.db_path)],
            ["job-fixed"],
        )
        self.assertEqual(
            [item["id"] for item in list_generation_jobs(path=self.db_path)],
            ["job-fixed"],
        )

    def test_same_conversation_has_only_one_active_job(self) -> None:
        self._create_job()
        second_message = insert_message(
            {
                "conversation_id": self.conversation["id"],
                "role": "user",
                "display_content": "第二题",
                "model_content": "第二题",
            },
            path=self.db_path,
        )

        with self.assertRaises(RepositoryError):
            self._create_job(message=second_message)

    def test_different_conversations_can_have_active_jobs(self) -> None:
        other_conversation, other_message = self._create_conversation_with_message(
            "会话二"
        )

        first = self._create_job()
        second = self._create_job(other_conversation, other_message)

        self.assertNotEqual(first["conversation_id"], second["conversation_id"])
        self.assertEqual(len(list_pending_generation_jobs(path=self.db_path)), 2)

    def test_claim_is_atomic_and_cannot_be_repeated(self) -> None:
        created = self._create_job()

        claimed = claim_generation_job(created["id"], path=self.db_path)
        duplicate = claim_generation_job(created["id"], path=self.db_path)

        self.assertEqual(claimed["status"], "running")
        self.assertEqual(claimed["attempts"], 1)
        self.assertIsNotNone(claimed["started_at"])
        self.assertIsNone(duplicate)
        self.assertEqual(
            get_active_generation_job(self.conversation["id"], path=self.db_path)[
                "status"
            ],
            "running",
        )

    def test_complete_finishes_running_job_and_cannot_retry(self) -> None:
        job = self._create_job()
        claim_generation_job(job["id"], path=self.db_path)

        completed = complete_generation_job(job["id"], path=self.db_path)

        self.assertEqual(completed["status"], "completed")
        self.assertIsNotNone(completed["finished_at"])
        self.assertIsNone(
            get_active_generation_job(self.conversation["id"], path=self.db_path)
        )
        with self.assertRaises(RepositoryError):
            retry_generation_job(job["id"], path=self.db_path)

    def test_failed_job_can_retry_and_be_claimed_again(self) -> None:
        job = self._create_job()
        claim_generation_job(job["id"], path=self.db_path)
        failed = fail_generation_job(
            job["id"],
            "model_api",
            "模型调用失败。",
            path=self.db_path,
        )

        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["error_type"], "model_api")
        retried = retry_generation_job(job["id"], path=self.db_path)
        self.assertEqual(retried["status"], "pending")
        self.assertEqual(retried["attempts"], 1)
        self.assertIsNone(retried["error_type"])
        self.assertIsNone(retried["started_at"])
        reclaimed = claim_generation_job(job["id"], path=self.db_path)
        self.assertEqual(reclaimed["attempts"], 2)

    def test_mark_running_interrupted_and_retry(self) -> None:
        other_conversation, other_message = self._create_conversation_with_message(
            "会话二"
        )
        first = self._create_job()
        second = self._create_job(other_conversation, other_message)
        claim_generation_job(first["id"], path=self.db_path)
        claim_generation_job(second["id"], path=self.db_path)

        changed = mark_running_interrupted(path=self.db_path)

        self.assertEqual(changed, 2)
        interrupted = list_generation_jobs(
            status=GenerationJobStatus.INTERRUPTED,
            path=self.db_path,
        )
        self.assertEqual(len(interrupted), 2)
        self.assertTrue(all(item["error_type"] == "worker_interrupted" for item in interrupted))
        retried = retry_generation_job(first["id"], path=self.db_path)
        self.assertEqual(retried["status"], "pending")

    def test_list_pending_limit_and_status_filter(self) -> None:
        other_conversation, other_message = self._create_conversation_with_message(
            "会话二"
        )
        first = self._create_job(id="job-1")
        second = self._create_job(
            other_conversation,
            other_message,
            id="job-2",
        )
        claim_generation_job(first["id"], path=self.db_path)

        pending = list_pending_generation_jobs(limit=1, path=self.db_path)
        running = list_generation_jobs(status="running", path=self.db_path)

        self.assertEqual([item["id"] for item in pending], [second["id"]])
        self.assertEqual([item["id"] for item in running], [first["id"]])

    def test_finalize_generation_job_is_atomic_on_insert_failure(self) -> None:
        job = self._create_job()
        claim_generation_job(job["id"], path=self.db_path)
        insert_agent_run(
            {
                "run_id": "duplicate-run",
                "conversation_id": self.conversation["id"],
                "user_message_id": self.user_message["id"],
                "status": "failed",
            },
            path=self.db_path,
        )

        with self.assertRaises(RepositoryError):
            finalize_generation_job(
                job["id"],
                self.conversation["id"],
                {
                    "conversation_id": self.conversation["id"],
                    "role": "assistant",
                    "display_content": "不应保存",
                    "model_content": "不应保存",
                },
                {
                    "run_id": "duplicate-run",
                    "conversation_id": self.conversation["id"],
                    "user_message_id": self.user_message["id"],
                    "status": "completed",
                },
                {
                    "conversation_id": self.conversation["id"],
                    "active_problem_text": "不应保存",
                },
                path=self.db_path,
            )

        messages = list_messages(self.conversation["id"], path=self.db_path)
        self.assertEqual([item["role"] for item in messages], ["user"])
        self.assertEqual(
            get_generation_job(job["id"], path=self.db_path)["status"],
            "running",
        )


if __name__ == "__main__":
    unittest.main()
