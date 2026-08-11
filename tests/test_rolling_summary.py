"""Stage 11.3 rolling-summary selection, trigger, and refresh tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.context.rolling_summary import (
    build_summary_model_messages,
    refresh_conversation_summary,
    select_unsummarized_bridge_turns,
    should_refresh_summary,
)
from src.storage import (
    create_conversation,
    get_conversation_summary,
    initialize_database,
    insert_message,
    upsert_conversation_summary,
)


def _messages(turn_count: int) -> list[dict]:
    records: list[dict] = []
    for index in range(1, turn_count + 1):
        records.extend(
            [
                {
                    "id": f"u{index}",
                    "conversation_id": "conv-1",
                    "role": "user",
                    "display_content": f"问题 {index}",
                    "model_content": f"问题 {index}",
                    "image_metadata_json": [],
                    "created_at": f"2026-08-10T00:{index:02d}:00+00:00",
                },
                {
                    "id": f"a{index}",
                    "conversation_id": "conv-1",
                    "role": "assistant",
                    "display_content": f"回答 {index}",
                    "model_content": f"回答 {index}",
                    "image_metadata_json": [],
                    "created_at": f"2026-08-10T00:{index:02d}:01+00:00",
                },
            ]
        )
    return records


def _existing_summary(boundary: str, covered_turn_count: int = 4) -> dict:
    return {
        "conversation_id": "conv-1",
        "summary_text": "旧摘要",
        "covered_until_message_id": boundary,
        "covered_turn_count": covered_turn_count,
        "summary_revision": 1,
        "model_name": "qwen-test",
        "created_at": "2026-08-10T01:00:00+00:00",
        "updated_at": "2026-08-10T01:00:00+00:00",
    }


def _fake_response(summary: str) -> SimpleNamespace:
    content = json.dumps({"summary": summary}, ensure_ascii=False)
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
    )


class RollingSummarySelectionTests(unittest.TestCase):
    def test_initial_below_threshold_and_isolated_user_do_not_trigger(self) -> None:
        messages = _messages(6)
        messages.append(
            {
                "id": "u7",
                "conversation_id": "conv-1",
                "role": "user",
                "display_content": "未回答问题",
                "model_content": "未回答问题",
                "image_metadata_json": [],
                "created_at": "2026-08-10T00:07:00+00:00",
            }
        )

        decision = should_refresh_summary(messages, [])

        self.assertFalse(decision.should_refresh)
        self.assertEqual(decision.new_turn_count, 3)
        self.assertEqual(
            [turn.assistant_message_id for turn in decision.eligible_turns],
            ["a1", "a2", "a3"],
        )

    def test_initial_four_old_turns_trigger(self) -> None:
        decision = should_refresh_summary(_messages(7), [])

        self.assertTrue(decision.should_refresh)
        self.assertEqual(decision.new_turn_count, 4)
        self.assertEqual(decision.reason, "initial_turn_threshold")
        self.assertEqual(
            [turn.assistant_message_id for turn in decision.eligible_turns],
            ["a1", "a2", "a3", "a4"],
        )

    def test_initial_character_threshold_triggers(self) -> None:
        messages = _messages(6)
        for index in (0, 2, 4):
            messages[index]["model_content"] = "长" * 2000

        decision = should_refresh_summary(messages, [])

        self.assertTrue(decision.should_refresh)
        self.assertEqual(decision.new_turn_count, 3)
        self.assertGreaterEqual(decision.eligible_chars, 6000)
        self.assertEqual(decision.reason, "initial_char_threshold")

    def test_covered_turns_are_not_repeated_and_recent_three_stay_raw(self) -> None:
        decision = should_refresh_summary(
            _messages(9),
            [],
            _existing_summary("a4"),
        )

        self.assertTrue(decision.should_refresh)
        self.assertEqual(decision.eligible_turn_count, 6)
        self.assertEqual(decision.new_turn_count, 2)
        self.assertEqual(
            [turn.assistant_message_id for turn in decision.eligible_turns],
            ["a5", "a6"],
        )

    def test_bridge_is_old_history_after_boundary_and_before_recent_window(self) -> None:
        bridge_v1 = select_unsummarized_bridge_turns(
            _messages(8),
            [],
            _existing_summary("a4"),
        )
        bridge_v2 = select_unsummarized_bridge_turns(
            _messages(8),
            [],
            _existing_summary("a6", covered_turn_count=6),
        )

        self.assertEqual(
            [turn.assistant_message_id for turn in bridge_v1.bridge_turns],
            ["a5"],
        )
        self.assertEqual(bridge_v1.boundary_status, "found")
        self.assertEqual(bridge_v2.bridge_turns, ())

    def test_unknown_boundary_keeps_old_raw_turns_visible_in_bridge(self) -> None:
        summary = _existing_summary("missing-assistant")

        bridge = select_unsummarized_bridge_turns(_messages(8), [], summary)
        decision = should_refresh_summary(_messages(8), [], summary)

        self.assertEqual(
            [turn.assistant_message_id for turn in bridge.bridge_turns],
            ["a1", "a2", "a3", "a4", "a5"],
        )
        self.assertEqual(bridge.boundary_status, "not_found")
        self.assertFalse(decision.should_refresh)

    def test_existing_summary_one_new_old_turn_does_not_trigger(self) -> None:
        decision = should_refresh_summary(
            _messages(8),
            [],
            _existing_summary("a4"),
        )

        self.assertFalse(decision.should_refresh)
        self.assertEqual(decision.new_turn_count, 1)
        self.assertEqual(decision.reason, "rolling_threshold_not_met")

    def test_existing_summary_one_long_new_turn_triggers(self) -> None:
        messages = _messages(8)
        messages[8]["model_content"] = "长" * 4000

        decision = should_refresh_summary(
            messages,
            [],
            _existing_summary("a4"),
        )

        self.assertTrue(decision.should_refresh)
        self.assertEqual(decision.new_turn_count, 1)
        self.assertGreaterEqual(decision.eligible_chars, 4000)
        self.assertEqual(decision.reason, "rolling_char_threshold")

    def test_non_completed_job_states_are_never_compressed(self) -> None:
        for status in ("pending", "running", "failed", "interrupted"):
            with self.subTest(status=status):
                decision = should_refresh_summary(
                    _messages(8),
                    [
                        {
                            "user_message_id": "u1",
                            "status": status,
                            "payload": {},
                        }
                    ],
                )
                self.assertNotIn(
                    "u1", [turn.user_message_id for turn in decision.eligible_turns]
                )

    def test_confirmed_image_context_is_the_only_job_context_in_prompt(self) -> None:
        decision = should_refresh_summary(
            _messages(7),
            [
                {
                    "user_message_id": "u1",
                    "status": "completed",
                    "payload": {
                        "image_context_available": True,
                        "image_context": "图中 A、B 两点等高。",
                        "question": "看图回答",
                    },
                }
            ],
        )

        prompt = json.loads(build_summary_model_messages(decision)[1]["content"])
        first_turn = prompt["newly_eligible_turns"][0]
        self.assertEqual(first_turn["confirmed_image_context"], "图中 A、B 两点等高。")
        self.assertNotIn("payload", first_turn)

    def test_missing_or_unknown_covered_boundary_fails_closed(self) -> None:
        for boundary, reason in (
            (None, "covered_boundary_missing"),
            ("missing-assistant", "covered_boundary_not_found"),
        ):
            with self.subTest(boundary=boundary):
                summary = _existing_summary("a4")
                summary["covered_until_message_id"] = boundary
                decision = should_refresh_summary(_messages(9), [], summary)
                self.assertFalse(decision.should_refresh)
                self.assertEqual(decision.new_turn_count, 0)
                self.assertEqual(decision.reason, reason)


class RollingSummaryServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp_dir.cleanup)
        self.db_path = str(Path(self._temp_dir.name) / "rolling-summary.db")
        initialize_database(self.db_path)

    def _conversation_with_turns(self, turn_count: int, title: str = "摘要测试") -> str:
        conversation = create_conversation(title, path=self.db_path)
        for index in range(1, turn_count + 1):
            self._append_turn(conversation["id"], index)
        return conversation["id"]

    def _append_turn(
        self,
        conversation_id: str,
        index: int,
        *,
        user_content: str | None = None,
    ) -> None:
        user_text = user_content or f"问题 {index}"
        insert_message(
            {
                "id": f"{conversation_id}-u{index}",
                "conversation_id": conversation_id,
                "role": "user",
                "display_content": user_text,
                "model_content": user_text,
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

    def _seed_existing_summary(self, conversation_id: str) -> dict:
        return upsert_conversation_summary(
            conversation_id,
            "旧摘要",
            covered_until_message_id=f"{conversation_id}-a4",
            covered_turn_count=4,
            model_name="qwen-old",
            path=self.db_path,
        )

    def test_below_threshold_never_calls_model(self) -> None:
        conversation_id = self._conversation_with_turns(6)

        def unexpected_call(**kwargs):
            self.fail("summary model must not be called below threshold")

        result = refresh_conversation_summary(
            conversation_id,
            completion_func=unexpected_call,
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(result["status"], "skipped")
        self.assertEqual(result["model_requests"], 0)
        self.assertIsNone(get_conversation_summary(conversation_id, path=self.db_path))

    def test_first_refresh_parses_fake_output_and_creates_revision_one(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        calls: list[dict] = []

        def complete(**kwargs):
            calls.append(kwargs)
            return _fake_response("首次滚动摘要")

        result = refresh_conversation_summary(
            conversation_id,
            completion_func=complete,
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(len(calls), 1)
        summary = result["summary"]
        self.assertEqual(summary["summary_revision"], 1)
        self.assertEqual(summary["covered_turn_count"], 4)
        self.assertEqual(
            summary["covered_until_message_id"], f"{conversation_id}-a4"
        )
        payload = json.loads(calls[0]["messages"][1]["content"])
        sent_ids = [
            turn["assistant_message_id"]
            for turn in payload["newly_eligible_turns"]
        ]
        self.assertEqual(
            sent_ids,
            [f"{conversation_id}-a{index}" for index in range(1, 5)],
        )

    @patch("src.context.rolling_summary.OpenAI")
    @patch("src.context.rolling_summary.load_qwen_config")
    def test_default_client_uses_real_tuple_config_contract(
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
            _fake_response("默认客户端摘要")
        )

        result = refresh_conversation_summary(
            conversation_id,
            path=self.db_path,
        )

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["summary"]["summary_revision"], 1)
        mock_openai.assert_called_once_with(
            api_key="test-api-key",
            base_url="https://example.invalid/v1",
        )
        request = mock_openai.return_value.chat.completions.create.call_args.kwargs
        self.assertEqual(request["model"], "qwen-test")

    def test_rolling_update_sends_old_summary_plus_only_new_delta(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        first = refresh_conversation_summary(
            conversation_id,
            completion_func=lambda **kwargs: _fake_response("第一版摘要"),
            model_name="qwen-test",
            path=self.db_path,
        )
        for index in (8, 9):
            self._append_turn(conversation_id, index)
        calls: list[dict] = []

        def complete(**kwargs):
            calls.append(kwargs)
            return _fake_response("第二版摘要")

        second = refresh_conversation_summary(
            conversation_id,
            completion_func=complete,
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(first["summary"]["summary_revision"], 1)
        self.assertEqual(second["summary"]["summary_revision"], 2)
        self.assertEqual(second["summary"]["covered_turn_count"], 6)
        self.assertEqual(
            second["summary"]["covered_until_message_id"],
            f"{conversation_id}-a6",
        )
        payload = json.loads(calls[0]["messages"][1]["content"])
        self.assertEqual(payload["existing_summary"], "第一版摘要")
        self.assertEqual(
            [turn["assistant_message_id"] for turn in payload["newly_eligible_turns"]],
            [f"{conversation_id}-a5", f"{conversation_id}-a6"],
        )
        serialized = calls[0]["messages"][1]["content"]
        self.assertNotIn(f"{conversation_id}-a1", serialized)
        self.assertNotIn(f"{conversation_id}-a7", serialized)

    def test_parse_and_schema_errors_preserve_existing_summary(self) -> None:
        conversation_id = self._conversation_with_turns(9)
        original = self._seed_existing_summary(conversation_id)
        invalid_responses = (
            SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content="not-json"))]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(
                            content=json.dumps(
                                {"summary": "摘要", "trace_json": {}},
                                ensure_ascii=False,
                            )
                        )
                    )
                ]
            ),
        )

        for response in invalid_responses:
            with self.subTest(content=response.choices[0].message.content):
                result = refresh_conversation_summary(
                    conversation_id,
                    completion_func=lambda **kwargs: response,
                    model_name="qwen-test",
                    path=self.db_path,
                )
                current = get_conversation_summary(
                    conversation_id, path=self.db_path
                )
                self.assertEqual(result["status"], "failed")
                self.assertEqual(current, original)

    def test_transient_failure_retries_once_then_succeeds(self) -> None:
        conversation_id = self._conversation_with_turns(7)
        attempts = 0

        class TransientError(RuntimeError):
            pass

        def complete(**kwargs):
            nonlocal attempts
            attempts += 1
            if attempts == 1:
                raise TransientError("temporary")
            return _fake_response("重试后摘要")

        result = refresh_conversation_summary(
            conversation_id,
            completion_func=complete,
            should_retry=lambda error: isinstance(error, TransientError),
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(result["status"], "updated")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(attempts, 2)

    def test_two_transient_failures_preserve_old_summary_and_boundary(self) -> None:
        conversation_id = self._conversation_with_turns(9)
        original = self._seed_existing_summary(conversation_id)
        attempts = 0

        class TransientError(RuntimeError):
            pass

        def complete(**kwargs):
            nonlocal attempts
            attempts += 1
            raise TransientError("temporary")

        result = refresh_conversation_summary(
            conversation_id,
            completion_func=complete,
            should_retry=lambda error: isinstance(error, TransientError),
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["error_type"], "summary_api_error")
        self.assertEqual(attempts, 2)
        self.assertEqual(
            get_conversation_summary(conversation_id, path=self.db_path), original
        )

    def test_unsafe_or_overlong_output_cannot_replace_old_summary(self) -> None:
        conversation_id = self._conversation_with_turns(9)
        original = self._seed_existing_summary(conversation_id)
        unsafe_outputs = (
            "data:image/png;base64,aGVsbG8=",
            "aGVsbG93b3JsZA==",
            "长" * 2201,
        )

        for output in unsafe_outputs:
            with self.subTest(length=len(output)):
                result = refresh_conversation_summary(
                    conversation_id,
                    completion_func=lambda **kwargs: _fake_response(output),
                    model_name="qwen-test",
                    path=self.db_path,
                )
                self.assertEqual(result["status"], "failed")
                self.assertEqual(result["error_type"], "summary_schema_error")
                self.assertEqual(
                    get_conversation_summary(conversation_id, path=self.db_path),
                    original,
                )

    def test_conversations_are_isolated(self) -> None:
        conversation_a = self._conversation_with_turns(7, "A")
        conversation_b = self._conversation_with_turns(7, "B")

        result = refresh_conversation_summary(
            conversation_a,
            completion_func=lambda **kwargs: _fake_response("A 摘要"),
            model_name="qwen-test",
            path=self.db_path,
        )

        self.assertEqual(result["status"], "updated")
        self.assertIsNotNone(
            get_conversation_summary(conversation_a, path=self.db_path)
        )
        self.assertIsNone(
            get_conversation_summary(conversation_b, path=self.db_path)
        )


if __name__ == "__main__":
    unittest.main()
