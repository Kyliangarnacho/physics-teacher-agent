"""Stage 11.3 会话摘要 Schema 测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.context import (
    ConversationSummary,
    HistoryRetrievalResult,
    RetrievedHistoryTurn,
    RollingSummaryOutput,
)


VALID_SUMMARY = {
    "conversation_id": "conv-1",
    "summary_text": "学生正在讨论欧姆定律，已完成串联电路部分。",
    "covered_until_message_id": "msg-4",
    "covered_turn_count": 2,
    "summary_revision": 1,
    "model_name": "qwen-summary",
    "created_at": "2026-08-10T08:00:00+00:00",
    "updated_at": "2026-08-10T08:00:00+00:00",
}


class ConversationSummarySchemaTests(unittest.TestCase):
    def test_valid_conversation_summary(self) -> None:
        summary = ConversationSummary.model_validate(VALID_SUMMARY)

        self.assertEqual(summary.conversation_id, "conv-1")
        self.assertEqual(summary.covered_turn_count, 2)
        self.assertEqual(summary.summary_revision, 1)

    def test_extra_field_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationSummary.model_validate(
                {**VALID_SUMMARY, "trace_json": {"status": "completed"}}
            )

    def test_negative_covered_turn_count_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationSummary.model_validate(
                {**VALID_SUMMARY, "covered_turn_count": -1}
            )

    def test_summary_revision_below_one_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationSummary.model_validate(
                {**VALID_SUMMARY, "summary_revision": 0}
            )

    def test_blank_summary_is_rejected(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationSummary.model_validate(
                {**VALID_SUMMARY, "summary_text": "   "}
            )

    def test_data_url_and_base64_summary_are_rejected(self) -> None:
        unsafe_values = (
            "data:image/png;base64,aGVsbG8=",
            "aGVsbG93b3JsZA==",
        )
        for value in unsafe_values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    ConversationSummary.model_validate(
                        {**VALID_SUMMARY, "summary_text": value}
                    )

    def test_rolling_summary_output_is_minimal_and_strict(self) -> None:
        output = RollingSummaryOutput.model_validate({"summary": "两轮对话摘要。"})
        self.assertEqual(output.summary, "两轮对话摘要。")

        with self.assertRaises(ValidationError):
            RollingSummaryOutput.model_validate(
                {"summary": "摘要", "tool_records": []}
            )


class HistoryRetrievalSchemaTests(unittest.TestCase):
    def test_valid_result_uses_minimal_safe_turn_projection(self) -> None:
        result = HistoryRetrievalResult(
            turns=[
                RetrievedHistoryTurn(
                    user_message_id="u1",
                    assistant_message_id="a1",
                    user_content="凸透镜成像规律是什么？",
                    assistant_content="旧回答提到了焦距和像距。",
                    score=1.25,
                )
            ],
            candidate_count=3,
            query="凸透镜 焦距",
            retrieval_used=True,
        )

        self.assertEqual(result.turns[0].assistant_message_id, "a1")
        self.assertEqual(
            set(result.turns[0].model_dump()),
            {
                "user_message_id",
                "assistant_message_id",
                "user_content",
                "assistant_content",
                "score",
            },
        )

    def test_extra_negative_score_and_unsafe_text_are_rejected(self) -> None:
        base = {
            "user_message_id": "u1",
            "assistant_message_id": "a1",
            "user_content": "问题",
            "assistant_content": "回答",
            "score": 1.0,
        }
        invalid_values = (
            {**base, "trace_json": {}},
            {**base, "score": -0.1},
            {**base, "assistant_content": "data:image/png;base64,aGVsbG8="},
            {**base, "user_content": "aGVsbG93b3JsZA=="},
        )
        for value in invalid_values:
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    RetrievedHistoryTurn.model_validate(value)

    def test_result_is_strict_and_rejects_unsafe_query(self) -> None:
        with self.assertRaises(ValidationError):
            HistoryRetrievalResult.model_validate(
                {
                    "turns": [],
                    "candidate_count": 0,
                    "query": "data:image/png;base64,aGVsbG8=",
                    "retrieval_used": False,
                }
            )
        with self.assertRaises(ValidationError):
            HistoryRetrievalResult.model_validate(
                {
                    "turns": [],
                    "candidate_count": 0,
                    "query": "问题",
                    "retrieval_used": 0,
                }
            )


if __name__ == "__main__":
    unittest.main()
