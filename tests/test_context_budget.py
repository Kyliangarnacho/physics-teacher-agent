"""Deterministic ContextBundle application-budget tests."""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.context.budget import (
    SUMMARY_TRUNCATION_MARKER,
    apply_context_budget,
    estimate_context_chars,
)
from src.context.schemas import (
    ContextBundle,
    ContextTurn,
    HistoryRetrievalResult,
    RetrievedHistoryTurn,
)


def _turn(index: int, size: int = 20) -> ContextTurn:
    return ContextTurn(
        user_message_id=f"u{index}",
        assistant_message_id=f"a{index}",
        user_content="问" * size,
        assistant_content="答" * size,
    )


def _retrieval(count: int, size: int = 30) -> HistoryRetrievalResult:
    return HistoryRetrievalResult(
        turns=[
            RetrievedHistoryTurn(
                user_message_id=f"ru{index}",
                assistant_message_id=f"ra{index}",
                user_content="检索问题" + "甲" * size,
                assistant_content="历史回答" + "乙" * size,
                score=float(count - index + 1),
            )
            for index in range(1, count + 1)
        ],
        candidate_count=count,
        query="检索问题",
        retrieval_used=True,
    )


def _apply(**overrides) -> ContextBundle:
    values = {
        "current_question": "当前问题",
        "rolling_summary": "旧摘要",
        "bridge_history": [_turn(1)],
        "recent_history": [_turn(2)],
        "retrieved_history": _retrieval(2),
        "teaching_state_context": "当前活动题目：测试",
        "learning_memory_context": "记忆参考：\n- 条目一\n- 条目二",
        "summary_revision": 1,
        "budget_limit": 10_000,
    }
    values.update(overrides)
    return apply_context_budget(**values)


class ContextBudgetTests(unittest.TestCase):
    def test_under_budget_keeps_everything_without_trim(self) -> None:
        bundle = _apply()

        self.assertEqual(bundle.trimmed_components, [])
        self.assertFalse(bundle.budget_exceeded)
        self.assertEqual(bundle.retrieved_turn_count, 2)
        self.assertLessEqual(bundle.estimated_chars, bundle.budget_limit)

    def test_same_input_has_deterministic_trim_result(self) -> None:
        first = _apply(budget_limit=180)
        second = _apply(budget_limit=180)

        self.assertEqual(first.model_dump(), second.model_dump())

    def test_retrieved_history_degrades_top_two_to_one_then_zero(self) -> None:
        retrieved = _retrieval(2, size=20)
        base_options = {
            "current_question": "问",
            "rolling_summary": None,
            "bridge_history": [],
            "recent_history": [],
            "teaching_state_context": None,
            "learning_memory_context": None,
        }
        base_cost = estimate_context_chars(
            **base_options,
            retrieved_history=_retrieval(0),
        )
        one_cost = estimate_context_chars(
            **base_options,
            retrieved_history=HistoryRetrievalResult(
                turns=[retrieved.turns[0]],
                candidate_count=2,
                query="检索问题",
                retrieval_used=True,
            ),
        )

        one = apply_context_budget(
            **base_options,
            retrieved_history=retrieved,
            summary_revision=None,
            budget_limit=one_cost,
        )
        zero = apply_context_budget(
            **base_options,
            retrieved_history=retrieved,
            summary_revision=None,
            budget_limit=base_cost,
        )

        self.assertEqual(one.retrieved_turn_count, 1)
        self.assertEqual(zero.retrieved_turn_count, 0)
        self.assertIn("retrieved_history", one.trimmed_components)
        self.assertIn("retrieved_history", zero.trimmed_components)

    def test_recent_bridge_and_state_survive_extreme_budget_as_complete_turns(self) -> None:
        bridge = [_turn(1, size=80)]
        recent = [_turn(2, size=80)]
        bundle = apply_context_budget(
            current_question="当前问题不可裁",
            rolling_summary=None,
            bridge_history=bridge,
            recent_history=recent,
            retrieved_history=_retrieval(0),
            teaching_state_context="当前活动题目必须保护",
            learning_memory_context=None,
            summary_revision=None,
            budget_limit=1,
        )

        self.assertEqual(bundle.bridge_history, bridge)
        self.assertEqual(bundle.recent_history, recent)
        self.assertEqual(bundle.teaching_state_context, "当前活动题目必须保护")
        self.assertEqual(bundle.current_question_chars, len("当前问题不可裁"))
        self.assertEqual(bundle.bridge_history[0].user_content, "问" * 80)
        self.assertEqual(bundle.bridge_history[0].assistant_content, "答" * 80)
        self.assertTrue(bundle.budget_exceeded)

    def test_learning_memory_is_trimmed_by_whole_entries(self) -> None:
        memory = "记忆标题\n- 第一条完整记忆\n- 第二条完整记忆\n- 第三条完整记忆"
        full = _apply(
            rolling_summary=None,
            summary_revision=None,
            bridge_history=[],
            recent_history=[],
            retrieved_history=_retrieval(0),
            teaching_state_context=None,
            learning_memory_context=memory,
        )
        target = full.estimated_chars - len("\n- 第三条完整记忆")
        trimmed = _apply(
            rolling_summary=None,
            summary_revision=None,
            bridge_history=[],
            recent_history=[],
            retrieved_history=_retrieval(0),
            teaching_state_context=None,
            learning_memory_context=memory,
            budget_limit=target,
        )

        self.assertIn("learning_memory_context", trimmed.trimmed_components)
        self.assertNotIn("第三条", trimmed.learning_memory_context or "")
        self.assertIn("第二条完整记忆", trimmed.learning_memory_context or "")

    def test_large_summary_is_safely_truncated_after_other_optional_context(self) -> None:
        bundle = apply_context_budget(
            current_question="问",
            rolling_summary="摘要内容" * 400,
            bridge_history=[],
            recent_history=[],
            retrieved_history=_retrieval(1),
            teaching_state_context=None,
            learning_memory_context="标题\n- 记忆一",
            summary_revision=3,
            budget_limit=400,
        )

        self.assertEqual(bundle.retrieved_turn_count, 0)
        self.assertIsNone(bundle.learning_memory_context)
        self.assertIn("rolling_summary", bundle.trimmed_components)
        self.assertTrue(bundle.rolling_summary.endswith(SUMMARY_TRUNCATION_MARKER))
        self.assertLessEqual(bundle.estimated_chars, 400)
        self.assertFalse(bundle.budget_exceeded)

    def test_context_schemas_reject_extra_and_unsafe_payloads(self) -> None:
        for unsafe_content in (
            "data:image/png;base64,aGVsbG8=",
            "sk-secret-value",
            "tool_records internal payload",
        ):
            with self.subTest(unsafe_content=unsafe_content):
                with self.assertRaises(ValidationError):
                    ContextTurn.model_validate(
                        {
                            "user_message_id": "u1",
                            "assistant_message_id": "a1",
                            "user_content": unsafe_content,
                            "assistant_content": "回答",
                        }
                    )
        with self.assertRaises(ValidationError):
            ContextTurn.model_validate(
                {
                    "user_message_id": "u1",
                    "assistant_message_id": "a1",
                    "user_content": "问题",
                    "assistant_content": "回答",
                    "trace_json": {},
                }
            )
        valid = _apply().model_dump()
        with self.assertRaises(ValidationError):
            ContextBundle.model_validate({**valid, "tool_records": []})


if __name__ == "__main__":
    unittest.main()
