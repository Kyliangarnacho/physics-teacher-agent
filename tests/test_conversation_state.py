"""Stage 10 Conversation State 纯逻辑单元测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.conversation import (
    ConversationState,
    ResolvedConversationState,
    build_state_update_after_turn,
    build_teaching_state_context,
    is_follow_up_message,
    requests_full_answer,
    resolve_conversation_state,
)


def previous_state(**overrides: object) -> ConversationState:
    payload: dict[str, object] = {
        "conversation_id": "conv-1",
        "active_problem_text": "电阻是 6Ω，电压 12V，求电流",
        "active_image_context": "这是一张串联电路图",
        "teaching_mode": "hint",
        "hint_step": 2,
        "updated_at": "2026-08-06T08:00:00+00:00",
    }
    payload.update(overrides)
    return ConversationState(**payload)


class FollowUpDetectionTests(unittest.TestCase):
    def test_is_follow_up_message_is_conservative(self) -> None:
        for text in (
            "继续",
            "继续讲",
            "下一步怎么做",
            "再提示一步",
            "那为什么电压表会有示数",
            "刚才那道题",
            "上一题的方法",
            "这张图怎么分析",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_follow_up_message(text))

        for text in (
            "",
            "   ",
            "凸透镜成像规律是什么",
            "电路断开后继续通电会怎样",
            "为什么电流会变大",
            "1+1等于几",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_follow_up_message(text))

    def test_requests_full_answer_detection(self) -> None:
        for text in (
            "直接告诉答案",
            "直接给答案",
            "告诉我答案",
            "完整解答",
            "不要提示了",
            "不用提示",
            "别提示，直接算完",
        ):
            with self.subTest(text=text):
                self.assertTrue(requests_full_answer(text))

        for text in ("", "继续", "下一步", "为什么电阻会变大"):
            with self.subTest(text=text):
                self.assertFalse(requests_full_answer(text))


class ResolveConversationStateTests(unittest.TestCase):
    def test_new_question_builds_fresh_state(self) -> None:
        resolved = resolve_conversation_state(
            "新的问题：凸透镜焦距 10cm",
            previous_state=previous_state(),
        )

        self.assertEqual(resolved.active_problem_text, "新的问题：凸透镜焦距 10cm")
        self.assertIsNone(resolved.active_image_context)
        self.assertFalse(resolved.is_follow_up)
        self.assertFalse(resolved.exits_hint)
        self.assertEqual(resolved.hint_step, 0)

    def test_continue_reuses_previous_problem_and_hint(self) -> None:
        for text in ("继续", "下一步怎么做"):
            with self.subTest(text=text):
                resolved = resolve_conversation_state(
                    text,
                    previous_state=previous_state(),
                )
                self.assertTrue(resolved.is_follow_up)
                self.assertEqual(
                    resolved.active_problem_text,
                    "电阻是 6Ω，电压 12V，求电流",
                )
                self.assertEqual(
                    resolved.active_image_context,
                    "这是一张串联电路图",
                )
                self.assertEqual(resolved.teaching_mode, "hint")
                self.assertEqual(resolved.hint_step, 2)

    def test_follow_up_reuses_image_without_new_image(self) -> None:
        resolved = resolve_conversation_state(
            "这张图里电流方向是什么",
            previous_state=previous_state(
                active_image_context="串联电路图已确认文字"
            ),
        )

        self.assertTrue(resolved.is_follow_up)
        self.assertEqual(resolved.active_image_context, "串联电路图已确认文字")

    def test_new_image_context_replaces_old(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(active_image_context="旧图上下文"),
            current_image_context="新图上下文",
        )

        self.assertEqual(resolved.active_image_context, "新图上下文")

        # 新问题带新图片：新图片仍优先。
        resolved = resolve_conversation_state(
            "完全新的题目",
            previous_state=previous_state(active_image_context="旧图上下文"),
            current_image_context="新图上下文",
        )
        self.assertEqual(resolved.active_image_context, "新图上下文")

    def test_unrelated_new_question_clears_old_image(self) -> None:
        resolved = resolve_conversation_state(
            "完全无关的新题目",
            previous_state=previous_state(active_image_context="旧图上下文"),
        )

        self.assertFalse(resolved.is_follow_up)
        self.assertEqual(resolved.active_problem_text, "完全无关的新题目")
        self.assertIsNone(resolved.active_image_context)
        self.assertEqual(resolved.hint_step, 0)

    def test_full_answer_request_exits_hint(self) -> None:
        resolved = resolve_conversation_state(
            "直接告诉我答案",
            previous_state=previous_state(),
        )

        self.assertTrue(resolved.exits_hint)
        self.assertEqual(resolved.teaching_mode, "solve")

    def test_explicit_mode_override_takes_priority(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(teaching_mode="hint"),
            mode_override="explain",
        )

        self.assertEqual(resolved.teaching_mode, "explain")

        # 显式覆盖优先于“退出 hint”默认 solve。
        resolved = resolve_conversation_state(
            "直接算完",
            previous_state=previous_state(),
            mode_override="diagnose",
        )
        self.assertTrue(resolved.exits_hint)
        self.assertEqual(resolved.teaching_mode, "diagnose")

    def test_empty_question_raises_value_error(self) -> None:
        for question in ("", "   "):
            with self.subTest(question=question):
                with self.assertRaises(ValueError):
                    resolve_conversation_state(question)

    def test_previous_state_is_not_mutated(self) -> None:
        previous = previous_state()
        before = previous.model_dump()

        resolve_conversation_state("继续", previous_state=previous)

        self.assertEqual(previous.model_dump(), before)


class StateUpdateAfterTurnTests(unittest.TestCase):
    def test_hint_success_increments_step(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(hint_step=1),
        )

        update = build_state_update_after_turn(
            resolved,
            succeeded=True,
            effective_mode="hint",
        )

        self.assertEqual(update["hint_step"], 2)
        self.assertEqual(update["teaching_mode"], "hint")
        self.assertEqual(update["conversation_id"], "conv-1")

    def test_hint_success_increments_step_repeatedly(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(hint_step=2),
        )
        update = build_state_update_after_turn(
            resolved,
            succeeded=True,
            effective_mode="hint",
        )

        self.assertEqual(update["hint_step"], 3)

    def test_failure_does_not_advance_step(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(hint_step=2),
        )

        self.assertIsNone(
            build_state_update_after_turn(
                resolved,
                succeeded=False,
                effective_mode="hint",
            )
        )

    def test_non_hint_success_resets_step(self) -> None:
        resolved = resolve_conversation_state(
            "直接告诉我答案",
            previous_state=previous_state(hint_step=2),
        )

        update = build_state_update_after_turn(
            resolved,
            succeeded=True,
            effective_mode=resolved.teaching_mode,
        )

        self.assertEqual(update["hint_step"], 0)
        self.assertEqual(update["teaching_mode"], "solve")


class TeachingStateContextTests(unittest.TestCase):
    def test_context_contains_safe_fields_only(self) -> None:
        resolved = resolve_conversation_state(
            "继续",
            previous_state=previous_state(),
        )

        context = build_teaching_state_context(resolved)

        self.assertIsNotNone(context)
        self.assertIn("当前活动题目：电阻是 6Ω，电压 12V，求电流", context)
        self.assertIn("已确认图片文字上下文：这是一张串联电路图", context)
        self.assertIn("当前教学模式：hint", context)
        self.assertIn("已完成提示步数：2", context)
        self.assertIn("不要重复已完成步骤", context)
        for forbidden in (
            "trace",
            "sources",
            "tool_records",
            "base64",
            "data:image",
            "image_hash",
        ):
            self.assertNotIn(forbidden, context)

    def test_context_returns_none_without_active_problem(self) -> None:
        empty = ResolvedConversationState(
            active_problem_text=None,
            active_image_context=None,
            teaching_mode=None,
            hint_step=0,
            is_follow_up=False,
            exits_hint=False,
        )

        self.assertIsNone(build_teaching_state_context(empty))


class ImageContextSafetyTests(unittest.TestCase):
    def test_resolve_rejects_unsafe_image_context(self) -> None:
        with self.assertRaises(ValueError):
            resolve_conversation_state(
                "继续",
                previous_state=previous_state(),
                current_image_context="data:image/png;base64,AAAA",
            )
        with self.assertRaises(ValueError):
            resolve_conversation_state(
                "继续",
                previous_state=previous_state(),
                current_image_context="QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVoxMjM0NTY=",
            )
        with self.assertRaises(ValueError):
            resolve_conversation_state(
                "继续",
                previous_state=previous_state(),
                current_image_context=b"raw bytes",
            )

    def test_schema_rejects_unsafe_active_image_context(self) -> None:
        with self.assertRaises(ValidationError):
            ConversationState(
                conversation_id="conv-1",
                active_image_context="data:image/png;base64,AAAA",
                updated_at="2026-08-06T08:00:00+00:00",
            )
        with self.assertRaises(ValidationError):
            ConversationState(
                conversation_id="conv-1",
                active_image_context=b"raw bytes",
                updated_at="2026-08-06T08:00:00+00:00",
            )


if __name__ == "__main__":
    unittest.main()
