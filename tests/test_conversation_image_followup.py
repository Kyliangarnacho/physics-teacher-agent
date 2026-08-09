"""图片题跨轮语义继承的 Conversation Service 集成测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.agent import run_teacher_agent
from src.conversation import run_conversation_turn
from src.storage import (
    create_conversation,
    get_conversation_state,
    initialize_database,
    upsert_conversation_state,
)


def analysis_payload(
    relation: str,
    *,
    needs_image: bool,
    image_required: bool = False,
) -> str:
    return json.dumps(
        {
            "teaching_mode": "solve",
            "physics_topic": "电学",
            "question_type": "跟进计算题",
            "needs_rag": False,
            "missing_conditions": False,
            "image_required": image_required,
            "student_work_provided": False,
            "short_reason": "根据当前问题与上一题状态判断上下文关系。",
            "calculation_required": False,
            "context_relation": relation,
            "needs_previous_image_context": needs_image,
        },
        ensure_ascii=False,
    )


class SemanticAnalyzer:
    def __init__(self, relation: str, *, needs_image: bool, image_required=False):
        self.relation = relation
        self.needs_image = needs_image
        self.image_required = image_required
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> str:
        self.calls.append({"question": question, **kwargs})
        return analysis_payload(
            self.relation,
            needs_image=self.needs_image,
            image_required=self.image_required,
        )


class CapturingAnswer:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def __call__(self, question: str, **kwargs) -> str:
        self.calls.append({"question": question, **kwargs})
        return "教师回答"


class ConversationImageFollowUpTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.db_path = str(Path(self.temp_dir.name) / "follow-up.db")
        initialize_database(self.db_path)

    def _create_stateful_conversation(self, *, with_image: bool = True) -> str:
        conversation = create_conversation("图片跟进", path=self.db_path)
        upsert_conversation_state(
            {
                "conversation_id": conversation["id"],
                "active_problem_text": "图示电路中铜片可转到 a、b 两点。",
                "active_image_context": (
                    "图中铜片连接 a、b 两点及电表位置关系。" if with_image else None
                ),
                "teaching_mode": "solve",
                "hint_step": 0,
            },
            path=self.db_path,
        )
        return conversation["id"]

    @staticmethod
    def _agent(analyzer: SemanticAnalyzer, answer: CapturingAnswer):
        def invoke(question: str, **kwargs):
            return run_teacher_agent(
                question,
                analyzer_func=analyzer,
                answer_func=answer,
                **kwargs,
            )

        return invoke

    def test_semantic_follow_up_phrases_inherit_previous_image(self) -> None:
        phrases = ("铜片转到a时，求电流", "第三问怎么做", "a点时呢", "那②呢")
        for phrase in phrases:
            with self.subTest(phrase=phrase):
                conversation_id = self._create_stateful_conversation()
                analyzer = SemanticAnalyzer(
                    "follow_up",
                    needs_image=True,
                    image_required=True,
                )
                answer = CapturingAnswer()

                result = run_conversation_turn(
                    conversation_id,
                    phrase,
                    phrase,
                    db_path=self.db_path,
                    agent_func=self._agent(analyzer, answer),
                )

                self.assertEqual(len(analyzer.calls), 1)
                self.assertIn(
                    "上一题存在已确认图片上下文：是",
                    analyzer.calls[0]["teaching_state_context"],
                )
                self.assertNotIn("图中铜片连接", analyzer.calls[0]["question"])
                self.assertIn("上一活动题目", answer.calls[0]["question"])
                self.assertIn("上一题已确认图片上下文", answer.calls[0]["question"])
                self.assertIn("图中铜片连接", answer.calls[0]["question"])
                self.assertTrue(result["resolved_state"].is_follow_up)
                self.assertIsNotNone(result["resolved_state"].active_image_context)

    def test_explicit_new_problem_does_not_inherit_old_image(self) -> None:
        conversation_id = self._create_stateful_conversation()
        analyzer = SemanticAnalyzer("new_problem", needs_image=False)
        answer = CapturingAnswer()
        question = "一辆小车 10 秒行驶 50 米，求平均速度。"

        result = run_conversation_turn(
            conversation_id,
            question,
            question,
            db_path=self.db_path,
            agent_func=self._agent(analyzer, answer),
        )

        self.assertEqual(len(analyzer.calls), 1)
        self.assertNotIn("图中铜片连接", answer.calls[0]["question"])
        self.assertFalse(result["resolved_state"].is_follow_up)
        self.assertIsNone(result["resolved_state"].active_image_context)
        state = get_conversation_state(conversation_id, path=self.db_path)
        self.assertIsNone(state["active_image_context"])

    def test_no_image_follow_up_keeps_normal_multiturn_behavior(self) -> None:
        conversation_id = self._create_stateful_conversation(with_image=False)
        analyzer = SemanticAnalyzer("follow_up", needs_image=False)
        answer = CapturingAnswer()

        result = run_conversation_turn(
            conversation_id,
            "第三问怎么做",
            "第三问怎么做",
            db_path=self.db_path,
            agent_func=self._agent(analyzer, answer),
        )

        self.assertEqual(len(analyzer.calls), 1)
        self.assertTrue(result["resolved_state"].is_follow_up)
        self.assertIsNone(result["resolved_state"].active_image_context)
        self.assertIn("上一活动题目", answer.calls[0]["question"])


if __name__ == "__main__":
    unittest.main()
