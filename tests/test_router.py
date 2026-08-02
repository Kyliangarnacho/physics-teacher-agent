"""Stage 06 Router 的单元测试。"""

from __future__ import annotations

import unittest

from src.router import route_question
from src.schemas import QuestionAnalysis, TeachingMode


def make_analysis(**overrides: object) -> QuestionAnalysis:
    data: dict[str, object] = {
        "teaching_mode": "explain",
        "physics_topic": "电学",
        "question_type": "概念题",
        "needs_rag": True,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "需要解释电路中的物理关系。",
        "calculation_required": False,
    }
    data.update(overrides)
    return QuestionAnalysis(**data)


class RouteQuestionTests(unittest.TestCase):
    def test_auto_uses_analyzer_teaching_mode(self) -> None:
        decision = route_question(make_analysis(teaching_mode="explain"))

        self.assertEqual(decision.teaching_mode, TeachingMode.EXPLAIN)

    def test_each_manual_mode_overrides_analyzer(self) -> None:
        for mode in TeachingMode:
            with self.subTest(mode=mode.value):
                decision = route_question(
                    make_analysis(teaching_mode="explain"),
                    mode_override=mode.value,
                )

                self.assertEqual(decision.teaching_mode, mode)

    def test_auto_rag_uses_analysis_true(self) -> None:
        decision = route_question(make_analysis(needs_rag=True))

        self.assertTrue(decision.use_rag)

    def test_auto_rag_uses_analysis_false(self) -> None:
        decision = route_question(make_analysis(needs_rag=False))

        self.assertFalse(decision.use_rag)

    def test_force_rag_enables_rag(self) -> None:
        decision = route_question(
            make_analysis(needs_rag=False),
            rag_policy="force",
        )

        self.assertTrue(decision.use_rag)

    def test_off_rag_disables_rag(self) -> None:
        decision = route_question(
            make_analysis(needs_rag=True),
            rag_policy="off",
        )

        self.assertFalse(decision.use_rag)

    def test_image_required_prevents_answer(self) -> None:
        decision = route_question(
            make_analysis(image_required=True, calculation_required=True)
        )

        self.assertFalse(decision.should_answer)
        self.assertFalse(decision.use_tools)

    def test_image_required_disables_rag_even_when_forced(self) -> None:
        decision = route_question(
            make_analysis(image_required=True),
            rag_policy="force",
        )

        self.assertFalse(decision.use_rag)

    def test_image_required_provides_user_message(self) -> None:
        decision = route_question(make_analysis(image_required=True))

        self.assertIsNotNone(decision.user_message)
        self.assertIn("题图", decision.user_message)
        self.assertIn("描述", decision.user_message)

    def test_missing_conditions_still_allows_answer(self) -> None:
        decision = route_question(make_analysis(missing_conditions=True))

        self.assertTrue(decision.should_answer)
        self.assertIsNone(decision.user_message)

    def test_no_calculation_requirement_disables_tools(self) -> None:
        decision = route_question(make_analysis(calculation_required=False))

        self.assertFalse(decision.use_tools)

    def test_complete_calculation_requirement_enables_tools(self) -> None:
        decision = route_question(
            make_analysis(
                calculation_required=True,
                missing_conditions=False,
            )
        )

        self.assertTrue(decision.use_tools)

    def test_missing_conditions_disable_tools_but_still_allow_answer(self) -> None:
        decision = route_question(
            make_analysis(
                calculation_required=True,
                missing_conditions=True,
            )
        )

        self.assertFalse(decision.use_tools)
        self.assertTrue(decision.should_answer)

    def test_rag_and_tools_can_both_be_enabled(self) -> None:
        decision = route_question(
            make_analysis(
                needs_rag=True,
                calculation_required=True,
            )
        )

        self.assertTrue(decision.use_rag)
        self.assertTrue(decision.use_tools)

    def test_rag_off_does_not_disable_tools(self) -> None:
        decision = route_question(
            make_analysis(calculation_required=True),
            rag_policy="off",
        )

        self.assertFalse(decision.use_rag)
        self.assertTrue(decision.use_tools)

    def test_forced_rag_does_not_force_tools(self) -> None:
        decision = route_question(
            make_analysis(
                needs_rag=False,
                calculation_required=False,
            ),
            rag_policy="force",
        )

        self.assertTrue(decision.use_rag)
        self.assertFalse(decision.use_tools)

    def test_hint_and_diagnose_can_still_use_tools(self) -> None:
        for mode in ("hint", "diagnose"):
            with self.subTest(mode=mode):
                decision = route_question(
                    make_analysis(
                        teaching_mode=mode,
                        calculation_required=True,
                        missing_conditions=False,
                    )
                )

                self.assertEqual(decision.teaching_mode.value, mode)
                self.assertTrue(decision.use_tools)

    def test_model_dump_contains_use_tools(self) -> None:
        decision = route_question(make_analysis(calculation_required=True))

        self.assertIn("use_tools", decision.model_dump(mode="json"))
        self.assertTrue(decision.model_dump(mode="json")["use_tools"])

    def test_auto_preserves_hint_mode(self) -> None:
        decision = route_question(make_analysis(teaching_mode="hint"))

        self.assertEqual(decision.teaching_mode, TeachingMode.HINT)

    def test_auto_preserves_diagnose_mode(self) -> None:
        decision = route_question(make_analysis(teaching_mode="diagnose"))

        self.assertEqual(decision.teaching_mode, TeachingMode.DIAGNOSE)

    def test_invalid_mode_override_raises_value_error(self) -> None:
        for value in ("chat", "", "SOLVE"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    route_question(make_analysis(), mode_override=value)

    def test_invalid_rag_policy_raises_value_error(self) -> None:
        for value in ("on", "", "FORCE"):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    route_question(make_analysis(), rag_policy=value)


if __name__ == "__main__":
    unittest.main()
