"""Stage 06 结构化数据模型的单元测试。"""

from __future__ import annotations

import unittest

from pydantic import ValidationError

from src.schemas import QuestionAnalysis, RouteDecision, TeachingMode


def valid_analysis_data() -> dict[str, object]:
    """返回一份可按测试需要修改的合法问题分析数据。"""
    return {
        "teaching_mode": "solve",
        "physics_topic": "力学",
        "question_type": "计算题",
        "needs_rag": True,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "题目条件完整，需要计算平均速度。",
    }


class TeachingModeTests(unittest.TestCase):
    def test_defines_exactly_four_string_modes(self) -> None:
        self.assertEqual(
            [(mode.name, mode.value) for mode in TeachingMode],
            [
                ("SOLVE", "solve"),
                ("EXPLAIN", "explain"),
                ("HINT", "hint"),
                ("DIAGNOSE", "diagnose"),
            ],
        )
        self.assertTrue(all(isinstance(mode, str) for mode in TeachingMode))


class QuestionAnalysisTests(unittest.TestCase):
    def test_accepts_valid_question_analysis(self) -> None:
        analysis = QuestionAnalysis(**valid_analysis_data())

        self.assertEqual(analysis.teaching_mode, TeachingMode.SOLVE)
        self.assertEqual(analysis.physics_topic, "力学")
        self.assertTrue(analysis.needs_rag)
        self.assertFalse(analysis.missing_conditions)

    def test_accepts_each_teaching_mode(self) -> None:
        for mode in TeachingMode:
            with self.subTest(mode=mode.value):
                data = valid_analysis_data()
                data["teaching_mode"] = mode.value

                analysis = QuestionAnalysis(**data)

                self.assertEqual(analysis.teaching_mode, mode)

    def test_calculation_required_accepts_explicit_booleans(self) -> None:
        for value in (True, False):
            with self.subTest(value=value):
                data = valid_analysis_data()
                data["calculation_required"] = value

                analysis = QuestionAnalysis(**data)

                self.assertIs(analysis.calculation_required, value)

    def test_calculation_required_defaults_to_false(self) -> None:
        analysis = QuestionAnalysis(**valid_analysis_data())

        self.assertFalse(analysis.calculation_required)

    def test_rejects_invalid_teaching_mode(self) -> None:
        data = valid_analysis_data()
        data["teaching_mode"] = "chat"

        with self.assertRaises(ValidationError):
            QuestionAnalysis(**data)

    def test_rejects_missing_required_field(self) -> None:
        data = valid_analysis_data()
        del data["physics_topic"]

        with self.assertRaises(ValidationError):
            QuestionAnalysis(**data)

    def test_rejects_short_reason_longer_than_80_characters(self) -> None:
        data = valid_analysis_data()
        data["short_reason"] = "理" * 81

        with self.assertRaises(ValidationError):
            QuestionAnalysis(**data)

    def test_rejects_empty_or_whitespace_text_fields(self) -> None:
        for field_name, value in (
            ("physics_topic", ""),
            ("physics_topic", "   "),
            ("question_type", ""),
            ("short_reason", "\t\r\n"),
        ):
            with self.subTest(field_name=field_name, value=repr(value)):
                data = valid_analysis_data()
                data[field_name] = value

                with self.assertRaises(ValidationError):
                    QuestionAnalysis(**data)

    def test_rejects_extra_field(self) -> None:
        data = valid_analysis_data()
        data["unexpected"] = "不允许"

        with self.assertRaises(ValidationError):
            QuestionAnalysis(**data)


class RouteDecisionTests(unittest.TestCase):
    def test_user_message_defaults_to_none(self) -> None:
        decision = RouteDecision(
            teaching_mode=TeachingMode.EXPLAIN,
            use_rag=False,
            should_answer=True,
        )

        self.assertIsNone(decision.user_message)

    def test_accepts_optional_user_message(self) -> None:
        decision = RouteDecision(
            teaching_mode="hint",
            use_rag=True,
            should_answer=False,
            user_message="请补充题目图片。",
        )

        self.assertEqual(decision.teaching_mode, TeachingMode.HINT)
        self.assertEqual(decision.user_message, "请补充题目图片。")

    def test_use_tools_accepts_explicit_booleans(self) -> None:
        for value in (True, False):
            with self.subTest(value=value):
                decision = RouteDecision(
                    teaching_mode="solve",
                    use_rag=False,
                    should_answer=True,
                    use_tools=value,
                )

                self.assertIs(decision.use_tools, value)

    def test_use_tools_defaults_to_false(self) -> None:
        decision = RouteDecision(
            teaching_mode="solve",
            use_rag=False,
            should_answer=True,
        )

        self.assertFalse(decision.use_tools)

    def test_model_dump_contains_new_defaulted_fields(self) -> None:
        analysis_dump = QuestionAnalysis(**valid_analysis_data()).model_dump()
        decision_dump = RouteDecision(
            teaching_mode="solve",
            use_rag=False,
            should_answer=True,
        ).model_dump()

        self.assertIn("calculation_required", analysis_dump)
        self.assertFalse(analysis_dump["calculation_required"])
        self.assertIn("use_tools", decision_dump)
        self.assertFalse(decision_dump["use_tools"])

    def test_new_fields_reject_non_boolean_types(self) -> None:
        analysis_data = valid_analysis_data()
        analysis_data["calculation_required"] = "true"
        with self.assertRaises(ValidationError):
            QuestionAnalysis(**analysis_data)

        with self.assertRaises(ValidationError):
            RouteDecision(
                teaching_mode="solve",
                use_rag=False,
                should_answer=True,
                use_tools=1,
            )

    def test_rejects_extra_field(self) -> None:
        with self.assertRaises(ValidationError):
            RouteDecision(
                teaching_mode="diagnose",
                use_rag=False,
                should_answer=True,
                unexpected=True,
            )


if __name__ == "__main__":
    unittest.main()
