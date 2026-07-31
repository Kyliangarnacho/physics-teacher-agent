import json
import unittest

from src.analyzer import analyze_question
from src.prompts import MODE_INSTRUCTIONS, QUESTION_ANALYZER_SYSTEM_PROMPT
from src.schemas import TeachingMode


def valid_payload(mode: str = "solve") -> dict:
    return {
        "teaching_mode": mode,
        "physics_topic": "力学",
        "question_type": "计算题",
        "needs_rag": True,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "这是一道条件完整的力学计算题。",
    }


def json_result(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)


class TestAnalyzeQuestion(unittest.TestCase):
    def test_analyzer_prompt_defines_diagnose_priority_and_examples(self):
        self.assertIn(
            "diagnose 优先于 explain",
            QUESTION_ANALYZER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "我写成 I=R/U，这一步哪里错了？只解释错误。",
            QUESTION_ANALYZER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "为什么电阻增大时电流减小？",
            QUESTION_ANALYZER_SYSTEM_PROMPT,
        )
        self.assertIn(
            "下一步怎么做？不要告诉我答案。",
            QUESTION_ANALYZER_SYSTEM_PROMPT,
        )

    def test_valid_json_returns_question_analysis(self):
        analysis, fallback = analyze_question(
            "一辆小车的平均速度是多少？",
            lambda question: json_result(valid_payload()),
        )

        self.assertEqual(analysis.teaching_mode, TeachingMode.SOLVE)
        self.assertEqual(analysis.physics_topic, "力学")
        self.assertTrue(analysis.needs_rag)
        self.assertFalse(fallback)

    def test_all_four_teaching_modes_are_supported(self):
        self.assertEqual(
            set(MODE_INSTRUCTIONS),
            {mode.value for mode in TeachingMode},
        )
        for mode in TeachingMode:
            with self.subTest(mode=mode.value):
                analysis, fallback = analyze_question(
                    "测试问题",
                    lambda question, value=mode.value: json_result(
                        valid_payload(value)
                    ),
                )
                self.assertEqual(analysis.teaching_mode, mode)
                self.assertFalse(fallback)

    def test_non_json_uses_fallback(self):
        analysis, fallback = analyze_question(
            "测试问题",
            lambda question: "这不是 JSON",
        )

        self.assertEqual(analysis.teaching_mode, TeachingMode.SOLVE)
        self.assertTrue(fallback)

    def test_missing_required_field_uses_fallback(self):
        payload = valid_payload()
        del payload["physics_topic"]

        analysis, fallback = analyze_question(
            "测试问题",
            lambda question: json_result(payload),
        )

        self.assertEqual(analysis.physics_topic, "综合")
        self.assertTrue(fallback)

    def test_invalid_mode_uses_fallback(self):
        analysis, fallback = analyze_question(
            "测试问题",
            lambda question: json_result(valid_payload("review")),
        )

        self.assertEqual(analysis.teaching_mode, TeachingMode.SOLVE)
        self.assertTrue(fallback)

    def test_overlong_short_reason_uses_fallback(self):
        payload = valid_payload()
        payload["short_reason"] = "长" * 81

        analysis, fallback = analyze_question(
            "测试问题",
            lambda question: json_result(payload),
        )

        self.assertEqual(analysis.short_reason, "问题分析失败，按普通完整解题处理。")
        self.assertTrue(fallback)

    def test_analyze_func_exception_uses_fallback(self):
        def failing_analyzer(question: str) -> str:
            raise RuntimeError("模拟模型异常")

        analysis, fallback = analyze_question("测试问题", failing_analyzer)

        self.assertEqual(analysis.physics_topic, "综合")
        self.assertTrue(fallback)

    def test_safe_fallback_has_expected_values(self):
        analysis, fallback = analyze_question(
            "测试问题",
            lambda question: "{}",
        )

        self.assertEqual(
            analysis.model_dump(mode="json"),
            {
                "teaching_mode": "solve",
                "physics_topic": "综合",
                "question_type": "未知",
                "needs_rag": False,
                "missing_conditions": False,
                "image_required": False,
                "student_work_provided": False,
                "short_reason": "问题分析失败，按普通完整解题处理。",
            },
        )
        self.assertTrue(fallback)

    def test_empty_question_does_not_call_analyze_func(self):
        calls = []

        def fake_analyzer(question: str) -> str:
            calls.append(question)
            return json_result(valid_payload())

        for question in ("", "   "):
            with self.subTest(question=repr(question)):
                with self.assertRaises(ValueError):
                    analyze_question(question, fake_analyzer)

        self.assertEqual(calls, [])

    def test_normal_result_sets_analysis_fallback_false(self):
        received_questions = []

        def fake_analyzer(question: str) -> str:
            received_questions.append(question)
            return json_result(valid_payload("explain"))

        analysis, fallback = analyze_question(
            "  为什么金属摸起来更凉？  ",
            fake_analyzer,
        )

        self.assertEqual(received_questions, ["为什么金属摸起来更凉？"])
        self.assertEqual(analysis.teaching_mode, TeachingMode.EXPLAIN)
        self.assertIs(fallback, False)


if __name__ == "__main__":
    unittest.main()
