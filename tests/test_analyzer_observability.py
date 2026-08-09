"""Analyzer 有限重试与步骤观测的纯本地测试。"""

from __future__ import annotations

import json
import unittest

from src.analyzer import analyze_question, analyze_question_with_trace
from src.schemas import StepStatus, TeachingMode


def valid_payload() -> dict:
    return {
        "teaching_mode": "solve",
        "physics_topic": "电学",
        "question_type": "计算题",
        "needs_rag": False,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "条件完整，需要计算电流。",
        "calculation_required": True,
    }


def valid_json() -> str:
    return json.dumps(valid_payload(), ensure_ascii=False)


def fallback_dump() -> dict:
    return {
        "teaching_mode": "solve",
        "physics_topic": "综合",
        "question_type": "未知",
        "needs_rag": False,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "问题分析失败，按普通完整解题处理。",
        "calculation_required": False,
        "context_relation": "uncertain",
        "needs_previous_image_context": False,
    }


class AnalyzerObservabilityTests(unittest.TestCase):
    def test_first_request_success_creates_success_trace(self) -> None:
        calls = 0

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            return valid_json()

        analysis, fallback, trace = analyze_question_with_trace(
            "求电流",
            analyze_func=analyzer,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(analysis.physics_topic, "电学")
        self.assertFalse(fallback)
        self.assertEqual(trace.status, StepStatus.SUCCESS)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 1)
        self.assertEqual(trace.metadata, {"fallback": False})

    def test_first_failure_second_success_creates_retry_success_trace(self) -> None:
        calls = 0

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise TimeoutError("temporary")
            return valid_json()

        analysis, fallback, trace = analyze_question_with_trace(
            "求电流",
            analyze_func=analyzer,
            should_retry=lambda error: True,
        )

        self.assertEqual(calls, 2)
        self.assertTrue(analysis.calculation_required)
        self.assertFalse(fallback)
        self.assertEqual(trace.status, StepStatus.RETRY_SUCCESS)
        self.assertEqual(trace.attempts, 2)
        self.assertEqual(trace.model_requests, 2)
        self.assertIsNone(trace.error_type)

    def test_two_retryable_call_failures_create_api_error_trace(self) -> None:
        calls = 0

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            raise TimeoutError("unsafe details must not enter trace")

        analysis, fallback, trace = analyze_question_with_trace(
            "求电流",
            analyze_func=analyzer,
            should_retry=lambda error: True,
        )

        self.assertEqual(calls, 2)
        self.assertEqual(analysis.model_dump(mode="json"), fallback_dump())
        self.assertTrue(fallback)
        self.assertEqual(trace.status, StepStatus.ERROR)
        self.assertEqual(trace.attempts, 2)
        self.assertEqual(trace.model_requests, 2)
        self.assertEqual(trace.error_type, "analyzer_api")
        self.assertEqual(trace.error_message, "问题分析模型调用失败。")
        self.assertNotIn("unsafe", trace.error_message)
        self.assertEqual(trace.metadata, {"fallback": True})

    def test_non_retryable_call_failure_runs_once(self) -> None:
        calls = 0

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            raise ValueError("invalid request")

        analysis, fallback, trace = analyze_question_with_trace(
            "求电流",
            analyze_func=analyzer,
            should_retry=lambda error: False,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(analysis.model_dump(mode="json"), fallback_dump())
        self.assertTrue(fallback)
        self.assertEqual(trace.status, StepStatus.ERROR)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 1)
        self.assertEqual(trace.error_type, "analyzer_api")

    def test_invalid_json_is_not_retried(self) -> None:
        calls = 0

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            return "not json"

        analysis, fallback, trace = analyze_question_with_trace(
            "测试问题",
            analyze_func=analyzer,
            should_retry=lambda error: True,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(analysis.model_dump(mode="json"), fallback_dump())
        self.assertTrue(fallback)
        self.assertEqual(trace.status, StepStatus.ERROR)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 1)
        self.assertEqual(trace.error_type, "analyzer_parse")
        self.assertEqual(trace.error_message, "问题分析结果不是有效 JSON。")

    def test_schema_error_is_not_retried(self) -> None:
        calls = 0
        payload = valid_payload()
        del payload["physics_topic"]

        def analyzer(question: str) -> str:
            nonlocal calls
            calls += 1
            return json.dumps(payload, ensure_ascii=False)

        analysis, fallback, trace = analyze_question_with_trace(
            "测试问题",
            analyze_func=analyzer,
            should_retry=lambda error: True,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(analysis.model_dump(mode="json"), fallback_dump())
        self.assertTrue(fallback)
        self.assertEqual(trace.status, StepStatus.ERROR)
        self.assertEqual(trace.attempts, 1)
        self.assertEqual(trace.model_requests, 1)
        self.assertEqual(trace.error_type, "analyzer_schema")
        self.assertEqual(trace.error_message, "问题分析结果不符合字段要求。")

    def test_legacy_entry_keeps_two_item_return_contract(self) -> None:
        result = analyze_question("求电流", analyze_func=lambda question: valid_json())

        self.assertEqual(len(result), 2)
        analysis, fallback = result
        self.assertIsInstance(analysis.teaching_mode, TeachingMode)
        self.assertFalse(fallback)


if __name__ == "__main__":
    unittest.main()
