"""统一 Agent 粗粒度运行 Trace 的纯本地测试。"""

from __future__ import annotations

import json
import unittest

from src.agent import run_teacher_agent
from src.tool_client import answer_with_tools
from tests.test_agent import (
    RAG_CARDS,
    FakeAnalyzer,
    FakeAnswer,
    FakeRetriever,
    FakeToolAnswer,
    analysis_json,
)
from tests.test_tool_client import FakeCompletion, completion_response, tool_call


def step_by_name(trace: dict, name: str) -> dict:
    return next(step for step in trace["steps"] if step["name"] == name)


class AgentObservabilityTests(unittest.TestCase):
    def test_plain_path_step_order_and_model_request_total(self) -> None:
        result = run_teacher_agent(
            "解释欧姆定律。",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(analysis_json()),
            answer_func=FakeAnswer(),
        )

        trace = result["trace"]
        self.assertEqual(
            [step["name"] for step in trace["steps"]],
            ["analyzer", "router", "rag_retrieval", "final_answer"],
        )
        self.assertEqual(trace["status"], "completed")
        self.assertEqual(trace["total_model_requests"], 2)
        self.assertEqual(
            step_by_name(trace, "rag_retrieval")["status"],
            "skipped",
        )
        self.assertEqual(
            step_by_name(trace, "final_answer")["model_requests"],
            1,
        )

    def test_rag_path_records_search_and_source_count(self) -> None:
        result = run_teacher_agent(
            "欧姆定律是什么？",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(analysis_json()),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
        )

        trace = result["trace"]
        retrieval = step_by_name(trace, "rag_retrieval")
        self.assertEqual(trace["rag_searches"], 1)
        self.assertEqual(retrieval["status"], "success")
        self.assertEqual(retrieval["model_requests"], 0)
        self.assertEqual(retrieval["metadata"]["source_count"], 1)
        self.assertEqual(trace["total_model_requests"], 2)

    def test_tool_path_records_three_total_requests_and_one_execution(self) -> None:
        result = run_teacher_agent(
            "电压为 12 V，电阻为 6 Ω，求电流。",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    teaching_mode="solve",
                    calculation_required=True,
                )
            ),
            answer_func=FakeAnswer(),
            tool_answer_func=FakeToolAnswer(),
        )

        trace = result["trace"]
        selection = step_by_name(trace, "tool_selection")
        execution = step_by_name(trace, "tool_execution")
        result_answer = step_by_name(trace, "tool_result_answer")
        self.assertEqual(
            [step["name"] for step in trace["steps"]],
            [
                "analyzer",
                "router",
                "rag_retrieval",
                "tool_selection",
                "tool_execution",
                "tool_result_answer",
            ],
        )
        self.assertEqual(selection["model_requests"], 1)
        self.assertEqual(execution["model_requests"], 0)
        self.assertEqual(execution["metadata"]["tool_record_count"], 1)
        self.assertEqual(result_answer["model_requests"], 1)
        self.assertEqual(trace["total_model_requests"], 3)
        self.assertEqual(trace["tool_executions"], 1)
        self.assertTrue(trace["use_tools"])

    def test_rag_and_tool_path_counts_one_search_without_extra_requests(self) -> None:
        result = run_teacher_agent(
            "根据资料计算电流。",
            analyzer_func=FakeAnalyzer(
                analysis_json(needs_rag=True, calculation_required=True)
            ),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
            tool_answer_func=FakeToolAnswer(),
        )

        trace = result["trace"]
        self.assertEqual(trace["rag_searches"], 1)
        self.assertEqual(trace["tool_executions"], 1)
        self.assertEqual(trace["total_model_requests"], 3)
        self.assertEqual(
            step_by_name(trace, "rag_retrieval")["metadata"]["source_count"],
            1,
        )

    def test_tool_result_retry_raises_agent_total_to_four_requests(self) -> None:
        completion = FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                TimeoutError("temporary result failure"),
                completion_response(tool_calls=None, content="电流为 2 A。"),
            ]
        )

        def retrying_tool_answer(
            question: str,
            context: str | None = None,
            mode_instruction: str | None = None,
        ) -> dict:
            return answer_with_tools(
                question,
                context=context,
                mode_instruction=mode_instruction,
                completion_func=completion,
            )

        result = run_teacher_agent(
            "电压为 12 V，电阻为 6 Ω，求电流。",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True)
            ),
            answer_func=FakeAnswer(),
            tool_answer_func=retrying_tool_answer,
        )

        trace = result["trace"]
        self.assertEqual(trace["total_model_requests"], 4)
        self.assertEqual(trace["tool_executions"], 1)
        self.assertEqual(result["tool_model_requests"], 3)
        result_step = step_by_name(trace, "tool_result_answer")
        self.assertEqual(result_step["status"], "retry_success")
        self.assertEqual(result_step["attempts"], 2)
        self.assertEqual(result_step["model_requests"], 2)

    def test_image_block_creates_blocked_trace(self) -> None:
        result = run_teacher_agent(
            "请看图回答。",
            analyzer_func=FakeAnalyzer(analysis_json(image_required=True)),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
            tool_answer_func=FakeToolAnswer(),
        )

        trace = result["trace"]
        self.assertEqual(trace["status"], "blocked")
        self.assertEqual(trace["rag_searches"], 0)
        self.assertEqual(trace["tool_executions"], 0)
        self.assertEqual(trace["total_model_requests"], 1)
        self.assertEqual(
            [step["name"] for step in trace["steps"]],
            ["analyzer", "router", "rag_retrieval"],
        )
        self.assertEqual(trace["steps"][-1]["status"], "skipped")

    def test_analyzer_fallback_sets_completed_with_fallback(self) -> None:
        result = run_teacher_agent(
            "测试问题",
            analyzer_func=FakeAnalyzer("不是 JSON"),
            answer_func=FakeAnswer(),
        )

        trace = result["trace"]
        self.assertTrue(result["analysis_fallback"])
        self.assertEqual(trace["status"], "completed_with_fallback")
        self.assertTrue(trace["analysis_fallback"])
        self.assertEqual(trace["steps"][0]["error_type"], "analyzer_parse")
        self.assertEqual(trace["total_model_requests"], 2)

    def test_case_id_route_summary_and_json_serialization(self) -> None:
        result = run_teacher_agent(
            "解释欧姆定律。",
            mode_override="explain",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(analysis_json()),
            answer_func=FakeAnswer(),
            case_id="case-stage08-001",
        )

        trace = result["trace"]
        router = step_by_name(trace, "router")
        self.assertEqual(trace["case_id"], "case-stage08-001")
        self.assertEqual(trace["teaching_mode"], "explain")
        self.assertFalse(trace["use_rag"])
        self.assertFalse(trace["use_tools"])
        self.assertEqual(
            router["metadata"],
            {
                "teaching_mode": "explain",
                "use_rag": False,
                "use_tools": False,
                "should_answer": True,
            },
        )
        self.assertIsInstance(json.dumps(trace, ensure_ascii=False), str)


if __name__ == "__main__":
    unittest.main()
