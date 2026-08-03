"""Stage 06 统一教师 Agent 编排入口的单元测试。"""

from __future__ import annotations

import json
import unittest

from src.agent import run_teacher_agent
from src.prompts import MODE_INSTRUCTIONS


def analysis_json(**overrides: object) -> str:
    payload: dict[str, object] = {
        "teaching_mode": "explain",
        "physics_topic": "电学",
        "question_type": "概念题",
        "needs_rag": False,
        "missing_conditions": False,
        "image_required": False,
        "student_work_provided": False,
        "short_reason": "需要解释电学概念。",
        "calculation_required": False,
    }
    payload.update(overrides)
    return json.dumps(payload, ensure_ascii=False)


class FakeAnalyzer:
    def __init__(self, result: str) -> None:
        self.result = result
        self.calls: list[str] = []

    def __call__(self, question: str) -> str:
        self.calls.append(question)
        return self.result


class FakeRetriever:
    def __init__(self, cards: list[dict[str, object]]) -> None:
        self.cards = cards
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        self.calls.append((question, top_k))
        return self.cards


class FakeAnswer:
    def __init__(self, answer: str = "测试回答") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
    ) -> str:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "mode_instruction": mode_instruction,
            }
        )
        return self.answer


class FakeToolAnswer:
    def __init__(self, answer: str = "工具教师回答") -> None:
        self.answer = answer
        self.calls: list[dict[str, object]] = []

    def __call__(
        self,
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
    ) -> dict[str, object]:
        self.calls.append(
            {
                "question": question,
                "context": context,
                "mode_instruction": mode_instruction,
            }
        )
        return {
            "answer": self.answer,
            "tool_records": [
                {
                    "tool_call_id": "call-1",
                    "name": "calculate_ohms_law",
                    "status": "success",
                }
            ],
            "model_requests": 2,
        }


RAG_CARDS = [
    {
        "id": "KB-ELEC-001",
        "chapter": "欧姆定律与动态电路",
        "topic": "欧姆定律公式",
        "keywords": ["欧姆定律"],
        "source": "物理课后总结.docx",
        "content": "欧姆定律写作 I=U/R，要分清电压与电阻的位置。",
        "score": 8.5,
    }
]


class RunTeacherAgentTests(unittest.TestCase):
    def test_plain_non_rag_flow(self) -> None:
        analyzer = FakeAnalyzer(analysis_json(needs_rag=False))
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer("普通回答")
        tool_answer = FakeToolAnswer()

        result = run_teacher_agent(
            "解释欧姆定律。",
            analyzer_func=analyzer,
            retriever=retriever,
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(result["answer"], "普通回答")
        self.assertEqual(result["sources"], [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(tool_answer.calls, [])
        self.assertEqual(result["tool_records"], [])
        self.assertEqual(result["tool_model_requests"], 0)

    def test_analyzer_mode_reaches_route_and_answer_layer(self) -> None:
        answer = FakeAnswer()

        result = run_teacher_agent(
            "我的公式哪里错了？",
            analyzer_func=FakeAnalyzer(
                analysis_json(teaching_mode="diagnose")
            ),
            rag_policy="off",
            answer_func=answer,
        )

        self.assertEqual(result["route"]["teaching_mode"], "diagnose")
        self.assertEqual(
            answer.calls[0]["mode_instruction"],
            MODE_INSTRUCTIONS["diagnose"],
        )

    def test_manual_mode_override_reaches_answer_layer(self) -> None:
        answer = FakeAnswer()

        result = run_teacher_agent(
            "解释这个现象。",
            mode_override="hint",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(teaching_mode="explain")
            ),
            answer_func=answer,
        )

        self.assertEqual(result["route"]["teaching_mode"], "hint")
        self.assertEqual(
            answer.calls[0]["mode_instruction"],
            MODE_INSTRUCTIONS["hint"],
        )

    def test_each_mode_instruction_is_passed_to_answer_func(self) -> None:
        for mode, instruction in MODE_INSTRUCTIONS.items():
            with self.subTest(mode=mode):
                answer = FakeAnswer()
                run_teacher_agent(
                    "测试问题",
                    mode_override=mode,
                    rag_policy="off",
                    analyzer_func=FakeAnalyzer(analysis_json()),
                    answer_func=answer,
                )

                self.assertEqual(
                    answer.calls[0]["mode_instruction"],
                    instruction,
                )

    def test_rag_flow_calls_retriever_and_passes_context(self) -> None:
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()

        run_teacher_agent(
            "欧姆定律公式是什么？",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(analysis_json()),
            retriever=retriever,
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(
            retriever.calls,
            [("欧姆定律公式是什么？", 3)],
        )
        self.assertIn("KB-ELEC-001", answer.calls[0]["context"])
        self.assertIn("I=U/R", answer.calls[0]["context"])
        self.assertEqual(tool_answer.calls, [])

    def test_tool_flow_calls_only_tool_client_with_mode_instruction(self) -> None:
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()
        question = "电压为 12 V，电阻为 6 Ω，求电流。"

        result = run_teacher_agent(
            question,
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    teaching_mode="solve",
                    calculation_required=True,
                )
            ),
            retriever=retriever,
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(answer.calls, [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(
            tool_answer.calls,
            [
                {
                    "question": question,
                    "context": None,
                    "mode_instruction": MODE_INSTRUCTIONS["solve"],
                }
            ],
        )
        self.assertEqual(result["answer"], "工具教师回答")
        self.assertEqual(result["tool_model_requests"], 2)
        self.assertEqual(result["tool_records"][0]["tool_call_id"], "call-1")

    def test_rag_and_tool_flow_retrieves_once_and_passes_same_context(self) -> None:
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()
        question = "根据资料计算电流。"

        result = run_teacher_agent(
            question,
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    needs_rag=True,
                    calculation_required=True,
                )
            ),
            retriever=retriever,
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(retriever.calls, [(question, 3)])
        self.assertEqual(answer.calls, [])
        self.assertIn("KB-ELEC-001", tool_answer.calls[0]["context"])
        self.assertEqual(result["sources"][0]["id"], "KB-ELEC-001")
        self.assertEqual(result["tool_model_requests"], 2)

    def test_missing_conditions_disable_tools_and_use_plain_answer(self) -> None:
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()

        result = run_teacher_agent(
            "一个物体受到力，它的加速度是多少？",
            analyzer_func=FakeAnalyzer(
                analysis_json(
                    calculation_required=True,
                    missing_conditions=True,
                )
            ),
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertFalse(result["route"]["use_tools"])
        self.assertEqual(len(answer.calls), 1)
        self.assertEqual(tool_answer.calls, [])

    def test_rag_off_does_not_disable_valid_tool_route(self) -> None:
        tool_answer = FakeToolAnswer()

        result = run_teacher_agent(
            "求电流。",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(
                analysis_json(calculation_required=True)
            ),
            answer_func=FakeAnswer(),
            tool_answer_func=tool_answer,
        )

        self.assertFalse(result["route"]["use_rag"])
        self.assertTrue(result["route"]["use_tools"])
        self.assertEqual(len(tool_answer.calls), 1)

    def test_rag_returns_reduced_sources(self) -> None:
        result = run_teacher_agent(
            "欧姆定律公式是什么？",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(analysis_json()),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
        )

        self.assertEqual(
            result["sources"],
            [
                {
                    "id": "KB-ELEC-001",
                    "topic": "欧姆定律公式",
                    "source": "物理课后总结.docx",
                    "score": 8.5,
                }
            ],
        )

    def test_rag_off_never_calls_retriever(self) -> None:
        retriever = FakeRetriever(RAG_CARDS)

        run_teacher_agent(
            "测试问题",
            rag_policy="off",
            analyzer_func=FakeAnalyzer(analysis_json(needs_rag=True)),
            retriever=retriever,
            answer_func=FakeAnswer(),
        )

        self.assertEqual(retriever.calls, [])

    def test_image_required_calls_neither_retriever_nor_answer(self) -> None:
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer()
        tool_answer = FakeToolAnswer()

        run_teacher_agent(
            "请看图回答。",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(
                analysis_json(image_required=True)
            ),
            retriever=retriever,
            answer_func=answer,
            tool_answer_func=tool_answer,
        )

        self.assertEqual(retriever.calls, [])
        self.assertEqual(answer.calls, [])
        self.assertEqual(tool_answer.calls, [])

    def test_image_required_answer_is_route_user_message(self) -> None:
        tool_answer = FakeToolAnswer()
        result = run_teacher_agent(
            "请看图回答。",
            analyzer_func=FakeAnalyzer(
                analysis_json(image_required=True)
            ),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
            tool_answer_func=tool_answer,
        )

        self.assertFalse(result["route"]["should_answer"])
        self.assertEqual(result["answer"], result["route"]["user_message"])
        self.assertIn("题图", result["answer"])
        self.assertEqual(result["tool_records"], [])
        self.assertEqual(result["tool_model_requests"], 0)
        self.assertEqual(tool_answer.calls, [])

    def test_analyzer_fallback_is_returned(self) -> None:
        result = run_teacher_agent(
            "测试问题",
            analyzer_func=FakeAnalyzer("不是 JSON"),
            answer_func=FakeAnswer(),
        )

        self.assertTrue(result["analysis_fallback"])
        self.assertEqual(result["analysis"]["teaching_mode"], "solve")

    def test_empty_question_calls_no_dependency(self) -> None:
        analyzer = FakeAnalyzer(analysis_json())
        retriever = FakeRetriever(RAG_CARDS)
        answer = FakeAnswer()

        for question in ("", "   "):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(ValueError, "问题不能为空"):
                    run_teacher_agent(
                        question,
                        analyzer_func=analyzer,
                        retriever=retriever,
                        answer_func=answer,
                    )

        self.assertEqual(analyzer.calls, [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(answer.calls, [])

    def test_result_has_complete_top_level_fields(self) -> None:
        result = run_teacher_agent(
            "测试问题",
            analyzer_func=FakeAnalyzer(analysis_json()),
            answer_func=FakeAnswer(),
        )

        self.assertEqual(
            set(result),
            {
                "answer",
                "analysis",
                "route",
                "sources",
                "analysis_fallback",
                "tool_records",
                "tool_model_requests",
                "trace",
            },
        )
        self.assertIsInstance(result["analysis"], dict)
        self.assertIsInstance(result["route"], dict)
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["analysis_fallback"], bool)
        self.assertIsInstance(result["tool_records"], list)
        self.assertIsInstance(result["tool_model_requests"], int)
        self.assertIsInstance(result["trace"], dict)


if __name__ == "__main__":
    unittest.main()
