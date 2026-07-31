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

        result = run_teacher_agent(
            "解释欧姆定律。",
            analyzer_func=analyzer,
            retriever=retriever,
            answer_func=answer,
        )

        self.assertEqual(result["answer"], "普通回答")
        self.assertEqual(result["sources"], [])
        self.assertEqual(retriever.calls, [])
        self.assertEqual(len(answer.calls), 1)

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

        run_teacher_agent(
            "欧姆定律公式是什么？",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(analysis_json()),
            retriever=retriever,
            answer_func=answer,
        )

        self.assertEqual(
            retriever.calls,
            [("欧姆定律公式是什么？", 3)],
        )
        self.assertIn("KB-ELEC-001", answer.calls[0]["context"])
        self.assertIn("I=U/R", answer.calls[0]["context"])

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

        run_teacher_agent(
            "请看图回答。",
            rag_policy="force",
            analyzer_func=FakeAnalyzer(
                analysis_json(image_required=True)
            ),
            retriever=retriever,
            answer_func=answer,
        )

        self.assertEqual(retriever.calls, [])
        self.assertEqual(answer.calls, [])

    def test_image_required_answer_is_route_user_message(self) -> None:
        result = run_teacher_agent(
            "请看图回答。",
            analyzer_func=FakeAnalyzer(
                analysis_json(image_required=True)
            ),
            retriever=FakeRetriever(RAG_CARDS),
            answer_func=FakeAnswer(),
        )

        self.assertFalse(result["route"]["should_answer"])
        self.assertEqual(result["answer"], result["route"]["user_message"])
        self.assertIn("题图", result["answer"])

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
            },
        )
        self.assertIsInstance(result["analysis"], dict)
        self.assertIsInstance(result["route"], dict)
        self.assertIsInstance(result["sources"], list)
        self.assertIsInstance(result["analysis_fallback"], bool)


if __name__ == "__main__":
    unittest.main()
