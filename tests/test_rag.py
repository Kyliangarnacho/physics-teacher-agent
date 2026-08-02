"""Stage 05 最小 RAG 编排层单元测试。"""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.rag import answer_with_rag, build_context, retrieve_rag_context


CARDS = [
    {
        "id": "KB-ELEC-002",
        "chapter": "欧姆定律与动态电路",
        "topic": "并联动态电路中的电表示数与功率",
        "keywords": ["并联电路", "电流表"],
        "source": "12.课后总结（物理）.docx",
        "content": "并联支路两端电压不变，应先判断电流表测量对象。",
        "score": 9.5,
    },
    {
        "id": "KB-ELEC-001",
        "chapter": "欧姆定律与动态电路",
        "topic": "开关状态变化时先画等效电路",
        "keywords": ["动态电路", "等效电路"],
        "source": "11.课后总结（物理）.docx",
        "content": "开关状态改变时，应分别画出对应的等效电路。",
        "score": 5.2,
    },
]


class FakeRetriever:
    def __init__(self, results: list[dict[str, object]]) -> None:
        self.results = results
        self.calls: list[tuple[str, int]] = []

    def search(
        self,
        question: str,
        top_k: int = 3,
    ) -> list[dict[str, object]]:
        self.calls.append((question, top_k))
        return self.results


class BuildContextTests(unittest.TestCase):
    def test_context_contains_id_topic_source_and_content(self) -> None:
        context = build_context(CARDS)

        for card in CARDS:
            self.assertIn(card["id"], context)
            self.assertIn(card["topic"], context)
            self.assertIn(card["source"], context)
            self.assertIn(card["content"], context)


class RetrieveRagContextTests(unittest.TestCase):
    def test_returns_context_and_reduced_sources(self) -> None:
        result = retrieve_rag_context(
            "并联电路怎样变化？",
            retriever=FakeRetriever(CARDS),
        )

        self.assertEqual(result["context"], build_context(CARDS))
        self.assertEqual(set(result), {"context", "sources"})
        for source in result["sources"]:
            self.assertEqual(set(source), {"id", "topic", "source", "score"})
            self.assertNotIn("content", source)

    def test_sources_keep_retrieval_order(self) -> None:
        result = retrieve_rag_context(
            "测试问题",
            retriever=FakeRetriever(CARDS),
        )

        self.assertEqual(
            [source["id"] for source in result["sources"]],
            ["KB-ELEC-002", "KB-ELEC-001"],
        )

    def test_top_k_is_forwarded_to_retriever(self) -> None:
        retriever = FakeRetriever(CARDS)

        retrieve_rag_context("测试问题", top_k=2, retriever=retriever)

        self.assertEqual(retriever.calls, [("测试问题", 2)])

    def test_no_results_returns_none_context_and_empty_sources(self) -> None:
        result = retrieve_rag_context(
            "无相关资料的问题",
            retriever=FakeRetriever([]),
        )

        self.assertEqual(result, {"context": None, "sources": []})

    def test_empty_question_is_rejected_before_retriever(self) -> None:
        retriever = Mock()

        for question in ("", " ", "\t\r\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(ValueError, "问题不能为空"):
                    retrieve_rag_context(question, retriever=retriever)

        retriever.search.assert_not_called()


class AnswerWithRagTests(unittest.TestCase):
    def test_answer_func_receives_original_question_and_context(self) -> None:
        question = "并联支路变化时，电流表怎样变化？"
        retriever = FakeRetriever(CARDS)
        answer_func = Mock(return_value="模型回答")

        answer_with_rag(
            question,
            retriever=retriever,
            answer_func=answer_func,
        )

        answer_func.assert_called_once_with(
            question,
            context=build_context(CARDS),
        )

    def test_return_value_contains_answer_and_reduced_sources(self) -> None:
        result = answer_with_rag(
            "测试问题",
            retriever=FakeRetriever(CARDS),
            answer_func=Mock(return_value="测试回答"),
        )

        self.assertEqual(result["answer"], "测试回答")
        self.assertEqual(set(result), {"answer", "sources"})
        for source in result["sources"]:
            self.assertEqual(
                set(source),
                {"id", "topic", "source", "score"},
            )
            self.assertNotIn("content", source)

    def test_sources_keep_retrieval_order(self) -> None:
        result = answer_with_rag(
            "测试问题",
            retriever=FakeRetriever(CARDS),
            answer_func=Mock(return_value="测试回答"),
        )

        self.assertEqual(
            [source["id"] for source in result["sources"]],
            ["KB-ELEC-002", "KB-ELEC-001"],
        )

    def test_no_results_uses_plain_answer_and_empty_sources(self) -> None:
        question = "完全无关的问题"
        answer_func = Mock(return_value="普通回答")

        result = answer_with_rag(
            question,
            retriever=FakeRetriever([]),
            answer_func=answer_func,
        )

        answer_func.assert_called_once_with(question, context=None)
        self.assertEqual(
            result,
            {"answer": "普通回答", "sources": []},
        )

    def test_top_k_is_forwarded_to_retriever(self) -> None:
        retriever = FakeRetriever(CARDS)

        answer_with_rag(
            "测试问题",
            top_k=2,
            retriever=retriever,
            answer_func=Mock(return_value="测试回答"),
        )

        self.assertEqual(retriever.calls, [("测试问题", 2)])

    def test_retrieval_is_executed_only_once(self) -> None:
        retriever = FakeRetriever(CARDS)

        answer_with_rag(
            "测试问题",
            retriever=retriever,
            answer_func=Mock(return_value="测试回答"),
        )

        self.assertEqual(retriever.calls, [("测试问题", 3)])

    def test_empty_question_is_rejected_before_dependencies_are_used(self) -> None:
        retriever = Mock()
        answer_func = Mock()

        for question in ("", " ", "\t\r\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(ValueError, "问题不能为空"):
                    answer_with_rag(
                        question,
                        retriever=retriever,
                        answer_func=answer_func,
                    )

        retriever.search.assert_not_called()
        answer_func.assert_not_called()


if __name__ == "__main__":
    unittest.main()
