"""Stage 05 BM25 知识检索器单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.retriever import KnowledgeRetriever, load_knowledge_base


EXPECTED_CARD_FIELDS = {
    "id",
    "chapter",
    "topic",
    "keywords",
    "source",
    "content",
}


class KnowledgeRetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.retriever = KnowledgeRetriever()

    def test_default_knowledge_base_loads_10_cards(self) -> None:
        self.assertEqual(len(load_knowledge_base()), 10)
        self.assertEqual(len(self.retriever.cards), 10)

    def test_parallel_branch_query_ranks_elec_002_first(self) -> None:
        results = self.retriever.search(
            "并联电路中一条支路电阻变大，电流表怎样变化？"
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "KB-ELEC-002")

    def test_convex_lens_query_ranks_opt_002_first(self) -> None:
        results = self.retriever.search(
            "凸透镜成实像时，光屏应该放在哪里？"
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "KB-OPT-002")

    def test_heater_gear_query_ranks_power_001_first(self) -> None:
        results = self.retriever.search(
            "电热器的高温档为什么对应的总电阻更小？"
        )

        self.assertTrue(results)
        self.assertEqual(results[0]["id"], "KB-POWER-001")
        self.assertNotIn(
            "KB-EXP-002",
            [result["id"] for result in results],
        )

    def test_results_are_descending_and_keep_card_fields(self) -> None:
        results = self.retriever.search(
            "并联电路中一条支路电阻变大，电流表怎样变化？"
        )

        scores = [result["score"] for result in results]
        self.assertEqual(scores, sorted(scores, reverse=True))
        for result in results:
            self.assertEqual(set(result), EXPECTED_CARD_FIELDS | {"score"})
            self.assertIsInstance(result["score"], float)
            self.assertGreater(result["score"], 0)

    def test_results_meet_default_relative_score_threshold(self) -> None:
        queries = (
            "电热器的高温档为什么对应的总电阻更小？",
            "凸透镜成实像时，光屏应该放在哪里？",
            "并联电路中一条支路电阻变大，电流表怎样变化？",
        )

        for query in queries:
            with self.subTest(query=query):
                results = self.retriever.search(query)
                self.assertTrue(results)
                threshold = results[0]["score"] * 0.3
                self.assertTrue(
                    all(result["score"] >= threshold for result in results)
                )

    def test_unrelated_question_returns_empty_list(self) -> None:
        self.assertEqual(
            self.retriever.search("企鹅在南极怎样孵蛋？"),
            [],
        )

    def test_empty_questions_raise_value_error(self) -> None:
        for question in ("", "   ", "\t\r\n"):
            with self.subTest(question=repr(question)):
                with self.assertRaisesRegex(ValueError, "不能为空"):
                    self.retriever.search(question)

    def test_invalid_top_k_values_raise_value_error(self) -> None:
        invalid_values = (0, -1, 1.5, True, 11)
        for top_k in invalid_values:
            with self.subTest(top_k=top_k):
                with self.assertRaises(ValueError):
                    self.retriever.search("凸透镜", top_k=top_k)  # type: ignore[arg-type]

    def test_invalid_min_score_ratio_values_raise_value_error(self) -> None:
        invalid_values = (-0.1, 1.1, "0.3", None, True, float("nan"))
        for min_score_ratio in invalid_values:
            with self.subTest(min_score_ratio=min_score_ratio):
                with self.assertRaisesRegex(ValueError, "min_score_ratio"):
                    self.retriever.search(
                        "凸透镜",
                        min_score_ratio=min_score_ratio,  # type: ignore[arg-type]
                    )

    def test_malformed_temporary_jsonl_is_rejected_without_pollution(self) -> None:
        valid_card = {
            "id": "TEMP-001",
            "chapter": "临时章节",
            "topic": "临时主题",
            "keywords": ["临时"],
            "source": "临时来源",
            "content": "临时内容",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_path = Path(temporary_directory) / "invalid.jsonl"
            temporary_path.write_text(
                json.dumps(valid_card, ensure_ascii=False)
                + "\n"
                + "{invalid json}\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "第 2 行"):
                load_knowledge_base(temporary_path)

        self.assertEqual(len(load_knowledge_base()), 10)


if __name__ == "__main__":
    unittest.main()
