"""Tests for batch-level image relation analysis without API calls."""

from __future__ import annotations

import json
from types import SimpleNamespace
import unittest

from src.vision.batch_relation import (
    BatchRelation,
    BatchRelationError,
    analyze_batch_relation,
)


def response(payload: object):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=json.dumps(payload, ensure_ascii=False)
                )
            )
        ]
    )


class VisionBatchRelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.summaries = [
            {"index": 1, "filename": "stem.png", "extracted_context": "题干"},
            {"index": 2, "filename": "options.png", "extracted_context": "A B C D"},
        ]

    def test_same_problem_stem_options(self) -> None:
        calls = []

        def fake(**kwargs):
            calls.append(kwargs)
            return response(
                {
                    "relationship": "same_problem",
                    "image_roles": [
                        {"index": 1, "role": "stem"},
                        {"index": 2, "role": "options"},
                    ],
                    "combined_context": "题干与 A/B/C/D 选项属于同一题。",
                    "short_reason": "第二张是第一张的选项。",
                }
            )

        result = analyze_batch_relation(
            self.summaries,
            completion_func=fake,
        )
        self.assertIs(result["relation"].relationship, BatchRelation.SAME_PROBLEM)
        self.assertEqual(result["model_requests"], 1)
        prompt = json.dumps(calls[0]["messages"], ensure_ascii=False)
        self.assertIn("不解题", prompt)
        self.assertIn("same_problem", prompt)

    def test_invalid_or_incomplete_relation_is_rejected(self) -> None:
        with self.assertRaises(BatchRelationError):
            analyze_batch_relation(
                self.summaries,
                completion_func=lambda **_kwargs: response(
                    {
                        "relationship": "same_problem",
                        "image_roles": [{"index": 1, "role": "stem"}],
                        "combined_context": "不完整",
                        "short_reason": "少一张标注。",
                    }
                ),
            )


if __name__ == "__main__":
    unittest.main()
