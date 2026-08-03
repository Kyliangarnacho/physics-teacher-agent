"""Stage 08 Agent 评测数据、Runner 与汇总器的纯本地测试。"""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_stage08_agent_evaluation import (
    run_evaluation,
    run_fake_case,
)
from evaluation.summarize_stage08_results import summarize_records
from evaluation.validate_stage08_agent_cases import (
    CASES_PATH,
    REQUIRED_FIELDS,
    load_cases,
    validate_cases,
)


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


class Stage08CaseValidationTests(unittest.TestCase):
    def test_eight_cases_are_valid_unique_and_cover_required_categories(self) -> None:
        cases = load_cases(CASES_PATH)

        validate_cases(cases)

        self.assertEqual(len(cases), 8)
        self.assertEqual(len({case["id"] for case in cases}), 8)
        self.assertTrue(all(set(case) == REQUIRED_FIELDS for case in cases))
        self.assertEqual(
            {case["category"] for case in cases},
            {
                "普通概念回答",
                "普通完整解题",
                "RAG",
                "Tool",
                "RAG + Tool",
                "hint",
                "diagnose",
                "缺图 blocked",
            },
        )

    def test_duplicate_id_is_rejected(self) -> None:
        cases = copy.deepcopy(load_cases(CASES_PATH))
        cases[1]["id"] = cases[0]["id"]

        with self.assertRaisesRegex(ValueError, "重复 id"):
            validate_cases(cases)


class Stage08RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp_path = Path(self.temporary_directory.name)

    def test_limit_and_case_id_select_expected_cases(self) -> None:
        limited_output = self.temp_path / "limited.jsonl"
        selected_output = self.temp_path / "selected.jsonl"

        added = run_evaluation(
            output_path=limited_output,
            mode="fake",
            limit=2,
        )
        selected_added = run_evaluation(
            output_path=selected_output,
            mode="fake",
            case_id="S08-AGENT-005",
        )

        self.assertEqual(added, 2)
        self.assertEqual(
            [record["case_id"] for record in read_jsonl(limited_output)],
            ["S08-AGENT-001", "S08-AGENT-002"],
        )
        self.assertEqual(selected_added, 1)
        self.assertEqual(
            read_jsonl(selected_output)[0]["case_id"],
            "S08-AGENT-005",
        )

    def test_resume_skips_existing_case_id(self) -> None:
        output = self.temp_path / "resume.jsonl"
        original = json.dumps(
            {"case_id": "S08-AGENT-001"},
            ensure_ascii=False,
        ) + "\n"
        output.write_text(original, encoding="utf-8")

        def fail_if_called(case: dict[str, object]) -> dict:
            self.fail("断点续跑不应再次执行已有 case_id")

        added = run_evaluation(
            output_path=output,
            mode="fake",
            limit=1,
            case_runner=fail_if_called,
        )

        self.assertEqual(added, 0)
        self.assertEqual(output.read_text(encoding="utf-8"), original)

    def test_jsonl_contains_required_result_fields_and_complete_trace(self) -> None:
        output = self.temp_path / "fields.jsonl"

        run_evaluation(output_path=output, mode="fake", limit=1)

        record = read_jsonl(output)[0]
        self.assertTrue(
            {
                "case_id",
                "question",
                "answer",
                "analysis",
                "route",
                "sources",
                "tool_records",
                "trace",
                "expectation_checks",
            }.issubset(record)
        )
        self.assertEqual(record["mode"], "fake")
        self.assertTrue(record["expectation_checks"]["all_matched"])
        self.assertIn("run_id", record["trace"])
        self.assertIsInstance(record["trace"]["steps"], list)
        self.assertIsInstance(json.dumps(record["trace"], ensure_ascii=False), str)

    def test_single_case_failure_is_written_and_next_case_continues(self) -> None:
        output = self.temp_path / "continue.jsonl"
        calls = 0

        def fail_once(case: dict[str, object]) -> dict:
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("private detail")
            return run_fake_case(case)

        added = run_evaluation(
            output_path=output,
            mode="fake",
            limit=2,
            case_runner=fail_once,
        )

        records = read_jsonl(output)
        self.assertEqual(added, 2)
        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["status"], "failed")
        self.assertIsNone(records[0]["trace"])
        self.assertNotIn("private detail", records[0]["error"])
        self.assertEqual(records[1]["status"], "completed")


class Stage08SummaryTests(unittest.TestCase):
    def test_summary_counts_routes_requests_retries_and_errors(self) -> None:
        records = [
            {
                "status": "completed",
                "expectation_checks": {"all_matched": True},
                "trace": {
                    "total_model_requests": 2,
                    "analysis_fallback": False,
                    "rag_searches": 1,
                    "tool_executions": 0,
                    "steps": [
                        {"status": "retry_success", "error_type": None}
                    ],
                },
            },
            {
                "status": "blocked",
                "expectation_checks": {"all_matched": False},
                "trace": {
                    "total_model_requests": 1,
                    "analysis_fallback": True,
                    "rag_searches": 0,
                    "tool_executions": 0,
                    "steps": [
                        {"status": "error", "error_type": "analyzer_parse"}
                    ],
                },
            },
            {
                "status": "failed",
                "expectation_checks": {"all_matched": False},
                "trace": None,
            },
        ]

        summary = summarize_records(records)

        self.assertEqual(summary["total_cases"], 3)
        self.assertEqual(summary["completed_cases"], 1)
        self.assertEqual(summary["blocked_cases"], 1)
        self.assertEqual(summary["failed_cases"], 1)
        self.assertEqual(summary["expected_route_matches"], 1)
        self.assertEqual(summary["total_model_requests"], 3)
        self.assertEqual(summary["average_model_requests"], 1.0)
        self.assertEqual(summary["retry_success_steps"], 1)
        self.assertEqual(summary["analysis_fallbacks"], 1)
        self.assertEqual(summary["rag_searches"], 1)
        self.assertEqual(summary["tool_executions"], 0)
        self.assertEqual(summary["error_types"], {"analyzer_parse": 1})


if __name__ == "__main__":
    unittest.main()
