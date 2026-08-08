from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from evaluation.run_stage11_tool_selection_evaluation import (
    CASES_PATH,
    STRATEGIES,
    load_cases,
    run_evaluation,
    run_selection_case,
    summarize_records,
    validate_cases,
    write_markdown_report,
)


def response(tool_names: list[str]):
    calls = [
        SimpleNamespace(function=SimpleNamespace(name=name))
        for name in tool_names
    ]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(tool_calls=calls))]
    )


class FakeCompletion:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        value = self.responses.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


class CaseTests(unittest.TestCase):
    def test_public_cases_are_valid_and_cover_required_shapes(self):
        cases = load_cases(CASES_PATH)
        validate_cases(cases)
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["id"] for case in cases}), 24)
        self.assertTrue(any(not case["expected_tools"] for case in cases))
        self.assertTrue(any(len(case["expected_tools"]) == 1 for case in cases))
        self.assertTrue(any(len(case["expected_tools"]) > 1 for case in cases))

    def test_strategy_uses_expected_choice_and_keeps_selection_only(self):
        case = load_cases(CASES_PATH)[0]
        fake = FakeCompletion([response([])])
        record = run_selection_case(case, strategy="auto_current", completion_func=fake)
        self.assertEqual(record["selected_tools"], [])
        self.assertEqual(fake.calls[0]["tool_choice"], "auto")
        self.assertIn("tools", fake.calls[0])
        self.assertFalse(fake.calls[0]["stream"])
        self.assertIn("不要为了调用而调用", fake.calls[0]["messages"][-2]["content"])

    def test_formal_strategy_uses_the_tool_client_prompt(self):
        case = load_cases(CASES_PATH)[8]
        fake = FakeCompletion([response(["calculate_average_speed"])])
        run_selection_case(case, strategy="auto_formal", completion_func=fake)
        self.assertIn("应主动调用", fake.calls[0]["messages"][-2]["content"])


class RunnerAndSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.path = Path(self.temporary_directory.name)

    def test_resume_and_jsonl_output(self):
        output = self.path / "result.jsonl"
        fake = FakeCompletion([response([]) for _ in range(24)])
        self.assertEqual(run_evaluation(strategy="auto_current", output_path=output, completion_func=fake), 24)
        self.assertEqual(run_evaluation(strategy="auto_current", output_path=output, completion_func=fake), 0)
        rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(rows), 24)
        self.assertTrue(all(row["model_requests"] == 1 for row in rows))

    def test_metrics_detect_precision_recall_and_failure_types(self):
        records = [
            {"strategy":"x","expected_tools":["calculate_ohms_law"],"allow_no_tool":False,"selected_tools":["calculate_ohms_law"],"status":"ok","model_requests":1},
            {"strategy":"x","expected_tools":["calculate_average_speed","calculate_density"],"allow_no_tool":False,"selected_tools":["calculate_average_speed"],"status":"ok","model_requests":1},
            {"strategy":"x","expected_tools":[],"allow_no_tool":True,"selected_tools":["calculate_density"],"status":"ok","model_requests":1},
        ]
        summary = summarize_records(records)["x"]
        self.assertAlmostEqual(summary["tool_precision"], 2 / 3)
        self.assertAlmostEqual(summary["tool_recall"], 2 / 3)
        self.assertEqual(summary["obvious_wrong_tool_calls"], 1)
        self.assertEqual(summary["failure_types"]["multi_tool_missing"], 1)
        self.assertEqual(summary["failure_types"]["should_not_call_but_did"], 1)

    def test_markdown_report_is_written(self):
        report = self.path / "report.md"
        write_markdown_report(
            [{"strategy":"x","expected_tools":[],"allow_no_tool":True,"selected_tools":[],"status":"ok","model_requests":1}],
            report,
        )
        text = report.read_text(encoding="utf-8")
        self.assertIn("Precision", text)
        self.assertIn("No-tool 正确率", text)
        self.assertIn("推荐", text)


if __name__ == "__main__":
    unittest.main()
