"""Stage 04 评测工具的纯本地单元测试。"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.build_stage04_review_sheet import (
    EXPECTED_PROMPT_VERSION,
    build_review_sheet,
)
from evaluation.run_stage04_text_evaluation import run_evaluation
from evaluation.validate_stage04_text_cases import (
    CASES_PATH,
    REQUIRED_FIELDS,
    load_cases,
    validate_cases,
)


class Stage04CaseValidationTests(unittest.TestCase):
    def test_all_15_cases_have_complete_fields_and_unique_ids(self) -> None:
        cases = load_cases(CASES_PATH)

        validate_cases(cases)

        self.assertEqual(len(cases), 15)
        self.assertEqual(len({case["id"] for case in cases}), 15)
        for case in cases:
            self.assertEqual(set(case), REQUIRED_FIELDS)


class Stage04RunnerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary_directory.cleanup)
        self.temp_path = Path(self.temporary_directory.name)

    def run_locally(
        self,
        output_path: Path,
        *,
        limit: int = 1,
        answer: str = "本地测试回答",
        prompt_version: str = "teacher_v3_personal_humor",
    ) -> int:
        return run_evaluation(
            cases_path=CASES_PATH,
            output_path=output_path,
            limit=limit,
            answer_fn=lambda _question: answer,
            model="local-test-model",
            prompt_version=prompt_version,
        )

    @staticmethod
    def read_jsonl(path: Path) -> list[dict[str, object]]:
        return [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def test_prompt_version_is_recorded_in_result(self) -> None:
        output_path = self.temp_path / "v3.jsonl"

        self.assertEqual(self.run_locally(output_path), 1)

        records = self.read_jsonl(output_path)
        self.assertEqual(records[0]["prompt_version"], "teacher_v3_personal_humor")

    def test_resume_skips_existing_id(self) -> None:
        output_path = self.temp_path / "resume.jsonl"
        first_case_id = str(load_cases(CASES_PATH)[0]["id"])
        original = json.dumps({"id": first_case_id}, ensure_ascii=False) + "\n"
        output_path.write_text(original, encoding="utf-8")

        def fail_if_called(_question: str) -> str:
            self.fail("断点续跑不应再次调用已有 ID")

        added = run_evaluation(
            cases_path=CASES_PATH,
            output_path=output_path,
            limit=1,
            answer_fn=fail_if_called,
            model="local-test-model",
            prompt_version="teacher_v3_personal_humor",
        )

        self.assertEqual(added, 0)
        self.assertEqual(output_path.read_text(encoding="utf-8"), original)

    def test_old_and_new_version_outputs_do_not_overwrite_each_other(self) -> None:
        old_output = self.temp_path / "teacher_v2.jsonl"
        new_output = self.temp_path / "teacher_v3.jsonl"

        self.run_locally(
            old_output,
            answer="旧版回答",
            prompt_version="teacher_v2_personal",
        )
        old_snapshot = old_output.read_text(encoding="utf-8")
        self.run_locally(
            new_output,
            answer="新版回答",
            prompt_version="teacher_v3_personal_humor",
        )

        self.assertNotEqual(old_output, new_output)
        self.assertEqual(old_output.read_text(encoding="utf-8"), old_snapshot)
        self.assertEqual(
            self.read_jsonl(old_output)[0]["prompt_version"],
            "teacher_v2_personal",
        )
        self.assertEqual(
            self.read_jsonl(new_output)[0]["prompt_version"],
            "teacher_v3_personal_humor",
        )


class Stage04ReviewSheetTests(unittest.TestCase):
    def test_review_sheet_merges_cases_and_results(self) -> None:
        cases = load_cases(CASES_PATH)[:5]
        results = {
            str(case["id"]): {
                "id": case["id"],
                "status": "ok",
                "answer": f"回答-{case['id']}",
                "prompt_version": EXPECTED_PROMPT_VERSION,
            }
            for case in cases
        }

        content, checks = build_review_sheet(cases, results)

        self.assertEqual(len(checks), 5)
        self.assertTrue(all(all(item.values()) for _, item in checks))
        for case in cases:
            self.assertIn(str(case["question"]), content)
            self.assertIn(f"回答-{case['id']}", content)


if __name__ == "__main__":
    unittest.main()
