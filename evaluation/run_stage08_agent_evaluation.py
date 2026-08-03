"""运行 Stage 08 统一 Agent 评测，并逐题保存 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluation.validate_stage08_agent_cases import (
    CASES_PATH,
    load_cases,
    validate_cases,
)
from src.agent import run_teacher_agent
from src.schemas import StepStatus, StepTrace


DEFAULT_RESULTS_DIR = Path(__file__).with_name("results")
CaseRunner = Callable[[dict[str, object]], dict[str, Any]]


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return number


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 Stage 08 Agent 评测集 V1。")
    parser.add_argument("--mode", choices=("fake", "real"), default="fake")
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--case-id")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()

    ids: set[str] = set()
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"结果文件第 {line_number} 行不是有效 JSON。"
                ) from error
            case_id = record.get("case_id")
            if isinstance(case_id, str):
                ids.add(case_id)
    return ids


class _FakeRetriever:
    def search(self, question: str, top_k: int = 3) -> list[dict[str, object]]:
        return [
            {
                "id": "KB-FAKE-001",
                "chapter": "本地测试",
                "topic": "确定性 Fake 资料",
                "keywords": ["评测"],
                "source": "stage08_fake_source",
                "content": "这是用于验证 RAG 编排的确定性本地资料。",
                "score": 1.0,
            }
        ]


def _fake_tool_result(case: dict[str, object]) -> dict[str, Any]:
    traces = [
        StepTrace(
            name="tool_selection",
            status=StepStatus.SUCCESS,
            attempts=1,
            duration_ms=0,
            model_requests=1,
        ),
        StepTrace(
            name="tool_execution",
            status=StepStatus.SUCCESS,
            attempts=1,
            duration_ms=0,
            model_requests=0,
            metadata={"tool_record_count": 1},
        ),
        StepTrace(
            name="tool_result_answer",
            status=StepStatus.SUCCESS,
            attempts=1,
            duration_ms=0,
            model_requests=1,
        ),
    ]
    return {
        "answer": f"Fake 工具回答：{'、'.join(case['answer_keywords'])}",
        "tool_records": [
            {
                "tool_call_id": f"fake-{case['id']}",
                "name": "calculate_ohms_law",
                "arguments": {"voltage_v": 12, "resistance_ohm": 6},
                "normalized_fields": [],
                "status": "success",
                "result": {
                    "formula": "I = U / R",
                    "raw_result": "2",
                    "display_value": "2",
                    "unit": "A",
                },
                "error": None,
            }
        ],
        "model_requests": 2,
        "step_traces": [trace.model_dump(mode="json") for trace in traces],
    }


def run_fake_case(case: dict[str, object]) -> dict[str, Any]:
    """通过依赖注入运行正式 Agent 编排，但不调用任何外部 API。"""
    is_blocked = case["expected_run_status"] == "blocked"
    analysis_payload = {
        "teaching_mode": case["expected_teaching_mode"],
        "physics_topic": "Stage 08 Fake",
        "question_type": case["category"],
        "needs_rag": case["expected_use_rag"],
        "missing_conditions": False,
        "image_required": is_blocked,
        "student_work_provided": case["expected_teaching_mode"] == "diagnose",
        "short_reason": "确定性本地评测分析。",
        "calculation_required": case["expected_use_tools"],
    }

    def fake_analyzer(question: str) -> str:
        return json.dumps(analysis_payload, ensure_ascii=False)

    def fake_answer(
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
    ) -> str:
        return f"Fake 教师回答：{'、'.join(case['answer_keywords'])}"

    def fake_tool_answer(
        question: str,
        context: str | None = None,
        mode_instruction: str | None = None,
    ) -> dict[str, Any]:
        return _fake_tool_result(case)

    return run_teacher_agent(
        str(case["question"]),
        analyzer_func=fake_analyzer,
        retriever=_FakeRetriever(),
        answer_func=fake_answer,
        tool_answer_func=fake_tool_answer,
        case_id=str(case["id"]),
    )


def run_real_case(case: dict[str, object]) -> dict[str, Any]:
    return run_teacher_agent(
        str(case["question"]),
        case_id=str(case["id"]),
    )


def expectation_checks(
    case: dict[str, object],
    result: dict[str, Any],
) -> dict[str, bool]:
    route = result["route"]
    trace = result["trace"]
    checks = {
        "teaching_mode": (
            route["teaching_mode"] == case["expected_teaching_mode"]
        ),
        "use_rag": route["use_rag"] is case["expected_use_rag"],
        "use_tools": route["use_tools"] is case["expected_use_tools"],
        "run_status": trace["status"] == case["expected_run_status"],
    }
    checks["all_matched"] = all(checks.values())
    return checks


def _failure_record(case: dict[str, object], mode: str, error: Exception) -> dict:
    return {
        "case_id": case["id"],
        "question": case["question"],
        "mode": mode,
        "status": "failed",
        "answer": "",
        "analysis": {},
        "route": {},
        "sources": [],
        "tool_records": [],
        "trace": None,
        "expectation_checks": {
            "teaching_mode": False,
            "use_rag": False,
            "use_tools": False,
            "run_status": False,
            "all_matched": False,
        },
        "error": f"{type(error).__name__}：单题执行失败。",
    }


def run_evaluation(
    *,
    cases_path: Path = CASES_PATH,
    output_path: Path,
    mode: str = "fake",
    limit: int | None = None,
    case_id: str | None = None,
    case_runner: CaseRunner | None = None,
) -> int:
    cases = load_cases(cases_path)
    validate_cases(cases)
    if mode not in {"fake", "real"}:
        raise ValueError("mode 只能是 fake 或 real。")

    selected = cases
    if case_id is not None:
        selected = [case for case in selected if case["id"] == case_id]
        if not selected:
            raise ValueError(f"未找到评测题：{case_id}")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit 必须为正整数。")
        selected = selected[:limit]

    runner = case_runner
    if runner is None:
        runner = run_fake_case if mode == "fake" else run_real_case

    finished = completed_case_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0
    with output_path.open("a", encoding="utf-8") as output:
        for case in selected:
            current_id = str(case["id"])
            if current_id in finished:
                print(f"跳过已完成题目：{current_id}")
                continue

            try:
                result = runner(case)
                record = {
                    "case_id": current_id,
                    "question": case["question"],
                    "mode": mode,
                    "status": result["trace"]["status"],
                    "answer": result["answer"],
                    "analysis": result["analysis"],
                    "route": result["route"],
                    "sources": result["sources"],
                    "tool_records": result["tool_records"],
                    "trace": result["trace"],
                    "expectation_checks": expectation_checks(case, result),
                    "error": None,
                }
            except Exception as error:
                record = _failure_record(case, mode, error)

            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            finished.add(current_id)
            new_count += 1
            print(f"完成：{current_id}，status={record['status']}")

    print(f"本次新增 {new_count} 条结果；结果文件：{output_path}")
    return new_count


def main() -> None:
    args = parse_args()
    output = args.output
    if output is None:
        output = DEFAULT_RESULTS_DIR / f"stage08_agent_v1_{args.mode}_results.jsonl"
    run_evaluation(
        output_path=output,
        mode=args.mode,
        limit=args.limit,
        case_id=args.case_id,
    )


if __name__ == "__main__":
    main()
