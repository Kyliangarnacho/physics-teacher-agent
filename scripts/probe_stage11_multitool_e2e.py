"""Run one real Stage 11.1 multi-tool Agent diagnostic without storing a prompt.

Pass a question through ``--question``, ``--question-file``, or standard input.
The script intentionally prints only safe summaries of tool records and traces;
it never reads or prints API credentials, prompts, or image data.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

# Allow direct ``python scripts/...`` execution from the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.agent import run_teacher_agent
from src.config import load_qwen_config

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")


def _read_question(args: argparse.Namespace) -> str:
    """Read exactly one question source without embedding private material."""
    supplied = sum(
        value is not None for value in (args.question, args.question_file)
    )
    if supplied > 1:
        raise ValueError("--question 与 --question-file 只能选择一个。")
    if args.question is not None:
        question = args.question
    elif args.question_file is not None:
        question = Path(args.question_file).read_text(encoding="utf-8")
    else:
        question = sys.stdin.read()
    if not question or not question.strip():
        raise ValueError("请通过参数或标准输入提供非空题目。")
    return question.strip()


def _safe_tool_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep only execution facts useful for diagnostics."""
    summaries: list[dict[str, Any]] = []
    for record in records:
        result = record.get("result")
        core_result = None
        if isinstance(result, dict):
            core_result = {
                key: result.get(key)
                for key in ("formula", "display_value", "unit")
                if key in result
            }
        summaries.append(
            {
                "tool_call_id": record.get("tool_call_id"),
                "name": record.get("name"),
                "status": record.get("status"),
                "normalized_fields": record.get("normalized_fields", []),
                "result": core_result,
                "error": record.get("error"),
            }
        )
    return summaries


def _safe_steps(trace: dict[str, Any]) -> list[dict[str, Any]]:
    """Return trace timing/counts without recording question contents."""
    fields = (
        "name",
        "status",
        "attempts",
        "duration_ms",
        "model_requests",
        "error_type",
        "error_message",
        "metadata",
    )
    return [
        {field: step.get(field) for field in fields if field in step}
        for step in trace.get("steps", [])
        if isinstance(step, dict)
    ]


def build_report(result: dict[str, Any], model: str) -> dict[str, Any]:
    """Build a JSON-serializable, non-sensitive report from Agent output."""
    analysis = result.get("analysis", {})
    route = result.get("route", {})
    trace = result.get("trace", {})
    records = result.get("tool_records", [])
    unique_call_ids = {
        record.get("tool_call_id")
        for record in records
        if isinstance(record, dict) and record.get("tool_call_id")
    }
    tool_steps = [
        step
        for step in trace.get("steps", [])
        if isinstance(step, dict) and step.get("name") == "tool_repair"
    ]
    final_steps = [
        step
        for step in trace.get("steps", [])
        if isinstance(step, dict) and step.get("name") == "tool_result_answer"
    ]
    model_only_fallback = any(
        isinstance(step, dict)
        and isinstance(step.get("metadata"), dict)
        and step["metadata"].get("fallback") == "model_only"
        for step in final_steps
    )
    return {
        "model": model,
        "analysis": {
            key: analysis.get(key)
            for key in (
                "teaching_mode",
                "physics_topic",
                "question_type",
                "calculation_required",
                "missing_conditions",
                "needs_rag",
                "image_required",
            )
        },
        "route": route,
        "analysis_fallback": result.get("analysis_fallback"),
        "tool_call_count": len(unique_call_ids),
        "tool_call_names": [
            record.get("name")
            for record in records
            if isinstance(record, dict)
            and record.get("tool_call_id") in unique_call_ids
        ],
        "tool_records": _safe_tool_records(
            record for record in records if isinstance(record, dict)
        ),
        "repair_attempted": bool(tool_steps),
        "repair": [step.get("metadata", {}) for step in tool_steps],
        "model_only_fallback": model_only_fallback,
        "tool_model_requests": result.get("tool_model_requests"),
        "trace_summary": {
            key: trace.get(key)
            for key in (
                "status",
                "total_model_requests",
                "tool_executions",
                "rag_searches",
            )
        },
        "steps": _safe_steps(trace),
        "final_answer": result.get("answer"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="运行一题真实 Stage 11.1 多工具 Agent 诊断。"
    )
    parser.add_argument("--question", help="直接提供题目文本")
    parser.add_argument("--question-file", help="从本地 UTF-8 文本文件读取题目")
    parser.add_argument("--mode", default="solve")
    parser.add_argument("--rag-policy", default="off")
    parser.add_argument("--case-id", default=None)
    parser.add_argument(
        "--output",
        help="可选：将同一份安全诊断 JSON 写入本地文件。",
    )
    args = parser.parse_args()

    try:
        question = _read_question(args)
        _, _, model = load_qwen_config()
        result = run_teacher_agent(
            question,
            mode_override=args.mode,
            rag_policy=args.rag_policy,
            case_id=args.case_id,
        )
    except Exception as error:
        print(
            json.dumps(
                {
                    "status": "error",
                    "error_type": type(error).__name__,
                    "message": "Agent 诊断调用失败，请检查本地配置或网络。",
                },
                ensure_ascii=False,
            )
        )
        return 2

    report = build_report(result, model)
    serialized = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(serialized + "\n", encoding="utf-8")
    print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
