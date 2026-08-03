"""汇总 Stage 08 Agent JSONL 评测结果。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def load_results(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
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
            if not isinstance(record, dict):
                raise ValueError(f"结果文件第 {line_number} 行必须是 JSON 对象。")
            records.append(record)
    return records


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    completed_statuses = {"completed", "completed_with_fallback"}
    total_requests = 0
    retry_success_steps = 0
    fallback_count = 0
    rag_searches = 0
    tool_executions = 0
    error_types: Counter[str] = Counter()

    for record in records:
        trace = record.get("trace")
        if not isinstance(trace, dict):
            continue
        total_requests += int(trace.get("total_model_requests", 0))
        fallback_count += int(bool(trace.get("analysis_fallback", False)))
        rag_searches += int(trace.get("rag_searches", 0))
        tool_executions += int(trace.get("tool_executions", 0))
        steps = trace.get("steps", [])
        if not isinstance(steps, list):
            continue
        for step in steps:
            if not isinstance(step, dict):
                continue
            retry_success_steps += int(step.get("status") == "retry_success")
            error_type = step.get("error_type")
            if isinstance(error_type, str) and error_type:
                error_types[error_type] += 1

    total = len(records)
    completed = sum(record.get("status") in completed_statuses for record in records)
    blocked = sum(record.get("status") == "blocked" for record in records)
    failed = sum(record.get("status") == "failed" for record in records)
    route_matches = sum(
        isinstance(record.get("expectation_checks"), dict)
        and bool(record["expectation_checks"].get("all_matched"))
        for record in records
    )
    average_requests = total_requests / total if total else 0.0

    return {
        "total_cases": total,
        "completed_cases": completed,
        "blocked_cases": blocked,
        "failed_cases": failed,
        "expected_route_matches": route_matches,
        "total_model_requests": total_requests,
        "average_model_requests": round(average_requests, 3),
        "retry_success_steps": retry_success_steps,
        "analysis_fallbacks": fallback_count,
        "rag_searches": rag_searches,
        "tool_executions": tool_executions,
        "error_types": dict(sorted(error_types.items())),
    }


def format_terminal(summary: dict[str, object]) -> str:
    errors = summary["error_types"]
    error_text = (
        "，".join(f"{name}={count}" for name, count in errors.items())
        if errors
        else "无"
    )
    lines = [
        "Stage 08 Agent 评测汇总",
        f"总题数：{summary['total_cases']}",
        f"完成数：{summary['completed_cases']}",
        f"blocked 数：{summary['blocked_cases']}",
        f"失败数：{summary['failed_cases']}",
        f"预期路由匹配数：{summary['expected_route_matches']}",
        f"模型请求总数：{summary['total_model_requests']}",
        f"平均模型请求数：{summary['average_model_requests']}",
        f"Retry 成功步骤数：{summary['retry_success_steps']}",
        f"Analyzer fallback 次数：{summary['analysis_fallbacks']}",
        f"RAG 检索总数：{summary['rag_searches']}",
        f"工具执行总数：{summary['tool_executions']}",
        f"错误类型：{error_text}",
    ]
    return "\n".join(lines)


def format_markdown(summary: dict[str, object]) -> str:
    errors = summary["error_types"]
    error_text = (
        "、".join(f"`{name}`：{count}" for name, count in errors.items())
        if errors
        else "无"
    )
    return "\n".join(
        [
            "# Stage 08 Agent 评测汇总",
            "",
            f"- 总题数：{summary['total_cases']}",
            f"- 完成数：{summary['completed_cases']}",
            f"- blocked 数：{summary['blocked_cases']}",
            f"- 失败数：{summary['failed_cases']}",
            f"- 预期路由匹配数：{summary['expected_route_matches']}",
            f"- 模型请求总数：{summary['total_model_requests']}",
            f"- 平均模型请求数：{summary['average_model_requests']}",
            f"- Retry 成功步骤数：{summary['retry_success_steps']}",
            f"- Analyzer fallback 次数：{summary['analysis_fallbacks']}",
            f"- RAG 检索总数：{summary['rag_searches']}",
            f"- 工具执行总数：{summary['tool_executions']}",
            f"- 错误类型：{error_text}",
            "",
        ]
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="汇总 Stage 08 Agent 评测结果。")
    parser.add_argument("input", type=Path)
    parser.add_argument("--markdown", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_records(load_results(args.input))
    print(format_terminal(summary))
    if args.markdown is not None:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(format_markdown(summary), encoding="utf-8")
        print(f"Markdown 汇总已写入：{args.markdown}")


if __name__ == "__main__":
    main()
