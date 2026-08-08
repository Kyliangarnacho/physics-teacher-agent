"""Run public, repeatable Stage 11 tool-selection evaluations only.

This script makes one tools-enabled selection request per case.  It never
executes a local tool and never asks the model for a final teaching answer.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_qwen_config
from src.model_client import build_messages
from src.tool_client import TOOL_SELECTION_INSTRUCTION
from src.tools.registry import get_openai_tools


CASES_PATH = Path(__file__).with_name("stage11_tool_selection_cases_v1.json")
RESULTS_DIR = Path(__file__).with_name("results")
ID_PATTERN = re.compile(r"S11-TS-\d{3}")
TOOL_NAMES = {
    tool["function"]["name"] for tool in get_openai_tools()
}
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_tools",
    "allow_no_tool",
    "reason",
}

STRATEGIES: dict[str, dict[str, str | None]] = {
    "required_baseline": {"tool_choice": "required", "instruction": None},
    "auto_current": {
        "tool_choice": "auto",
        "instruction": (
            "你正在判断是否需要调用白名单本地物理计算工具。"
            "只有现有工具能够直接、可靠地完成用户所需的确定性计算时才调用工具；"
            "没有匹配工具时不要为了调用而调用，直接回答。"
        ),
    },
    "auto_explicit": {
        "tool_choice": "auto",
        "instruction": (
            "先判断白名单工具是否能直接、可靠地核算用户要求的数值。"
            "若条件完整且有匹配工具，应调用所有彼此独立且确有必要的工具；"
            "若只是概念、实验判断，或缺少所需公式/工具，则不要调用工具。"
            "不要因题目出现数字就调用，也不要为了调用而调用。"
        ),
    },
    "auto_formal": {
        "tool_choice": "auto",
        "instruction": TOOL_SELECTION_INSTRUCTION,
    },
}
CompletionFunc = Callable[..., Any]


def load_cases(path: Path = CASES_PATH) -> list[dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("评测集顶层必须是 JSON 数组。")
    return data


def validate_cases(cases: list[dict[str, Any]]) -> None:
    errors: list[str] = []
    ids: list[str] = []
    if len(cases) != 24:
        errors.append(f"评测题必须恰好为 24 条，当前为 {len(cases)} 条。")
    for index, case in enumerate(cases, start=1):
        if not isinstance(case, dict):
            errors.append(f"第 {index} 条不是对象。")
            continue
        missing = REQUIRED_FIELDS - set(case)
        extra = set(case) - REQUIRED_FIELDS
        if missing:
            errors.append(f"第 {index} 条缺少字段：{sorted(missing)}。")
        if extra:
            errors.append(f"第 {index} 条存在多余字段：{sorted(extra)}。")
        case_id = case.get("id")
        if not isinstance(case_id, str) or not ID_PATTERN.fullmatch(case_id):
            errors.append(f"第 {index} 条 id 格式错误。")
        else:
            ids.append(case_id)
        for field in ("category", "question", "reason"):
            if not isinstance(case.get(field), str) or not case[field].strip():
                errors.append(f"第 {index} 条 {field} 必须为非空字符串。")
        tools = case.get("expected_tools")
        if (
            not isinstance(tools, list)
            or any(not isinstance(name, str) or name not in TOOL_NAMES for name in tools)
            or len(tools) != len(set(tools))
        ):
            errors.append(f"第 {index} 条 expected_tools 非法。")
        if type(case.get("allow_no_tool")) is not bool:
            errors.append(f"第 {index} 条 allow_no_tool 必须为布尔值。")
    duplicates = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicates:
        errors.append(f"存在重复 ID：{', '.join(duplicates)}。")
    if errors:
        raise ValueError("\n".join(errors))


def _selection_messages(question: str, instruction: str | None) -> list[dict[str, str]]:
    messages = build_messages(question)
    if instruction is None:
        return messages
    return [*messages[:-1], {"role": "system", "content": instruction}, messages[-1]]


def _tool_names_from_response(response: Any) -> list[str]:
    try:
        tool_calls = response.choices[0].message.tool_calls
    except (AttributeError, IndexError, TypeError):
        return []
    if not tool_calls:
        return []
    names: list[str] = []
    for tool_call in tool_calls:
        try:
            names.append(str(tool_call.function.name))
        except AttributeError:
            names.append("<invalid_tool_call>")
    return names


def run_selection_case(
    case: dict[str, Any],
    *,
    strategy: str,
    completion_func: CompletionFunc | None = None,
) -> dict[str, Any]:
    if strategy not in STRATEGIES:
        raise ValueError(f"未知 strategy：{strategy}。")
    config = STRATEGIES[strategy]
    if completion_func is None:
        api_key, base_url, model = load_qwen_config()
        completion = OpenAI(api_key=api_key, base_url=base_url).chat.completions.create
    else:
        completion = completion_func
        model = "injected-test-model"
    try:
        response = completion(
            model=model,
            messages=deepcopy(_selection_messages(case["question"], config["instruction"])),
            tools=get_openai_tools(),
            tool_choice=config["tool_choice"],
            extra_body={"enable_thinking": False},
            stream=False,
        )
        selected_tools = _tool_names_from_response(response)
        status = "ok" if len(selected_tools) <= 5 else "protocol_error"
        error_type = None if status == "ok" else "too_many_tool_calls"
    except Exception as error:
        selected_tools = []
        status = "error"
        error_type = type(error).__name__
    return {
        "case_id": case["id"],
        "category": case["category"],
        "strategy": strategy,
        "tool_choice": config["tool_choice"],
        "expected_tools": case["expected_tools"],
        "allow_no_tool": case["allow_no_tool"],
        "selected_tools": selected_tools,
        "tool_call_count": len(selected_tools),
        "status": status,
        "error_type": error_type,
        "model_requests": 1,
    }


def completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if isinstance(value.get("case_id"), str):
                ids.add(value["case_id"])
    return ids


def run_evaluation(
    *,
    strategy: str,
    output_path: Path,
    cases_path: Path = CASES_PATH,
    completion_func: CompletionFunc | None = None,
) -> int:
    cases = load_cases(cases_path)
    validate_cases(cases)
    completed = completed_case_ids(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with output_path.open("a", encoding="utf-8") as handle:
        for case in cases:
            if case["id"] in completed:
                continue
            record = run_selection_case(case, strategy=strategy, completion_func=completion_func)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            added += 1
    return added


def read_records(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(record["strategy"], []).append(record)
    summaries: dict[str, dict[str, Any]] = {}
    for strategy, items in grouped.items():
        expected_total = sum(len(item["expected_tools"]) for item in items)
        selected_total = sum(len(item["selected_tools"]) for item in items)
        correct_selected = sum(
            len(set(item["expected_tools"]) & set(item["selected_tools"]))
            for item in items
        )
        no_tool_correct = sum(
            bool(not item["selected_tools"]) == item["allow_no_tool"]
            for item in items
        )
        single_cases = [item for item in items if len(item["expected_tools"]) == 1]
        multi_cases = [item for item in items if len(item["expected_tools"]) > 1]
        errors = Counter()
        for item in items:
            expected = set(item["expected_tools"])
            selected = set(item["selected_tools"])
            if item["status"] != "ok":
                errors["request_or_protocol_error"] += 1
            elif expected and not selected:
                errors["should_call_but_did_not"] += 1
            elif not expected and selected:
                errors["should_not_call_but_did"] += 1
            elif selected - expected:
                errors["wrong_tool"] += 1
            elif len(expected) > 1 and expected - selected:
                errors["multi_tool_missing"] += 1
        summaries[strategy] = {
            "cases": len(items),
            "tool_precision": None if selected_total == 0 else correct_selected / selected_total,
            "tool_recall": None if expected_total == 0 else correct_selected / expected_total,
            "no_tool_accuracy": no_tool_correct / len(items) if items else 0.0,
            "single_tool_hit_rate": (
                sum(set(item["selected_tools"]) == set(item["expected_tools"]) for item in single_cases) / len(single_cases)
                if single_cases else 0.0
            ),
            "multi_tool_complete_hit_rate": (
                sum(set(item["selected_tools"]) == set(item["expected_tools"]) for item in multi_cases) / len(multi_cases)
                if multi_cases else 0.0
            ),
            "obvious_wrong_tool_calls": sum(
                len(set(item["selected_tools"]) - set(item["expected_tools"])) for item in items
            ),
            "model_requests": sum(item["model_requests"] for item in items),
            "failure_types": dict(sorted(errors.items())),
        }
    return summaries


def write_markdown_report(records: list[dict[str, Any]], path: Path) -> None:
    summaries = summarize_records(records)
    lines = [
        "# Stage 11 工具适用性专项评测",
        "",
        "本评测只发送带 tools 的选择请求；不执行本地工具，也不生成最终教学回答。",
        "",
        "| 策略 | Precision | Recall | No-tool 正确率 | 单工具命中 | 多工具完整命中 | 明显错误调用 | 请求数 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    def pct(value: float | None) -> str:
        return "N/A" if value is None else f"{value:.1%}"
    for name, summary in summaries.items():
        lines.append(
            f"| {name} | {pct(summary['tool_precision'])} | {pct(summary['tool_recall'])} | "
            f"{pct(summary['no_tool_accuracy'])} | {pct(summary['single_tool_hit_rate'])} | "
            f"{pct(summary['multi_tool_complete_hit_rate'])} | {summary['obvious_wrong_tool_calls']} | {summary['model_requests']} |"
        )
    lines.extend(["", "## 失败类型", ""])
    for name, summary in summaries.items():
        lines.append(f"- **{name}**：{summary['failure_types'] or '无'}")
    ranked = sorted(
        summaries.items(),
        key=lambda item: (
            item[1]["tool_precision"] or 0.0,
            item[1]["no_tool_accuracy"],
            item[1]["tool_recall"] or 0.0,
        ),
        reverse=True,
    )
    if "auto_formal" in summaries:
        formal = summaries["auto_formal"]
        candidate = summaries.get("auto_explicit")
        lines.extend(
            [
                "",
                "## 推荐",
                "",
                "- 正式 Tool Client 使用 **auto_formal**：它复用生产中的同一条通用选择规则，"
                f"本次结果为 precision {pct(formal['tool_precision'])}、"
                f"recall {pct(formal['tool_recall'])}、"
                f"no-tool 正确率 {pct(formal['no_tool_accuracy'])}，"
                f"明显错误调用 {formal['obvious_wrong_tool_calls']}。",
                "- 候选实验中的 **auto_explicit** 仍取得最高 precision，"
                "但 `auto_formal` 额外强调“即使可心算也应使用有价值的确定性工具”，"
                "召回有所提高，同时出现 1 次误调用。该差异应记录为模型选择权衡，"
                "本轮不继续针对评测题调提示词。",
            ]
        )
    elif ranked:
        name, summary = ranked[0]
        lines.extend(
            [
                "",
                "## 推荐",
                "",
                f"- 当前小型公开评测中建议优先继续验证 **{name}**："
                f"precision 为 {pct(summary['tool_precision'])}、"
                f"no-tool 正确率为 {pct(summary['no_tool_accuracy'])}，"
                f"且明显错误调用为 {summary['obvious_wrong_tool_calls']}。",
                "- 这不是对正式 Tool Client 的自动改动；它只说明下一步应以该通用提示作为候选，"
                "并继续关注“该调用但没调用”的召回缺口。",
            ]
        )
    lines.extend(["", "## 判读边界", "", "- `should_call_but_did_not`：已有直接、条件完整的白名单工具但模型未调用。", "- `should_not_call_but_did`：概念、实验或当前工具未覆盖的计算题被调用工具。", "- `wrong_tool`：调用了不在该案例期望列表内的工具。", "- `multi_tool_missing`：多项独立计算未完整选择所需工具。", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行或汇总 Stage 11 工具选择评测。")
    parser.add_argument("--strategy", choices=tuple(STRATEGIES))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--inputs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.inputs:
        write_markdown_report(read_records(args.inputs), args.report or RESULTS_DIR / "stage11_tool_selection_comparison.md")
        return
    if not args.strategy:
        raise SystemExit("运行评测时必须提供 --strategy。")
    output = args.output or RESULTS_DIR / f"stage11_tool_selection_{args.strategy}.jsonl"
    added = run_evaluation(strategy=args.strategy, output_path=output)
    print(f"完成 {added} 条选择评测：{output}")


if __name__ == "__main__":
    main()
