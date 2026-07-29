"""合并 Stage 04 题目与现有结果，生成前五题人工评审表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
DEFAULT_CASES = EVALUATION_DIR / "stage04_text_cases_v1.json"
DEFAULT_RESULTS = EVALUATION_DIR / "results" / "stage04_text_v1_results.jsonl"
DEFAULT_OUTPUT = EVALUATION_DIR / "reviews" / "stage04_first5_review.md"
EXPECTED_PROMPT_VERSION = "teacher_v2_personal"
REVIEW_COUNT = 5


def parse_args() -> argparse.Namespace:
    """解析本地文件参数。"""
    parser = argparse.ArgumentParser(
        description="仅合并现有 JSON/JSONL，生成 Stage 04 前五题人工评审表。"
    )
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    parser.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取评测题 JSON。"""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or len(cases) < REVIEW_COUNT:
        raise ValueError(f"评测题必须是至少包含 {REVIEW_COUNT} 项的 JSON 数组。")
    return cases


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """读取 JSONL，并保证结果 ID 非空且唯一。"""
    results: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"结果文件第 {line_number} 行不是有效 JSON。") from error
            result_id = record.get("id")
            if not isinstance(result_id, str) or not result_id:
                raise ValueError(f"结果文件第 {line_number} 行缺少有效 id。")
            if result_id in results:
                raise ValueError(f"结果文件存在重复 id：{result_id}。")
            results[result_id] = record
    return results


def marker(passed: bool) -> str:
    """把布尔结果转换为表格中的中文标记。"""
    return "通过" if passed else "失败"


def bullet_list(items: object) -> list[str]:
    """把字符串数组渲染为 Markdown 列表。"""
    if not isinstance(items, list):
        return ["- [数据格式错误：应为数组]"]
    return [f"- {item}" for item in items]


def quote_answer(answer: str) -> list[str]:
    """把原始回答完整放入 Markdown 引用块。"""
    lines = answer.splitlines() or [""]
    return [f"> {line}" if line else ">" for line in lines]


def precheck(
    case: dict[str, Any], result: dict[str, Any] | None
) -> dict[str, bool]:
    """执行不涉及物理正确性的确定性检查。"""
    return {
        "id_match": result is not None and result.get("id") == case.get("id"),
        "status_ok": result is not None and result.get("status") == "ok",
        "answer_nonempty": (
            result is not None
            and isinstance(result.get("answer"), str)
            and bool(result["answer"].strip())
        ),
        "prompt_version_ok": (
            result is not None
            and result.get("prompt_version") == EXPECTED_PROMPT_VERSION
        ),
    }


def build_review_sheet(
    cases: list[dict[str, Any]], results: dict[str, dict[str, Any]]
) -> tuple[str, list[tuple[str, dict[str, bool]]]]:
    """生成评审表文本和预检查结果。"""
    selected = cases[:REVIEW_COUNT]
    checks: list[tuple[str, dict[str, bool]]] = []
    lines = [
        "# Stage 04 前五题人工评审表",
        "",
        "- 评分标准：[`STAGE04_SCORING_RUBRIC.md`](../STAGE04_SCORING_RUBRIC.md)",
        f"- 预期提示词版本：`{EXPECTED_PROMPT_VERSION}`",
        "- 说明：确定性预检查只检查工程数据，不判断物理答案正确性。",
        "",
        "## 确定性预检查",
        "",
        "| id | status=ok | 回答非空 | id 匹配 | prompt_version 匹配 |",
        "| --- | --- | --- | --- | --- |",
    ]

    for case in selected:
        case_id = str(case.get("id", ""))
        result = results.get(case_id)
        check = precheck(case, result)
        checks.append((case_id, check))
        lines.append(
            "| {id} | {status} | {answer} | {id_match} | {version} |".format(
                id=case_id,
                status=marker(check["status_ok"]),
                answer=marker(check["answer_nonempty"]),
                id_match=marker(check["id_match"]),
                version=marker(check["prompt_version_ok"]),
            )
        )

    for index, case in enumerate(selected, 1):
        case_id = str(case.get("id", ""))
        result = results.get(case_id)
        answer = (
            str(result.get("answer", ""))
            if result is not None
            else "[缺少与该 id 匹配的结果记录]"
        )
        lines.extend(
            [
                "",
                "---",
                "",
                f"## {index}. {case_id}｜{case.get('topic', '')}",
                "",
                "### Question",
                "",
                str(case.get("question", "")),
                "",
                "### Expected key points",
                "",
                *bullet_list(case.get("expected_key_points")),
                "",
                "### Common errors",
                "",
                *bullet_list(case.get("common_errors")),
                "",
                "### Style checks",
                "",
                *bullet_list(case.get("style_checks")),
                "",
                "### Agent 完整回答",
                "",
                *quote_answer(answer),
                "",
                "### 人工评分",
                "",
                "| 评分维度 | 分数（0～2） | 评分依据 |",
                "| --- | --- | --- |",
                "| 物理正确性 |  |  |",
                "| 推理与答案完整性 |  |  |",
                "| 条件、研究对象和易错点处理 |  |  |",
                "| 个人教学语言风格 |  |  |",
                "| **总分（0～8）** |  |  |",
                "",
                "- **是否存在致命物理错误：** [ ] 是　[ ] 否",
                "- **漏掉的关键点：**",
                "- **命中的常见错误：**",
                "- **机械或不自然的个人句式：**",
                "- **人工总评：**",
            ]
        )

    return "\n".join(lines) + "\n", checks


def main() -> None:
    """读取现有数据并写出评审表，不调用任何 API。"""
    args = parse_args()
    cases = load_cases(args.cases)
    results = load_results(args.results)
    content, checks = build_review_sheet(cases, results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(content, encoding="utf-8", newline="\n")

    print(f"已生成评审表：{args.output}")
    for case_id, check in checks:
        summary = "，".join(f"{name}={marker(value)}" for name, value in check.items())
        print(f"{case_id}：{summary}")


if __name__ == "__main__":
    main()
