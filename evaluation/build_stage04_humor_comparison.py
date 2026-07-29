"""生成 Stage 04 前五题个人幽默提示词新旧回答对比表。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


EVALUATION_DIR = Path(__file__).resolve().parent
CASES_PATH = EVALUATION_DIR / "stage04_text_cases_v1.json"
OLD_RESULTS_PATH = EVALUATION_DIR / "results" / "stage04_text_v1_results.jsonl"
NEW_RESULTS_PATH = (
    EVALUATION_DIR
    / "results"
    / "stage04_text_v1_teacher_v3_personal_humor_results.jsonl"
)
OUTPUT_PATH = EVALUATION_DIR / "reviews" / "stage04_first5_v2_vs_v3_humor.md"
OLD_VERSION = "teacher_v2_personal"
NEW_VERSION = "teacher_v3_personal_humor"
REVIEW_COUNT = 5


def load_cases(path: Path) -> list[dict[str, Any]]:
    """读取评测题。"""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list) or len(cases) < REVIEW_COUNT:
        raise ValueError(f"评测集至少需要 {REVIEW_COUNT} 道题。")
    return cases[:REVIEW_COUNT]


def load_results(path: Path) -> dict[str, dict[str, Any]]:
    """读取结果 JSONL，并检查 ID 唯一。"""
    rows: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path} 第 {line_number} 行不是有效 JSON。") from error
            case_id = row.get("id")
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{path} 第 {line_number} 行缺少有效 id。")
            if case_id in rows:
                raise ValueError(f"{path} 存在重复 id：{case_id}。")
            rows[case_id] = row
    return rows


def quote(text: str) -> list[str]:
    """把回答完整放入 Markdown 引用块。"""
    return [f"> {line}" if line else ">" for line in text.splitlines()]


def result_ok(row: dict[str, Any] | None, expected_version: str) -> bool:
    """执行不涉及物理正确性的结果完整性检查。"""
    return bool(
        row
        and row.get("status") == "ok"
        and isinstance(row.get("answer"), str)
        and row["answer"].strip()
        and row.get("prompt_version") == expected_version
    )


def main() -> None:
    """合并新旧结果并生成留有人工判断栏的对比报告。"""
    cases = load_cases(CASES_PATH)
    old_results = load_results(OLD_RESULTS_PATH)
    new_results = load_results(NEW_RESULTS_PATH)

    lines = [
        "# Stage 04 前五题个人幽默提示词对比",
        "",
        f"- 旧版本：`{OLD_VERSION}`",
        f"- 新版本：`{NEW_VERSION}`",
        "- 题目：两版使用完全相同的 Stage 04 前五题",
        "- 说明：结果完整性由脚本检查；风格变化和物理退化由人工复核。",
        "",
        "## 确定性预检查",
        "",
        "| id | 旧结果完整 | 新结果完整 |",
        "| --- | --- | --- |",
    ]

    for case in cases:
        case_id = str(case["id"])
        old_ok = "通过" if result_ok(old_results.get(case_id), OLD_VERSION) else "失败"
        new_ok = "通过" if result_ok(new_results.get(case_id), NEW_VERSION) else "失败"
        lines.append(f"| {case_id} | {old_ok} | {new_ok} |")

    for index, case in enumerate(cases, 1):
        case_id = str(case["id"])
        old = old_results.get(case_id)
        new = new_results.get(case_id)
        if old is None or new is None:
            raise ValueError(f"缺少 {case_id} 的新旧结果，无法生成对比。")

        lines.extend(
            [
                "",
                "---",
                "",
                f"## {index}. {case_id}｜{case['topic']}",
                "",
                "### Question",
                "",
                str(case["question"]),
                "",
                f"### 旧回答（{OLD_VERSION}）",
                "",
                *quote(str(old["answer"])),
                "",
                f"### 新回答（{NEW_VERSION}）",
                "",
                *quote(str(new["answer"])),
                "",
                "### 对比检查",
                "",
                "- **是否减少通用模板句：** 待填写",
                "- **是否出现更自然的个人身份锚点：** 待填写",
                "- **物理内容是否发生退化：** 待填写",
                "- **对比备注：**",
                "",
                "### 新版人工评分",
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

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print(f"已生成对比报告：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
