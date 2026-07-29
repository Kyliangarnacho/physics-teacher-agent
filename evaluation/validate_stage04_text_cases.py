"""校验 Stage 04 文本评测集 V1。"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path


CASES_PATH = Path(__file__).with_name("stage04_text_cases_v1.json")
REQUIRED_FIELDS = {
    "id",
    "topic",
    "question",
    "expected_key_points",
    "common_errors",
    "style_checks",
    "source_note",
}
LIST_FIELDS = {"expected_key_points", "common_errors", "style_checks"}
ID_PATTERN = re.compile(r"S04-TXT-\d{3}")


def load_cases(path: Path = CASES_PATH) -> list[dict[str, object]]:
    """读取评测集并确认顶层结构。"""
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("评测数据顶层必须是 JSON 数组。")
    return cases


def validate_cases(cases: list[dict[str, object]]) -> None:
    """校验题目数量、字段、类型和 ID 唯一性。"""
    errors: list[str] = []
    ids: list[str] = []

    if len(cases) != 15:
        errors.append(f"题目数量应为 15，实际为 {len(cases)}。")

    for index, case in enumerate(cases, 1):
        if not isinstance(case, dict):
            errors.append(f"第 {index} 项不是 JSON 对象。")
            continue

        missing = REQUIRED_FIELDS - case.keys()
        extra = case.keys() - REQUIRED_FIELDS
        if missing:
            errors.append(f"第 {index} 题缺少字段：{', '.join(sorted(missing))}。")
        if extra:
            errors.append(f"第 {index} 题存在未知字段：{', '.join(sorted(extra))}。")

        case_id = case.get("id")
        if not isinstance(case_id, str) or not ID_PATTERN.fullmatch(case_id):
            errors.append(f"第 {index} 题 id 格式错误：{case_id!r}。")
        else:
            ids.append(case_id)

        for field in REQUIRED_FIELDS - LIST_FIELDS - {"id"}:
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"第 {index} 题字段 {field} 必须是非空字符串。")

        for field in LIST_FIELDS:
            value = case.get(field)
            if (
                not isinstance(value, list)
                or not value
                or any(not isinstance(item, str) or not item.strip() for item in value)
            ):
                errors.append(f"第 {index} 题字段 {field} 必须是非空字符串数组。")

    duplicate_ids = sorted(case_id for case_id, count in Counter(ids).items() if count > 1)
    if duplicate_ids:
        errors.append(f"存在重复 id：{', '.join(duplicate_ids)}。")

    if errors:
        raise ValueError("\n".join(errors))


def main() -> None:
    """运行本地校验并输出简要分布。"""
    cases = load_cases()
    validate_cases(cases)
    topic_counts = Counter(str(case["topic"]).split("·", 1)[0] for case in cases)
    distribution = "，".join(
        f"{topic} {count} 题" for topic, count in sorted(topic_counts.items())
    )
    print(f"校验通过：共 {len(cases)} 题，id 唯一且字段完整。")
    print(f"一级知识点分布：{distribution}")


if __name__ == "__main__":
    main()
