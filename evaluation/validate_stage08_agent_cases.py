"""校验 Stage 08 统一 Agent 代表题评测集 V1。"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.schemas import RunStatus, TeachingMode


CASES_PATH = Path(__file__).with_name("stage08_agent_cases_v1.json")
REQUIRED_FIELDS = {
    "id",
    "category",
    "question",
    "expected_teaching_mode",
    "expected_use_rag",
    "expected_use_tools",
    "expected_run_status",
    "answer_keywords",
}
ID_PATTERN = re.compile(r"S08-AGENT-\d{3}")
VALID_MODES = {mode.value for mode in TeachingMode}
VALID_RUN_STATUSES = {status.value for status in RunStatus}


def load_cases(path: Path = CASES_PATH) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as file:
        cases = json.load(file)
    if not isinstance(cases, list):
        raise ValueError("评测数据顶层必须是 JSON 数组。")
    return cases


def validate_cases(cases: list[dict[str, object]]) -> None:
    errors: list[str] = []
    ids: list[str] = []

    if len(cases) != 8:
        errors.append(f"题目数量应为 8，实际为 {len(cases)}。")

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

        for field in ("category", "question"):
            value = case.get(field)
            if not isinstance(value, str) or not value.strip():
                errors.append(f"第 {index} 题字段 {field} 必须是非空字符串。")

        if case.get("expected_teaching_mode") not in VALID_MODES:
            errors.append(f"第 {index} 题教学模式不合法。")
        if case.get("expected_run_status") not in VALID_RUN_STATUSES:
            errors.append(f"第 {index} 题运行状态不合法。")
        for field in ("expected_use_rag", "expected_use_tools"):
            if type(case.get(field)) is not bool:
                errors.append(f"第 {index} 题字段 {field} 必须是布尔值。")

        keywords = case.get("answer_keywords")
        if (
            not isinstance(keywords, list)
            or not keywords
            or any(not isinstance(item, str) or not item.strip() for item in keywords)
        ):
            errors.append(f"第 {index} 题 answer_keywords 必须是非空字符串数组。")

    duplicates = sorted(
        case_id for case_id, count in Counter(ids).items() if count > 1
    )
    if duplicates:
        errors.append(f"存在重复 id：{', '.join(duplicates)}。")

    if errors:
        raise ValueError("\n".join(errors))


def main() -> None:
    cases = load_cases()
    validate_cases(cases)
    categories = "，".join(str(case["category"]) for case in cases)
    print(f"校验通过：共 {len(cases)} 题，ID 唯一且字段、枚举合法。")
    print(f"覆盖类别：{categories}")


if __name__ == "__main__":
    main()
