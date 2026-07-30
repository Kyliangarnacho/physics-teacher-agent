"""校验 Stage 05 JSONL 知识库。"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_KNOWLEDGE_BASE = PROJECT_ROOT / "knowledge" / "physics_notes_v1.jsonl"
REQUIRED_FIELDS = {"id", "chapter", "topic", "keywords", "source", "content"}
TEXT_FIELDS = {"id", "chapter", "topic", "source", "content"}


def validate_knowledge_base(
    path: Path,
) -> tuple[int, Counter[str], list[str]]:
    """校验知识库，返回记录数、章节分布和错误列表。"""
    errors: list[str] = []
    chapter_counts: Counter[str] = Counter()
    seen_ids: dict[str, int] = {}
    record_count = 0

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        return 0, chapter_counts, [f"无法读取文件：{error}"]

    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue

        try:
            record: Any = json.loads(line)
        except json.JSONDecodeError as error:
            errors.append(
                f"第 {line_number} 行：不是合法 JSON（{error.msg}，列 {error.colno}）。"
            )
            continue

        if not isinstance(record, dict):
            errors.append(f"第 {line_number} 行：JSON 顶层必须是对象。")
            continue

        record_count += 1
        missing_fields = sorted(REQUIRED_FIELDS - record.keys())
        if missing_fields:
            errors.append(
                f"第 {line_number} 行：缺少字段 {', '.join(missing_fields)}。"
            )

        for field in sorted(TEXT_FIELDS):
            if field not in record:
                continue
            value = record[field]
            if not isinstance(value, str) or not value.strip():
                errors.append(f"第 {line_number} 行：{field} 必须是非空字符串。")

        if "keywords" in record:
            keywords = record["keywords"]
            if (
                not isinstance(keywords, list)
                or not keywords
                or any(
                    not isinstance(keyword, str) or not keyword.strip()
                    for keyword in keywords
                )
            ):
                errors.append(
                    f"第 {line_number} 行：keywords 必须是非空字符串列表。"
                )

        record_id = record.get("id")
        if isinstance(record_id, str) and record_id.strip():
            if record_id in seen_ids:
                errors.append(
                    f"第 {line_number} 行：id {record_id!r} 重复，"
                    f"首次出现在第 {seen_ids[record_id]} 行。"
                )
            else:
                seen_ids[record_id] = line_number

        chapter = record.get("chapter")
        if isinstance(chapter, str) and chapter.strip():
            chapter_counts[chapter] += 1

    return record_count, chapter_counts, errors


def parse_args() -> argparse.Namespace:
    """解析可选的知识库文件路径。"""
    parser = argparse.ArgumentParser(description="校验 Stage 05 JSONL 知识库。")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=DEFAULT_KNOWLEDGE_BASE,
        help="待校验的 JSONL 文件；默认使用 knowledge/physics_notes_v1.jsonl。",
    )
    return parser.parse_args()


def main() -> int:
    """运行校验并返回进程退出码。"""
    args = parse_args()
    record_count, chapter_counts, errors = validate_knowledge_base(args.path)

    if errors:
        print(f"知识库校验失败：{args.path}", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 1

    distribution = "，".join(
        f"{chapter} {count} 条" for chapter, count in sorted(chapter_counts.items())
    )
    print(f"知识库校验通过：共 {record_count} 条。")
    print(f"章节分布：{distribution or '无记录'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
