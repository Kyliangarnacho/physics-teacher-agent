"""运行 Stage 04 文本评测集，并将回答逐条保存为 JSONL。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_qwen_config
from src.model_client import answer_question
from src.prompts import PROMPT_VERSION

from evaluation.validate_stage04_text_cases import (
    CASES_PATH,
    load_cases,
    validate_cases,
)


DEFAULT_OUTPUT = Path(__file__).with_name("results") / "stage04_text_v1_results.jsonl"
ERROR_PREFIXES = (
    "认证失败",
    "网络连接失败",
    "请求过于频繁",
    "千问 API 请求失败",
    "调用千问时发生未知错误",
    "千问返回了空回答",
)


def positive_int(value: str) -> int:
    """解析正整数命令行参数。"""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("必须是正整数。")
    return number


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="运行 Stage 04 文本评测集 V1。")
    parser.add_argument(
        "--limit",
        type=positive_int,
        default=5,
        help="只评测数据集前 N 题；默认为 5，避免意外调用全部题目。",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="JSONL 结果文件路径。",
    )
    return parser.parse_args()


def completed_ids(path: Path) -> set[str]:
    """读取已完成结果，供断点续跑时跳过。"""
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
                raise ValueError(f"结果文件第 {line_number} 行不是有效 JSON。") from error
            case_id = record.get("id")
            if isinstance(case_id, str):
                ids.add(case_id)
    return ids


def run_evaluation(
    *,
    cases_path: Path,
    output_path: Path,
    limit: int,
    answer_fn: Callable[[str], str],
    model: str,
    prompt_version: str,
) -> int:
    """运行评测并返回新增记录数；调用函数可注入，便于纯本地测试。"""
    cases = load_cases(cases_path)
    validate_cases(cases)
    selected = cases[: min(limit, len(cases))]
    finished = completed_ids(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    new_count = 0

    with output_path.open("a", encoding="utf-8") as output:
        for case in selected:
            case_id = str(case["id"])
            if case_id in finished:
                print(f"跳过已完成题目：{case_id}")
                continue

            print(f"正在评测：{case_id}（{case['topic']}）")
            started = perf_counter()
            answer = answer_fn(str(case["question"]))
            elapsed_seconds = round(perf_counter() - started, 3)
            status = "error" if answer.startswith(ERROR_PREFIXES) else "ok"

            record = {
                "id": case_id,
                "topic": case["topic"],
                "question": case["question"],
                "answer": answer,
                "status": status,
                "prompt_version": prompt_version,
                "model": model,
                "elapsed_seconds": elapsed_seconds,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            output.write(json.dumps(record, ensure_ascii=False) + "\n")
            output.flush()
            finished.add(case_id)
            new_count += 1
            print(f"完成：{case_id}，status={status}，answer_length={len(answer)}")

    print(f"本次新增 {new_count} 条结果；结果文件：{output_path}")
    return new_count


def main() -> None:
    """校验数据、调用模型，并在每题完成后立即写入结果。"""
    args = parse_args()
    _, _, model = load_qwen_config()
    run_evaluation(
        cases_path=CASES_PATH,
        output_path=args.output,
        limit=args.limit,
        answer_fn=answer_question,
        model=model,
        prompt_version=PROMPT_VERSION,
    )


if __name__ == "__main__":
    main()
