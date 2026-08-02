"""Stage 07 unified Agent one-shot end-to-end acceptance probe."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import patch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.agent as agent_module
from src.analyzer import analyze_question as real_analyze_question
from src.config import load_qwen_config


QUESTION = "某电阻两端电压为 12 V，电阻为 6 Ω，求通过它的电流。"
EXPECTED_TOOL_NAME = "calculate_ohms_law"


def _print_json(label: str, value: Any) -> None:
    print(f"{label}：{json.dumps(value, ensure_ascii=False)}")


def _failure(layer: str, message: str) -> int:
    print("验收结果：失败")
    print(f"失败层级：{layer}")
    print(f"错误现象：{message}")
    print("处理方式：按要求停止，不重试、不切换模型。")
    return 1


def main() -> int:
    try:
        _, _, model = load_qwen_config()
    except Exception as exc:
        return _failure(
            "模型配置",
            f"{type(exc).__name__}：无法加载当前 QWEN_MODEL 配置。",
        )

    analyzer_calls = 0
    plain_answer_called = False

    def counted_real_analyzer(
        question: str,
        analyze_func: Any = None,
    ) -> Any:
        nonlocal analyzer_calls
        analyzer_calls += 1
        return real_analyze_question(question, analyze_func=analyze_func)

    def forbidden_plain_answer(*args: Any, **kwargs: Any) -> str:
        nonlocal plain_answer_called
        plain_answer_called = True
        raise AssertionError("工具路径错误调用了普通 answer_func。")

    print(f"实际模型名：{model}")
    try:
        with patch.object(
            agent_module,
            "analyze_question",
            side_effect=counted_real_analyzer,
        ):
            result = agent_module.run_teacher_agent(
                QUESTION,
                mode_override="solve",
                rag_policy="off",
                answer_func=forbidden_plain_answer,
            )
    except AssertionError as exc:
        print(f"Analyzer 调用次数：{analyzer_calls}")
        print(f"普通 answer_func 是否被调用：{plain_answer_called}")
        return _failure("Router 或 Agent 路径选择", str(exc))
    except Exception as exc:
        print(f"Analyzer 调用次数：{analyzer_calls}")
        print(f"普通 answer_func 是否被调用：{plain_answer_called}")
        return _failure(
            "Analyzer、Tool Client 或最终结果回传",
            f"{type(exc).__name__}：端到端调用未完成。",
        )

    analysis = result.get("analysis", {})
    route = result.get("route", {})
    tool_records = result.get("tool_records", [])
    tool_model_requests = result.get("tool_model_requests", 0)
    sources = result.get("sources", [])
    answer = result.get("answer", "")
    analysis_fallback = result.get("analysis_fallback", False)
    total_model_requests = analyzer_calls + tool_model_requests

    print(f"Analyzer 调用次数：{analyzer_calls}")
    print(
        "analysis.calculation_required："
        f"{analysis.get('calculation_required')}"
    )
    print(
        "analysis.missing_conditions："
        f"{analysis.get('missing_conditions')}"
    )
    print(f"route.use_tools：{route.get('use_tools')}")
    print(f"route.use_rag：{route.get('use_rag')}")
    print(f"工具记录数量：{len(tool_records)}")

    first_record = tool_records[0] if len(tool_records) == 1 else {}
    if not isinstance(first_record, dict):
        first_record = {}
    print(f"工具名：{first_record.get('name')}")
    _print_json("arguments", first_record.get("arguments", {}))
    _print_json(
        "normalized_fields",
        first_record.get("normalized_fields", []),
    )
    print(f"工具执行状态：{first_record.get('status')}")
    _print_json("工具结果", first_record.get("result"))
    print(f"Tool Client 内部模型请求数：{tool_model_requests}")
    print(f"整个 Agent 模型请求数：{total_model_requests}")
    print(f"最终教师回答：{answer}")
    print(f"sources 数量：{len(sources)}")
    print(f"analysis_fallback：{analysis_fallback}")
    print(f"普通 answer_func 是否被调用：{plain_answer_called}")

    tool_result = first_record.get("result")
    got_two_amperes = isinstance(tool_result, dict) and (
        tool_result.get("display_value") == "2"
        and tool_result.get("unit") == "A"
    )
    failures = []
    checks = [
        (analyzer_calls == 1, "Analyzer 调用次数不是 1"),
        (
            analysis.get("calculation_required") is True,
            "calculation_required 不是 True",
        ),
        (
            analysis.get("missing_conditions") is False,
            "missing_conditions 不是 False",
        ),
        (route.get("use_tools") is True, "route.use_tools 不是 True"),
        (route.get("use_rag") is False, "route.use_rag 不是 False"),
        (plain_answer_called is False, "普通 answer_func 被误调用"),
        (len(tool_records) == 1, "工具记录数量不是 1"),
        (
            first_record.get("name") == EXPECTED_TOOL_NAME,
            "工具名不是 calculate_ohms_law",
        ),
        (first_record.get("status") == "success", "工具状态不是 success"),
        (got_two_amperes, "工具结果不是 2 A"),
        (tool_model_requests == 2, "Tool Client 模型请求数不是 2"),
        (total_model_requests == 3, "整个 Agent 模型请求总数不是 3"),
        (isinstance(answer, str) and bool(answer.strip()), "最终回答为空"),
        (sources == [], "sources 不是空列表"),
        (analysis_fallback is False, "Analyzer 发生了 fallback"),
    ]
    failures.extend(message for passed, message in checks if not passed)

    if failures:
        print("未满足的验收项：")
        for failure in failures:
            print(f"- {failure}")
        return _failure("端到端验收标准", "存在未满足的验收项。")

    print("验收结果：成功，全部标准满足。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
