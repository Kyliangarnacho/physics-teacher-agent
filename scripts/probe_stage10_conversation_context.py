"""Stage 10 历史与教学状态上下文真实探针。

需要 .env 中配置有效的千问 API（DASHSCOPE_API_KEY / QWEN_BASE_URL / QWEN_MODEL）。
本脚本只向 stdout 打印结果，不写任何 Git 跟踪文件，不输出 API Key。
真实 API 失败会按错误类别报告，不使用修改业务逻辑的方式掩盖。
"""

from __future__ import annotations

import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agent import run_teacher_agent


ERROR_PREFIXES = (
    "认证失败",
    "网络连接失败",
    "请求过于频繁",
    "千问 API 请求失败",
    "调用千问时发生未知错误",
    "千问返回了空回答",
)

HISTORY_SPEED = [
    {"role": "user", "content": "一辆小车 10 秒行驶 50 米，求平均速度。"},
    {"role": "assistant", "content": "平均速度 v = s / t = 50 m / 10 s = 5 m/s。"},
]

HISTORY_OHMS = [
    {"role": "user", "content": "电压 12 V，电阻 6 Ω，求电流。"},
    {
        "role": "assistant",
        "content": "第一步：先写出欧姆定律公式 I = U / R。",
    },
]

STATE_OHMS = (
    "当前活动题目：电压 12 V，电阻 6 Ω，求电流。\n"
    "当前教学模式：hint\n"
    "已完成提示步数：1"
)


def classify_exception(exc: Exception) -> str:
    message = str(exc)
    if "配置缺失" in message:
        return "配置缺失（请在 .env 中检查 API Key / Base URL / 模型）"
    name = type(exc).__name__
    if "AuthenticationError" in name:
        return "认证失败（API Key 无效）"
    if "RateLimitError" in name:
        return "额度或限流"
    if "APIConnectionError" in name:
        return "网络连接失败"
    if "APIError" in name:
        return "API 服务错误"
    return "代码或未知错误"


def run_probe(
    name: str,
    question: str,
    *,
    history,
    state,
) -> None:
    print(f"--- {name} ---")
    print(f"question: {question}")
    print(
        "history_turns: "
        f"{len(history) if history else 0}, "
        f"state_used: {bool(state)}"
    )
    try:
        result = run_teacher_agent(
            question,
            conversation_history=history,
            teaching_state_context=state,
        )
    except Exception as exc:  # noqa: BLE001 - 探针需要区分错误类别
        print(f"ERROR: {classify_exception(exc)}")
        print(f"detail: {str(exc)[:300]}")
        return

    route = result["route"]
    trace = result["trace"]
    answer = result["answer"]
    print(
        "route: "
        f"mode={route['teaching_mode']} "
        f"use_rag={route['use_rag']} "
        f"use_tools={route['use_tools']} "
        f"should_answer={route['should_answer']}"
    )
    print(
        "trace: "
        f"status={trace['status']} "
        f"total_model_requests={trace['total_model_requests']} "
        f"rag_searches={trace['rag_searches']} "
        f"tool_executions={trace['tool_executions']} "
        f"analysis_fallback={result['analysis_fallback']}"
    )
    if result.get("tool_records"):
        for record in result["tool_records"]:
            print(
                f"tool_record: name={record.get('name')} "
                f"status={record.get('status')} "
                f"arguments={record.get('arguments')} "
                f"error={record.get('error')}"
            )
    print(f"answer: {answer[:600]}")
    for prefix in ERROR_PREFIXES:
        if answer.startswith(prefix):
            print(f"NOTICE: 回答看起来是错误提示（{prefix}）")


def main() -> int:
    print("Stage 10 conversation context probe (real API)")
    run_probe(
        "探针 A：历史承接（平均速度 → 为什么要除以时间）",
        "为什么要除以时间？",
        history=HISTORY_SPEED,
        state=None,
    )
    run_probe(
        "探针 B：hint 状态承接（欧姆定律 → 继续下一步）",
        "继续下一步",
        history=HISTORY_OHMS,
        state=STATE_OHMS,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
