"""One-shot Stage 07 probe for Qwen-compatible Function Calling support."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from src.config import load_qwen_config
from src.prompts import JUNIOR_PHYSICS_SYSTEM_PROMPT
from src.tools.registry import (
    ToolExecutionStatus,
    execute_tool_call,
    get_openai_tools,
)


QUESTION = "某电阻两端电压为 12 V，电阻为 6 Ω，请使用计算工具求电流。"
EXPECTED_TOOL_NAME = "calculate_ohms_law"
TOOL_ARGUMENT_TYPE_INSTRUCTION = """调用工具时必须严格遵守工具的 parameters JSON Schema。
所有 type: number 字段必须输出 JSON 数字字面量，例如：
{"voltage_v": 12, "resistance_ohm": 6}
禁止写成：
{"voltage_v": "12", "resistance_ohm": "6"}
不得增加 Schema 未定义的字段。"""


def _usage_data(usage: Any) -> dict[str, Any] | None:
    if usage is None:
        return None
    return usage.model_dump() if hasattr(usage, "model_dump") else None


def _print_json(label: str, value: Any) -> None:
    print(f"{label}：{json.dumps(value, ensure_ascii=False)}")


def _schema_type_summary(field_schema: dict[str, Any]) -> str:
    if "type" in field_schema:
        return str(field_schema["type"])
    types = [
        branch["type"]
        for branch in field_schema.get("anyOf", [])
        if isinstance(branch, dict) and "type" in branch
    ]
    return " | ".join(types) if types else "未声明"


def _fail(layer: str, message: str, request_count: int) -> int:
    print("探针结果：失败")
    print(f"失败层级：{layer}")
    print(f"错误现象：{message}")
    print(f"总请求数：{request_count}")
    return 1


def main() -> int:
    request_count = 0

    try:
        api_key, base_url, configured_model = load_qwen_config()
        model = os.getenv("QWEN_TOOL_MODEL", "").strip() or configured_model
        client = OpenAI(api_key=api_key, base_url=base_url)
    except Exception as exc:
        return _fail("模型配置", f"{type(exc).__name__}：无法加载当前千问配置。", 0)

    tools = get_openai_tools()
    ohms_law_tool = next(
        tool
        for tool in tools
        if tool["function"]["name"] == EXPECTED_TOOL_NAME
    )
    ohms_law_properties = ohms_law_tool["function"]["parameters"]["properties"]

    print(f"模型名：{model}")
    print(
        "voltage_v Schema 类型："
        f"{_schema_type_summary(ohms_law_properties['voltage_v'])}"
    )
    print(
        "resistance_ohm Schema 类型："
        f"{_schema_type_summary(ohms_law_properties['resistance_ohm'])}"
    )
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": JUNIOR_PHYSICS_SYSTEM_PROMPT},
        {"role": "system", "content": TOOL_ARGUMENT_TYPE_INSTRUCTION},
        {"role": "user", "content": QUESTION},
    ]

    try:
        request_count += 1
        first_response = client.chat.completions.create(
            model=model,
            messages=messages,
            tools=tools,
            tool_choice="required",
            extra_body={"enable_thinking": False},
            stream=False,
        )
    except Exception as exc:
        return _fail(
            "第一次 API 请求或工具 Schema",
            f"{type(exc).__name__}：API 未接受第一轮 Function Calling 请求。",
            request_count,
        )

    first_choice = first_response.choices[0]
    assistant_message = first_choice.message
    tool_calls = assistant_message.tool_calls or []
    print(f"第一轮 finish_reason：{first_choice.finish_reason}")
    print(f"tool_calls 数量：{len(tool_calls)}")
    _print_json("第一轮 usage", _usage_data(first_response.usage))

    if len(tool_calls) != 1:
        return _fail(
            "模型工具选择",
            f"预期恰好一个工具调用，实际为 {len(tool_calls)} 个。",
            request_count,
        )

    tool_call = tool_calls[0]
    tool_name = tool_call.function.name
    arguments_json = tool_call.function.arguments
    print(f"tool_call_id：{tool_call.id}")
    print(f"工具名：{tool_name}")
    print(f"arguments 原始 JSON：{arguments_json}")

    try:
        parsed_arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return _fail(
            "参数解析",
            "arguments 不是合法 JSON，无法检查参数类型。",
            request_count,
        )
    if not isinstance(parsed_arguments, dict):
        return _fail(
            "参数解析",
            "arguments 解析后不是 JSON 对象。",
            request_count,
        )
    _print_json(
        "json.loads 后参数 Python 类型",
        {key: type(value).__name__ for key, value in parsed_arguments.items()},
    )

    if tool_name != EXPECTED_TOOL_NAME:
        return _fail(
            "模型工具选择",
            f"预期工具 {EXPECTED_TOOL_NAME}，实际为 {tool_name}。",
            request_count,
        )

    execution_record = execute_tool_call(
        tool_call_id=tool_call.id,
        name=tool_name,
        arguments_json=arguments_json,
    )
    execution_data = execution_record.model_dump(mode="json")
    _print_json("本地工具执行记录", execution_data)
    _print_json("normalized_fields", execution_record.normalized_fields)

    if execution_record.status is not ToolExecutionStatus.SUCCESS:
        return _fail(
            "参数解析或本地执行",
            execution_record.error or "本地工具执行失败。",
            request_count,
        )
    if not execution_record.result or (
        execution_record.result.get("display_value") != "2"
        or execution_record.result.get("unit") != "A"
    ):
        return _fail(
            "本地执行结果",
            "本地工具执行成功，但结果不是预期的 2 A。",
            request_count,
        )

    assistant_tool_calls = [
        {
            "id": call.id,
            "type": call.type,
            "function": {
                "name": call.function.name,
                "arguments": call.function.arguments,
            },
        }
        for call in tool_calls
    ]
    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": assistant_tool_calls,
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(execution_record.result, ensure_ascii=False),
        }
    )

    try:
        request_count += 1
        second_response = client.chat.completions.create(
            model=model,
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=False,
        )
    except Exception as exc:
        return _fail(
            "第二次 API 请求或结果回传",
            f"{type(exc).__name__}：API 未接受 role=tool 结果消息。",
            request_count,
        )

    final_answer = second_response.choices[0].message.content
    _print_json("第二轮 usage", _usage_data(second_response.usage))
    print(f"总请求数：{request_count}")
    if not final_answer or not final_answer.strip():
        return _fail("最终回答", "第二轮返回了空回答。", request_count)

    print("role=tool 接受情况：成功")
    print(f"最终回答：{final_answer}")
    print("探针结果：成功，全部标准满足。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
