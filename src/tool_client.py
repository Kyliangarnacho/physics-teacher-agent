"""Two-request Function Calling client for one local physics tool call."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from openai import OpenAI

from src.config import load_qwen_config
from src.model_client import build_messages
from src.tools.registry import (
    ToolExecutionStatus,
    execute_tool_call,
    get_openai_tools,
)


CompletionFunc = Callable[..., Any]


def _tool_call_payload(tool_call: Any) -> dict[str, Any]:
    return {
        "id": tool_call.id,
        "type": tool_call.type,
        "function": {
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        },
    }


def _result(
    answer: str,
    tool_records: list[dict[str, Any]],
    model_requests: int,
) -> dict[str, Any]:
    return {
        "answer": answer,
        "tool_records": tool_records,
        "model_requests": model_requests,
    }


def answer_with_tools(
    question: str,
    context: str | None = None,
    mode_instruction: str | None = None,
    completion_func: CompletionFunc | None = None,
) -> dict[str, Any]:
    """Run one model-selected local tool and ask the model for a final answer."""
    messages: list[dict[str, Any]] = list(
        build_messages(question, context, mode_instruction)
    )
    request_options: dict[str, Any] = {}

    if completion_func is None:
        api_key, base_url, model = load_qwen_config()
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create
        request_options["model"] = model
    else:
        completion = completion_func

    model_requests = 0
    try:
        model_requests += 1
        first_response = completion(
            **request_options,
            messages=[message.copy() for message in messages],
            tools=get_openai_tools(),
            tool_choice="required",
            extra_body={"enable_thinking": False},
            stream=False,
        )
        assistant_message = first_response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
    except Exception:
        return _result("工具调用请求失败，请稍后重试。", [], model_requests)

    if len(tool_calls) == 0:
        return _result("模型没有返回工具调用，本次未执行计算。", [], model_requests)
    if len(tool_calls) != 1:
        return _result("模型返回了多个工具调用，本阶段不执行。", [], model_requests)

    tool_call = tool_calls[0]
    try:
        record = execute_tool_call(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            arguments_json=tool_call.function.arguments,
        )
    except Exception:
        return _result("本地工具执行失败，请检查题目条件。", [], model_requests)

    record_data = record.model_dump(mode="json")
    tool_records = [record_data]
    if record.status is not ToolExecutionStatus.SUCCESS:
        return _result(
            record.error or "本地工具执行失败，请检查题目条件。",
            tool_records,
            model_requests,
        )

    messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [_tool_call_payload(tool_call)],
        }
    )
    messages.append(
        {
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": json.dumps(record_data, ensure_ascii=False),
        }
    )

    try:
        model_requests += 1
        second_response = completion(
            **request_options,
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=False,
        )
        final_answer = second_response.choices[0].message.content
    except Exception:
        return _result("工具结果回传失败，请稍后重试。", tool_records, model_requests)

    answer = final_answer.strip() if final_answer and final_answer.strip() else "模型返回了空回答。"
    return _result(answer, tool_records, model_requests)
