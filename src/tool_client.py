"""Two-request Function Calling client for one local physics tool call."""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from typing import Any

from openai import OpenAI

from src.config import load_qwen_config
from src.model_client import build_messages
from src.observability import ErrorType, StepTimer, create_skipped_step
from src.retry import run_with_one_retry
from src.schemas import StepStatus, StepTrace
from src.tools.registry import (
    ToolExecutionStatus,
    execute_tool_call,
    get_openai_tools,
)


CompletionFunc = Callable[..., Any]
ShouldRetryFunc = Callable[[Exception], bool]


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
    step_traces: list[StepTrace],
) -> dict[str, Any]:
    return {
        "answer": answer,
        "tool_records": tool_records,
        "model_requests": model_requests,
        "step_traces": [trace.model_dump(mode="json") for trace in step_traces],
    }


def _request_status(attempts: int) -> StepStatus:
    return StepStatus.SUCCESS if attempts == 1 else StepStatus.RETRY_SUCCESS


def _skipped_remaining_steps(
    traces: list[StepTrace],
    *names: str,
    reason: str,
) -> None:
    for name in names:
        traces.append(create_skipped_step(name, {"reason": reason}))


def _record_error_type(error_message: str | None) -> ErrorType:
    if error_message and error_message.startswith("工具参数"):
        return ErrorType.TOOL_VALIDATION
    if error_message and "未知工具" in error_message:
        return ErrorType.TOOL_PROTOCOL
    return ErrorType.TOOL_EXECUTION


def answer_with_tools(
    question: str,
    context: str | None = None,
    mode_instruction: str | None = None,
    completion_func: CompletionFunc | None = None,
    selection_should_retry: ShouldRetryFunc | None = None,
    result_should_retry: ShouldRetryFunc | None = None,
    *,
    teaching_state_context: str | None = None,
    learning_memory_context: str | None = None,
    conversation_history: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Run one model-selected local tool and ask the model for a final answer."""
    kwargs: dict[str, object] = {}
    if teaching_state_context is not None:
        kwargs["teaching_state_context"] = teaching_state_context
    if learning_memory_context is not None:
        kwargs["learning_memory_context"] = learning_memory_context
    if conversation_history is not None:
        kwargs["conversation_history"] = conversation_history
    messages: list[dict[str, Any]] = list(
        build_messages(question, context, mode_instruction, **kwargs)
    )
    request_options: dict[str, Any] = {}

    if completion_func is None:
        api_key, base_url, model = load_qwen_config()
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create
        request_options["model"] = model
    else:
        completion = completion_func

    step_traces: list[StepTrace] = []
    retry_selection = (
        selection_should_retry
        if selection_should_retry is not None
        else lambda error: True
    )
    retry_result = (
        result_should_retry
        if result_should_retry is not None
        else lambda error: True
    )

    selection_timer = StepTimer("tool_selection")

    def request_tool_selection() -> Any:
        return completion(
            **request_options,
            messages=deepcopy(messages),
            tools=get_openai_tools(),
            tool_choice="required",
            extra_body={"enable_thinking": False},
            stream=False,
        )

    selection_outcome = run_with_one_retry(
        request_tool_selection,
        retry_selection,
        delay_seconds=0.0,
    )
    if not selection_outcome.succeeded:
        step_traces.append(
            selection_timer.finish(
                status=StepStatus.ERROR,
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                error_type=ErrorType.TOOL_SELECTION_API,
                error_message="工具选择模型调用失败。",
            )
        )
        _skipped_remaining_steps(
            step_traces,
            "tool_execution",
            "tool_result_answer",
            reason="tool_selection_failed",
        )
        return _result(
            "工具调用请求失败，请稍后重试。",
            [],
            selection_outcome.attempts,
            step_traces,
        )

    try:
        first_response = selection_outcome.value
        assistant_message = first_response.choices[0].message
        tool_calls = assistant_message.tool_calls or []
    except Exception:
        step_traces.append(
            selection_timer.finish(
                status=StepStatus.ERROR,
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                error_type=ErrorType.TOOL_PROTOCOL,
                error_message="模型工具调用结构无效。",
            )
        )
        _skipped_remaining_steps(
            step_traces,
            "tool_execution",
            "tool_result_answer",
            reason="tool_protocol_error",
        )
        return _result(
            "模型工具调用结构无效，本次未执行计算。",
            [],
            selection_outcome.attempts,
            step_traces,
        )

    if len(tool_calls) == 0:
        step_traces.append(
            selection_timer.finish(
                status=StepStatus.ERROR,
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                error_type=ErrorType.TOOL_PROTOCOL,
                error_message="模型没有返回工具调用。",
            )
        )
        _skipped_remaining_steps(
            step_traces,
            "tool_execution",
            "tool_result_answer",
            reason="no_tool_call",
        )
        return _result(
            "模型没有返回工具调用，本次未执行计算。",
            [],
            selection_outcome.attempts,
            step_traces,
        )
    if len(tool_calls) != 1:
        step_traces.append(
            selection_timer.finish(
                status=StepStatus.ERROR,
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                error_type=ErrorType.TOOL_PROTOCOL,
                error_message="模型返回的工具调用数量不受支持。",
            )
        )
        _skipped_remaining_steps(
            step_traces,
            "tool_execution",
            "tool_result_answer",
            reason="multiple_tool_calls",
        )
        return _result(
            "模型返回了多个工具调用，本阶段不执行。",
            [],
            selection_outcome.attempts,
            step_traces,
        )

    step_traces.append(
        selection_timer.finish(
            status=_request_status(selection_outcome.attempts),
            attempts=selection_outcome.attempts,
            model_requests=selection_outcome.attempts,
        )
    )

    tool_call = tool_calls[0]
    execution_timer = StepTimer("tool_execution")
    try:
        record = execute_tool_call(
            tool_call_id=tool_call.id,
            name=tool_call.function.name,
            arguments_json=tool_call.function.arguments,
        )
    except Exception:
        step_traces.append(
            execution_timer.finish(
                status=StepStatus.ERROR,
                attempts=1,
                model_requests=0,
                error_type=ErrorType.TOOL_EXECUTION,
                error_message="本地工具执行失败。",
            )
        )
        step_traces.append(
            create_skipped_step(
                "tool_result_answer",
                {"reason": "tool_execution_failed"},
            )
        )
        return _result(
            "本地工具执行失败，请检查题目条件。",
            [],
            selection_outcome.attempts,
            step_traces,
        )

    record_data = record.model_dump(mode="json")
    tool_records = [record_data]
    if record.status is not ToolExecutionStatus.SUCCESS:
        error_type = _record_error_type(record.error)
        error_message = (
            "本地工具参数校验失败。"
            if error_type is ErrorType.TOOL_VALIDATION
            else "本地工具执行失败。"
        )
        step_traces.append(
            execution_timer.finish(
                status=StepStatus.ERROR,
                attempts=1,
                model_requests=0,
                error_type=error_type,
                error_message=error_message,
                metadata={"tool_record_count": 1},
            )
        )
        step_traces.append(
            create_skipped_step(
                "tool_result_answer",
                {"reason": "tool_record_error"},
            )
        )
        return _result(
            record.error or "本地工具执行失败，请检查题目条件。",
            tool_records,
            selection_outcome.attempts,
            step_traces,
        )

    step_traces.append(
        execution_timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
            metadata={"tool_record_count": 1},
        )
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

    result_messages = deepcopy(messages)
    result_timer = StepTimer("tool_result_answer")

    def request_result_answer() -> Any:
        return completion(
            **request_options,
            messages=deepcopy(result_messages),
            extra_body={"enable_thinking": False},
            stream=False,
        )

    result_outcome = run_with_one_retry(
        request_result_answer,
        retry_result,
        delay_seconds=0.0,
    )
    model_requests = selection_outcome.attempts + result_outcome.attempts
    if not result_outcome.succeeded:
        step_traces.append(
            result_timer.finish(
                status=StepStatus.ERROR,
                attempts=result_outcome.attempts,
                model_requests=result_outcome.attempts,
                error_type=ErrorType.TOOL_RESULT_API,
                error_message="工具结果回传模型调用失败。",
            )
        )
        return _result(
            "工具结果回传失败，请稍后重试。",
            tool_records,
            model_requests,
            step_traces,
        )

    try:
        second_response = result_outcome.value
        final_answer = second_response.choices[0].message.content
    except Exception:
        step_traces.append(
            result_timer.finish(
                status=StepStatus.ERROR,
                attempts=result_outcome.attempts,
                model_requests=result_outcome.attempts,
                error_type=ErrorType.TOOL_RESULT_API,
                error_message="工具结果回传响应结构无效。",
            )
        )
        return _result(
            "工具结果回传失败，请稍后重试。",
            tool_records,
            model_requests,
            step_traces,
        )

    if not final_answer or not final_answer.strip():
        step_traces.append(
            result_timer.finish(
                status=StepStatus.ERROR,
                attempts=result_outcome.attempts,
                model_requests=result_outcome.attempts,
                error_type=ErrorType.EMPTY_ANSWER,
                error_message="工具结果回传后模型回答为空。",
            )
        )
        return _result(
            "模型返回了空回答。",
            tool_records,
            model_requests,
            step_traces,
        )

    step_traces.append(
        result_timer.finish(
            status=_request_status(result_outcome.attempts),
            attempts=result_outcome.attempts,
            model_requests=result_outcome.attempts,
        )
    )
    return _result(
        final_answer.strip(),
        tool_records,
        model_requests,
        step_traces,
    )
