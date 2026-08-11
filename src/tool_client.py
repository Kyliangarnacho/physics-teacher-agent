"""Bounded local-physics Function Calling client."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from copy import deepcopy
from decimal import Decimal, InvalidOperation
from typing import Any

from openai import OpenAI

from src.config import load_qwen_config
from src.context.adapters import build_tool_context_view
from src.context.schemas import ContextBundle
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
MAX_TOOL_CALLS = 5
TOOL_SELECTION_INSTRUCTION = (
    "上下文仅用于理解当前用户问题的必要条件。"
    "历史中已经完成的计算、旧问题、旧工具需求不得视为本轮待执行任务。"
    "只为当前用户请求选择必要工具。"
    "不要因为历史上下文中出现多个可计算表达式而重复调用对应工具。"
    "若当前用户明确要求不计算、只解释，则不要调用计算工具。"
    "若历史讨论过 A、B、C 而当前只问其中一项，只处理当前明确指定的那一项。"
    "工具参数只能来自当前请求的明确条件或为当前指代恢复的必要条件；"
    "若存在多个合理解释或会实质改变题意，不要猜测参数或调用工具。"
    "学生写出的公式或概念不能仅因来自当前 user 就当作正确先验。"
    "先判断白名单工具是否能直接、可靠地核算用户要求的关键数值。"
    "若条件完整且有匹配工具，应主动调用；即使可以心算，也不要放弃有价值的确定性工具。"
    "若多个彼此独立的计算都被现有工具覆盖，可以一次返回多个 tool_calls。"
    "若只是概念、实验判断，或缺少所需公式/工具，则不要调用工具。"
    "不要因题目出现数字就调用，也不要为了调用而调用。"
)
_JSON_NUMBER_WITH_OPTIONAL_UNIT = re.compile(
    r"\s*(-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*(?:[A-Za-zΩμ³/]+)?\s*\Z"
)


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


def _build_tool_selection_messages(
    messages: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add a bounded tool-use policy without changing the final-answer prompt."""
    return [
        *deepcopy(messages[:-1]),
        {"role": "system", "content": TOOL_SELECTION_INSTRUCTION},
        deepcopy(messages[-1]),
    ]


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


def _as_decimal_fact(value: Any) -> Decimal | None:
    """Return a comparable numeric value without interpreting expressions."""
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float, Decimal)):
        try:
            return Decimal(str(value))
        except InvalidOperation:
            return None
    if isinstance(value, str):
        match = _JSON_NUMBER_WITH_OPTIONAL_UNIT.fullmatch(value)
        if match is not None:
            try:
                return Decimal(match.group(1))
            except InvalidOperation:
                return None
    return None


def _repair_preserves_known_facts(
    original_arguments_json: str,
    repaired_arguments: dict[str, Any],
) -> bool:
    """Reject repaired values that change facts already present in valid JSON."""
    try:
        original_arguments = json.loads(original_arguments_json)
    except (json.JSONDecodeError, TypeError):
        return True
    if not isinstance(original_arguments, dict):
        return True

    for field_name, original_value in original_arguments.items():
        if field_name not in repaired_arguments:
            continue
        repaired_value = repaired_arguments[field_name]
        original_number = _as_decimal_fact(original_value)
        repaired_number = _as_decimal_fact(repaired_value)
        if original_number is not None or repaired_number is not None:
            if original_number is None or repaired_number is None:
                return False
            if original_number != repaired_number:
                return False
        elif original_value != repaired_value:
            return False
    return True


def _build_repair_messages(
    messages: list[dict[str, Any]],
    candidates: list[tuple[Any, dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Build a bounded repair request without exposing a new planning surface."""
    candidate_data = [
        {
            "tool_call_id": tool_call.id,
            "name": tool_call.function.name,
            "arguments": tool_call.function.arguments,
        }
        for tool_call, _ in candidates
    ]
    repair_instruction = (
        "你只负责修复下列已选定工具调用的参数合同，不解题、不规划新工具。"
        "只能保留给出的 tool_call_id 和 name；不得增加、删除或替换调用。"
        "不得修改题目中的已知物理事实或数值；仅可修正 JSON 对象、字段缺失、"
        "多余字段、字段类型或取值范围问题。"
        "只返回 JSON：{\"repairs\":[{\"tool_call_id\":\"...\","
        "\"name\":\"...\",\"arguments\":{...}}]}。"
        "每个待修复调用必须恰好出现一次，arguments 必须是 JSON 对象。"
        "待修复调用："
        + json.dumps(candidate_data, ensure_ascii=False)
    )
    return [
        *deepcopy(messages[:-1]),
        {"role": "system", "content": repair_instruction},
        deepcopy(messages[-1]),
    ]


def _parse_repair_response(
    response: Any,
    candidates: list[tuple[Any, dict[str, Any]]],
) -> dict[str, str]:
    """Validate a one-round repair response for the original failed calls only."""
    try:
        content = response.choices[0].message.content
        payload = json.loads(content)
    except (AttributeError, IndexError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("repair response is not valid JSON") from error

    if not isinstance(payload, dict) or set(payload) != {"repairs"}:
        raise ValueError("repair response has an invalid top-level shape")
    repairs = payload["repairs"]
    if not isinstance(repairs, list) or len(repairs) != len(candidates):
        raise ValueError("repair response has an invalid repair count")

    original_by_id = {tool_call.id: tool_call for tool_call, _ in candidates}
    repaired_arguments_by_id: dict[str, str] = {}
    for repair in repairs:
        if not isinstance(repair, dict) or set(repair) != {
            "tool_call_id",
            "name",
            "arguments",
        }:
            raise ValueError("repair entry has an invalid shape")
        tool_call_id = repair["tool_call_id"]
        name = repair["name"]
        arguments = repair["arguments"]
        original_call = original_by_id.get(tool_call_id)
        if (
            not isinstance(tool_call_id, str)
            or not isinstance(name, str)
            or not isinstance(arguments, dict)
            or original_call is None
            or original_call.function.name != name
            or tool_call_id in repaired_arguments_by_id
            or not _repair_preserves_known_facts(
                original_call.function.arguments,
                arguments,
            )
        ):
            raise ValueError("repair entry is not allowed")
        repaired_arguments_by_id[tool_call_id] = json.dumps(
            arguments,
            ensure_ascii=False,
            allow_nan=False,
        )
    if set(repaired_arguments_by_id) != set(original_by_id):
        raise ValueError("repair response does not match all candidates")
    return repaired_arguments_by_id


def _run_final_answer(
    *,
    completion: CompletionFunc,
    request_options: dict[str, Any],
    messages: list[dict[str, Any]],
    retry_result: ShouldRetryFunc,
    model_requests_before: int,
    tool_records: list[dict[str, Any]],
    step_traces: list[StepTrace],
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Make exactly one final-answer operation (with the existing bounded retry)."""
    result_timer = StepTimer("tool_result_answer")

    def request_result_answer() -> Any:
        return completion(
            **request_options,
            messages=deepcopy(messages),
            extra_body={"enable_thinking": False},
            stream=False,
        )

    result_outcome = run_with_one_retry(
        request_result_answer,
        retry_result,
        delay_seconds=0.0,
    )
    model_requests = model_requests_before + result_outcome.attempts
    if not result_outcome.succeeded:
        step_traces.append(
            result_timer.finish(
                status=StepStatus.ERROR,
                attempts=result_outcome.attempts,
                model_requests=result_outcome.attempts,
                error_type=ErrorType.TOOL_RESULT_API,
                error_message="最终回答模型调用失败。",
                metadata=metadata,
            )
        )
        return _result(
            "最终回答生成失败，请稍后重试。",
            tool_records,
            model_requests,
            step_traces,
        )

    try:
        final_answer = result_outcome.value.choices[0].message.content
    except Exception:
        step_traces.append(
            result_timer.finish(
                status=StepStatus.ERROR,
                attempts=result_outcome.attempts,
                model_requests=result_outcome.attempts,
                error_type=ErrorType.TOOL_RESULT_API,
                error_message="最终回答响应结构无效。",
                metadata=metadata,
            )
        )
        return _result(
            "最终回答生成失败，请稍后重试。",
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
                error_message="最终回答为空。",
                metadata=metadata,
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
            metadata=metadata,
        )
    )
    return _result(
        final_answer.strip(),
        tool_records,
        model_requests,
        step_traces,
    )


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
    context_bundle_context: str | None = None,
    context_bundle: ContextBundle | Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute one to five selected tools, then request one final teacher answer."""
    has_explicit_projection = any(
        value is not None
        for value in (
            context_bundle_context,
            teaching_state_context,
            learning_memory_context,
            conversation_history,
        )
    )
    if context_bundle is not None and not has_explicit_projection:
        tool_inputs = build_tool_context_view(
            context_bundle,
            {"context_relation": "uncertain"},
        )
        context_bundle_context = tool_inputs["context_bundle_context"]
        teaching_state_context = tool_inputs["teaching_state_context"]
        learning_memory_context = tool_inputs["learning_memory_context"]
        conversation_history = tool_inputs["conversation_history"]
    kwargs: dict[str, object] = {}
    if teaching_state_context is not None:
        kwargs["teaching_state_context"] = teaching_state_context
    if learning_memory_context is not None:
        kwargs["learning_memory_context"] = learning_memory_context
    if conversation_history is not None:
        kwargs["conversation_history"] = conversation_history
    if context_bundle_context is not None:
        kwargs["context_bundle_context"] = context_bundle_context
    if context_bundle is not None:
        kwargs["context_bundle"] = context_bundle
    messages: list[dict[str, Any]] = list(
        build_messages(question, context, mode_instruction, **kwargs)
    )
    tool_messages = _build_tool_selection_messages(messages)
    request_options: dict[str, Any] = {}

    if completion_func is None:
        api_key, base_url, model = load_qwen_config()
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create
        request_options["model"] = model
    else:
        completion = completion_func

    step_traces: list[StepTrace] = []
    retry_selection = selection_should_retry or (lambda error: True)
    retry_result = result_should_retry or (lambda error: True)
    selection_timer = StepTimer("tool_selection")

    def request_tool_selection() -> Any:
        return completion(
            **request_options,
            messages=deepcopy(tool_messages),
            tools=get_openai_tools(),
            tool_choice="auto",
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
        assistant_message = selection_outcome.value.choices[0].message
        raw_tool_calls = getattr(assistant_message, "tool_calls", [])
        tool_calls = [] if raw_tool_calls is None else raw_tool_calls
    except Exception:
        assistant_message = None
        tool_calls = None

    protocol_reason: str | None = None
    if tool_calls is None:
        protocol_reason = "tool_protocol_error"
    elif len(tool_calls) > MAX_TOOL_CALLS:
        protocol_reason = "too_many_tool_calls"

    if tool_calls == []:
        selection_content = getattr(assistant_message, "content", None)
        selection_metadata = {
            "decision": "no_tool_needed",
            "answer_strategy": "model_only",
        }
        step_traces.append(
            selection_timer.finish(
                status=_request_status(selection_outcome.attempts),
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                metadata=selection_metadata,
            )
        )
        step_traces.append(
            create_skipped_step(
                "tool_execution",
                {"reason": "no_tool_needed"},
            )
        )
        if isinstance(selection_content, str) and selection_content.strip():
            step_traces.append(
                create_skipped_step(
                    "tool_result_answer",
                    {
                        "reason": "selection_answer_reused",
                        "answer_strategy": "model_only",
                    },
                )
            )
            return _result(
                selection_content.strip(),
                [],
                selection_outcome.attempts,
                step_traces,
            )
        return _run_final_answer(
            completion=completion,
            request_options=request_options,
            messages=tool_messages,
            retry_result=retry_result,
            model_requests_before=selection_outcome.attempts,
            tool_records=[],
            step_traces=step_traces,
            metadata={
                "reason": "no_tool_needed",
                "answer_strategy": "model_only",
            },
        )

    if protocol_reason is not None:
        step_traces.append(
            selection_timer.finish(
                status=StepStatus.ERROR,
                attempts=selection_outcome.attempts,
                model_requests=selection_outcome.attempts,
                error_type=ErrorType.TOOL_PROTOCOL,
                error_message="模型工具调用协议无效。",
                metadata={"reason": protocol_reason},
            )
        )
        step_traces.append(
            create_skipped_step("tool_execution", {"reason": protocol_reason})
        )
        return _run_final_answer(
            completion=completion,
            request_options=request_options,
            messages=tool_messages,
            retry_result=retry_result,
            model_requests_before=selection_outcome.attempts,
            tool_records=[],
            step_traces=step_traces,
            metadata={"fallback": "model_only", "reason": protocol_reason},
        )

    step_traces.append(
        selection_timer.finish(
            status=_request_status(selection_outcome.attempts),
            attempts=selection_outcome.attempts,
            model_requests=selection_outcome.attempts,
        )
    )

    execution_timer = StepTimer("tool_execution")
    tool_records: list[dict[str, Any]] = []
    success_count = 0
    first_error_type: ErrorType | None = None

    for tool_call in tool_calls:
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
                    metadata={"tool_record_count": len(tool_records)},
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
                tool_records,
                selection_outcome.attempts,
                step_traces,
            )

        record_data = record.model_dump(mode="json")
        tool_records.append(record_data)
        if record.status is ToolExecutionStatus.SUCCESS:
            success_count += 1
        elif first_error_type is None:
            first_error_type = _record_error_type(record.error)

    final_records_by_id = {
        record["tool_call_id"]: record for record in tool_records
    }
    repair_candidates = [
        (tool_call, final_records_by_id[tool_call.id])
        for tool_call in tool_calls
        if final_records_by_id[tool_call.id]["status"] == "error"
        and _record_error_type(final_records_by_id[tool_call.id].get("error"))
        is ErrorType.TOOL_VALIDATION
    ]
    repair_model_requests = 0
    repair_success_count = 0
    repair_attempted = bool(repair_candidates)

    if repair_candidates:
        repair_timer = StepTimer("tool_repair")
        repair_model_requests = 1
        try:
            repair_response = completion(
                **request_options,
                messages=_build_repair_messages(
                    tool_messages,
                    repair_candidates,
                ),
                extra_body={"enable_thinking": False},
                stream=False,
            )
            repaired_arguments_by_id = _parse_repair_response(
                repair_response,
                repair_candidates,
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            step_traces.append(
                repair_timer.finish(
                    status=StepStatus.ERROR,
                    attempts=1,
                    model_requests=1,
                    error_type=ErrorType.TOOL_PROTOCOL,
                    error_message="工具参数修复响应无效。",
                    metadata={
                        "repair_requested": 1,
                        "repair_candidate_count": len(repair_candidates),
                        "repaired_success_count": 0,
                    },
                )
            )
        except Exception:
            step_traces.append(
                repair_timer.finish(
                    status=StepStatus.ERROR,
                    attempts=1,
                    model_requests=1,
                    error_type=ErrorType.TOOL_REPAIR_API,
                    error_message="工具参数修复模型调用失败。",
                    metadata={
                        "repair_requested": 1,
                        "repair_candidate_count": len(repair_candidates),
                        "repaired_success_count": 0,
                    },
                )
            )
        else:
            for tool_call, _ in repair_candidates:
                repaired_record = execute_tool_call(
                    tool_call_id=tool_call.id,
                    name=tool_call.function.name,
                    arguments_json=repaired_arguments_by_id[tool_call.id],
                )
                repaired_record_data = repaired_record.model_dump(mode="json")
                tool_records.append(repaired_record_data)
                final_records_by_id[tool_call.id] = repaired_record_data
                if repaired_record.status is ToolExecutionStatus.SUCCESS:
                    success_count += 1
                    repair_success_count += 1

            step_traces.append(
                repair_timer.finish(
                    status=StepStatus.SUCCESS,
                    attempts=1,
                    model_requests=1,
                    metadata={
                        "repair_requested": 1,
                        "repair_candidate_count": len(repair_candidates),
                        "repaired_success_count": repair_success_count,
                        "final_failed_count": len(tool_calls) - success_count,
                    },
                )
            )

    error_count = len(tool_records) - success_count
    execution_metadata = {
        "tool_record_count": len(tool_records),
        "success_count": success_count,
        "error_count": error_count,
        "repair_attempted": repair_attempted,
        "repair_model_requests": repair_model_requests,
        "repair_candidate_count": len(repair_candidates),
        "repaired_success_count": repair_success_count,
        "final_failed_count": len(tool_calls) - success_count,
    }
    if success_count == 0:
        step_traces.append(
            execution_timer.finish(
                status=StepStatus.ERROR,
                attempts=1,
                model_requests=0,
                error_type=first_error_type or ErrorType.TOOL_EXECUTION,
                error_message="所有本地工具均未成功执行。",
                metadata=execution_metadata,
            )
        )
        return _run_final_answer(
            completion=completion,
            request_options=request_options,
            messages=tool_messages,
            retry_result=retry_result,
            model_requests_before=(
                selection_outcome.attempts + repair_model_requests
            ),
            tool_records=tool_records,
            step_traces=step_traces,
            metadata={
                **execution_metadata,
                "fallback": "model_only",
                "reason": "all_tools_failed",
            },
        )

    step_traces.append(
        execution_timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
            metadata=execution_metadata,
        )
    )
    tool_messages.append(
        {
            "role": "assistant",
            "content": assistant_message.content,
            "tool_calls": [_tool_call_payload(tool_call) for tool_call in tool_calls],
        }
    )
    for tool_call in tool_calls:
        record_data = final_records_by_id[tool_call.id]
        tool_messages.append(
            {
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(record_data, ensure_ascii=False),
            }
        )

    return _run_final_answer(
        completion=completion,
        request_options=request_options,
        messages=tool_messages,
        retry_result=retry_result,
        model_requests_before=(
            selection_outcome.attempts + repair_model_requests
        ),
        tool_records=tool_records,
        step_traces=step_traces,
        metadata=execution_metadata,
    )
