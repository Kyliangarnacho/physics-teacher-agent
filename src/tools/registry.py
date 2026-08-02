"""Explicit whitelist registry for the Stage 07 local physics tools."""

from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from decimal import Decimal
from enum import Enum
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from src.tools.physics_calculators import (
    calculate_average_speed,
    calculate_density,
    calculate_electric_power,
    calculate_ohms_law,
    convert_physics_unit,
)
from src.tools.schemas import (
    AverageSpeedParameters,
    DensityParameters,
    ElectricPowerParameters,
    OhmsLawParameters,
    UnitConversionParameters,
)


class ToolExecutionStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"


class ToolExecutionRecord(BaseModel):
    """JSON-serializable outcome of one local tool execution."""

    model_config = ConfigDict(extra="forbid")

    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    normalized_fields: list[str] = Field(default_factory=list)
    status: ToolExecutionStatus
    result: dict[str, Any] | None = None
    error: str | None = None

    @model_validator(mode="after")
    def validate_outcome(self) -> ToolExecutionRecord:
        if self.status is ToolExecutionStatus.SUCCESS:
            if self.result is None or self.error is not None:
                raise ValueError("成功记录必须包含结果且不能包含错误。")
        elif self.result is not None or not self.error:
            raise ValueError("失败记录必须包含错误摘要且不能包含结果。")
        return self


ToolFunction = Callable[..., dict[str, str]]


@dataclass(frozen=True)
class _ToolRegistration:
    name: str
    description: str
    parameters_model: type[BaseModel]
    function: ToolFunction
    numeric_fields: tuple[str, ...]


_TOOL_REGISTRY: dict[str, _ToolRegistration] = {
    "calculate_average_speed": _ToolRegistration(
        name="calculate_average_speed",
        description="根据路程和时间计算平均速度。",
        parameters_model=AverageSpeedParameters,
        function=calculate_average_speed,
        numeric_fields=("distance_m", "time_s"),
    ),
    "calculate_density": _ToolRegistration(
        name="calculate_density",
        description="根据质量和体积计算物质密度。",
        parameters_model=DensityParameters,
        function=calculate_density,
        numeric_fields=("mass_kg", "volume_m3"),
    ),
    "calculate_ohms_law": _ToolRegistration(
        name="calculate_ohms_law",
        description="根据欧姆定律的两个已知量计算第三个量。",
        parameters_model=OhmsLawParameters,
        function=calculate_ohms_law,
        numeric_fields=("voltage_v", "current_a", "resistance_ohm"),
    ),
    "calculate_electric_power": _ToolRegistration(
        name="calculate_electric_power",
        description="根据电压、电流或电阻中的两个已知量计算电功率。",
        parameters_model=ElectricPowerParameters,
        function=calculate_electric_power,
        numeric_fields=("voltage_v", "current_a", "resistance_ohm"),
    ),
    "convert_physics_unit": _ToolRegistration(
        name="convert_physics_unit",
        description="在受支持的同类物理单位之间进行换算。",
        parameters_model=UnitConversionParameters,
        function=convert_physics_unit,
        numeric_fields=("value",),
    ),
}


def get_openai_tools() -> list[dict]:
    """Return isolated OpenAI Function Calling definitions for all whitelisted tools."""
    definitions = [
        {
            "type": "function",
            "function": {
                "name": registration.name,
                "description": registration.description,
                "parameters": registration.parameters_model.model_json_schema(),
            },
        }
        for registration in _TOOL_REGISTRY.values()
    ]
    return copy.deepcopy(definitions)


def _error_record(
    tool_call_id: str,
    name: str,
    error: str,
    arguments: dict[str, Any] | None = None,
    normalized_fields: list[str] | None = None,
) -> ToolExecutionRecord:
    return ToolExecutionRecord(
        tool_call_id=tool_call_id,
        name=name,
        arguments=arguments or {},
        normalized_fields=normalized_fields or [],
        status=ToolExecutionStatus.ERROR,
        result=None,
        error=error,
    )


def _safe_record_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    """Keep error-record arguments only when they are strict JSON data."""
    try:
        json.dumps(arguments, ensure_ascii=False, allow_nan=False)
    except (TypeError, ValueError):
        return {}
    return arguments


_JSON_NUMBER_PATTERN = re.compile(
    r"-?(?:0|[1-9]\d*)(?:\.\d+)?(?:[eE][+-]?\d+)?\Z"
)


def _normalize_numeric_arguments(
    arguments: dict[str, Any],
    numeric_fields: tuple[str, ...],
) -> tuple[dict[str, Any], list[str]]:
    """Normalize only registered, pure JSON-number strings to Decimal."""
    normalized = dict(arguments)
    normalized_fields: list[str] = []
    for field_name in numeric_fields:
        value = arguments.get(field_name)
        if isinstance(value, str) and _JSON_NUMBER_PATTERN.fullmatch(value):
            normalized[field_name] = Decimal(value)
            normalized_fields.append(field_name)
    return normalized, normalized_fields


def execute_tool_call(
    tool_call_id: str,
    name: str,
    arguments_json: str,
) -> ToolExecutionRecord:
    """Validate and execute one explicitly registered local tool call."""
    registration = _TOOL_REGISTRY.get(name)
    if registration is None:
        return _error_record(tool_call_id, name, "未知工具，无法执行。")

    try:
        parsed_arguments = json.loads(arguments_json)
    except (json.JSONDecodeError, TypeError):
        return _error_record(tool_call_id, name, "工具参数不是合法的 JSON。")

    if not isinstance(parsed_arguments, dict):
        return _error_record(tool_call_id, name, "工具参数必须是 JSON 对象。")

    normalized_arguments, normalized_fields = _normalize_numeric_arguments(
        parsed_arguments,
        registration.numeric_fields,
    )
    try:
        validated = registration.parameters_model.model_validate(normalized_arguments)
    except ValidationError:
        return _error_record(
            tool_call_id,
            name,
            "工具参数校验失败，请检查必填字段、字段类型和取值范围。",
            _safe_record_arguments(parsed_arguments),
            normalized_fields,
        )

    execution_arguments = validated.model_dump()
    record_arguments = validated.model_dump(mode="json")
    try:
        result = registration.function(**execution_arguments)
        json.dumps(result, ensure_ascii=False, allow_nan=False)
    except ValueError:
        return _error_record(
            tool_call_id,
            name,
            "工具计算失败，参数不满足计算要求。",
            record_arguments,
            normalized_fields,
        )
    except Exception:
        return _error_record(
            tool_call_id,
            name,
            "工具执行失败，请检查参数后重试。",
            record_arguments,
            normalized_fields,
        )

    return ToolExecutionRecord(
        tool_call_id=tool_call_id,
        name=name,
        arguments=record_arguments,
        normalized_fields=normalized_fields,
        status=ToolExecutionStatus.SUCCESS,
        result=result,
        error=None,
    )
