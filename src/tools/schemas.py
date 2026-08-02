"""Pydantic parameter contracts for the local physics calculators."""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    WithJsonSchema,
    model_validator,
)


def _finite_decimal(value: object) -> Decimal:
    """Convert a real numeric input to a finite Decimal without coercing strings."""
    if isinstance(value, bool) or not isinstance(value, (int, float, Decimal)):
        raise ValueError("必须提供数值，不能使用字符串、布尔值或其他类型。")

    try:
        number = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError("必须提供有效数值。") from exc

    if not number.is_finite():
        raise ValueError("数值必须是有限值，不能使用 NaN 或无穷值。")
    return number


FiniteNumber = Annotated[
    Decimal,
    BeforeValidator(_finite_decimal),
    WithJsonSchema({"type": "number"}),
]
NonNegativeNumber = Annotated[
    Decimal,
    BeforeValidator(_finite_decimal),
    Field(ge=0),
    WithJsonSchema({"type": "number", "minimum": 0}),
]
PositiveNumber = Annotated[
    Decimal,
    BeforeValidator(_finite_decimal),
    Field(gt=0),
    WithJsonSchema({"type": "number", "exclusiveMinimum": 0}),
]

SupportedUnit = Literal[
    "m",
    "cm",
    "mm",
    "km",
    "s",
    "min",
    "h",
    "m/s",
    "km/h",
    "A",
    "mA",
    "Ω",
    "kΩ",
    "W",
    "kW",
    "kg",
    "g",
    "m³",
    "cm³",
]

_UNIT_CATEGORY_BY_UNIT = {
    "m": "length",
    "cm": "length",
    "mm": "length",
    "km": "length",
    "s": "time",
    "min": "time",
    "h": "time",
    "m/s": "speed",
    "km/h": "speed",
    "A": "current",
    "mA": "current",
    "Ω": "resistance",
    "kΩ": "resistance",
    "W": "power",
    "kW": "power",
    "kg": "mass",
    "g": "mass",
    "m³": "volume",
    "cm³": "volume",
}


class _ToolParameters(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AverageSpeedParameters(_ToolParameters):
    """Parameters for average-speed calculation."""

    distance_m: NonNegativeNumber
    time_s: PositiveNumber


class DensityParameters(_ToolParameters):
    """Parameters for density calculation."""

    mass_kg: NonNegativeNumber
    volume_m3: PositiveNumber


class OhmsLawParameters(_ToolParameters):
    """Exactly two known quantities for Ohm's law."""

    voltage_v: NonNegativeNumber | None = None
    current_a: NonNegativeNumber | None = None
    resistance_ohm: NonNegativeNumber | None = None

    @model_validator(mode="after")
    def validate_known_quantities(self) -> OhmsLawParameters:
        provided = sum(
            value is not None
            for value in (self.voltage_v, self.current_a, self.resistance_ohm)
        )
        if provided != 2:
            raise ValueError("电压、电流和电阻必须恰好提供两个。")

        if self.current_a is None and self.resistance_ohm == 0:
            raise ValueError("求电流时，已知电阻必须大于 0。")
        if self.resistance_ohm is None and self.current_a == 0:
            raise ValueError("求电阻时，已知电流必须大于 0。")
        return self


class ElectricPowerParameters(_ToolParameters):
    """Exactly two known quantities for electric-power calculation."""

    voltage_v: NonNegativeNumber | None = None
    current_a: NonNegativeNumber | None = None
    resistance_ohm: NonNegativeNumber | None = None

    @model_validator(mode="after")
    def validate_known_quantities(self) -> ElectricPowerParameters:
        provided = sum(
            value is not None
            for value in (self.voltage_v, self.current_a, self.resistance_ohm)
        )
        if provided != 2:
            raise ValueError("电压、电流和电阻必须恰好提供两个。")

        if self.current_a is None and self.resistance_ohm == 0:
            raise ValueError("使用 U²/R 计算电功率时，电阻必须大于 0。")
        return self


class UnitConversionParameters(_ToolParameters):
    """Parameters for conversion within one supported physical-quantity category."""

    value: FiniteNumber
    from_unit: SupportedUnit
    to_unit: SupportedUnit

    @model_validator(mode="after")
    def validate_same_category(self) -> UnitConversionParameters:
        if _UNIT_CATEGORY_BY_UNIT[self.from_unit] != _UNIT_CATEGORY_BY_UNIT[self.to_unit]:
            raise ValueError("只能在同一物理量类别内进行单位换算。")
        return self
