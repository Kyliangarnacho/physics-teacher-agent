"""使用 Decimal 实现的 Stage 07 纯本地物理计算函数。"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any


CalculationResult = dict[str, str]


UNIT_CATEGORIES: dict[str, dict[str, Decimal]] = {
    "长度": {
        "m": Decimal("1"),
        "cm": Decimal("0.01"),
        "mm": Decimal("0.001"),
        "km": Decimal("1000"),
    },
    "时间": {
        "s": Decimal("1"),
        "min": Decimal("60"),
        "h": Decimal("3600"),
    },
    "速度": {
        "m/s": Decimal("1"),
        "km/h": Decimal("5") / Decimal("18"),
    },
    "电流": {
        "A": Decimal("1"),
        "mA": Decimal("0.001"),
    },
    "电阻": {
        "Ω": Decimal("1"),
        "kΩ": Decimal("1000"),
    },
    "功率": {
        "W": Decimal("1"),
        "kW": Decimal("1000"),
    },
    "质量": {
        "kg": Decimal("1"),
        "g": Decimal("0.001"),
    },
    "体积": {
        "m³": Decimal("1"),
        "cm³": Decimal("0.000001"),
    },
}


def _to_decimal(value: Any, field_name: str) -> Decimal:
    """按 Decimal(str(value)) 解析有限数值，并统一转换异常。"""
    try:
        number = Decimal(str(value))
    except (InvalidOperation, TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} 必须是有效数值。") from exc
    if not number.is_finite():
        raise ValueError(f"{field_name} 必须是有限数值。")
    return number


def _display_decimal(value: Decimal) -> str:
    """使用普通十进制表示，并移除无意义的小数尾零。"""
    displayed = format(value, "f")
    if "." in displayed:
        displayed = displayed.rstrip("0").rstrip(".")
    return "0" if displayed in {"-0", ""} else displayed


def _build_result(
    formula: str,
    result: Decimal,
    unit: str,
) -> CalculationResult:
    return {
        "formula": formula,
        "raw_result": str(result),
        "display_value": _display_decimal(result),
        "unit": unit,
    }


def _require_non_negative(value: Decimal, field_name: str) -> None:
    if value < 0:
        raise ValueError(f"{field_name}不得为负数。")


def calculate_average_speed(distance_m: Any, time_s: Any) -> CalculationResult:
    """根据路程和时间计算平均速度，单位为 m/s。"""
    distance = _to_decimal(distance_m, "路程")
    time = _to_decimal(time_s, "时间")
    _require_non_negative(distance, "路程")
    if time <= 0:
        raise ValueError("时间必须大于 0。")
    return _build_result("v = s / t", distance / time, "m/s")


def calculate_density(mass_kg: Any, volume_m3: Any) -> CalculationResult:
    """根据质量和体积计算密度，单位为 kg/m³。"""
    mass = _to_decimal(mass_kg, "质量")
    volume = _to_decimal(volume_m3, "体积")
    _require_non_negative(mass, "质量")
    if volume <= 0:
        raise ValueError("体积必须大于 0。")
    return _build_result("ρ = m / V", mass / volume, "kg/m³")


def calculate_mechanical_power(work_j: Any, time_s: Any) -> CalculationResult:
    """根据功和时间计算机械功率，单位为 W。"""
    work = _to_decimal(work_j, "功")
    time = _to_decimal(time_s, "时间")
    _require_non_negative(work, "功")
    if time <= 0:
        raise ValueError("时间必须大于 0。")
    return _build_result("P = W / t", work / time, "W")


def calculate_pulley_efficiency(
    weight_n: Any,
    height_m: Any,
    force_n: Any,
    distance_m: Any,
) -> CalculationResult:
    """按已给定的有用功与总功计算滑轮效率，不推导绳端移动距离。"""
    weight = _to_decimal(weight_n, "物重")
    height = _to_decimal(height_m, "提升高度")
    force = _to_decimal(force_n, "拉力")
    distance = _to_decimal(distance_m, "绳端移动距离")
    _require_non_negative(weight, "物重")
    _require_non_negative(height, "提升高度")
    if force <= 0:
        raise ValueError("拉力必须大于 0。")
    if distance <= 0:
        raise ValueError("绳端移动距离必须大于 0。")

    efficiency_ratio = weight * height / (force * distance)
    if efficiency_ratio > Decimal("1"):
        raise ValueError("滑轮效率超过 100%，请检查物理条件。")
    return _build_result(
        "η = Gh / (F × s) × 100%",
        efficiency_ratio * Decimal("100"),
        "%",
    )


def calculate_ohms_law(
    voltage_v: Any = None,
    current_a: Any = None,
    resistance_ohm: Any = None,
) -> CalculationResult:
    """恰好使用欧姆定律的两个已知量计算第三个量。"""
    provided = {
        "电压": voltage_v,
        "电流": current_a,
        "电阻": resistance_ohm,
    }
    if sum(value is not None for value in provided.values()) != 2:
        raise ValueError("电压、电流和电阻必须恰好提供两个。")

    values = {
        name: _to_decimal(value, name)
        for name, value in provided.items()
        if value is not None
    }
    for name, value in values.items():
        _require_non_negative(value, name)

    if voltage_v is None:
        result = values["电流"] * values["电阻"]
        return _build_result("U = I × R", result, "V")
    if current_a is None:
        resistance = values["电阻"]
        if resistance <= 0:
            raise ValueError("计算电流时，作为分母的电阻必须大于 0。")
        return _build_result("I = U / R", values["电压"] / resistance, "A")

    current = values["电流"]
    if current <= 0:
        raise ValueError("计算电阻时，作为分母的电流必须大于 0。")
    return _build_result("R = U / I", values["电压"] / current, "Ω")


def calculate_electric_power(
    voltage_v: Any = None,
    current_a: Any = None,
    resistance_ohm: Any = None,
) -> CalculationResult:
    """根据恰好两个已知电学量选择公式计算电功率。"""
    provided = {
        "电压": voltage_v,
        "电流": current_a,
        "电阻": resistance_ohm,
    }
    if sum(value is not None for value in provided.values()) != 2:
        raise ValueError("电压、电流和电阻必须恰好提供两个。")

    values = {
        name: _to_decimal(value, name)
        for name, value in provided.items()
        if value is not None
    }
    for name, value in values.items():
        _require_non_negative(value, name)

    if resistance_ohm is None:
        result = values["电压"] * values["电流"]
        return _build_result("P = U × I", result, "W")
    if voltage_v is None:
        result = values["电流"] ** 2 * values["电阻"]
        return _build_result("P = I² × R", result, "W")

    resistance = values["电阻"]
    if resistance <= 0:
        raise ValueError("使用 P = U² / R 时，作为分母的电阻必须大于 0。")
    result = values["电压"] ** 2 / resistance
    return _build_result("P = U² / R", result, "W")


def _find_unit_category(unit: str) -> tuple[str, Decimal] | None:
    for category, units in UNIT_CATEGORIES.items():
        if unit in units:
            return category, units[unit]
    return None


def convert_physics_unit(
    value: Any,
    from_unit: str,
    to_unit: str,
) -> CalculationResult:
    """在受支持的同一物理量类别内进行单位换算。"""
    if not isinstance(from_unit, str) or not from_unit.strip():
        raise ValueError("原单位必须是受支持的非空单位字符串。")
    if not isinstance(to_unit, str) or not to_unit.strip():
        raise ValueError("目标单位必须是受支持的非空单位字符串。")

    normalized_from = from_unit.strip()
    normalized_to = to_unit.strip()
    from_info = _find_unit_category(normalized_from)
    to_info = _find_unit_category(normalized_to)
    if from_info is None:
        raise ValueError(f"不支持的原单位：{normalized_from}。")
    if to_info is None:
        raise ValueError(f"不支持的目标单位：{normalized_to}。")
    if from_info[0] != to_info[0]:
        raise ValueError(
            f"不能在不同物理量类别之间换算：{normalized_from} → {normalized_to}。"
        )

    number = _to_decimal(value, "换算数值")
    result = number * from_info[1] / to_info[1]
    return _build_result(
        f"{normalized_from} → {normalized_to}",
        result,
        normalized_to,
    )
