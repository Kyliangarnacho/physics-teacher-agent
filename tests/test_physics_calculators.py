"""Stage 07 纯本地物理计算工具测试。"""

from __future__ import annotations

import json
import unittest
from decimal import Decimal

from src.tools.physics_calculators import (
    calculate_average_speed,
    calculate_density,
    calculate_electric_power,
    calculate_ohms_law,
    convert_physics_unit,
)


EXPECTED_RESULT_FIELDS = {
    "formula",
    "raw_result",
    "display_value",
    "unit",
}


class AverageSpeedTests(unittest.TestCase):
    def test_calculates_average_speed(self) -> None:
        result = calculate_average_speed(50, 10)

        self.assertEqual(result["formula"], "v = s / t")
        self.assertEqual(result["display_value"], "5")
        self.assertEqual(result["unit"], "m/s")

    def test_rejects_zero_or_negative_time(self) -> None:
        for time in (0, -1):
            with self.subTest(time=time):
                with self.assertRaisesRegex(ValueError, "时间必须大于 0"):
                    calculate_average_speed(50, time)

    def test_rejects_negative_distance(self) -> None:
        with self.assertRaisesRegex(ValueError, "路程不得为负数"):
            calculate_average_speed(-1, 2)


class DensityTests(unittest.TestCase):
    def test_calculates_density(self) -> None:
        result = calculate_density("2.4", "0.003")

        self.assertEqual(result["formula"], "ρ = m / V")
        self.assertEqual(result["display_value"], "800")
        self.assertEqual(result["unit"], "kg/m³")

    def test_rejects_zero_or_negative_volume(self) -> None:
        for volume in (0, -1):
            with self.subTest(volume=volume):
                with self.assertRaisesRegex(ValueError, "体积必须大于 0"):
                    calculate_density(2, volume)

    def test_rejects_negative_mass(self) -> None:
        with self.assertRaisesRegex(ValueError, "质量不得为负数"):
            calculate_density(-1, 1)


class OhmsLawTests(unittest.TestCase):
    def test_calculates_current(self) -> None:
        result = calculate_ohms_law(voltage_v=12, resistance_ohm=4)

        self.assertEqual(result["formula"], "I = U / R")
        self.assertEqual(result["display_value"], "3")
        self.assertEqual(result["unit"], "A")

    def test_calculates_voltage(self) -> None:
        result = calculate_ohms_law(current_a="1.5", resistance_ohm=8)

        self.assertEqual(result["formula"], "U = I × R")
        self.assertEqual(result["display_value"], "12")
        self.assertEqual(result["unit"], "V")

    def test_calculates_resistance(self) -> None:
        result = calculate_ohms_law(voltage_v=12, current_a="0.5")

        self.assertEqual(result["formula"], "R = U / I")
        self.assertEqual(result["display_value"], "24")
        self.assertEqual(result["unit"], "Ω")

    def test_requires_exactly_two_known_values(self) -> None:
        invalid_arguments = (
            {"voltage_v": 12},
            {"voltage_v": 12, "current_a": 2, "resistance_ohm": 6},
        )
        for arguments in invalid_arguments:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "恰好提供两个"):
                    calculate_ohms_law(**arguments)

    def test_rejects_negative_known_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "电流不得为负数"):
            calculate_ohms_law(current_a=-1, resistance_ohm=2)

    def test_rejects_zero_denominators(self) -> None:
        cases = (
            {"voltage_v": 12, "resistance_ohm": 0},
            {"voltage_v": 12, "current_a": 0},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "分母"):
                    calculate_ohms_law(**arguments)


class ElectricPowerTests(unittest.TestCase):
    def test_calculates_power_with_voltage_and_current(self) -> None:
        result = calculate_electric_power(voltage_v=12, current_a=2)

        self.assertEqual(result["formula"], "P = U × I")
        self.assertEqual(result["display_value"], "24")

    def test_calculates_power_with_current_and_resistance(self) -> None:
        result = calculate_electric_power(current_a=2, resistance_ohm=3)

        self.assertEqual(result["formula"], "P = I² × R")
        self.assertEqual(result["display_value"], "12")

    def test_calculates_power_with_voltage_and_resistance(self) -> None:
        result = calculate_electric_power(voltage_v=12, resistance_ohm=6)

        self.assertEqual(result["formula"], "P = U² / R")
        self.assertEqual(result["display_value"], "24")

    def test_requires_exactly_two_known_values(self) -> None:
        for arguments in ({"voltage_v": 12}, {}):
            with self.subTest(arguments=arguments):
                with self.assertRaisesRegex(ValueError, "恰好提供两个"):
                    calculate_electric_power(**arguments)

    def test_rejects_negative_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "电压不得为负数"):
            calculate_electric_power(voltage_v=-12, current_a=2)

    def test_rejects_zero_resistance_as_denominator(self) -> None:
        with self.assertRaisesRegex(ValueError, "分母"):
            calculate_electric_power(voltage_v=12, resistance_ohm=0)


class UnitConversionTests(unittest.TestCase):
    def test_converts_representative_units_for_each_category(self) -> None:
        cases = (
            (100, "cm", "m", "1"),
            (2, "min", "s", "120"),
            (1500, "mA", "A", "1.5"),
            (2, "kΩ", "Ω", "2000"),
            (1.5, "kW", "W", "1500"),
            (2500, "g", "kg", "2.5"),
        )
        for value, from_unit, to_unit, expected in cases:
            with self.subTest(from_unit=from_unit, to_unit=to_unit):
                result = convert_physics_unit(value, from_unit, to_unit)
                self.assertEqual(result["display_value"], expected)
                self.assertEqual(result["unit"], to_unit)

    def test_converts_speed_in_both_directions(self) -> None:
        self.assertEqual(
            convert_physics_unit(10, "m/s", "km/h")["display_value"],
            "36",
        )
        self.assertEqual(
            convert_physics_unit(36, "km/h", "m/s")["display_value"],
            "10",
        )

    def test_converts_volume_in_both_directions(self) -> None:
        self.assertEqual(
            convert_physics_unit(1, "m³", "cm³")["display_value"],
            "1000000",
        )
        self.assertEqual(
            convert_physics_unit(1, "cm³", "m³")["display_value"],
            "0.000001",
        )

    def test_rejects_unsupported_units(self) -> None:
        for from_unit, to_unit in (("ft", "m"), ("m", "ft")):
            with self.subTest(from_unit=from_unit, to_unit=to_unit):
                with self.assertRaisesRegex(ValueError, "不支持"):
                    convert_physics_unit(1, from_unit, to_unit)

    def test_rejects_cross_category_conversion(self) -> None:
        with self.assertRaisesRegex(ValueError, "不同物理量类别"):
            convert_physics_unit(1, "m", "s")


class DecimalAndResultFormatTests(unittest.TestCase):
    def test_decimal_inputs_remove_only_display_trailing_zeros(self) -> None:
        result = convert_physics_unit("1200.0", "mA", "A")

        self.assertEqual(result["raw_result"], "1.2000")
        self.assertEqual(result["display_value"], "1.2")

    def test_decimal_instance_and_repeated_calls_are_consistent(self) -> None:
        first = calculate_average_speed(Decimal("12.00"), Decimal("4.0"))
        second = calculate_average_speed(Decimal("12.00"), Decimal("4.0"))

        self.assertEqual(first, second)
        self.assertEqual(first["display_value"], "3")

    def test_result_is_json_serializable_and_uses_string_values(self) -> None:
        result = calculate_density("2.50", "0.50")

        self.assertEqual(set(result), EXPECTED_RESULT_FIELDS)
        self.assertTrue(all(isinstance(value, str) for value in result.values()))
        self.assertIsInstance(json.dumps(result, ensure_ascii=False), str)

    def test_rejects_non_finite_or_invalid_numbers(self) -> None:
        for value in ("NaN", "Infinity", "not-a-number"):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "数值"):
                    calculate_average_speed(value, 1)


if __name__ == "__main__":
    unittest.main()
