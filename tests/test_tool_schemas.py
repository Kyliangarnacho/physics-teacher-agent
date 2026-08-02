import json
import unittest
from decimal import Decimal

from pydantic import ValidationError

from src.tools.schemas import (
    AverageSpeedParameters,
    DensityParameters,
    ElectricPowerParameters,
    OhmsLawParameters,
    UnitConversionParameters,
)


class ToolSchemaValidationTests(unittest.TestCase):
    def test_five_parameter_models_accept_valid_inputs(self):
        self.assertEqual(
            AverageSpeedParameters(distance_m=50, time_s=10).distance_m,
            Decimal("50"),
        )
        self.assertEqual(
            DensityParameters(mass_kg=2.7, volume_m3=0.001).mass_kg,
            Decimal("2.7"),
        )
        self.assertIsNone(
            OhmsLawParameters(voltage_v=6, resistance_ohm=3).current_a
        )
        self.assertIsNone(
            ElectricPowerParameters(voltage_v=220, current_a=1).resistance_ohm
        )
        self.assertEqual(
            UnitConversionParameters(value=1.5, from_unit="m", to_unit="cm").to_unit,
            "cm",
        )

    def test_extra_fields_are_forbidden(self):
        with self.assertRaises(ValidationError):
            AverageSpeedParameters(distance_m=10, time_s=2, extra_value=1)

    def test_numeric_strings_booleans_and_wrong_types_are_rejected(self):
        invalid_cases = (
            {"distance_m": "10", "time_s": 2},
            {"distance_m": True, "time_s": 2},
            {"distance_m": [10], "time_s": 2},
        )
        for parameters in invalid_cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    AverageSpeedParameters(**parameters)

    def test_nan_and_infinity_are_rejected(self):
        for value in (float("nan"), float("inf"), float("-inf"), Decimal("NaN")):
            with self.subTest(value=value):
                with self.assertRaises(ValidationError):
                    UnitConversionParameters(value=value, from_unit="m", to_unit="cm")

    def test_average_speed_and_density_boundaries(self):
        self.assertEqual(
            AverageSpeedParameters(distance_m=0, time_s=1).distance_m,
            Decimal("0"),
        )
        self.assertEqual(
            DensityParameters(mass_kg=0, volume_m3=1).mass_kg,
            Decimal("0"),
        )
        invalid_models = (
            (AverageSpeedParameters, {"distance_m": -1, "time_s": 1}),
            (AverageSpeedParameters, {"distance_m": 1, "time_s": 0}),
            (DensityParameters, {"mass_kg": -1, "volume_m3": 1}),
            (DensityParameters, {"mass_kg": 1, "volume_m3": 0}),
        )
        for model, parameters in invalid_models:
            with self.subTest(model=model.__name__, parameters=parameters):
                with self.assertRaises(ValidationError):
                    model(**parameters)

    def test_ohms_law_accepts_all_three_valid_combinations(self):
        cases = (
            {"voltage_v": 12, "current_a": 2},
            {"voltage_v": 12, "resistance_ohm": 6},
            {"current_a": 2, "resistance_ohm": 6},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                self.assertIsInstance(OhmsLawParameters(**parameters), OhmsLawParameters)

    def test_ohms_law_rejects_one_or_three_known_values(self):
        for parameters in (
            {"voltage_v": 12},
            {"voltage_v": 12, "current_a": 2, "resistance_ohm": 6},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    OhmsLawParameters(**parameters)

    def test_ohms_law_rejects_negative_values_and_zero_denominators(self):
        cases = (
            {"voltage_v": -1, "current_a": 2},
            {"voltage_v": 12, "current_a": 0},
            {"voltage_v": 12, "resistance_ohm": 0},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    OhmsLawParameters(**parameters)

    def test_ohms_law_allows_zero_without_division(self):
        model = OhmsLawParameters(current_a=0, resistance_ohm=5)
        self.assertEqual(model.current_a, Decimal("0"))

    def test_electric_power_accepts_all_three_formula_combinations(self):
        cases = (
            {"voltage_v": 12, "current_a": 2},
            {"current_a": 2, "resistance_ohm": 3},
            {"voltage_v": 12, "resistance_ohm": 3},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                self.assertIsInstance(
                    ElectricPowerParameters(**parameters), ElectricPowerParameters
                )

    def test_electric_power_rejects_invalid_combinations_and_values(self):
        cases = (
            {"voltage_v": 12},
            {"voltage_v": 12, "current_a": 2, "resistance_ohm": 3},
            {"voltage_v": -1, "current_a": 2},
            {"voltage_v": 12, "resistance_ohm": 0},
        )
        for parameters in cases:
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    ElectricPowerParameters(**parameters)

    def test_optional_fields_support_omission_and_explicit_none(self):
        omitted = OhmsLawParameters(voltage_v=12, resistance_ohm=6)
        explicit_null = OhmsLawParameters.model_validate_json(
            '{"voltage_v": 12, "current_a": null, "resistance_ohm": 6}'
        )
        self.assertIsNone(omitted.current_a)
        self.assertIsNone(explicit_null.current_a)
        self.assertEqual(omitted, explicit_null)

    def test_each_supported_unit_category_and_same_unit_are_allowed(self):
        cases = (
            ("m", "km"),
            ("s", "h"),
            ("m/s", "km/h"),
            ("A", "mA"),
            ("Ω", "kΩ"),
            ("W", "kW"),
            ("kg", "g"),
            ("m³", "cm³"),
            ("cm", "cm"),
        )
        for from_unit, to_unit in cases:
            with self.subTest(from_unit=from_unit, to_unit=to_unit):
                model = UnitConversionParameters(
                    value=1, from_unit=from_unit, to_unit=to_unit
                )
                self.assertEqual(model.to_unit, to_unit)

    def test_unknown_unit_and_cross_category_are_rejected(self):
        for parameters in (
            {"value": 1, "from_unit": "inch", "to_unit": "m"},
            {"value": 1, "from_unit": "m", "to_unit": "s"},
        ):
            with self.subTest(parameters=parameters):
                with self.assertRaises(ValidationError):
                    UnitConversionParameters(**parameters)


class ToolJsonSchemaTests(unittest.TestCase):
    def test_required_numeric_schema_is_openai_tool_friendly(self):
        schema = AverageSpeedParameters.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["properties"]), {"distance_m", "time_s"})
        self.assertEqual(set(schema["required"]), {"distance_m", "time_s"})
        self.assertEqual(schema["properties"]["distance_m"]["type"], "number")
        self.assertEqual(schema["properties"]["distance_m"]["minimum"], 0)
        self.assertEqual(schema["properties"]["time_s"]["type"], "number")
        self.assertEqual(schema["properties"]["time_s"]["exclusiveMinimum"], 0)
        json.dumps(schema)

    def test_optional_fields_are_nullable_and_not_required(self):
        schema = OhmsLawParameters.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(
            set(schema["properties"]),
            {"voltage_v", "current_a", "resistance_ohm"},
        )
        self.assertFalse(schema.get("required"))
        for field_schema in schema["properties"].values():
            self.assertEqual(
                {branch.get("type") for branch in field_schema["anyOf"]},
                {"number", "null"},
            )

    def test_unit_schema_exposes_required_fields_and_supported_enum(self):
        schema = UnitConversionParameters.model_json_schema()
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"value", "from_unit", "to_unit"})
        self.assertEqual(schema["properties"]["value"]["type"], "number")
        self.assertIn("m/s", schema["properties"]["from_unit"]["enum"])
        self.assertIn("cm³", schema["properties"]["to_unit"]["enum"])


if __name__ == "__main__":
    unittest.main()
