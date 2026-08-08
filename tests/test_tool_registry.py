import json
import unittest
from dataclasses import replace
from unittest.mock import patch

from src.tools import registry
from src.tools.registry import (
    ToolExecutionStatus,
    execute_tool_call,
    get_openai_tools,
)


class OpenAIToolDefinitionsTests(unittest.TestCase):
    def test_exactly_seven_unique_tools_are_registered(self):
        tools = get_openai_tools()
        names = [tool["function"]["name"] for tool in tools]
        self.assertEqual(len(tools), 7)
        self.assertEqual(len(set(names)), 7)
        self.assertEqual(
            set(names),
            {
                "calculate_average_speed",
                "calculate_density",
                "calculate_mechanical_power",
                "calculate_ohms_law",
                "calculate_electric_power",
                "calculate_pulley_efficiency",
                "convert_physics_unit",
            },
        )

    def test_every_definition_has_complete_openai_structure(self):
        for tool in get_openai_tools():
            with self.subTest(name=tool["function"]["name"]):
                self.assertEqual(tool["type"], "function")
                function = tool["function"]
                self.assertIsInstance(function["name"], str)
                self.assertTrue(function["description"])
                parameters = function["parameters"]
                self.assertEqual(parameters["type"], "object")
                self.assertTrue(parameters["properties"])
                self.assertFalse(parameters["additionalProperties"])
                json.dumps(tool, ensure_ascii=False)

    def test_returned_definitions_cannot_pollute_later_calls(self):
        first = get_openai_tools()
        first[0]["function"]["name"] = "tampered"
        first[0]["function"]["parameters"]["properties"].clear()

        second = get_openai_tools()
        self.assertNotEqual(second[0]["function"]["name"], "tampered")
        self.assertTrue(second[0]["function"]["parameters"]["properties"])

    def test_openai_schema_still_declares_numeric_fields_as_number(self):
        ohms_law = next(
            tool
            for tool in get_openai_tools()
            if tool["function"]["name"] == "calculate_ohms_law"
        )
        properties = ohms_law["function"]["parameters"]["properties"]
        for field_name in ("voltage_v", "current_a", "resistance_ohm"):
            types = {branch["type"] for branch in properties[field_name]["anyOf"]}
            self.assertEqual(types, {"number", "null"})


class ToolExecutionTests(unittest.TestCase):
    def test_all_seven_tools_execute_successfully(self):
        cases = (
            (
                "calculate_average_speed",
                {"distance_m": 50, "time_s": 10},
                "5",
            ),
            (
                "calculate_density",
                {"mass_kg": 2, "volume_m3": 0.5},
                "4",
            ),
            (
                "calculate_mechanical_power",
                {"work_j": 120, "time_s": 10},
                "12",
            ),
            (
                "calculate_ohms_law",
                {"voltage_v": 12, "resistance_ohm": 6},
                "2",
            ),
            (
                "calculate_electric_power",
                {"voltage_v": 12, "current_a": 2},
                "24",
            ),
            (
                "calculate_pulley_efficiency",
                {"weight_n": 100, "height_m": 2, "force_n": 50, "distance_m": 5},
                "80",
            ),
            (
                "convert_physics_unit",
                {"value": 1, "from_unit": "m", "to_unit": "cm"},
                "100",
            ),
        )
        for index, (name, arguments, expected) in enumerate(cases, start=1):
            with self.subTest(name=name):
                record = execute_tool_call(
                    f"call-{index}", name, json.dumps(arguments, ensure_ascii=False)
                )
                self.assertEqual(record.status, ToolExecutionStatus.SUCCESS)
                self.assertEqual(record.result["display_value"], expected)
                self.assertIsNone(record.error)

    def test_call_id_and_name_are_preserved(self):
        record = execute_tool_call(
            "call-speed",
            "calculate_average_speed",
            '{"distance_m": 50, "time_s": 10}',
        )
        self.assertEqual(record.tool_call_id, "call-speed")
        self.assertEqual(record.name, "calculate_average_speed")

    def test_validated_arguments_and_result_are_json_serializable(self):
        record = execute_tool_call(
            "call-density",
            "calculate_density",
            '{"mass_kg": 2.5, "volume_m3": 0.5}',
        )
        json.dumps(record.arguments, ensure_ascii=False)
        json.dumps(record.result, ensure_ascii=False)
        json.dumps(record.model_dump(mode="json"), ensure_ascii=False)

    def test_unknown_tool_returns_error_without_dynamic_execution(self):
        record = execute_tool_call(
            "call-unknown", "builtins.eval", '{"expression": "1 + 1"}'
        )
        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertEqual(record.name, "builtins.eval")
        self.assertIsNone(record.result)
        self.assertIn("未知工具", record.error)

    def test_invalid_json_returns_error(self):
        record = execute_tool_call(
            "call-json", "calculate_average_speed", "{not valid json"
        )
        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertIn("JSON", record.error)

    def test_array_and_scalar_arguments_return_error(self):
        for value in ([50, 10], 5, "text", None):
            with self.subTest(value=value):
                record = execute_tool_call(
                    "call-object",
                    "calculate_average_speed",
                    json.dumps(value, ensure_ascii=False),
                )
                self.assertEqual(record.status, ToolExecutionStatus.ERROR)
                self.assertIn("JSON 对象", record.error)

    def test_missing_extra_and_invalid_parameters_return_error(self):
        cases = (
            {"distance_m": 50},
            {"distance_m": 50, "time_s": 10, "extra": 1},
            {"distance_m": 50, "time_s": 0},
            {"distance_m": 50, "time_s": float("nan")},
        )
        for arguments in cases:
            with self.subTest(arguments=arguments):
                record = execute_tool_call(
                    "call-invalid",
                    "calculate_average_speed",
                    json.dumps(arguments, ensure_ascii=False),
                )
                self.assertEqual(record.status, ToolExecutionStatus.ERROR)
                self.assertIn("参数校验失败", record.error)
                json.dumps(record.model_dump(mode="json"), allow_nan=False)

    def test_numeric_strings_are_normalized_and_calculate_two_amperes(self):
        record = execute_tool_call(
            "call-string-ohms",
            "calculate_ohms_law",
            '{"voltage_v": "12", "resistance_ohm": "6"}',
        )
        self.assertEqual(record.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(record.result["display_value"], "2")
        self.assertEqual(record.result["unit"], "A")
        self.assertEqual(
            record.normalized_fields,
            ["voltage_v", "resistance_ohm"],
        )

    def test_decimal_and_scientific_notation_strings_are_normalized(self):
        speed = execute_tool_call(
            "call-scientific",
            "calculate_average_speed",
            '{"distance_m": "1.2e3", "time_s": "0.5"}',
        )
        conversion = execute_tool_call(
            "call-negative-decimal",
            "convert_physics_unit",
            '{"value": "-0.5", "from_unit": "m", "to_unit": "cm"}',
        )
        self.assertEqual(speed.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(speed.result["display_value"], "2400")
        self.assertEqual(speed.normalized_fields, ["distance_m", "time_s"])
        self.assertEqual(conversion.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(conversion.result["display_value"], "-50")
        self.assertEqual(conversion.normalized_fields, ["value"])

    def test_native_json_numbers_do_not_need_normalization(self):
        record = execute_tool_call(
            "call-native-number",
            "calculate_average_speed",
            '{"distance_m": 50, "time_s": 10}',
        )
        self.assertEqual(record.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(record.normalized_fields, [])

    def test_null_is_unchanged_and_not_recorded_as_normalized(self):
        record = execute_tool_call(
            "call-null",
            "calculate_ohms_law",
            '{"voltage_v": "12", "current_a": null, "resistance_ohm": "6"}',
        )
        self.assertEqual(record.status, ToolExecutionStatus.SUCCESS)
        self.assertIsNone(record.arguments["current_a"])
        self.assertEqual(
            record.normalized_fields,
            ["voltage_v", "resistance_ohm"],
        )

    def test_non_numeric_strings_are_not_normalized_and_are_rejected(self):
        for value in ("12 V", "1/2", "", "十二", "NaN", "Infinity"):
            with self.subTest(value=value):
                record = execute_tool_call(
                    "call-unsafe-string",
                    "calculate_average_speed",
                    json.dumps({"distance_m": value, "time_s": 10}, ensure_ascii=False),
                )
                self.assertEqual(record.status, ToolExecutionStatus.ERROR)
                self.assertEqual(record.normalized_fields, [])

    def test_new_tools_normalize_only_pure_numeric_strings(self):
        mechanical = execute_tool_call(
            "call-mechanical-string",
            "calculate_mechanical_power",
            '{"work_j": "120", "time_s": "10"}',
        )
        pulley = execute_tool_call(
            "call-pulley-invalid-string",
            "calculate_pulley_efficiency",
            '{"weight_n": "100", "height_m": "2", "force_n": "None", "distance_m": "5"}',
        )
        self.assertEqual(mechanical.status, ToolExecutionStatus.SUCCESS)
        self.assertEqual(mechanical.normalized_fields, ["work_j", "time_s"])
        self.assertEqual(mechanical.result["display_value"], "12")
        self.assertEqual(pulley.status, ToolExecutionStatus.ERROR)
        self.assertEqual(pulley.normalized_fields, ["weight_n", "height_m", "distance_m"])

    def test_boolean_is_not_normalized_and_is_rejected(self):
        record = execute_tool_call(
            "call-bool",
            "calculate_average_speed",
            '{"distance_m": true, "time_s": 10}',
        )
        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertEqual(record.normalized_fields, [])

    def test_unknown_field_cannot_bypass_extra_forbid(self):
        record = execute_tool_call(
            "call-extra-string",
            "calculate_average_speed",
            '{"distance_m": "50", "time_s": "10", "extra": "5"}',
        )
        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertEqual(record.normalized_fields, ["distance_m", "time_s"])
        self.assertEqual(record.arguments["extra"], "5")

    def test_calculator_value_error_becomes_safe_error_record(self):
        original = registry._TOOL_REGISTRY["calculate_average_speed"]

        def raise_value_error(**_arguments):
            raise ValueError("sensitive calculator detail")

        broken = replace(original, function=raise_value_error)
        with patch.dict(
            registry._TOOL_REGISTRY,
            {"calculate_average_speed": broken},
            clear=False,
        ):
            record = execute_tool_call(
                "call-value-error",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            )

        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertIn("计算失败", record.error)
        self.assertNotIn("sensitive", record.error)

    def test_unexpected_exception_becomes_safe_error_without_traceback(self):
        original = registry._TOOL_REGISTRY["calculate_average_speed"]

        def raise_unexpected(**_arguments):
            raise RuntimeError("C:\\private\\internal.py secret detail")

        broken = replace(original, function=raise_unexpected)
        with patch.dict(
            registry._TOOL_REGISTRY,
            {"calculate_average_speed": broken},
            clear=False,
        ):
            record = execute_tool_call(
                "call-unexpected",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            )

        self.assertEqual(record.status, ToolExecutionStatus.ERROR)
        self.assertIn("执行失败", record.error)
        self.assertNotIn("Traceback", record.error)
        self.assertNotIn("private", record.error)


if __name__ == "__main__":
    unittest.main()
