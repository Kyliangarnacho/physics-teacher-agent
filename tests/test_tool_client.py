import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.tool_client import answer_with_tools


def tool_call(
    call_id: str = "call-ohms",
    name: str = "calculate_ohms_law",
    arguments: str = '{"voltage_v": "12", "resistance_ohm": "6"}',
):
    return SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def completion_response(*, tool_calls=None, content=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                )
            )
        ]
    )


class FakeCompletion:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ToolClientTests(unittest.TestCase):
    def successful_completion(self) -> FakeCompletion:
        return FakeCompletion(
            [
                completion_response(tool_calls=[tool_call()]),
                completion_response(
                    tool_calls=None,
                    content="根据欧姆定律，电流为 2 A。",
                ),
            ]
        )

    def test_first_request_keeps_question_mode_context_and_optional_tools(self):
        fake = self.successful_completion()

        answer_with_tools(
            "某电阻两端电压为 12 V，电阻为 6 Ω，求电流。",
            context="参考资料：欧姆定律 I=U/R。",
            mode_instruction="给出完整答案和必要单位。",
            completion_func=fake,
        )

        first = fake.calls[0]
        self.assertEqual(
            [message["role"] for message in first["messages"]],
            ["system", "system", "system", "system", "user"],
        )
        self.assertEqual(
            first["messages"][-1]["content"],
            "某电阻两端电压为 12 V，电阻为 6 Ω，求电流。",
        )
        self.assertIn("完整答案", first["messages"][1]["content"])
        self.assertIn("欧姆定律", first["messages"][2]["content"])
        self.assertIn("应主动调用", first["messages"][3]["content"])
        self.assertIn("不要为了调用而调用", first["messages"][3]["content"])
        self.assertEqual(first["tool_choice"], "auto")
        self.assertEqual(len(first["tools"]), 7)
        self.assertFalse(first["stream"])
        self.assertEqual(first["extra_body"], {"enable_thinking": False})

    def test_successful_tool_call_returns_two_amperes_and_two_requests(self):
        fake = self.successful_completion()

        result = answer_with_tools("电压 12 V，电阻 6 Ω，求电流。", completion_func=fake)

        self.assertEqual(result["answer"], "根据欧姆定律，电流为 2 A。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(len(result["tool_records"]), 1)
        record = result["tool_records"][0]
        self.assertEqual(record["status"], "success")
        self.assertEqual(record["result"]["display_value"], "2")
        self.assertEqual(record["result"]["unit"], "A")
        self.assertEqual(record["normalized_fields"], ["voltage_v", "resistance_ohm"])

    def test_second_request_message_order_and_tool_call_id_match(self):
        fake = self.successful_completion()
        question = "电压 12 V，电阻 6 Ω，求电流。"

        answer_with_tools(question, completion_func=fake)

        messages = fake.calls[1]["messages"]
        self.assertEqual(
            [message["role"] for message in messages],
            ["system", "system", "user", "assistant", "tool"],
        )
        self.assertEqual(messages[2]["content"], question)
        self.assertEqual(messages[3]["tool_calls"][0]["id"], "call-ohms")
        self.assertEqual(messages[4]["tool_call_id"], "call-ohms")
        tool_content = json.loads(messages[4]["content"])
        self.assertEqual(tool_content["result"]["display_value"], "2")

    def test_second_request_omits_tools_and_tool_choice(self):
        fake = self.successful_completion()

        answer_with_tools("电压 12 V，电阻 6 Ω，求电流。", completion_func=fake)

        second = fake.calls[1]
        self.assertNotIn("tools", second)
        self.assertNotIn("tool_choice", second)
        self.assertFalse(second["stream"])

    def test_two_tool_calls_execute_in_model_order_and_return_once(self):
        calls = [
            tool_call(
                "call-speed",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            ),
            tool_call(
                "call-density",
                "calculate_density",
                '{"mass_kg": 2, "volume_m3": 0.5}',
            ),
        ]
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(tool_calls=None, content="两个结果已计算。"),
            ]
        )

        result = answer_with_tools("请分别计算。", completion_func=fake)

        self.assertEqual(result["answer"], "两个结果已计算。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(
            [record["tool_call_id"] for record in result["tool_records"]],
            ["call-speed", "call-density"],
        )
        self.assertEqual(
            [record["name"] for record in result["tool_records"]],
            ["calculate_average_speed", "calculate_density"],
        )
        self.assertTrue(
            all(record["status"] == "success" for record in result["tool_records"])
        )
        self.assertEqual(len(fake.calls), 2)

        result_messages = fake.calls[1]["messages"]
        self.assertEqual(
            [message["role"] for message in result_messages],
            ["system", "system", "user", "assistant", "tool", "tool"],
        )
        self.assertEqual(
            [call["id"] for call in result_messages[3]["tool_calls"]],
            ["call-speed", "call-density"],
        )
        self.assertEqual(
            [message["tool_call_id"] for message in result_messages[4:]],
            ["call-speed", "call-density"],
        )

    def test_three_tool_calls_keep_all_records_and_call_final_once(self):
        calls = [
            tool_call(
                "call-speed",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            ),
            tool_call(
                "call-power",
                "calculate_mechanical_power",
                '{"work_j": 120, "time_s": 20}',
            ),
            tool_call(
                "call-convert",
                "convert_physics_unit",
                '{"value": 36, "from_unit": "km/h", "to_unit": "m/s"}',
            ),
        ]
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(tool_calls=None, content="三个结果已计算。"),
            ]
        )

        result = answer_with_tools("请计算三项。", completion_func=fake)

        self.assertEqual(result["answer"], "三个结果已计算。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(len(result["tool_records"]), 3)
        self.assertEqual(
            [record["tool_call_id"] for record in result["tool_records"]],
            ["call-speed", "call-power", "call-convert"],
        )
        self.assertEqual(len(fake.calls), 2)
        self.assertEqual(
            [message["tool_call_id"] for message in fake.calls[1]["messages"][-3:]],
            ["call-speed", "call-power", "call-convert"],
        )

    def test_empty_question_is_rejected_before_completion(self):
        fake = FakeCompletion([])

        for question in ("", "   "):
            with self.subTest(question=repr(question)):
                with self.assertRaises(ValueError):
                    answer_with_tools(question, completion_func=fake)

        self.assertEqual(fake.calls, [])

    def test_completion_exception_returns_safe_result_without_traceback(self):
        fake = FakeCompletion(
            [
                RuntimeError("private internal detail"),
                RuntimeError("private internal detail again"),
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(result["tool_records"], [])
        self.assertNotIn("private", result["answer"])


class MultiToolClientBoundaryTests(unittest.TestCase):
    def test_single_validation_error_repairs_once_and_succeeds(self):
        call = tool_call(
            "call-ohms",
            "calculate_ohms_law",
            '{"voltage_v": "12 V", "resistance_ohm": 6}',
        )
        repair_payload = {
            "repairs": [
                {
                    "tool_call_id": "call-ohms",
                    "name": "calculate_ohms_law",
                    "arguments": {"voltage_v": 12, "resistance_ohm": 6},
                }
            ]
        }
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[call]),
                completion_response(content=json.dumps(repair_payload)),
                completion_response(tool_calls=None, content="电流为 2 A。"),
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["answer"], "电流为 2 A。")
        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["error", "success"],
        )
        self.assertEqual(result["tool_records"][-1]["result"]["display_value"], "2")
        self.assertNotIn("tools", fake.calls[1])
        repair_trace = next(
            trace for trace in result["step_traces"] if trace["name"] == "tool_repair"
        )
        self.assertEqual(repair_trace["model_requests"], 1)
        self.assertEqual(repair_trace["metadata"]["repair_candidate_count"], 1)
        self.assertEqual(repair_trace["metadata"]["repaired_success_count"], 1)

    def test_two_invalid_tools_share_one_repair_request(self):
        calls = [
            tool_call(
                "call-speed",
                "calculate_average_speed",
                '{"distance_m": 50}',
            ),
            tool_call(
                "call-density",
                "calculate_density",
                '{"mass_kg": 2}',
            ),
        ]
        repair_payload = {
            "repairs": [
                {
                    "tool_call_id": "call-speed",
                    "name": "calculate_average_speed",
                    "arguments": {"distance_m": 50, "time_s": 10},
                },
                {
                    "tool_call_id": "call-density",
                    "name": "calculate_density",
                    "arguments": {"mass_kg": 2, "volume_m3": 0.5},
                },
            ]
        }
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(content=json.dumps(repair_payload)),
                completion_response(tool_calls=None, content="两个结果均已计算。"),
            ]
        )

        result = answer_with_tools("请计算两项。", completion_func=fake)

        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["error", "error", "success", "success"],
        )
        repair_trace = next(
            trace for trace in result["step_traces"] if trace["name"] == "tool_repair"
        )
        self.assertEqual(repair_trace["metadata"]["repair_candidate_count"], 2)
        self.assertEqual(repair_trace["metadata"]["repaired_success_count"], 2)

    def test_partial_success_repairs_only_failed_call(self):
        calls = [
            tool_call(
                "call-speed",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            ),
            tool_call(
                "call-density",
                "calculate_density",
                '{"mass_kg": 2}',
            ),
        ]
        repair_payload = {
            "repairs": [
                {
                    "tool_call_id": "call-density",
                    "name": "calculate_density",
                    "arguments": {"mass_kg": 2, "volume_m3": 0.5},
                }
            ]
        }
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(content=json.dumps(repair_payload)),
                completion_response(tool_calls=None, content="结果已说明。"),
            ]
        )

        with patch("src.tool_client.execute_tool_call") as mocked_execute:
            from src.tools.registry import execute_tool_call

            mocked_execute.side_effect = execute_tool_call
            result = answer_with_tools("请计算。", completion_func=fake)

        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(mocked_execute.call_count, 3)
        self.assertEqual(
            [call.kwargs["tool_call_id"] for call in mocked_execute.call_args_list],
            ["call-speed", "call-density", "call-density"],
        )
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["success", "error", "success"],
        )

    def test_failed_repair_is_not_retried_and_all_failed_uses_fallback(self):
        call = tool_call(
            "call-density",
            "calculate_density",
            '{"mass_kg": 2}',
        )
        repair_payload = {
            "repairs": [
                {
                    "tool_call_id": "call-density",
                    "name": "calculate_density",
                    "arguments": {"mass_kg": 2, "volume_m3": 0},
                }
            ]
        }
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[call]),
                completion_response(content=json.dumps(repair_payload)),
                completion_response(tool_calls=None, content="请补充有效体积后再计算。"),
            ]
        )

        result = answer_with_tools("求密度。", completion_func=fake)

        self.assertEqual(result["answer"], "请补充有效体积后再计算。")
        self.assertEqual(result["model_requests"], 3)
        self.assertEqual(len(fake.calls), 3)
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["error", "error"],
        )
        self.assertNotIn("tools", fake.calls[2])
        self.assertEqual(
            result["step_traces"][-1]["metadata"]["fallback"],
            "model_only",
        )

    def test_unknown_tool_never_enters_repair(self):
        call = tool_call("call-unknown", "not_registered", "{}")
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[call]),
                completion_response(tool_calls=None, content="无法使用该本地工具。"),
            ]
        )

        result = answer_with_tools("请计算。", completion_func=fake)

        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(len(fake.calls), 2)
        self.assertFalse(
            any(trace["name"] == "tool_repair" for trace in result["step_traces"])
        )

    def test_partial_success_executes_all_records_and_returns_once(self):
        calls = [
            tool_call(
                "call-speed",
                "calculate_average_speed",
                '{"distance_m": 50, "time_s": 10}',
            ),
            tool_call(
                "call-invalid",
                "calculate_pulley_efficiency",
                '{"weight_n": 100, "height_m": 1, "force_n": 1, "distance_m": 1}',
            ),
            tool_call(
                "call-density",
                "calculate_density",
                '{"mass_kg": 2, "volume_m3": 0.5}',
            ),
        ]
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(tool_calls=None, content="已分别说明三个计算结果。"),
            ]
        )

        result = answer_with_tools("请计算三项。", completion_func=fake)

        self.assertEqual(result["answer"], "已分别说明三个计算结果。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(
            [record["tool_call_id"] for record in result["tool_records"]],
            ["call-speed", "call-invalid", "call-density"],
        )
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["success", "error", "success"],
        )
        self.assertEqual(len(fake.calls), 2)
        tool_messages = fake.calls[1]["messages"][-3:]
        self.assertEqual(
            [message["tool_call_id"] for message in tool_messages],
            ["call-speed", "call-invalid", "call-density"],
        )
        self.assertEqual(
            [json.loads(message["content"])["status"] for message in tool_messages],
            ["success", "error", "success"],
        )

    def test_all_failed_tools_use_model_only_fallback(self):
        calls = [
            tool_call(
                "call-invalid-1",
                "not_registered",
                "{}",
            ),
            tool_call(
                "call-invalid-2",
                "calculate_pulley_efficiency",
                '{"weight_n": 100, "height_m": 1, "force_n": 1, "distance_m": 1}',
            ),
        ]
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(tool_calls=None, content="请补充或核对题目条件后再计算。"),
            ]
        )

        result = answer_with_tools("请计算。", completion_func=fake)

        self.assertEqual(result["answer"], "请补充或核对题目条件后再计算。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(
            [record["status"] for record in result["tool_records"]],
            ["error", "error"],
        )
        fallback_request = fake.calls[1]
        self.assertNotIn("tools", fallback_request)
        self.assertNotIn("tool_choice", fallback_request)
        self.assertEqual(
            [message["role"] for message in fallback_request["messages"]],
            ["system", "system", "user"],
        )
        self.assertEqual(
            result["step_traces"][-1]["metadata"]["fallback"],
            "model_only",
        )

    def test_no_tool_call_reuses_model_only_answer_without_repair(self):
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[], content="我会按题意直接说明。"),
            ]
        )

        result = answer_with_tools("请说明。", completion_func=fake)

        self.assertEqual(result["answer"], "我会按题意直接说明。")
        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(result["tool_records"], [])
        self.assertEqual(len(fake.calls), 1)
        self.assertEqual(
            result["step_traces"][0]["metadata"]["decision"],
            "no_tool_needed",
        )
        self.assertFalse(any(trace["name"] == "tool_repair" for trace in result["step_traces"]))

    def test_empty_no_tool_response_uses_clean_model_only_follow_up(self):
        fake = FakeCompletion(
            [
                completion_response(tool_calls=[], content=""),
                completion_response(tool_calls=None, content="我会按题意直接说明。"),
            ]
        )

        result = answer_with_tools("请说明。", completion_func=fake)

        self.assertEqual(result["answer"], "我会按题意直接说明。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(result["tool_records"], [])
        self.assertNotIn("tools", fake.calls[1])
        selection_trace = result["step_traces"][0]
        self.assertEqual(selection_trace["status"], "success")
        self.assertEqual(selection_trace["metadata"]["decision"], "no_tool_needed")
        self.assertEqual(
            result["step_traces"][-1]["metadata"]["answer_strategy"],
            "model_only",
        )

    def test_more_than_five_tool_calls_are_rejected_before_execution(self):
        calls = [tool_call(f"call-{index}") for index in range(6)]
        fake = FakeCompletion(
            [
                completion_response(tool_calls=calls),
                completion_response(tool_calls=None, content="请将问题分开描述。"),
            ]
        )

        with patch("src.tool_client.execute_tool_call") as mocked_execute:
            result = answer_with_tools("请计算。", completion_func=fake)

        self.assertEqual(result["answer"], "请将问题分开描述。")
        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(result["tool_records"], [])
        self.assertEqual(len(fake.calls), 2)
        mocked_execute.assert_not_called()
        self.assertNotIn("tools", fake.calls[1])
        self.assertEqual(result["step_traces"][0]["error_type"], "tool_protocol")
        self.assertFalse(
            any(trace["name"] == "tool_repair" for trace in result["step_traces"])
        )


if __name__ == "__main__":
    unittest.main()
