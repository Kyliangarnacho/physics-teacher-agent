import json
import unittest
from types import SimpleNamespace

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

    def test_first_request_keeps_question_mode_context_and_required_tools(self):
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
            ["system", "system", "system", "user"],
        )
        self.assertEqual(
            first["messages"][-1]["content"],
            "某电阻两端电压为 12 V，电阻为 6 Ω，求电流。",
        )
        self.assertIn("完整答案", first["messages"][1]["content"])
        self.assertIn("欧姆定律", first["messages"][2]["content"])
        self.assertEqual(first["tool_choice"], "required")
        self.assertEqual(len(first["tools"]), 5)
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
            ["system", "user", "assistant", "tool"],
        )
        self.assertEqual(messages[1]["content"], question)
        self.assertEqual(messages[2]["tool_calls"][0]["id"], "call-ohms")
        self.assertEqual(messages[3]["tool_call_id"], "call-ohms")
        tool_content = json.loads(messages[3]["content"])
        self.assertEqual(tool_content["result"]["display_value"], "2")

    def test_second_request_omits_tools_and_tool_choice(self):
        fake = self.successful_completion()

        answer_with_tools("电压 12 V，电阻 6 Ω，求电流。", completion_func=fake)

        second = fake.calls[1]
        self.assertNotIn("tools", second)
        self.assertNotIn("tool_choice", second)
        self.assertFalse(second["stream"])

    def test_no_tool_call_stops_safely_after_one_request(self):
        fake = FakeCompletion([completion_response(tool_calls=[], content="直接回答")])

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(result["tool_records"], [])
        self.assertIn("没有返回工具调用", result["answer"])

    def test_multiple_tool_calls_stop_safely_without_execution(self):
        fake = FakeCompletion(
            [completion_response(tool_calls=[tool_call("call-1"), tool_call("call-2")])]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(result["tool_records"], [])
        self.assertIn("多个工具调用", result["answer"])

    def test_failed_tool_execution_returns_record_without_second_request(self):
        fake = FakeCompletion(
            [
                completion_response(
                    tool_calls=[
                        tool_call(arguments='{"voltage_v": "12 V", "resistance_ohm": "6"}')
                    ]
                )
            ]
        )

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(len(result["tool_records"]), 1)
        self.assertEqual(result["tool_records"][0]["status"], "error")
        self.assertEqual(len(fake.calls), 1)

    def test_empty_question_is_rejected_before_completion(self):
        fake = FakeCompletion([])

        for question in ("", "   "):
            with self.subTest(question=repr(question)):
                with self.assertRaises(ValueError):
                    answer_with_tools(question, completion_func=fake)

        self.assertEqual(fake.calls, [])

    def test_completion_exception_returns_safe_result_without_traceback(self):
        fake = FakeCompletion([RuntimeError("private internal detail")])

        result = answer_with_tools("求电流。", completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(result["tool_records"], [])
        self.assertNotIn("private", result["answer"])


if __name__ == "__main__":
    unittest.main()
