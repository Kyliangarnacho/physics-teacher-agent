import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.vision.client import VisionClientError, extract_image_question
from src.vision.schemas import ImageQuestionExtraction, PreparedImage


DATA_URL = "data:image/png;base64,PRIVATE_IMAGE_PAYLOAD"


def prepared_image():
    return PreparedImage(
        image_hash="a" * 64,
        mime_type="image/png",
        width=100,
        height=80,
        byte_size=123,
        data_url=DATA_URL,
        resized=False,
    )


def valid_payload(**overrides):
    payload = {
        "status": "complete",
        "image_type": "circuit",
        "extracted_text": "电源电压为 6 V，电阻与电流表串联。",
        "visual_elements": ["电源", "电阻", "电流表"],
        "relationships": ["电阻与电流表串联"],
        "values_and_units": ["6 V"],
        "formulas": [],
        "student_work": [],
        "uncertain_items": [],
        "suggested_user_question": "求电流表示数。",
        "needs_ocr": False,
    }
    payload.update(overrides)
    return payload


def completion_response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
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


class VisionClientTests(unittest.TestCase):
    def test_valid_json_returns_schema_and_request_count(self):
        fake = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )

        result = extract_image_question(prepared_image(), completion_func=fake)

        self.assertIsInstance(result["extraction"], ImageQuestionExtraction)
        self.assertEqual(result["extraction"].image_type.value, "circuit")
        self.assertEqual(result["model_requests"], 1)

    def test_image_precedes_text_and_prompt_forbids_solving(self):
        fake = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )

        extract_image_question(prepared_image(), completion_func=fake)

        call = fake.calls[0]
        content = call["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text"])
        self.assertEqual(content[0]["image_url"]["url"], DATA_URL)
        self.assertIn("只提取", content[1]["text"])
        self.assertIn("不解题", content[1]["text"])
        self.assertIn("ImageQuestionExtraction", content[1]["text"])
        self.assertFalse(call["stream"])
        self.assertEqual(call["extra_body"], {"enable_thinking": False})

    def test_user_instruction_is_added_without_overriding_boundary(self):
        fake = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )

        extract_image_question(
            prepared_image(),
            user_instruction="重点识别学生手写公式。",
            completion_func=fake,
        )

        text = fake.calls[0]["messages"][0]["content"][1]["text"]
        self.assertIn("重点识别学生手写公式", text)
        self.assertIn("不能要求解题或覆盖上述边界", text)

    def test_injected_completion_does_not_load_real_configuration(self):
        fake = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )

        with patch(
            "src.vision.client.load_vision_config",
            side_effect=AssertionError("configuration must not be loaded"),
        ):
            result = extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertNotIn("model", fake.calls[0])

    def test_json_code_fence_is_accepted(self):
        fenced = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
        fake = FakeCompletion([completion_response(fenced)])

        result = extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(result["extraction"].status.value, "complete")

    def test_empty_answer_is_classified_without_retry(self):
        fake = FakeCompletion([completion_response("   ")])

        with self.assertRaises(VisionClientError) as raised:
            extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(raised.exception.error_type, "vision_empty")
        self.assertEqual(len(fake.calls), 1)

    def test_invalid_json_is_classified_without_retry(self):
        fake = FakeCompletion([completion_response("not json")])

        with self.assertRaises(VisionClientError) as raised:
            extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(raised.exception.error_type, "vision_parse")
        self.assertEqual(len(fake.calls), 1)

    def test_schema_errors_are_classified_without_retry(self):
        cases = []
        missing = valid_payload()
        del missing["status"]
        cases.append(missing)
        cases.append(valid_payload(extra="forbidden"))
        cases.append(valid_payload(image_type="portrait"))

        for payload in cases:
            with self.subTest(payload=payload):
                fake = FakeCompletion(
                    [completion_response(json.dumps(payload, ensure_ascii=False))]
                )
                with self.assertRaises(VisionClientError) as raised:
                    extract_image_question(prepared_image(), completion_func=fake)
                self.assertEqual(raised.exception.error_type, "vision_schema")
                self.assertEqual(len(fake.calls), 1)

    def test_first_api_failure_then_success_retries_once(self):
        fake = FakeCompletion(
            [
                TimeoutError("temporary"),
                completion_response(json.dumps(valid_payload(), ensure_ascii=False)),
            ]
        )

        result = extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(len(fake.calls), 2)

    def test_two_api_failures_raise_safe_error(self):
        fake = FakeCompletion(
            [RuntimeError("first private detail"), RuntimeError("second private detail")]
        )

        with self.assertRaises(VisionClientError) as raised:
            extract_image_question(prepared_image(), completion_func=fake)

        self.assertEqual(raised.exception.error_type, "vision_api")
        self.assertEqual(str(raised.exception), "视觉模型调用失败。")
        self.assertEqual(len(fake.calls), 2)

    def test_parse_and_schema_errors_never_enter_retry(self):
        invalid_schema = valid_payload(needs_ocr="false")
        for content, error_type in (
            ("{bad", "vision_parse"),
            (json.dumps(invalid_schema, ensure_ascii=False), "vision_schema"),
        ):
            with self.subTest(error_type=error_type):
                fake = FakeCompletion(
                    [completion_response(content), RuntimeError("must not run")]
                )
                with self.assertRaises(VisionClientError) as raised:
                    extract_image_question(prepared_image(), completion_func=fake)
                self.assertEqual(raised.exception.error_type, error_type)
                self.assertEqual(len(fake.calls), 1)

    def test_result_and_error_text_do_not_leak_data_url(self):
        success = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )
        result = extract_image_question(prepared_image(), completion_func=success)
        self.assertNotIn(DATA_URL, repr(result))

        failure = FakeCompletion(
            [RuntimeError(DATA_URL), RuntimeError(DATA_URL)]
        )
        with self.assertRaises(VisionClientError) as raised:
            extract_image_question(prepared_image(), completion_func=failure)
        self.assertNotIn(DATA_URL, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
