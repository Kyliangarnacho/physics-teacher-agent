import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from src.vision.ocr_client import OCRClientError, extract_image_text
from src.vision.schemas import OCRResult, PreparedImage


DATA_URL = "data:image/png;base64,PRIVATE_OCR_IMAGE_PAYLOAD"


def prepared_image():
    return PreparedImage(
        image_hash="b" * 64,
        mime_type="image/png",
        width=120,
        height=90,
        byte_size=234,
        data_url=DATA_URL,
        resized=False,
    )


def valid_payload(**overrides):
    payload = {
        "text": "电源电压为 6 V，求电流。",
        "formulas": ["I=U/R"],
        "tables": ["次数 | 电流/A"],
        "uncertain_fragments": ["右下角数字模糊"],
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


class OCRClientTests(unittest.TestCase):
    def test_valid_json_returns_ocr_schema(self):
        fake = FakeCompletion(
            [completion_response(json.dumps(valid_payload(), ensure_ascii=False))]
        )

        result = extract_image_text(prepared_image(), completion_func=fake)

        self.assertIsInstance(result["ocr_result"], OCRResult)
        self.assertEqual(result["ocr_result"].formulas, ["I=U/R"])
        self.assertEqual(result["model_requests"], 1)

    def test_json_code_fence_is_accepted(self):
        content = "```json\n" + json.dumps(valid_payload(), ensure_ascii=False) + "\n```"
        fake = FakeCompletion([completion_response(content)])

        result = extract_image_text(prepared_image(), completion_func=fake)

        self.assertEqual(result["ocr_result"].text, valid_payload()["text"])

    def test_plain_nonempty_text_uses_safe_fallback(self):
        plain_text = "电源电压为 6 V。\n手写公式：I=U/R"
        fake = FakeCompletion([completion_response(plain_text)])

        result = extract_image_text(prepared_image(), completion_func=fake)

        ocr_result = result["ocr_result"]
        self.assertEqual(ocr_result.text, plain_text)
        self.assertEqual(ocr_result.formulas, [])
        self.assertEqual(ocr_result.tables, [])
        self.assertEqual(ocr_result.uncertain_fragments, [])

    def test_image_precedes_text_and_prompt_has_safety_boundary(self):
        fake = FakeCompletion([completion_response("普通识别文字")])

        extract_image_text(prepared_image(), completion_func=fake)

        call = fake.calls[0]
        content = call["messages"][0]["content"]
        self.assertEqual([item["type"] for item in content], ["image_url", "text"])
        self.assertEqual(content[0]["image_url"]["url"], DATA_URL)
        self.assertIn("不解题", content[1]["text"])
        self.assertIn("不推断图形连接关系", content[1]["text"])
        self.assertIn("不编造缺失文字", content[1]["text"])
        self.assertFalse(call["stream"])
        self.assertEqual(call["extra_body"], {"enable_thinking": False})

    def test_user_instruction_is_added_without_overriding_boundary(self):
        fake = FakeCompletion([completion_response("识别内容")])

        extract_image_text(
            prepared_image(),
            user_instruction="重点识别右下角手写单位。",
            completion_func=fake,
        )

        prompt = fake.calls[0]["messages"][0]["content"][1]["text"]
        self.assertIn("重点识别右下角手写单位", prompt)
        self.assertIn("不能要求解题、推断或编造", prompt)

    def test_injected_completion_does_not_load_real_configuration(self):
        fake = FakeCompletion([completion_response("识别内容")])

        with patch(
            "src.vision.ocr_client.load_ocr_config",
            side_effect=AssertionError("configuration must not be loaded"),
        ):
            result = extract_image_text(prepared_image(), completion_func=fake)

        self.assertEqual(result["model_requests"], 1)
        self.assertNotIn("model", fake.calls[0])

    def test_empty_answer_is_classified_without_retry(self):
        fake = FakeCompletion([completion_response("  ")])

        with self.assertRaises(OCRClientError) as raised:
            extract_image_text(prepared_image(), completion_func=fake)

        self.assertEqual(raised.exception.error_type, "ocr_empty")
        self.assertEqual(len(fake.calls), 1)

    def test_damaged_json_is_parse_error_not_plain_text(self):
        for content in ('{"text": "缺少结尾"', "```json\n{bad\n```"):
            with self.subTest(content=content):
                fake = FakeCompletion([completion_response(content)])
                with self.assertRaises(OCRClientError) as raised:
                    extract_image_text(prepared_image(), completion_func=fake)
                self.assertEqual(raised.exception.error_type, "ocr_parse")
                self.assertEqual(len(fake.calls), 1)

    def test_json_schema_errors_are_rejected_without_retry(self):
        cases = (
            {"formulas": []},
            valid_payload(extra="forbidden"),
            valid_payload(formulas="I=U/R"),
        )
        for payload in cases:
            with self.subTest(payload=payload):
                fake = FakeCompletion(
                    [completion_response(json.dumps(payload, ensure_ascii=False))]
                )
                with self.assertRaises(OCRClientError) as raised:
                    extract_image_text(prepared_image(), completion_func=fake)
                self.assertEqual(raised.exception.error_type, "ocr_schema")
                self.assertEqual(len(fake.calls), 1)

    def test_first_api_failure_then_success_retries_once(self):
        fake = FakeCompletion(
            [TimeoutError("temporary"), completion_response("识别内容")]
        )

        result = extract_image_text(prepared_image(), completion_func=fake)

        self.assertEqual(result["model_requests"], 2)
        self.assertEqual(len(fake.calls), 2)

    def test_two_api_failures_raise_safe_error(self):
        fake = FakeCompletion(
            [RuntimeError("private first"), RuntimeError("private second")]
        )

        with self.assertRaises(OCRClientError) as raised:
            extract_image_text(prepared_image(), completion_func=fake)

        self.assertEqual(raised.exception.error_type, "ocr_api")
        self.assertEqual(str(raised.exception), "OCR 模型调用失败。")
        self.assertEqual(len(fake.calls), 2)

    def test_parse_and_schema_errors_do_not_retry(self):
        for content, expected_error in (
            ("{bad", "ocr_parse"),
            (json.dumps({"text": 123}), "ocr_schema"),
        ):
            with self.subTest(expected_error=expected_error):
                fake = FakeCompletion(
                    [completion_response(content), RuntimeError("must not run")]
                )
                with self.assertRaises(OCRClientError) as raised:
                    extract_image_text(prepared_image(), completion_func=fake)
                self.assertEqual(raised.exception.error_type, expected_error)
                self.assertEqual(len(fake.calls), 1)

    def test_result_and_error_text_do_not_leak_data_url(self):
        success = FakeCompletion([completion_response("识别内容")])
        result = extract_image_text(prepared_image(), completion_func=success)
        self.assertNotIn(DATA_URL, repr(result))

        failure = FakeCompletion([RuntimeError(DATA_URL), RuntimeError(DATA_URL)])
        with self.assertRaises(OCRClientError) as raised:
            extract_image_text(prepared_image(), completion_func=failure)
        self.assertNotIn(DATA_URL, str(raised.exception))


if __name__ == "__main__":
    unittest.main()
