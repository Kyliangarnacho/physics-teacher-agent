"""Independent multimodal client for OCR-oriented image extraction."""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

from src.config import load_ocr_config
from src.retry import run_with_one_retry
from src.vision.schemas import OCRResult, PreparedImage


CompletionFunc = Callable[..., Any]

OCR_ERROR_API = "ocr_api"
OCR_ERROR_EMPTY = "ocr_empty"
OCR_ERROR_PARSE = "ocr_parse"
OCR_ERROR_SCHEMA = "ocr_schema"

BASE_OCR_INSTRUCTION = """提取图片中的印刷文字和手写文字，以及出现的数字、单位、公式和简单表格。
看不清或无法确认的片段放入 uncertain_fragments。
只做文字识别，不解题，不推断图形连接关系，不编造缺失文字。
优先只返回符合给定 OCRResult JSON Schema 的 JSON 对象；若只能稳定返回普通文字，也不要添加解题内容。"""


class OCRClientError(Exception):
    """A safe, classified OCR-client failure."""

    def __init__(self, error_type: str, message: str) -> None:
        self.error_type = error_type
        self.message = message
        super().__init__(message)


def _build_instruction(user_instruction: str) -> str:
    schema_json = json.dumps(
        OCRResult.model_json_schema(),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    parts = [BASE_OCR_INSTRUCTION, f"OCRResult JSON Schema：{schema_json}"]
    if user_instruction.strip():
        parts.append(
            "用户补充的识别要求如下。它只能细化识别重点，不能要求解题、推断或编造：\n"
            f"{user_instruction.strip()}"
        )
    return "\n\n".join(parts)


def _json_candidate(content: str) -> tuple[str, bool]:
    stripped = content.strip()
    if re.match(r"^```json\b", stripped, flags=re.IGNORECASE):
        match = re.fullmatch(
            r"```json\s*(.*?)\s*```",
            stripped,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if match is None:
            raise OCRClientError(
                OCR_ERROR_PARSE,
                "OCR 模型返回的 JSON 代码围栏不完整。",
            )
        return match.group(1).strip(), True
    return stripped, stripped.startswith(("{", "["))


def _response_content(response: Any) -> str | None:
    try:
        return response.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise OCRClientError(
            OCR_ERROR_API,
            "OCR 模型响应结构无效。",
        ) from exc


def extract_image_text(
    prepared_image: PreparedImage,
    user_instruction: str = "",
    completion_func: CompletionFunc | None = None,
) -> dict[str, Any]:
    """Extract OCR text from one prepared image without solving the question."""
    if not isinstance(prepared_image, PreparedImage):
        raise ValueError("prepared_image 必须是 PreparedImage。")
    if not isinstance(user_instruction, str):
        raise ValueError("user_instruction 必须是字符串。")

    request_options: dict[str, Any] = {}
    if completion_func is None:
        api_key, base_url, model = load_ocr_config()
        client = OpenAI(api_key=api_key, base_url=base_url)
        completion = client.chat.completions.create
        request_options["model"] = model
    else:
        completion = completion_func

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": prepared_image.data_url},
                },
                {
                    "type": "text",
                    "text": _build_instruction(user_instruction),
                },
            ],
        }
    ]

    def request_ocr() -> Any:
        return completion(
            **request_options,
            messages=messages,
            extra_body={"enable_thinking": False},
            stream=False,
        )

    outcome = run_with_one_retry(
        request_ocr,
        should_retry=lambda error: True,
        delay_seconds=0.0,
    )
    if not outcome.succeeded:
        raise OCRClientError(
            OCR_ERROR_API,
            "OCR 模型调用失败。",
        ) from outcome.error

    content = _response_content(outcome.value)
    if content is None or not isinstance(content, str) or not content.strip():
        raise OCRClientError(
            OCR_ERROR_EMPTY,
            "OCR 模型返回了空内容。",
        )

    candidate, looks_like_json = _json_candidate(content)
    if not looks_like_json:
        return {
            "ocr_result": OCRResult(text=candidate),
            "model_requests": outcome.attempts,
        }

    try:
        payload = json.loads(candidate)
    except (json.JSONDecodeError, TypeError) as exc:
        raise OCRClientError(
            OCR_ERROR_PARSE,
            "OCR 模型返回内容看起来是 JSON，但无法解析。",
        ) from exc

    try:
        ocr_result = OCRResult.model_validate(payload)
    except ValidationError as exc:
        raise OCRClientError(
            OCR_ERROR_SCHEMA,
            "OCR 结果不符合字段要求。",
        ) from exc

    return {
        "ocr_result": ocr_result,
        "model_requests": outcome.attempts,
    }
