"""One-shot Stage 09 probe for Qwen-compatible image understanding."""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from openai import OpenAI

from src.config import load_qwen_config


FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}
REQUIRED_RESPONSE_FIELDS = {
    "extracted_text",
    "visual_elements",
    "values_and_units",
    "uncertain_items",
}
EXTRACTION_INSTRUCTION = """只提取图片中的题目内容，不要解题，也不要补充图片中没有的信息。
只返回一个 JSON 对象，不使用 Markdown 代码块。JSON 至少包含以下字段：
extracted_text：提取出的题目文字；
visual_elements：图形、表格、电路、标注等视觉元素；
values_and_units：出现的数值及单位；
uncertain_items：无法确认或看不清的内容。"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="探测千问视觉模型的图片题目提取能力。")
    parser.add_argument("--image", required=True, type=Path, help="JPEG、PNG 或 WEBP 图片路径。")
    return parser.parse_args()


def inspect_image(image_path: Path) -> tuple[str, str, tuple[int, int], bytes]:
    """Validate an image and return format, MIME, dimensions, and original bytes."""
    if not image_path.is_file():
        raise FileNotFoundError(f"图片不存在：{image_path}")

    image_bytes = image_path.read_bytes()
    try:
        with Image.open(image_path) as image:
            image_format = (image.format or "").upper()
            image_size = image.size
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("图片无法被 Pillow 识别或文件已损坏。") from exc

    if image_format not in FORMAT_TO_MIME:
        supported = ", ".join(FORMAT_TO_MIME)
        raise ValueError(f"不支持的图片格式：{image_format or '未知'}；仅支持 {supported}。")

    return image_format, FORMAT_TO_MIME[image_format], image_size, image_bytes


def build_data_url(mime_type: str, image_bytes: bytes) -> str:
    """Build an in-memory Base64 data URL without writing encoded data to disk."""
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def safe_error_message(exc: Exception, data_url: str | None = None) -> str:
    """Return useful exception text while redacting any embedded Base64 payload."""
    message = str(exc)
    if data_url:
        message = message.replace(data_url, "[BASE64_DATA_URL_REDACTED]")
    return re.sub(
        r"data:image/(?:jpeg|png|webp);base64,[A-Za-z0-9+/=]+",
        "[BASE64_DATA_URL_REDACTED]",
        message,
    )


def validate_response(content: str | None) -> dict[str, Any]:
    """Validate non-empty JSON content and required extraction fields."""
    if content is None or not content.strip():
        raise ValueError("视觉模型返回了空内容。")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise ValueError(f"视觉模型返回内容不是合法 JSON：{exc.msg}。") from exc
    if not isinstance(parsed, dict):
        raise ValueError("视觉模型返回的 JSON 顶层必须是对象。")
    missing = sorted(REQUIRED_RESPONSE_FIELDS - parsed.keys())
    if missing:
        raise ValueError(f"视觉模型 JSON 缺少字段：{', '.join(missing)}。")
    return parsed


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parse_args()
    data_url: str | None = None

    try:
        image_format, mime_type, image_size, image_bytes = inspect_image(args.image)
        data_url = build_data_url(mime_type, image_bytes)

        api_key, base_url, _ = load_qwen_config()
        model = os.getenv("QWEN_VISION_MODEL", "").strip()
        if not model:
            raise RuntimeError("配置缺失：请在 .env 中设置 QWEN_VISION_MODEL。")

        client = OpenAI(api_key=api_key, base_url=base_url)
        started = time.perf_counter()
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": EXTRACTION_INSTRUCTION},
                    ],
                }
            ],
            extra_body={"enable_thinking": False},
            stream=False,
        )
        elapsed_ms = (time.perf_counter() - started) * 1000
        parsed = validate_response(response.choices[0].message.content)

        print(f"模型名：{model}")
        print(f"图片格式：{image_format}")
        print(f"图片尺寸：{image_size[0]}x{image_size[1]}")
        print(f"图片字节数：{len(image_bytes)}")
        print(f"耗时：{elapsed_ms:.2f} ms")
        print("解析结果：")
        print(json.dumps(parsed, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"探针失败：{type(exc).__name__}：{safe_error_message(exc, data_url)}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
