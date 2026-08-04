"""千问 API 配置加载。"""

import os

from dotenv import load_dotenv


def load_qwen_config() -> tuple[str, str, str]:
    """从 .env 加载并校验千问 API 配置。"""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_MODEL")

    config = {
        "DASHSCOPE_API_KEY": api_key,
        "QWEN_BASE_URL": base_url,
        "QWEN_MODEL": model,
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(f"配置缺失：请在 .env 中设置 {', '.join(missing)}。")

    return api_key, base_url, model


def load_vision_config() -> tuple[str, str, str]:
    """从 .env 加载并校验独立的千问视觉模型配置。"""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_VISION_MODEL")

    config = {
        "DASHSCOPE_API_KEY": api_key,
        "QWEN_BASE_URL": base_url,
        "QWEN_VISION_MODEL": model,
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(f"视觉配置缺失：请在 .env 中设置 {', '.join(missing)}。")

    return api_key, base_url, model


def load_ocr_config() -> tuple[str, str, str]:
    """从 .env 加载并校验独立的千问 OCR 模型配置。"""
    load_dotenv()

    api_key = os.getenv("DASHSCOPE_API_KEY")
    base_url = os.getenv("QWEN_BASE_URL")
    model = os.getenv("QWEN_OCR_MODEL")

    config = {
        "DASHSCOPE_API_KEY": api_key,
        "QWEN_BASE_URL": base_url,
        "QWEN_OCR_MODEL": model,
    }
    missing = [name for name, value in config.items() if not value]
    if missing:
        raise RuntimeError(f"OCR 配置缺失：请在 .env 中设置 {', '.join(missing)}。")

    return api_key, base_url, model
