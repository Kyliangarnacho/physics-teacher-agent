"""Safe, in-memory preprocessing for supported question images."""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO

from PIL import Image, ImageOps, UnidentifiedImageError

from src.vision.schemas import PreparedImage


MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_DECODE_PIXELS = 40_000_000
MAX_LONG_EDGE = 2048

FORMAT_TO_MIME = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


def _normalize_color_mode(image: Image.Image, image_format: str) -> Image.Image:
    if image_format == "JPEG":
        if image.mode in {"RGBA", "LA"} or (
            image.mode == "P" and "transparency" in image.info
        ):
            rgba = image.convert("RGBA")
            background = Image.new("RGBA", rgba.size, "white")
            return Image.alpha_composite(background, rgba).convert("RGB")
        return image if image.mode in {"RGB", "L"} else image.convert("RGB")

    if image_format in {"PNG", "WEBP"}:
        if image.mode in {"RGB", "RGBA", "L", "LA"}:
            return image
        if image.mode == "P" and "transparency" in image.info:
            return image.convert("RGBA")
        return image.convert("RGB")

    raise ValueError("图片格式不受支持。")


def _encode_image(image: Image.Image, image_format: str) -> bytes:
    output = BytesIO()
    save_options: dict[str, object] = {"format": image_format}
    if image_format == "JPEG":
        save_options.update(quality=95, optimize=True, progressive=False)
    elif image_format == "PNG":
        save_options.update(optimize=True)
    elif image_format == "WEBP":
        save_options.update(quality=95, method=6)
    image.save(output, **save_options)
    return output.getvalue()


def prepare_image(image_bytes: bytes) -> PreparedImage:
    """Validate and normalize supported image bytes without writing to disk."""
    if not isinstance(image_bytes, bytes):
        raise ValueError("图片内容必须是 bytes。")
    if not image_bytes:
        raise ValueError("图片内容不能为空。")
    if len(image_bytes) > MAX_UPLOAD_BYTES:
        raise ValueError("图片大小不能超过 8 MB。")

    try:
        with Image.open(BytesIO(image_bytes)) as opened_image:
            image_format = (opened_image.format or "").upper()
            if image_format not in FORMAT_TO_MIME:
                raise ValueError("仅支持真实的 JPEG、PNG 或 WEBP 图片。")

            width, height = opened_image.size
            if width <= 0 or height <= 0:
                raise ValueError("图片尺寸无效。")
            if width * height > MAX_DECODE_PIXELS:
                raise ValueError("图片解码像素数过大。")

            opened_image.load()
            normalized = ImageOps.exif_transpose(opened_image)
            normalized = _normalize_color_mode(normalized, image_format)

            resized = max(normalized.size) > MAX_LONG_EDGE
            if resized:
                normalized.thumbnail(
                    (MAX_LONG_EDGE, MAX_LONG_EDGE),
                    Image.Resampling.LANCZOS,
                )

            final_bytes = _encode_image(normalized, image_format)
            final_width, final_height = normalized.size
    except ValueError:
        raise
    except (UnidentifiedImageError, OSError, SyntaxError) as exc:
        raise ValueError("图片无法识别或文件已损坏。") from exc

    mime_type = FORMAT_TO_MIME[image_format]
    encoded = base64.b64encode(final_bytes).decode("ascii")
    return PreparedImage(
        image_hash=hashlib.sha256(final_bytes).hexdigest(),
        mime_type=mime_type,
        width=final_width,
        height=final_height,
        byte_size=len(final_bytes),
        data_url=f"data:{mime_type};base64,{encoded}",
        resized=resized,
    )
