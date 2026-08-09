"""Local image attachment storage for persisted chat messages.

Only small validated JPEG/PNG/WEBP files are stored. SQLite keeps the returned
safe metadata; raw bytes and Data URLs never enter message or generation-job
payloads.
"""

from __future__ import annotations

from hashlib import sha256
from io import BytesIO
from pathlib import Path
import os
import re
import shutil
from typing import Any, Iterable
from uuid import uuid4

from PIL import Image, UnidentifiedImageError


MAX_ATTACHMENT_BYTES = 8 * 1024 * 1024
MAX_ATTACHMENT_PIXELS = 25_000_000
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ATTACHMENT_ROOT = _PROJECT_ROOT / "data" / "runtime" / "attachments"
_SAFE_ID = re.compile(r"^[A-Za-z0-9_-]+$")
_FORMAT_INFO = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}


class AttachmentStoreError(ValueError):
    """Safe attachment validation/storage error."""


def _attachment_root(root: Path | str | None) -> tuple[Path, bool]:
    if root is not None:
        return Path(root), True
    configured = os.getenv("PHYSICS_AGENT_ATTACHMENT_ROOT")
    if configured:
        return Path(configured), True
    return DEFAULT_ATTACHMENT_ROOT, False


def _safe_conversation_id(conversation_id: str) -> str:
    if not isinstance(conversation_id, str) or not _SAFE_ID.fullmatch(
        conversation_id
    ):
        raise AttachmentStoreError("会话 ID 格式无效。")
    return conversation_id


def _safe_filename(filename: object, fallback: str) -> str:
    if not isinstance(filename, str):
        return fallback
    cleaned = "".join(
        char for char in filename.strip() if char >= " " and char != "\x7f"
    )
    return cleaned[:255] or fallback


def _inspect_image(image_bytes: bytes) -> tuple[str, str, int, int]:
    if not isinstance(image_bytes, bytes) or not image_bytes:
        raise AttachmentStoreError("图片内容不能为空。")
    if len(image_bytes) > MAX_ATTACHMENT_BYTES:
        raise AttachmentStoreError("单张图片不能超过 8 MB。")
    try:
        with Image.open(BytesIO(image_bytes)) as image:
            image_format = str(image.format or "").upper()
            if image_format not in _FORMAT_INFO:
                raise AttachmentStoreError("只支持 JPEG、PNG 或 WEBP 图片。")
            width, height = image.size
            if width <= 0 or height <= 0 or width * height > MAX_ATTACHMENT_PIXELS:
                raise AttachmentStoreError("图片像素尺寸超出安全限制。")
            image.verify()
    except AttachmentStoreError:
        raise
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise AttachmentStoreError("图片已损坏或格式无效。") from exc
    mime_type, suffix = _FORMAT_INFO[image_format]
    return mime_type, suffix, width, height


def save_image_attachments(
    conversation_id: str,
    images: Iterable[dict[str, Any]],
    *,
    root: Path | str | None = None,
) -> list[dict[str, object]]:
    """Persist validated originals and return SQLite-safe metadata."""
    safe_id = _safe_conversation_id(conversation_id)
    attachment_root, custom_root = _attachment_root(root)
    conversation_dir = attachment_root / safe_id
    saved_paths: list[Path] = []
    metadata: list[dict[str, object]] = []
    try:
        for index, image in enumerate(images, start=1):
            image_bytes = image.get("bytes") if isinstance(image, dict) else None
            if not isinstance(image_bytes, bytes):
                raise AttachmentStoreError("图片内容格式无效。")
            mime_type, suffix, width, height = _inspect_image(image_bytes)
            attachment_id = uuid4().hex
            stored_name = f"{attachment_id}{suffix}"
            conversation_dir.mkdir(parents=True, exist_ok=True)
            stored_path = conversation_dir / stored_name
            stored_path.write_bytes(image_bytes)
            saved_paths.append(stored_path)
            metadata.append(
                {
                    "attachment_id": attachment_id,
                    "relative_path": str(
                        (Path(safe_id) / stored_name)
                        if custom_root
                        else (
                            Path("data")
                            / "runtime"
                            / "attachments"
                            / safe_id
                            / stored_name
                        )
                    ).replace("\\", "/"),
                    "filename": _safe_filename(
                        image.get("filename") if isinstance(image, dict) else None,
                        f"图片{index}{suffix}",
                    ),
                    "mime_type": mime_type,
                    "sha256": sha256(image_bytes).hexdigest(),
                    "byte_size": len(image_bytes),
                    "width": width,
                    "height": height,
                }
            )
    except Exception:
        for path in saved_paths:
            path.unlink(missing_ok=True)
        if conversation_dir.exists() and not any(conversation_dir.iterdir()):
            conversation_dir.rmdir()
        raise
    return metadata


def resolve_attachment_path(
    metadata: dict[str, object],
    *,
    root: Path | str | None = None,
) -> Path | None:
    """Resolve validated metadata without allowing traversal outside the store."""
    relative = metadata.get("relative_path") if isinstance(metadata, dict) else None
    if not isinstance(relative, str) or not relative:
        return None
    attachment_root, custom_root = _attachment_root(root)
    allowed_root = attachment_root.resolve()
    candidate = (
        (allowed_root / Path(relative)).resolve()
        if custom_root
        else (_PROJECT_ROOT.resolve() / Path(relative)).resolve()
    )
    try:
        candidate.relative_to(allowed_root)
    except ValueError:
        return None
    if not candidate.is_file() or candidate.stat().st_size > MAX_ATTACHMENT_BYTES:
        return None
    return candidate


def delete_conversation_attachments(
    conversation_id: str,
    *,
    root: Path | str | None = None,
) -> None:
    """Remove only the validated attachment directory for one conversation."""
    safe_id = _safe_conversation_id(conversation_id)
    attachment_root, _custom_root = _attachment_root(root)
    target = (attachment_root / safe_id).resolve()
    try:
        target.relative_to(attachment_root.resolve())
    except ValueError as exc:
        raise AttachmentStoreError("附件目录路径无效。") from exc
    if target.is_dir():
        shutil.rmtree(target)


def delete_saved_attachments(
    attachments: Iterable[dict[str, object]],
    *,
    root: Path | str | None = None,
) -> None:
    """Best-effort rollback for a just-saved attachment batch."""
    for metadata in attachments:
        path = resolve_attachment_path(metadata, root=root)
        if path is None:
            continue
        parent = path.parent
        path.unlink(missing_ok=True)
        if parent.is_dir() and not any(parent.iterdir()):
            parent.rmdir()
