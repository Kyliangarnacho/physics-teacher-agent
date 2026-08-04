"""Safe, local orchestration helpers for small image batches."""

from __future__ import annotations

from collections.abc import Callable, Iterable, MutableMapping
from hashlib import sha256
from typing import Any
from uuid import uuid4

from src.vision.context import build_image_context_draft
from src.vision.schemas import ExtractionStatus, ImageQuestionExtraction
from src.vision.service import analyze_uploaded_image


MAX_BATCH_IMAGES = 3
SUPPORTED_IMAGE_MIME_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def merge_image_inputs(
    pasted_images: Iterable[dict[str, Any]],
    attached_images: Iterable[dict[str, Any]],
    *,
    max_images: int = MAX_BATCH_IMAGES,
) -> list[dict[str, Any]]:
    """Merge pasted then attached bytes, deduplicating by raw SHA-256."""
    merged: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for item in [*pasted_images, *attached_images]:
        filename = item.get("filename")
        image_bytes = item.get("bytes")
        mime_type = item.get("mime_type", "")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("图片文件名不能为空。")
        if not isinstance(image_bytes, bytes) or not image_bytes:
            raise ValueError("图片内容不能为空。")
        if mime_type and mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError("只支持 PNG、JPEG 或 WEBP 图片。")
        raw_hash = sha256(image_bytes).hexdigest()
        if raw_hash in seen_hashes:
            continue
        seen_hashes.add(raw_hash)
        merged.append(
            {
                "filename": filename.strip(),
                "bytes": image_bytes,
                "mime_type": mime_type,
                "raw_hash": raw_hash,
            }
        )
    if len(merged) > max_images:
        raise ValueError(f"一次最多提交 {max_images} 张图片。")
    return merged


def _safe_trace_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    safe_steps: list[dict[str, Any]] = []
    for step in value:
        if not isinstance(step, dict):
            continue
        safe_steps.append(
            {
                key: step.get(key)
                for key in (
                    "name",
                    "status",
                    "attempts",
                    "duration_ms",
                    "model_requests",
                    "error_type",
                    "error_message",
                )
            }
        )
    return safe_steps


def _cache_key(raw_hash: str, mode: str, user_instruction: str) -> str:
    instruction_hash = sha256(user_instruction.encode("utf-8")).hexdigest()
    return f"{raw_hash}:{mode}:{instruction_hash}"


def _safe_cached_result(result: dict[str, Any]) -> dict[str, Any]:
    prepared = result.get("prepared_image")
    extraction = result.get("extraction")
    if not isinstance(prepared, dict) or not isinstance(
        extraction,
        ImageQuestionExtraction,
    ):
        raise ValueError("题图识别结果格式无效。")
    image_hash = prepared.get("image_hash")
    if not isinstance(image_hash, str) or not image_hash:
        raise ValueError("题图识别结果缺少图片哈希。")
    return {
        "image_hash": image_hash,
        "extraction": extraction.model_dump(mode="json"),
        "vision_run_id": str(result.get("vision_run_id", "")),
        "model_requests": int(result.get("model_requests", 0)),
        "vision_step_traces": _safe_trace_list(
            result.get("vision_step_traces")
        ),
        "ocr_used": bool(result.get("ocr_used", False)),
    }


def process_image_batch(
    images: Iterable[dict[str, Any]],
    *,
    mode: str = "auto",
    user_instruction: str = "",
    cache: MutableMapping[str, dict[str, Any]] | None = None,
    analyze_func: Callable[..., dict[str, Any]] | None = None,
    progress_func: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Analyze up to three unique images and return only safe results."""
    unique_images = merge_image_inputs([], images)
    if not unique_images:
        raise ValueError("图片 Batch 不能为空。")
    cache_store = cache if cache is not None else {}
    vision_call = analyze_func or analyze_uploaded_image
    records: list[dict[str, Any]] = []
    actual_model_requests = 0
    total = len(unique_images)

    for index, item in enumerate(unique_images, start=1):
        key = _cache_key(item["raw_hash"], mode, user_instruction)
        cached = cache_store.get(key)
        cache_hit = cached is not None
        if cached is None:
            if progress_func is not None:
                progress_func(index, total)
            raw_result = vision_call(
                item["bytes"],
                item["filename"],
                mode=mode,
                user_instruction=user_instruction,
            )
            cached = _safe_cached_result(raw_result)
            cache_store[key] = cached
            actual_model_requests += cached["model_requests"]

        extraction = ImageQuestionExtraction.model_validate(
            cached["extraction"]
        )
        records.append(
            {
                "index": index,
                "filename": item["filename"],
                "raw_hash": item["raw_hash"],
                "image_hash": cached["image_hash"],
                "extraction": extraction,
                "vision_run_id": cached["vision_run_id"],
                "model_requests": cached["model_requests"],
                "vision_step_traces": [
                    dict(step) for step in cached["vision_step_traces"]
                ],
                "ocr_used": cached["ocr_used"],
                "cache_hit": cache_hit,
            }
        )

    return {
        "batch_id": uuid4().hex,
        "images": records,
        "model_requests": actual_model_requests,
    }


def image_needs_confirmation(record: dict[str, Any]) -> bool:
    extraction = record.get("extraction")
    if not isinstance(extraction, ImageQuestionExtraction):
        raise ValueError("Batch 图片缺少有效 extraction。")
    return (
        extraction.status is ExtractionStatus.NEEDS_CONFIRMATION
        or bool(extraction.uncertain_items)
    )


def image_is_unreadable(record: dict[str, Any]) -> bool:
    extraction = record.get("extraction")
    if not isinstance(extraction, ImageQuestionExtraction):
        raise ValueError("Batch 图片缺少有效 extraction。")
    return extraction.status is ExtractionStatus.UNREADABLE


def build_batch_image_context(
    records: Iterable[dict[str, Any]],
    *,
    context_overrides: dict[int, str] | None = None,
) -> str:
    """Combine per-image drafts in stable order without semantic guessing."""
    overrides = context_overrides or {}
    sections: list[str] = []
    for fallback_index, record in enumerate(records, start=1):
        index = record.get("index", fallback_index)
        filename = record.get("filename")
        extraction = record.get("extraction")
        if not isinstance(index, int) or index < 1:
            raise ValueError("图片 index 必须是正整数。")
        if not isinstance(filename, str) or not filename.strip():
            raise ValueError("图片 filename 不能为空。")
        if not isinstance(extraction, ImageQuestionExtraction):
            raise ValueError("Batch 图片缺少有效 extraction。")
        context = overrides.get(index)
        if context is None:
            context = build_image_context_draft(extraction)
        context = context.strip()
        if not context:
            raise ValueError(f"图片 {index} 的上下文不能为空。")
        sections.append(f"【图片 {index}：{filename}】\n{context}")
    if len(sections) > 1:
        answer_rule = (
            "【多图回答规则】\n"
            "多张图片若是独立题目，请按图片编号分别回答；"
            "若属于同一道题，请结合各图信息回答。"
        )
        sections.insert(0, answer_rule)
    return "\n\n".join(sections)
