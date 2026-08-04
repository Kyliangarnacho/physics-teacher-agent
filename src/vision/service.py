"""Pure local composition of visual extraction and optional OCR results."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any
from uuid import uuid4

from src.config import load_ocr_config
from src.observability import StepTimer, create_skipped_step
from src.schemas import StepStatus, StepTrace
from src.vision.client import extract_image_question
from src.vision.image_utils import prepare_image
from src.vision.ocr_client import extract_image_text
from src.vision.schemas import (
    ExtractionStatus,
    ImageQuestionExtraction,
    OCRResult,
)


OCR_TEXT_HEADING = "[OCR 补充文字]"
OCR_TABLE_HEADING = "[OCR 表格]"
OCR_UNCERTAIN_PREFIX = "OCR 不确定："
TEXT_CONFLICT_MESSAGE = "视觉提取文字与 OCR 文字存在差异，请核对原图。"
VALID_ANALYSIS_MODES = {"auto", "vision", "ocr_enhanced"}


def _normalize_text(text: str) -> str:
    return "".join(text.split())


def _stable_unique(items: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _append_section(text: str, heading: str, content: str) -> str:
    section = f"{heading}\n{content}"
    return f"{text.rstrip()}\n\n{section}" if text.strip() else section


def merge_vision_and_ocr(
    extraction: ImageQuestionExtraction,
    ocr_result: OCRResult | None,
) -> ImageQuestionExtraction:
    """Merge optional OCR text into a new visual extraction without guessing."""
    if not isinstance(extraction, ImageQuestionExtraction):
        raise ValueError("extraction 必须是 ImageQuestionExtraction。")
    if ocr_result is not None and not isinstance(ocr_result, OCRResult):
        raise ValueError("ocr_result 必须是 OCRResult 或 None。")
    if ocr_result is None:
        return extraction.model_copy(deep=True)

    merged_text = extraction.extracted_text
    normalized_visual_text = _normalize_text(extraction.extracted_text)
    ocr_text = ocr_result.text.strip()
    normalized_ocr_text = _normalize_text(ocr_text)

    text_conflict = bool(
        normalized_visual_text
        and normalized_ocr_text
        and normalized_visual_text not in normalized_ocr_text
        and normalized_ocr_text not in normalized_visual_text
    )
    ocr_text_already_present = bool(
        normalized_ocr_text
        and normalized_ocr_text in normalized_visual_text
    )
    if normalized_ocr_text and not ocr_text_already_present:
        merged_text = _append_section(
            merged_text,
            OCR_TEXT_HEADING,
            ocr_text,
        )

    tables = _stable_unique(
        table for table in ocr_result.tables if table.strip()
    )
    if tables:
        merged_text = _append_section(
            merged_text,
            OCR_TABLE_HEADING,
            "\n".join(tables),
        )

    formulas = _stable_unique(
        [*extraction.formulas, *ocr_result.formulas]
    )
    prefixed_uncertain = [
        f"{OCR_UNCERTAIN_PREFIX}{fragment}"
        for fragment in ocr_result.uncertain_fragments
        if fragment.strip()
    ]
    uncertain_items = _stable_unique(
        [*extraction.uncertain_items, *prefixed_uncertain]
    )
    added_uncertain = any(
        item not in extraction.uncertain_items for item in prefixed_uncertain
    )
    if text_conflict and TEXT_CONFLICT_MESSAGE not in uncertain_items:
        uncertain_items.append(TEXT_CONFLICT_MESSAGE)

    has_usable_ocr_content = bool(
        normalized_ocr_text
        or any(formula.strip() for formula in ocr_result.formulas)
        or tables
    )
    if extraction.status is ExtractionStatus.UNREADABLE:
        status = (
            ExtractionStatus.NEEDS_CONFIRMATION
            if has_usable_ocr_content
            else ExtractionStatus.UNREADABLE
        )
    elif text_conflict or added_uncertain:
        status = ExtractionStatus.NEEDS_CONFIRMATION
    else:
        status = extraction.status

    payload = extraction.model_dump(mode="python")
    payload.update(
        status=status,
        extracted_text=merged_text,
        formulas=formulas,
        uncertain_items=uncertain_items,
        needs_ocr=False,
    )
    return ImageQuestionExtraction.model_validate(payload)


def _request_status(model_requests: int) -> StepStatus:
    if model_requests == 1:
        return StepStatus.SUCCESS
    if model_requests == 2:
        return StepStatus.RETRY_SUCCESS
    raise ValueError("Client 的 model_requests 只能是 1 或 2。")


def _safe_prepared_metadata(prepared_image: Any) -> dict[str, Any]:
    return {
        "image_hash": prepared_image.image_hash,
        "mime_type": prepared_image.mime_type,
        "width": prepared_image.width,
        "height": prepared_image.height,
        "byte_size": prepared_image.byte_size,
        "resized": prepared_image.resized,
    }


def analyze_uploaded_image(
    image_bytes: bytes,
    filename: str,
    mode: str = "auto",
    user_instruction: str = "",
    vision_func=None,
    ocr_func=None,
) -> dict[str, Any]:
    """Prepare and analyze one uploaded image with optional OCR enhancement."""
    if mode not in VALID_ANALYSIS_MODES:
        raise ValueError("mode 只允许 auto、vision 或 ocr_enhanced。")
    if not isinstance(filename, str):
        raise ValueError("filename 必须是字符串。")
    if not isinstance(user_instruction, str):
        raise ValueError("user_instruction 必须是字符串。")

    run_id = uuid4().hex
    step_traces: list[StepTrace] = []

    preprocess_timer = StepTimer("image_preprocess")
    prepared_image = prepare_image(image_bytes)
    step_traces.append(
        preprocess_timer.finish(
            status=StepStatus.SUCCESS,
            attempts=1,
            model_requests=0,
            metadata={
                "filename": filename,
                "mime_type": prepared_image.mime_type,
                "resized": prepared_image.resized,
            },
        )
    )

    vision_timer = StepTimer("vision_extract")
    vision_call = vision_func if vision_func is not None else extract_image_question
    vision_result = vision_call(
        prepared_image,
        user_instruction=user_instruction,
    )
    extraction = vision_result.get("extraction")
    vision_requests = vision_result.get("model_requests")
    if not isinstance(extraction, ImageQuestionExtraction):
        raise ValueError("Vision Client 必须返回 ImageQuestionExtraction。")
    vision_status = _request_status(vision_requests)
    step_traces.append(
        vision_timer.finish(
            status=vision_status,
            attempts=vision_requests,
            model_requests=vision_requests,
            metadata={"image_type": extraction.image_type.value},
        )
    )

    should_try_ocr = mode == "ocr_enhanced" or (
        mode == "auto" and extraction.needs_ocr
    )
    ocr_result: OCRResult | None = None
    ocr_requests = 0
    ocr_used = False

    if mode == "vision":
        step_traces.append(
            create_skipped_step("ocr_extract", {"reason": "mode_vision"})
        )
    elif not should_try_ocr:
        step_traces.append(
            create_skipped_step("ocr_extract", {"reason": "not_requested"})
        )
    else:
        ocr_call = ocr_func
        if ocr_call is None:
            try:
                load_ocr_config()
            except RuntimeError:
                step_traces.append(
                    create_skipped_step(
                        "ocr_extract",
                        {"reason": "ocr_not_configured"},
                    )
                )
            else:
                ocr_call = extract_image_text

        if ocr_call is not None:
            ocr_timer = StepTimer("ocr_extract")
            ocr_response = ocr_call(
                prepared_image,
                user_instruction=user_instruction,
            )
            ocr_result = ocr_response.get("ocr_result")
            ocr_requests = ocr_response.get("model_requests")
            if not isinstance(ocr_result, OCRResult):
                raise ValueError("OCR Client 必须返回 OCRResult。")
            ocr_status = _request_status(ocr_requests)
            step_traces.append(
                ocr_timer.finish(
                    status=ocr_status,
                    attempts=ocr_requests,
                    model_requests=ocr_requests,
                )
            )
            ocr_used = True

    final_extraction = (
        merge_vision_and_ocr(extraction, ocr_result)
        if ocr_used
        else extraction.model_copy(deep=True)
    )
    return {
        "prepared_image": _safe_prepared_metadata(prepared_image),
        "extraction": final_extraction,
        "ocr_result": ocr_result.model_copy(deep=True) if ocr_result else None,
        "ocr_used": ocr_used,
        "vision_run_id": run_id,
        "vision_step_traces": [
            trace.model_dump(mode="json") for trace in step_traces
        ],
        "model_requests": vision_requests + ocr_requests,
    }
