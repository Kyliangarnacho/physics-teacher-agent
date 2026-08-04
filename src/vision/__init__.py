"""Image preprocessing utilities for the physics teacher project."""

from src.vision.client import VisionClientError, extract_image_question
from src.vision.batch import (
    MAX_BATCH_IMAGES,
    build_batch_image_context,
    image_is_unreadable,
    image_needs_confirmation,
    merge_image_inputs,
    process_image_batch,
)
from src.vision.context import build_image_context_draft
from src.vision.image_utils import prepare_image
from src.vision.ocr_client import OCRClientError, extract_image_text
from src.vision.service import analyze_uploaded_image, merge_vision_and_ocr
from src.vision.schemas import (
    ExtractionStatus,
    ImageQuestionExtraction,
    OCRResult,
    PhysicsImageType,
    PreparedImage,
)

__all__ = [
    "ExtractionStatus",
    "ImageQuestionExtraction",
    "OCRClientError",
    "OCRResult",
    "PhysicsImageType",
    "PreparedImage",
    "VisionClientError",
    "MAX_BATCH_IMAGES",
    "analyze_uploaded_image",
    "build_batch_image_context",
    "build_image_context_draft",
    "extract_image_question",
    "extract_image_text",
    "image_is_unreadable",
    "image_needs_confirmation",
    "merge_vision_and_ocr",
    "merge_image_inputs",
    "prepare_image",
    "process_image_batch",
]
