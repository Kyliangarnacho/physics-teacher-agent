"""Strict data contracts for prepared images and visual extraction results."""

from enum import Enum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    Strict,
    StringConstraints,
    model_validator,
)


Sha256Text = Annotated[
    StrictStr,
    StringConstraints(pattern=r"^[0-9a-f]{64}$"),
]
PositiveInt = Annotated[StrictInt, Field(gt=0)]
StrictStringList = Annotated[list[StrictStr], Strict()]


class PhysicsImageType(str, Enum):
    """The dominant physics-specific content represented by an image."""

    TEXT_ONLY = "text_only"
    CIRCUIT = "circuit"
    OPTICS = "optics"
    EXPERIMENT = "experiment"
    GRAPH = "graph"
    HANDWRITING = "handwriting"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ExtractionStatus(str, Enum):
    """Whether an image extraction can be used without user clarification."""

    COMPLETE = "complete"
    NEEDS_CONFIRMATION = "needs_confirmation"
    UNREADABLE = "unreadable"


class PreparedImage(BaseModel):
    """A validated image payload ready for an in-memory multimodal request."""

    model_config = ConfigDict(extra="forbid")

    image_hash: Sha256Text
    mime_type: Literal["image/jpeg", "image/png", "image/webp"]
    width: PositiveInt
    height: PositiveInt
    byte_size: PositiveInt
    data_url: StrictStr = Field(min_length=1, repr=False)
    resized: StrictBool

    @model_validator(mode="after")
    def validate_data_url_mime(self) -> "PreparedImage":
        expected_prefix = f"data:{self.mime_type};base64,"
        if not self.data_url.startswith(expected_prefix):
            raise ValueError("data_url 的 MIME 必须与 mime_type 一致。")
        return self


class ImageQuestionExtraction(BaseModel):
    """A flat, structured extraction of a physics question image."""

    model_config = ConfigDict(extra="forbid")

    status: ExtractionStatus
    image_type: PhysicsImageType
    extracted_text: StrictStr
    visual_elements: StrictStringList = Field(default_factory=list)
    relationships: StrictStringList = Field(default_factory=list)
    values_and_units: StrictStringList = Field(default_factory=list)
    formulas: StrictStringList = Field(default_factory=list)
    student_work: StrictStringList = Field(default_factory=list)
    uncertain_items: StrictStringList = Field(default_factory=list)
    suggested_user_question: StrictStr
    needs_ocr: StrictBool


class OCRResult(BaseModel):
    """A strict, flat result returned by a future OCR capability."""

    model_config = ConfigDict(extra="forbid")

    text: StrictStr
    formulas: StrictStringList = Field(default_factory=list)
    tables: StrictStringList = Field(default_factory=list)
    uncertain_fragments: StrictStringList = Field(default_factory=list)
