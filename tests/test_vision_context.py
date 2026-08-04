"""Tests for editable context built from structured image extraction."""

from __future__ import annotations

import unittest

from src.vision.context import build_image_context_draft
from src.vision.schemas import ImageQuestionExtraction


def make_extraction(**overrides) -> ImageQuestionExtraction:
    payload = {
        "status": "needs_confirmation",
        "image_type": "circuit",
        "extracted_text": "如图，电源电压为 6 V。",
        "visual_elements": ["电源", "电阻 R"],
        "relationships": ["电源与电阻串联"],
        "values_and_units": ["U=6 V"],
        "formulas": ["I=U/R"],
        "student_work": ["学生写出 I=R/U"],
        "uncertain_items": ["电表量程不清楚"],
        "suggested_user_question": "请判断学生公式哪里错了。",
        "needs_ocr": False,
    }
    payload.update(overrides)
    return ImageQuestionExtraction.model_validate(payload)


class ImageContextDraftTests(unittest.TestCase):
    def test_all_supported_sections_are_formatted(self) -> None:
        draft = build_image_context_draft(make_extraction())

        for heading in (
            "【题目文字】",
            "【图中元素】",
            "【图形关系】",
            "【数值与单位】",
            "【公式】",
            "【学生过程】",
            "【不确定内容】",
            "【建议问题】",
        ):
            self.assertIn(heading, draft)
        self.assertIn("- 电源", draft)
        self.assertIn("学生写出 I=R/U", draft)

    def test_empty_fields_are_skipped(self) -> None:
        extraction = make_extraction(
            visual_elements=[],
            relationships=[],
            formulas=[],
            student_work=[],
            uncertain_items=[],
            suggested_user_question="",
        )

        draft = build_image_context_draft(extraction)

        self.assertIn("【题目文字】", draft)
        self.assertIn("【数值与单位】", draft)
        for heading in (
            "【图中元素】",
            "【图形关系】",
            "【公式】",
            "【学生过程】",
            "【不确定内容】",
            "【建议问题】",
        ):
            self.assertNotIn(heading, draft)

    def test_draft_uses_no_prepared_image_or_data_url_field(self) -> None:
        draft = build_image_context_draft(make_extraction())

        self.assertNotIn("data_url", draft)
        self.assertNotIn("base64", draft.lower())
        self.assertNotIn("image_hash", draft)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "ImageQuestionExtraction"):
            build_image_context_draft({})  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
