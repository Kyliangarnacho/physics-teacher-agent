import unittest

from pydantic import ValidationError

from src.vision import (
    ExtractionStatus,
    ImageQuestionExtraction,
    OCRResult,
    PhysicsImageType,
    PreparedImage,
)


def make_extraction(**overrides):
    data = {
        "status": "complete",
        "image_type": "circuit",
        "extracted_text": "如图所示，电源电压为 6 V。",
        "visual_elements": ["电源", "开关", "电阻"],
        "relationships": ["电阻与电流表串联"],
        "values_and_units": ["电源电压 6 V"],
        "formulas": ["I=U/R"],
        "student_work": [],
        "uncertain_items": [],
        "suggested_user_question": "请根据图中电路求电流。",
        "needs_ocr": False,
    }
    data.update(overrides)
    return data


class VisionEnumTests(unittest.TestCase):
    def test_physics_image_type_contains_all_expected_values(self):
        self.assertEqual(
            {item.value for item in PhysicsImageType},
            {
                "text_only",
                "circuit",
                "optics",
                "experiment",
                "graph",
                "handwriting",
                "mixed",
                "unknown",
            },
        )

    def test_extraction_status_contains_all_expected_values(self):
        self.assertEqual(
            {item.value for item in ExtractionStatus},
            {"complete", "needs_confirmation", "unreadable"},
        )


class ImageQuestionExtractionTests(unittest.TestCase):
    def test_complete_extraction_accepts_all_fields(self):
        extraction = ImageQuestionExtraction.model_validate(make_extraction())

        self.assertIs(extraction.status, ExtractionStatus.COMPLETE)
        self.assertIs(extraction.image_type, PhysicsImageType.CIRCUIT)
        self.assertEqual(extraction.values_and_units, ["电源电压 6 V"])
        self.assertFalse(extraction.needs_ocr)

    def test_missing_required_field_is_rejected(self):
        data = make_extraction()
        del data["extracted_text"]

        with self.assertRaises(ValidationError):
            ImageQuestionExtraction.model_validate(data)

    def test_extra_field_is_rejected(self):
        with self.assertRaises(ValidationError):
            ImageQuestionExtraction.model_validate(
                make_extraction(confidence=0.9)
            )

    def test_wrong_scalar_and_list_types_are_rejected(self):
        invalid_cases = (
            make_extraction(extracted_text=123),
            make_extraction(needs_ocr="false"),
            make_extraction(visual_elements=("电源",)),
            make_extraction(values_and_units=[6]),
        )
        for data in invalid_cases:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    ImageQuestionExtraction.model_validate(data)

    def test_list_defaults_are_not_shared(self):
        required = {
            "status": "complete",
            "image_type": "text_only",
            "extracted_text": "题目文字",
            "suggested_user_question": "请回答图片中的问题。",
            "needs_ocr": False,
        }
        first = ImageQuestionExtraction.model_validate(required)
        second = ImageQuestionExtraction.model_validate(required)

        first.visual_elements.append("新增元素")

        self.assertEqual(first.visual_elements, ["新增元素"])
        self.assertEqual(second.visual_elements, [])
        self.assertIsNot(first.visual_elements, second.visual_elements)

    def test_confirmation_and_unreadable_statuses_are_valid(self):
        for status in ("needs_confirmation", "unreadable"):
            with self.subTest(status=status):
                extraction = ImageQuestionExtraction.model_validate(
                    make_extraction(
                        status=status,
                        image_type="unknown",
                        extracted_text="",
                        uncertain_items=["题目右侧模糊"],
                        needs_ocr=True,
                    )
                )
                self.assertEqual(extraction.status.value, status)


class OCRResultTests(unittest.TestCase):
    def test_valid_ocr_result_and_safe_defaults(self):
        result = OCRResult(
            text="电源电压为 6 V",
            formulas=["I=U/R"],
            tables=["表头：次数、电流"],
            uncertain_fragments=["末尾单位不清晰"],
        )
        defaulted = OCRResult(text="纯文字")

        self.assertEqual(result.formulas, ["I=U/R"])
        self.assertEqual(defaulted.formulas, [])
        self.assertEqual(defaulted.tables, [])

    def test_invalid_ocr_inputs_are_rejected(self):
        invalid_cases = (
            {},
            {"text": 123},
            {"text": "内容", "formulas": "I=U/R"},
            {"text": "内容", "tables": [1]},
            {"text": "内容", "extra": "不允许"},
        )
        for data in invalid_cases:
            with self.subTest(data=data):
                with self.assertRaises(ValidationError):
                    OCRResult.model_validate(data)


class PreparedImageCompatibilityTests(unittest.TestCase):
    def test_prepared_image_behavior_is_preserved(self):
        prepared = PreparedImage(
            image_hash="a" * 64,
            mime_type="image/png",
            width=10,
            height=20,
            byte_size=12,
            data_url="data:image/png;base64,AAAA",
            resized=False,
        )

        self.assertNotIn("data_url", repr(prepared))
        with self.assertRaises(ValidationError):
            PreparedImage(
                image_hash="a" * 64,
                mime_type="image/png",
                width=10,
                height=20,
                byte_size=12,
                data_url="data:image/png;base64,AAAA",
                resized=False,
                extra="forbidden",
            )


if __name__ == "__main__":
    unittest.main()
