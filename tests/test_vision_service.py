import json
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from src.vision.schemas import ImageQuestionExtraction, OCRResult
from src.vision.service import (
    OCR_TABLE_HEADING,
    OCR_TEXT_HEADING,
    TEXT_CONFLICT_MESSAGE,
    analyze_uploaded_image,
    merge_vision_and_ocr,
)


def make_extraction(**overrides):
    data = {
        "status": "complete",
        "image_type": "circuit",
        "extracted_text": "电源电压为 6 V，求电流。",
        "visual_elements": ["电源", "电阻", "电流表"],
        "relationships": ["电阻和电流表串联"],
        "values_and_units": ["6 V"],
        "formulas": ["I=U/R"],
        "student_work": ["学生写出 U=IR"],
        "uncertain_items": [],
        "suggested_user_question": "请回答图中问题。",
        "needs_ocr": True,
    }
    data.update(overrides)
    return ImageQuestionExtraction.model_validate(data)


def make_png_bytes(size=(32, 20)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


class FakeVision:
    def __init__(self, extraction=None, model_requests=1):
        self.extraction = extraction or make_extraction(needs_ocr=False)
        self.model_requests = model_requests
        self.calls = []

    def __call__(self, prepared_image, user_instruction=""):
        self.calls.append((prepared_image, user_instruction))
        return {
            "extraction": self.extraction,
            "model_requests": self.model_requests,
        }


class FakeOCR:
    def __init__(self, ocr_result=None, model_requests=1):
        self.ocr_result = ocr_result or OCRResult(text="OCR 补充题目")
        self.model_requests = model_requests
        self.calls = []

    def __call__(self, prepared_image, user_instruction=""):
        self.calls.append((prepared_image, user_instruction))
        return {
            "ocr_result": self.ocr_result,
            "model_requests": self.model_requests,
        }


class VisionServiceTests(unittest.TestCase):
    def test_none_ocr_returns_equivalent_independent_copy(self):
        extraction = make_extraction()

        merged = merge_vision_and_ocr(extraction, None)

        self.assertEqual(merged, extraction)
        self.assertIsNot(merged, extraction)
        self.assertIsNot(merged.formulas, extraction.formulas)
        self.assertTrue(merged.needs_ocr)

    def test_inputs_are_not_modified(self):
        extraction = make_extraction()
        ocr = OCRResult(
            text="补充文字",
            formulas=["P=UI"],
            tables=["次数 | 电流/A"],
            uncertain_fragments=["末尾模糊"],
        )
        extraction_before = extraction.model_dump(mode="json")
        ocr_before = ocr.model_dump(mode="json")

        merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(extraction.model_dump(mode="json"), extraction_before)
        self.assertEqual(ocr.model_dump(mode="json"), ocr_before)

    def test_same_normalized_text_is_not_duplicated(self):
        extraction = make_extraction(extracted_text="电源电压为 6 V，求电流。")
        ocr = OCRResult(text=" 电源电压为 6 V，\n求电流。 ")

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.extracted_text, extraction.extracted_text)
        self.assertNotIn(OCR_TEXT_HEADING, merged.extracted_text)

    def test_ocr_supplement_is_appended_with_heading(self):
        extraction = make_extraction(extracted_text="电源电压为 6 V")
        ocr = OCRResult(text="电源电压为 6 V，开关闭合后求电流。")

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertIn(OCR_TEXT_HEADING, merged.extracted_text)
        self.assertTrue(merged.extracted_text.endswith(ocr.text))
        self.assertEqual(merged.status.value, "complete")

    def test_mutually_distinct_text_adds_conflict(self):
        extraction = make_extraction(extracted_text="电源电压为 6 V。")
        ocr = OCRResult(text="滑动变阻器阻值为 10 Ω。")

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertIn(OCR_TEXT_HEADING, merged.extracted_text)
        self.assertIn(TEXT_CONFLICT_MESSAGE, merged.uncertain_items)

    def test_formulas_are_stably_deduplicated(self):
        extraction = make_extraction(formulas=["I=U/R", "U=IR"])
        ocr = OCRResult(text="", formulas=["U=IR", "P=UI", "P=UI"])

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.formulas, ["I=U/R", "U=IR", "P=UI"])

    def test_ocr_tables_are_appended_to_text_not_visual_elements(self):
        extraction = make_extraction()
        ocr = OCRResult(text="", tables=["次数 | 电流/A", "次数 | 电流/A"])

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertIn(OCR_TABLE_HEADING, merged.extracted_text)
        self.assertEqual(merged.extracted_text.count("次数 | 电流/A"), 1)
        self.assertEqual(merged.visual_elements, extraction.visual_elements)

    def test_ocr_uncertain_fragments_are_prefixed_and_deduplicated(self):
        extraction = make_extraction(uncertain_items=["图中开关状态不清晰"])
        ocr = OCRResult(
            text="",
            uncertain_fragments=["末尾单位模糊", "末尾单位模糊"],
        )

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(
            merged.uncertain_items,
            ["图中开关状态不清晰", "OCR 不确定：末尾单位模糊"],
        )
        self.assertEqual(merged.status.value, "needs_confirmation")

    def test_complete_status_becomes_confirmation_on_text_conflict(self):
        extraction = make_extraction(status="complete", extracted_text="甲文字")
        ocr = OCRResult(text="乙文字")

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.status.value, "needs_confirmation")

    def test_unreadable_status_is_partially_recovered_by_ocr(self):
        extraction = make_extraction(
            status="unreadable",
            extracted_text="",
            image_type="unknown",
        )
        ocr = OCRResult(text="识别出的题目文字")

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.status.value, "needs_confirmation")
        self.assertFalse(merged.needs_ocr)
        self.assertIn("识别出的题目文字", merged.extracted_text)

    def test_unreadable_without_usable_ocr_content_stays_unreadable(self):
        extraction = make_extraction(
            status="unreadable",
            extracted_text="",
            image_type="unknown",
        )
        ocr = OCRResult(text="", uncertain_fragments=["整张图模糊"])

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.status.value, "unreadable")
        self.assertFalse(merged.needs_ocr)
        self.assertEqual(merged.uncertain_items, ["OCR 不确定：整张图模糊"])

    def test_visual_relationship_fields_remain_authoritative(self):
        extraction = make_extraction()
        expected = {
            "image_type": extraction.image_type,
            "visual_elements": list(extraction.visual_elements),
            "relationships": list(extraction.relationships),
            "values_and_units": list(extraction.values_and_units),
            "student_work": list(extraction.student_work),
            "suggested_user_question": extraction.suggested_user_question,
        }
        ocr = OCRResult(
            text="另一段文字",
            formulas=["P=UI"],
            tables=["表格内容"],
        )

        merged = merge_vision_and_ocr(extraction, ocr)

        self.assertEqual(merged.image_type, expected["image_type"])
        self.assertEqual(merged.visual_elements, expected["visual_elements"])
        self.assertEqual(merged.relationships, expected["relationships"])
        self.assertEqual(merged.values_and_units, expected["values_and_units"])
        self.assertEqual(merged.student_work, expected["student_work"])
        self.assertEqual(
            merged.suggested_user_question,
            expected["suggested_user_question"],
        )


class AnalyzeUploadedImageTests(unittest.TestCase):
    def test_invalid_mode_is_rejected_before_clients(self):
        vision = FakeVision()
        ocr = FakeOCR()

        with self.assertRaisesRegex(ValueError, "mode"):
            analyze_uploaded_image(
                make_png_bytes(),
                "question.png",
                mode="invalid",
                vision_func=vision,
                ocr_func=ocr,
            )

        self.assertEqual(vision.calls, [])
        self.assertEqual(ocr.calls, [])

    def test_preprocessing_returns_only_safe_image_metadata(self):
        result = analyze_uploaded_image(
            make_png_bytes(size=(64, 40)),
            "question.fake-jpg",
            mode="vision",
            vision_func=FakeVision(),
        )

        self.assertEqual(
            set(result["prepared_image"]),
            {"image_hash", "mime_type", "width", "height", "byte_size", "resized"},
        )
        self.assertEqual(result["prepared_image"]["mime_type"], "image/png")
        self.assertEqual(
            (result["prepared_image"]["width"], result["prepared_image"]["height"]),
            (64, 40),
        )

    def test_return_and_traces_do_not_contain_data_url(self):
        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="ocr_enhanced",
            vision_func=FakeVision(),
            ocr_func=FakeOCR(),
        )
        safe_payload = {
            **result,
            "extraction": result["extraction"].model_dump(mode="json"),
            "ocr_result": result["ocr_result"].model_dump(mode="json"),
        }
        serialized = json.dumps(safe_payload, ensure_ascii=False)

        self.assertNotIn("data_url", serialized)
        self.assertNotIn("base64", serialized)

    def test_auto_without_ocr_need_skips_ocr(self):
        vision = FakeVision(make_extraction(needs_ocr=False))
        ocr = FakeOCR()

        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="auto",
            vision_func=vision,
            ocr_func=ocr,
        )

        self.assertFalse(result["ocr_used"])
        self.assertIsNone(result["ocr_result"])
        self.assertEqual(ocr.calls, [])
        self.assertEqual(
            result["vision_step_traces"][2]["metadata"],
            {"reason": "not_requested"},
        )

    def test_auto_with_ocr_need_calls_ocr(self):
        vision = FakeVision(make_extraction(needs_ocr=True))
        ocr = FakeOCR(OCRResult(text="补充文字"))

        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="auto",
            vision_func=vision,
            ocr_func=ocr,
        )

        self.assertTrue(result["ocr_used"])
        self.assertEqual(len(ocr.calls), 1)

    def test_vision_mode_never_calls_ocr(self):
        vision = FakeVision(make_extraction(needs_ocr=True))
        ocr = FakeOCR()

        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="vision",
            vision_func=vision,
            ocr_func=ocr,
        )

        self.assertEqual(ocr.calls, [])
        self.assertEqual(
            result["vision_step_traces"][2]["metadata"],
            {"reason": "mode_vision"},
        )

    def test_ocr_enhanced_always_calls_ocr(self):
        vision = FakeVision(make_extraction(needs_ocr=False))
        ocr = FakeOCR()

        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="ocr_enhanced",
            vision_func=vision,
            ocr_func=ocr,
        )

        self.assertTrue(result["ocr_used"])
        self.assertEqual(len(ocr.calls), 1)

    def test_missing_default_ocr_configuration_skips_safely(self):
        vision = FakeVision(make_extraction(needs_ocr=True))

        with patch(
            "src.vision.service.load_ocr_config",
            side_effect=RuntimeError("QWEN_OCR_MODEL missing"),
        ), patch(
            "src.vision.service.extract_image_text",
            side_effect=AssertionError("OCR client must not run"),
        ):
            result = analyze_uploaded_image(
                make_png_bytes(),
                "question.png",
                mode="auto",
                vision_func=vision,
            )

        self.assertFalse(result["ocr_used"])
        self.assertEqual(result["model_requests"], 1)
        self.assertEqual(
            result["vision_step_traces"][2]["metadata"],
            {"reason": "ocr_not_configured"},
        )

    def test_injected_ocr_does_not_read_configuration(self):
        ocr = FakeOCR()

        with patch(
            "src.vision.service.load_ocr_config",
            side_effect=AssertionError("configuration must not be read"),
        ):
            result = analyze_uploaded_image(
                make_png_bytes(),
                "question.png",
                mode="ocr_enhanced",
                vision_func=FakeVision(),
                ocr_func=ocr,
            )

        self.assertTrue(result["ocr_used"])

    def test_user_instruction_is_forwarded_unchanged_to_both_clients(self):
        vision = FakeVision(make_extraction(needs_ocr=True))
        ocr = FakeOCR()
        instruction = "  重点识别右侧手写单位。  "

        analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="auto",
            user_instruction=instruction,
            vision_func=vision,
            ocr_func=ocr,
        )

        self.assertEqual(vision.calls[0][1], instruction)
        self.assertEqual(ocr.calls[0][1], instruction)

    def test_model_requests_are_summed_and_retry_statuses_recorded(self):
        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="ocr_enhanced",
            vision_func=FakeVision(model_requests=2),
            ocr_func=FakeOCR(model_requests=2),
        )

        self.assertEqual(result["model_requests"], 4)
        self.assertRegex(result["vision_run_id"], r"^[0-9a-f]{32}$")
        self.assertEqual(
            [step["name"] for step in result["vision_step_traces"]],
            ["image_preprocess", "vision_extract", "ocr_extract"],
        )
        self.assertEqual(
            [step["status"] for step in result["vision_step_traces"]],
            ["success", "retry_success", "retry_success"],
        )
        self.assertEqual(
            [step["model_requests"] for step in result["vision_step_traces"]],
            [0, 2, 2],
        )

    def test_final_extraction_uses_merged_ocr_result(self):
        vision_extraction = make_extraction(
            needs_ocr=True,
            extracted_text="视觉文字",
            formulas=["I=U/R"],
        )
        ocr_result = OCRResult(text="OCR 新文字", formulas=["P=UI"])

        result = analyze_uploaded_image(
            make_png_bytes(),
            "question.png",
            mode="auto",
            vision_func=FakeVision(vision_extraction),
            ocr_func=FakeOCR(ocr_result),
        )

        self.assertIn(OCR_TEXT_HEADING, result["extraction"].extracted_text)
        self.assertEqual(result["extraction"].formulas, ["I=U/R", "P=UI"])
        self.assertFalse(result["extraction"].needs_ocr)

    def test_filename_is_metadata_only_and_does_not_affect_format_or_disk(self):
        suspicious_name = "../../never-create-this.jpg"
        unexpected_path = Path("never-create-this.jpg")
        self.assertFalse(unexpected_path.exists())

        result = analyze_uploaded_image(
            make_png_bytes(),
            suspicious_name,
            mode="vision",
            vision_func=FakeVision(),
        )

        self.assertEqual(result["prepared_image"]["mime_type"], "image/png")
        self.assertEqual(
            result["vision_step_traces"][0]["metadata"]["filename"],
            suspicious_name,
        )
        self.assertFalse(unexpected_path.exists())

    def test_input_bytes_and_client_models_are_not_modified(self):
        image_bytes = make_png_bytes()
        original_bytes = bytes(image_bytes)
        extraction = make_extraction(needs_ocr=True)
        ocr_result = OCRResult(text="补充文字", formulas=["P=UI"])
        extraction_before = extraction.model_dump(mode="json")
        ocr_before = ocr_result.model_dump(mode="json")

        analyze_uploaded_image(
            image_bytes,
            "question.png",
            mode="auto",
            vision_func=FakeVision(extraction),
            ocr_func=FakeOCR(ocr_result),
        )

        self.assertEqual(image_bytes, original_bytes)
        self.assertEqual(extraction.model_dump(mode="json"), extraction_before)
        self.assertEqual(ocr_result.model_dump(mode="json"), ocr_before)


if __name__ == "__main__":
    unittest.main()
