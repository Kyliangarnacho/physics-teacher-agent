"""Tests for safe multi-image Vision Service orchestration."""

from __future__ import annotations

import json
import unittest

from src.vision.batch import (
    build_batch_image_context,
    image_is_unreadable,
    image_needs_confirmation,
    merge_image_inputs,
    process_image_batch,
)
from src.vision.schemas import ImageQuestionExtraction


def extraction(status: str = "complete", uncertain=None, text="题目文字"):
    return ImageQuestionExtraction(
        status=status,
        image_type="circuit",
        extracted_text=text,
        visual_elements=["电源", "电阻"],
        relationships=["电源与电阻串联"],
        values_and_units=["U=6 V"],
        formulas=[],
        student_work=[],
        uncertain_items=uncertain or [],
        suggested_user_question="求电流。",
        needs_ocr=False,
    )


def service_result(status="complete", text="题目文字"):
    return {
        "prepared_image": {
            "image_hash": "a" * 64,
            "mime_type": "image/png",
            "width": 20,
            "height": 10,
            "byte_size": 10,
            "resized": False,
        },
        "extraction": extraction(status=status, text=text),
        "ocr_used": False,
        "vision_run_id": f"run-{text}",
        "vision_step_traces": [
            {
                "name": "vision_extract",
                "status": "success",
                "attempts": 1,
                "duration_ms": 2,
                "model_requests": 1,
                "metadata": {"data_url": "must-not-be-cached"},
            }
        ],
        "model_requests": 1,
    }


def image(name: str, content: bytes) -> dict:
    return {"filename": name, "bytes": content, "mime_type": "image/png"}


class VisionBatchTests(unittest.TestCase):
    def test_pasted_then_attached_order_is_stable(self) -> None:
        merged = merge_image_inputs(
            [image("paste.png", b"paste")],
            [image("attach.png", b"attach")],
        )
        self.assertEqual(
            [item["filename"] for item in merged],
            ["paste.png", "attach.png"],
        )

    def test_same_batch_duplicate_is_removed(self) -> None:
        merged = merge_image_inputs(
            [image("first.png", b"same")],
            [image("duplicate.png", b"same")],
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["filename"], "first.png")

    def test_more_than_three_unique_images_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "最多提交 3 张"):
            merge_image_inputs(
                [],
                [image(f"{i}.png", bytes([i])) for i in range(1, 5)],
            )

    def test_batch_records_required_safe_fields_in_order(self) -> None:
        calls: list[str] = []

        def fake(_bytes, filename, **_kwargs):
            calls.append(filename)
            return service_result(text=filename)

        batch = process_image_batch(
            [image("1.png", b"one"), image("2.png", b"two")],
            analyze_func=fake,
        )

        self.assertTrue(batch["batch_id"])
        self.assertEqual(calls, ["1.png", "2.png"])
        self.assertEqual([r["index"] for r in batch["images"]], [1, 2])
        for record in batch["images"]:
            for key in (
                "filename",
                "raw_hash",
                "image_hash",
                "extraction",
                "vision_run_id",
                "model_requests",
                "vision_step_traces",
            ):
                self.assertIn(key, record)
        serialized = json.dumps(batch, default=lambda value: value.model_dump(mode="json"))
        self.assertNotIn("data_url", serialized)
        self.assertNotIn("base64", serialized.lower())

    def test_cache_avoids_repeated_vision_call(self) -> None:
        calls = 0
        cache = {}

        def fake(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return service_result()

        first = process_image_batch(
            [image("a.png", b"same")],
            mode="auto",
            user_instruction="核对电路",
            cache=cache,
            analyze_func=fake,
        )
        second = process_image_batch(
            [image("again.png", b"same")],
            mode="auto",
            user_instruction="核对电路",
            cache=cache,
            analyze_func=fake,
        )

        self.assertEqual(calls, 1)
        self.assertEqual(first["model_requests"], 1)
        self.assertEqual(second["model_requests"], 0)
        self.assertTrue(second["images"][0]["cache_hit"])

    def test_cache_key_changes_with_mode_or_instruction(self) -> None:
        calls = 0
        cache = {}

        def fake(*_args, **_kwargs):
            nonlocal calls
            calls += 1
            return service_result()

        for mode, instruction in (
            ("auto", "A"),
            ("vision", "A"),
            ("vision", "B"),
        ):
            process_image_batch(
                [image("a.png", b"same")],
                mode=mode,
                user_instruction=instruction,
                cache=cache,
                analyze_func=fake,
            )
        self.assertEqual(calls, 3)

    def test_batch_context_format_and_order(self) -> None:
        batch = process_image_batch(
            [image("first.png", b"1"), image("second.png", b"2")],
            analyze_func=lambda _b, filename, **_k: service_result(text=filename),
        )
        context = build_batch_image_context(batch["images"])
        self.assertLess(context.index("【图片 1：first.png】"), context.index("【图片 2：second.png】"))
        self.assertIn("【题目文字】", context)
        self.assertIn("【图形关系】", context)
        self.assertIn("独立题目，请按图片编号分别回答", context)
        self.assertIn("属于同一道题，请结合各图信息回答", context)

    def test_context_override_changes_only_selected_image(self) -> None:
        batch = process_image_batch(
            [image("1.png", b"1"), image("2.png", b"2")],
            analyze_func=lambda _b, filename, **_k: service_result(text=filename),
        )
        context = build_batch_image_context(
            batch["images"],
            context_overrides={2: "人工核对后的第二张图"},
        )
        self.assertIn("1.png", context)
        self.assertIn("人工核对后的第二张图", context)

    def test_confirmation_and_unreadable_classification(self) -> None:
        complete = {"extraction": extraction()}
        uncertain = {"extraction": extraction(uncertain=["需核对"])}
        confirmation = {"extraction": extraction("needs_confirmation")}
        unreadable = {"extraction": extraction("unreadable")}
        self.assertFalse(image_needs_confirmation(complete))
        self.assertTrue(image_needs_confirmation(uncertain))
        self.assertTrue(image_needs_confirmation(confirmation))
        self.assertTrue(image_is_unreadable(unreadable))


if __name__ == "__main__":
    unittest.main()
