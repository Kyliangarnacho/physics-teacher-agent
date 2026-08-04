"""Tests for the safe Python and fixed-JS paste image bridge."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.ui import paste_images


def payload() -> dict:
    return {
        "images": [
            {
                "filename": "pasted.png",
                "mime_type": "image/png",
                "bytes": [0, 1, 2, 255],
            }
        ]
    }


class PasteImageTests(unittest.TestCase):
    def test_binary_array_is_decoded_without_base64(self) -> None:
        decoded = paste_images.decode_pasted_images(payload())
        self.assertEqual(decoded[0]["bytes"], b"\x00\x01\x02\xff")
        self.assertNotIn("base64", str(decoded).lower())

    def test_invalid_byte_or_mime_is_ignored(self) -> None:
        invalid = payload()
        invalid["images"][0]["bytes"] = [False, 999]
        self.assertEqual(paste_images.decode_pasted_images(invalid), [])
        invalid = payload()
        invalid["images"][0]["mime_type"] = "image/gif"
        self.assertEqual(paste_images.decode_pasted_images(invalid), [])

    def test_decode_limits_to_three_images(self) -> None:
        items = [
            {
                "filename": f"pasted-{index}.png",
                "mime_type": "image/png",
                "bytes": [index],
            }
            for index in range(5)
        ]
        decoded = paste_images.decode_pasted_images({"images": items})
        self.assertEqual(len(decoded), 3)

    def test_duplicate_pasted_image_is_kept_once_by_sha256(self) -> None:
        item = payload()["images"][0]
        duplicate = dict(item, filename="same-again.png")
        decoded = paste_images.decode_pasted_images(
            {"images": [item, duplicate]}
        )
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["filename"], "pasted.png")

    def test_remove_returns_independent_ordered_list(self) -> None:
        images = [
            {"filename": "1.png"},
            {"filename": "2.png"},
            {"filename": "3.png"},
        ]
        result = paste_images.remove_pasted_image(images, 1)
        self.assertEqual([item["filename"] for item in result], ["1.png", "3.png"])
        result[0]["filename"] = "changed.png"
        self.assertEqual(images[0]["filename"], "1.png")

    def test_plain_text_paste_is_not_prevented(self) -> None:
        script = paste_images._PASTE_BRIDGE_JS
        self.assertLess(
            script.index("if (!files.length) return"),
            script.index("event.preventDefault()"),
        )

    def test_js_uses_fixed_dom_operations_and_supports_delete(self) -> None:
        script = paste_images._PASTE_BRIDGE_JS
        self.assertIn('document.addEventListener("paste"', script)
        self.assertIn('remove.textContent = "×"', script)
        self.assertIn("setStateValue(\"images\"", script)
        self.assertIn('crypto.subtle.digest("SHA-256"', script)
        self.assertIn('addEventListener("paste", onPaste, true)', script)
        self.assertIn("event.stopImmediatePropagation()", script)
        self.assertIn("name.textContent", script)
        self.assertNotIn("innerHTML", script)
        self.assertNotIn("eval(", script)

    def test_custom_preview_css_is_not_fixed_overlay(self) -> None:
        from pathlib import Path

        app_source = Path("app.py").read_text(encoding="utf-8")
        style_start = app_source.index(".st-key-paste_image_bridge {")
        style_end = app_source.index("}", style_start)
        paste_style = app_source[style_start:style_end]
        self.assertNotIn("position: fixed", paste_style)
        self.assertIn("margin:", paste_style)

    def test_unavailable_component_has_safe_failure(self) -> None:
        with patch.object(paste_images, "_PASTE_COMPONENT", None):
            with self.assertRaises(paste_images.PasteBridgeUnavailable):
                paste_images.paste_image_bridge(key="test")


if __name__ == "__main__":
    unittest.main()
