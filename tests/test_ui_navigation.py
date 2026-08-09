"""Tests for lightweight sidebar-to-message navigation helpers."""

import unittest
from pathlib import Path

from src.ui.navigation import message_anchor


class UiNavigationTests(unittest.TestCase):
    def test_anchor_is_stable_and_sanitized(self) -> None:
        self.assertEqual(message_anchor("abc-123_def"), "message-abc-123_def")
        self.assertEqual(message_anchor("abc-123_def"), "message-abc-123_def")
        self.assertEqual(message_anchor('<script id="x">'), "message-scriptidx")

    def test_empty_anchor_stays_empty(self) -> None:
        self.assertEqual(message_anchor("<>"), "")

    def test_sidebar_scroll_region_and_scroll_guard_are_present(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "app.py").read_text(
            encoding="utf-8"
        )
        self.assertIn(".st-key-current_conversation_outline", source)
        self.assertIn("max-height: 18rem", source)
        self.assertIn("overflow-y: auto", source)
        self.assertIn("sessionStorage", source)
        self.assertIn("nearBottom", source)


if __name__ == "__main__":
    unittest.main()
