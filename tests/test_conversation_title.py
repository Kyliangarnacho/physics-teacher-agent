"""Tests for the minimal double-click conversation-title UI helper."""

from __future__ import annotations

import unittest

from src.ui.conversation_title import (
    _TITLE_JS,
    editable_conversation_title,
)


class ConversationTitleTests(unittest.TestCase):
    def test_component_declares_double_click_and_enter_submit(self) -> None:
        self.assertIn("dblclick", _TITLE_JS)
        self.assertIn('event.key === "Enter"', _TITLE_JS)
        self.assertIn('setTriggerValue("submitted"', _TITLE_JS)

    def test_title_input_requires_string(self) -> None:
        with self.assertRaises(ValueError):
            editable_conversation_title(123, current=False, key="test")


if __name__ == "__main__":
    unittest.main()
