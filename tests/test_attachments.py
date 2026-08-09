"""Tests for safe local chat-image attachment persistence."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
import tempfile
import unittest

from PIL import Image

from src.storage.attachments import (
    AttachmentStoreError,
    delete_conversation_attachments,
    save_image_attachments,
)


def png_bytes(color: str = "white") -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (24, 16), color).save(buffer, format="PNG")
    return buffer.getvalue()


class AttachmentStoreTests(unittest.TestCase):
    def test_save_returns_safe_metadata_and_original_file(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "attachments"
            data = png_bytes()
            items = save_image_attachments(
                "conversation-1",
                [{"filename": "题图.png", "bytes": data}],
                root=root,
            )
            self.assertEqual(len(items), 1)
            self.assertEqual(items[0]["mime_type"], "image/png")
            self.assertEqual(items[0]["byte_size"], len(data))
            self.assertNotIn("bytes", items[0])
            self.assertNotIn("base64", str(items).lower())
            stored = list((root / "conversation-1").iterdir())
            self.assertEqual(len(stored), 1)
            self.assertEqual(stored[0].read_bytes(), data)

    def test_rejects_invalid_image_and_conversation_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "attachments"
            with self.assertRaises(AttachmentStoreError):
                save_image_attachments(
                    "../escape",
                    [{"filename": "x.png", "bytes": png_bytes()}],
                    root=root,
                )
            with self.assertRaises(AttachmentStoreError):
                save_image_attachments(
                    "safe",
                    [{"filename": "x.png", "bytes": b"not-image"}],
                    root=root,
                )

    def test_delete_only_target_conversation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir) / "attachments"
            for conversation_id in ("a", "b"):
                save_image_attachments(
                    conversation_id,
                    [{"filename": "x.png", "bytes": png_bytes(conversation_id == "a" and "red" or "blue")}],
                    root=root,
                )
            delete_conversation_attachments("a", root=root)
            self.assertFalse((root / "a").exists())
            self.assertTrue((root / "b").exists())


if __name__ == "__main__":
    unittest.main()
