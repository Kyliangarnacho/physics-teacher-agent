import base64
import hashlib
import unittest
from io import BytesIO
from unittest.mock import patch

from PIL import Image, features
from pydantic import ValidationError

from src.vision.image_utils import (
    MAX_UPLOAD_BYTES,
    prepare_image,
)
from src.vision.schemas import PreparedImage


def make_image_bytes(
    image_format="PNG",
    size=(32, 20),
    mode="RGB",
    color="navy",
    exif=None,
):
    output = BytesIO()
    image = Image.new(mode, size, color)
    save_options = {"format": image_format}
    if exif is not None:
        save_options["exif"] = exif
    image.save(output, **save_options)
    return output.getvalue()


class NamedBytes(bytes):
    """Bytes carrying a misleading name to prove names are ignored."""


class PreparedImageTests(unittest.TestCase):
    def test_jpeg_is_prepared_with_real_mime(self):
        prepared = prepare_image(make_image_bytes("JPEG"))

        self.assertEqual(prepared.mime_type, "image/jpeg")
        self.assertEqual((prepared.width, prepared.height), (32, 20))
        self.assertFalse(prepared.resized)

    def test_png_is_prepared_with_real_mime(self):
        prepared = prepare_image(make_image_bytes("PNG"))

        self.assertEqual(prepared.mime_type, "image/png")

    @unittest.skipUnless(features.check("webp"), "Pillow build lacks WEBP support")
    def test_webp_is_prepared_with_real_mime(self):
        prepared = prepare_image(make_image_bytes("WEBP"))

        self.assertEqual(prepared.mime_type, "image/webp")

    def test_misleading_filename_attribute_does_not_affect_mime(self):
        payload = NamedBytes(make_image_bytes("PNG"))
        payload.name = "actually-not-jpeg.jpg"

        prepared = prepare_image(payload)

        self.assertEqual(prepared.mime_type, "image/png")

    def test_empty_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "不能为空"):
            prepare_image(b"")

    def test_non_image_content_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "无法识别|已损坏"):
            prepare_image(b"this is not an image")

    def test_truncated_image_is_rejected(self):
        payload = make_image_bytes("JPEG", size=(80, 60))

        with self.assertRaisesRegex(ValueError, "无法识别|已损坏"):
            prepare_image(payload[: len(payload) // 2])

    def test_upload_larger_than_eight_mb_is_rejected_before_decode(self):
        with self.assertRaisesRegex(ValueError, "8 MB"):
            prepare_image(b"0" * (MAX_UPLOAD_BYTES + 1))

    def test_decoded_pixel_limit_is_enforced(self):
        payload = make_image_bytes("PNG", size=(11, 10))

        with patch("src.vision.image_utils.MAX_DECODE_PIXELS", 100):
            with self.assertRaisesRegex(ValueError, "像素数过大"):
                prepare_image(payload)

    def test_exif_orientation_is_applied(self):
        exif = Image.Exif()
        exif[274] = 6
        payload = make_image_bytes("JPEG", size=(40, 20), exif=exif)

        prepared = prepare_image(payload)

        self.assertEqual((prepared.width, prepared.height), (20, 40))

    def test_long_edge_is_resized_proportionally(self):
        payload = make_image_bytes("PNG", size=(4096, 1024))

        prepared = prepare_image(payload)

        self.assertEqual((prepared.width, prepared.height), (2048, 512))
        self.assertTrue(prepared.resized)

    def test_hash_and_output_are_stable_across_repeated_calls(self):
        payload = make_image_bytes("PNG", size=(40, 30), mode="RGBA")

        first = prepare_image(payload)
        second = prepare_image(payload)

        self.assertEqual(first.image_hash, second.image_hash)
        self.assertEqual(first.data_url, second.data_url)
        self.assertEqual(first.byte_size, second.byte_size)

    def test_data_url_decodes_to_final_image_bytes(self):
        prepared = prepare_image(make_image_bytes("PNG"))
        prefix, encoded = prepared.data_url.split(",", 1)
        decoded = base64.b64decode(encoded, validate=True)

        self.assertEqual(prefix, "data:image/png;base64")
        self.assertEqual(len(decoded), prepared.byte_size)
        self.assertEqual(hashlib.sha256(decoded).hexdigest(), prepared.image_hash)
        with Image.open(BytesIO(decoded)) as decoded_image:
            self.assertEqual(decoded_image.format, "PNG")
            self.assertEqual(decoded_image.size, (32, 20))

    def test_prepared_image_is_strict_and_hides_data_url_from_repr(self):
        prepared = prepare_image(make_image_bytes("PNG"))
        self.assertNotIn("data_url", repr(prepared))
        self.assertNotIn("base64", repr(prepared))

        data = prepared.model_dump()
        data["unexpected"] = "value"
        with self.assertRaises(ValidationError):
            PreparedImage.model_validate(data)


if __name__ == "__main__":
    unittest.main()
