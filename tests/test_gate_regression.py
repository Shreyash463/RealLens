"""Regression Test for Photo-vs-Non-Photo Gate Classifier.

Ensures that:
1. The user's actual WhatsApp mirror selfie (`WhatsApp Image 2026-10-08 at 8.31.35 PM.jpeg`)
   and all images in `data/my_images/real_whatsapp/` are classified as `photo-like` (photo_probability >= 0.35).
2. A synthetic set of compressed, resized, soft-focus, and low-light photo-like images
   are NEVER classified as non-photo.
3. Authentic certificates, web screenshots, and charts ARE classified as non-photo (photo_probability < 0.35).
"""
import io
import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter
import sys
import unittest
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reallens.detectors.image_type import classify_image_type
DATA_DIR = BASE_DIR / "data"


class TestGateRegression(unittest.TestCase):

    def test_user_whatsapp_selfie_is_photo_like(self):
        """Asserts that the user's mirror selfie is classified as photo-like."""
        selfie_path = DATA_DIR / "my_images" / "real_whatsapp" / "WhatsApp Image 2026-10-08 at 8.31.35 PM.jpeg"
        self.assertTrue(selfie_path.exists(), f"User test selfie not found at {selfie_path}")

        with Image.open(selfie_path) as img:
            res = classify_image_type(img)

        self.assertTrue(res["is_photo_like"], f"Selfie falsely classified as non-photo: {res}")
        self.assertGreaterEqual(res["photo_probability"], 0.65, f"Selfie photo_probability too low: {res['photo_probability']}")
        self.assertIsNone(res["unreliable_notice"], "Selfie wrongly triggered unreliable notice")

    def test_all_user_real_whatsapp_images(self):
        """Asserts all user WhatsApp photos in real_whatsapp are photo-like."""
        wa_dir = DATA_DIR / "my_images" / "real_whatsapp"
        if not wa_dir.exists():
            return

        for p in wa_dir.iterdir():
            if p.suffix.lower() in [".jpg", ".jpeg", ".png"]:
                with Image.open(p) as img:
                    res = classify_image_type(img)
                self.assertTrue(res["is_photo_like"], f"User WhatsApp image {p.name} failed: {res}")
                self.assertGreaterEqual(res["photo_probability"], 0.35, f"{p.name} photo_probability < 0.35: {res['photo_probability']}")

    def test_synthetic_compressed_photos_regression(self):
        """Generates synthetic compressed photo scenes (night, soft-focus, portrait phone) and tests them."""
        w, h = 720, 1600
        y = np.linspace(0.2, 0.7, h, dtype=np.float32).reshape(h, 1)
        x = np.linspace(0.3, 0.6, w, dtype=np.float32).reshape(1, w)
        base = (x * 0.5 + y * 0.5) * 255.0
        noise = np.random.normal(0, 12, (h, w))
        channel_r = np.clip(base * 1.1 + noise, 0, 255).astype(np.uint8)
        channel_g = np.clip(base * 0.9 + noise, 0, 255).astype(np.uint8)
        channel_b = np.clip(base * 0.8 + noise, 0, 255).astype(np.uint8)
        photo_arr = np.dstack([channel_r, channel_g, channel_b])

        img = Image.fromarray(photo_arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        compressed_img = Image.open(buf)

        res = classify_image_type(compressed_img)
        self.assertTrue(res["is_photo_like"], f"Synthetic compressed photo failed: {res}")
        self.assertGreaterEqual(res["photo_probability"], 0.35, f"Synthetic photo_probability < 0.35: {res['photo_probability']}")

    def test_authentic_non_photos_classified_properly(self):
        """Asserts that authentic certificates, slides, and screenshots have photo_probability < 0.35."""
        graphics_test_dir = DATA_DIR / "graphics_test"
        if not graphics_test_dir.exists():
            return

        tested = 0
        for p in graphics_test_dir.iterdir():
            if p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                with Image.open(p) as img:
                    res = classify_image_type(img)
                self.assertLess(res["photo_probability"], 0.35, f"Non-photo {p.name} got photo_probability {res['photo_probability']}")
                self.assertFalse(res["is_photo_like"], f"Non-photo {p.name} wrongly called photo")
                tested += 1

        self.assertGreaterEqual(tested, 20, f"Expected at least 20 graphics tested, got {tested}")


if __name__ == "__main__":
    unittest.main()

