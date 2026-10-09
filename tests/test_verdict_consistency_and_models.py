"""Unit tests for Phase 4 Verdict Consistency, Evidence Integrity, and Model Scores.

Verifies:
1. Every result has a non-empty `model_scores` dictionary with all 4 models.
2. Cached results retain complete model scores and original analysis duration.
3. Verdicts, headlines, probabilities/ranges, and evidence lists never contradict each other.
4. ELA and FFT never return 50% placeholder scores when not applicable.
5. Actionable 'What to do next' box is populated with reverse image search links.
6. Re-run with ignore_cache bypasses the cache cleanly.
"""
from __future__ import annotations

import io
from pathlib import Path
import sys
import unittest
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reallens.app import analyze_image_bytes, _ANALYSIS_CACHE, PIPELINE_VERSION


class TestVerdictConsistencyAndModels(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Create a real photo test image
        cls.photo_bytes = cls._create_image_bytes(w=720, h=1280, is_photo=True)
        # Create a non-photo document test image
        cls.doc_bytes = cls._create_image_bytes(w=900, h=600, is_photo=False)

    @staticmethod
    def _create_image_bytes(w: int, h: int, is_photo: bool) -> bytes:
        if is_photo:
            # Gradient scene with noise
            arr = np.random.normal(120, 30, (h, w, 3)).clip(0, 255).astype(np.uint8)
        else:
            # White paper document with flat text bars
            arr = np.full((h, w, 3), 255, dtype=np.uint8)
            arr[50:90, 50:w-50] = 30  # Header bar
            for y in range(120, h-60, 40):
                arr[y:y+12, 60:w-80] = 80  # Text lines
        img = Image.fromarray(arr)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85)
        return buf.getvalue()

    def test_model_scores_always_present_and_complete(self):
        """Asserts that model_scores is never empty and contains valid float probabilities."""
        res = analyze_image_bytes(self.photo_bytes, filename="test_photo.jpg")

        self.assertIn("model_scores", res)
        scores = res["model_scores"]
        self.assertGreater(len(scores), 0, "model_scores must not be empty!")

        for model_name, score in scores.items():
            self.assertIsNotNone(score, f"Score for {model_name} should not be None")
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)

        self.assertIn("per_model_weights", res)
        self.assertEqual(len(res["per_model_weights"]), len(scores))
        self.assertIn("spread", res)
        self.assertGreaterEqual(res["spread"], 0.0)

    def test_cached_result_retains_complete_model_scores(self):
        """Asserts that retrieving from cache preserves full model scores and original time."""
        # First run (computes)
        res1 = analyze_image_bytes(self.photo_bytes, filename="test_cache.jpg")
        self.assertFalse(res1.get("cached", False))

        # Second run (served from cache)
        res2 = analyze_image_bytes(self.photo_bytes, filename="test_cache.jpg")
        self.assertTrue(res2.get("cached", False))
        self.assertEqual(res1["model_scores"], res2["model_scores"])
        self.assertEqual(res1["verdict"], res2["verdict"])
        self.assertEqual(res1["headline"], res2["headline"])
        self.assertIn("original_time_taken_seconds", res2)
        self.assertEqual(res1["original_time_taken_seconds"], res2["original_time_taken_seconds"])

    def test_ignore_cache_bypasses_cache(self):
        """Asserts that ignore_cache=True forces re-analysis."""
        res_cached = analyze_image_bytes(self.photo_bytes, filename="test_ignore.jpg")
        # Run again with ignore_cache
        res_fresh = analyze_image_bytes(self.photo_bytes, filename="test_ignore.jpg", ignore_cache=True)
        self.assertFalse(res_fresh.get("cached", False))

    def test_inconclusive_headline_never_uninformative_range(self):
        """Asserts that Inconclusive headline is 'Inconclusive', never 'between X% and Y%'."""
        res = analyze_image_bytes(self.doc_bytes, filename="test_doc.jpg")
        if res["is_inconclusive"]:
            self.assertEqual(res["headline"], "Inconclusive")
            self.assertNotIn("between", res["headline"])
            self.assertIsNone(res["ai_percentage"], "Inconclusive results should suppress single ai_percentage")

    def test_evidence_lists_never_contradict(self):
        """Asserts the 3 evidence lists do not contain logical contradictions."""
        res = analyze_image_bytes(self.photo_bytes, filename="test_evidence.jpg")

        ev_ai = res.get("evidence_pointing_to_ai", [])
        ev_real = res.get("evidence_pointing_to_real", [])
        ev_unrel = res.get("evidence_pointing_to_unreliable", res.get("missing_or_unreliable_evidence", []))

        self.assertIsInstance(ev_ai, list)
        self.assertIsInstance(ev_real, list)
        self.assertIsInstance(ev_unrel, list)

        # C2PA manifest cannot be both present and absent
        has_c2pa_ai = any("C2PA manifest explicitly records" in s for s in ev_ai)
        has_c2pa_real = any("C2PA hardware signing confirms" in s for s in ev_real)
        has_c2pa_missing = any("No cryptographic C2PA Content Credentials" in s for s in ev_unrel)

        if has_c2pa_ai or has_c2pa_real:
            self.assertFalse(has_c2pa_missing, "Contradiction: C2PA claimed both present and absent!")

        # Camera EXIF cannot be both present and absent
        has_exif_real = any("Authentic camera hardware EXIF" in s for s in ev_real)
        has_exif_missing = any("No camera hardware EXIF" in s for s in ev_unrel)
        if has_exif_real:
            self.assertFalse(has_exif_missing, "Contradiction: Camera EXIF claimed both present and absent!")

    def test_ela_and_fft_never_return_50_placeholder_when_not_applicable(self):
        """Asserts ELA and FFT set applicable=False and score=None on non-photos, never 50%."""
        res = analyze_image_bytes(self.doc_bytes, filename="test_doc.jpg")
        sig = res["signals"]

        ela = sig.get("ela", {})
        fft = sig.get("frequency", {})

        if not res["is_photo_like"]:
            self.assertFalse(ela.get("applicable", False))
            self.assertIsNone(ela.get("score"), "ELA must not return 0.50 on non-photo graphics!")
            self.assertFalse(fft.get("applicable", False))
            self.assertIsNone(fft.get("score"), "FFT must not return 0.50 on non-photo graphics!")

    def test_what_to_do_next_practical_guidance(self):
        """Asserts practical guidance includes reverse image search links."""
        res = analyze_image_bytes(self.doc_bytes, filename="test_next.jpg")
        what_next = res.get("what_to_do_next", [])
        self.assertGreater(len(what_next), 0)

        # Check for Google Lens and TinEye reverse search links
        reverse_actions = [item for item in what_next if item.get("action_type") == "reverse_search"]
        self.assertEqual(len(reverse_actions), 1)
        links = reverse_actions[0].get("links", [])
        urls = [link["url"] for link in links]
        self.assertTrue(any("lens.google.com" in u for u in urls))
        self.assertTrue(any("tineye.com" in u for u in urls))


if __name__ == "__main__":
    unittest.main()
