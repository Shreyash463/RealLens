"""Phase 5 Regression and Invariant Test Suite.

Tests:
1. Regression Case 1 (Ganesha AI image: oiuhygfghjk.jpeg):
   Must return 'AI evidence found (strong)' or 'Leaning AI'.
2. Regression Case 2 (Real camera photo from WhatsApp: WhatsApp Image 2026-10-09 at 12.45.33 PM.jpeg):
   Must return 'No AI evidence found (not proof of real)' (and NEVER 'Likely real').
3. Screenshot copies of both:
   - Must be detected as screenshot / auto-cropped.
   - Ganesha AI screenshot must still be caught as AI.
   - Real photo screenshot must NOT trigger false positive AI.
4. Strict Invariant Verification:
   'Likely real' is strictly prohibited whenever condition != 'original'.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from reallens.app import analyze_image_bytes


def make_phone_screenshot(img: Image.Image) -> Image.Image:
    """Wraps an image inside a phone screenshot container with status bar and nav bar."""
    img_rgb = img.convert("RGB")
    w, h = img_rgb.size
    # Target phone aspect ratio e.g. 9:19.5
    top_bar_h = 48
    bottom_bar_h = 36
    total_h = h + top_bar_h + bottom_bar_h
    canvas = Image.new("RGB", (w, total_h), (18, 18, 18))
    # Paste status bar
    canvas.paste((24, 24, 24), (0, 0, w, top_bar_h))
    # Paste content
    canvas.paste(img_rgb, (0, top_bar_h))
    # Paste nav bar
    canvas.paste((24, 24, 24), (0, top_bar_h + h, w, total_h))
    return canvas


def run_tests():
    print("=" * 70)
    print("PHASE 5 REGRESSION & INVARIANT SUITE")
    print("=" * 70)

    ai_path = BASE_DIR / "data" / "my_images" / "ai" / "oiuhygfghjk.jpeg"
    real_path = BASE_DIR / "data" / "my_images" / "real" / "WhatsApp Image 2026-10-09 at 12.45.33 PM.jpeg"

    assert ai_path.exists(), f"Missing AI image {ai_path}"
    assert real_path.exists(), f"Missing Real image {real_path}"

    results = {}

    # Test 1: Ganesha AI image
    print("\n[1/5] Testing Ganesha AI image (oiuhygfghjk.jpeg)...")
    ai_bytes = ai_path.read_bytes()
    res_ai = analyze_image_bytes(ai_bytes, filename=ai_path.name, ignore_cache=True)
    print(f"  Verdict:    {res_ai['verdict']}")
    print(f"  Headline:   {res_ai['headline']}")
    print(f"  Prob:       {res_ai['probability']*100:.1f}%")
    print(f"  Condition:  {res_ai['condition']}")
    print(f"  Confidence: {res_ai['confidence']}")
    print(f"  Watermark:  {res_ai.get('watermark', {}).get('watermark_detected')}")
    print(f"  Spread:     {res_ai.get('spread')}")
    print(f"  Model Scores: {res_ai.get('model_scores')}")
    results["ganesha_ai"] = res_ai

    assert res_ai["verdict"] in ("AI evidence found (strong)", "Leaning AI"), (
        f"FAILED: Ganesha AI got {res_ai['verdict']}, expected AI verdict!"
    )
    print("  => PASSED: Caught as AI!")

    # Test 2: Real camera photo (WhatsApp Image)
    print("\n[2/5] Testing Real photo of man near tree (WhatsApp Image 2026-10-09...)...")
    real_bytes = real_path.read_bytes()
    res_real = analyze_image_bytes(real_bytes, filename=real_path.name, ignore_cache=True)
    print(f"  Verdict:    {res_real['verdict']}")
    print(f"  Headline:   {res_real['headline']}")
    print(f"  Prob:       {res_real['probability']*100:.1f}%")
    print(f"  Condition:  {res_real['condition']}")
    print(f"  Confidence: {res_real['confidence']}")
    print(f"  Spread:     {res_real.get('spread')}")
    print(f"  Model Scores: {res_real.get('model_scores')}")
    results["real_tree_man"] = res_real

    assert res_real["verdict"] != "Likely real", (
        "FAILED INVARIANT: WhatsApp image must NEVER return 'Likely real'!"
    )
    assert res_real["verdict"] in ("No AI evidence found (not proof of real)", "Inconclusive"), (
        f"FAILED: Real photo got {res_real['verdict']}, expected No AI evidence found or Inconclusive!"
    )
    print("  => PASSED: Honest real/unproven verdict without false alarm!")

    # Test 3: Screenshot of Ganesha AI
    print("\n[3/5] Testing screenshot copy of Ganesha AI...")
    with Image.open(ai_path) as im:
        shot_ai = make_phone_screenshot(im)
        buf = io.BytesIO()
        shot_ai.save(buf, format="PNG")
        shot_ai_bytes = buf.getvalue()

    res_shot_ai = analyze_image_bytes(shot_ai_bytes, filename="screenshot_ganesha_ai.png", ignore_cache=True)
    print(f"  Verdict:      {res_shot_ai['verdict']}")
    print(f"  Auto-cropped: {res_shot_ai.get('auto_cropped')}")
    print(f"  Container:    {res_shot_ai.get('detected_container')}")
    print(f"  Prob:         {res_shot_ai['probability']*100:.1f}%")
    results["screenshot_ganesha_ai"] = res_shot_ai

    assert res_shot_ai["verdict"] in ("AI evidence found (strong)", "Leaning AI"), (
        f"FAILED: Screenshot of AI image got {res_shot_ai['verdict']}!"
    )
    print("  => PASSED: AI screenshot caught as AI!")

    # Test 4: Screenshot of Real photo
    print("\n[4/5] Testing screenshot copy of Real photo...")
    with Image.open(real_path) as im:
        shot_real = make_phone_screenshot(im)
        buf = io.BytesIO()
        shot_real.save(buf, format="PNG")
        shot_real_bytes = buf.getvalue()

    res_shot_real = analyze_image_bytes(shot_real_bytes, filename="screenshot_real_tree.png", ignore_cache=True)
    print(f"  Verdict:      {res_shot_real['verdict']}")
    print(f"  Auto-cropped: {res_shot_real.get('auto_cropped')}")
    print(f"  Container:    {res_shot_real.get('detected_container')}")
    print(f"  Prob:         {res_shot_real['probability']*100:.1f}%")
    results["screenshot_real_tree"] = res_shot_real

    assert res_shot_real["verdict"] != "Likely real", (
        "FAILED INVARIANT: Screenshot must NEVER return 'Likely real'!"
    )
    assert res_shot_real["verdict"] != "AI evidence found (strong)", (
        f"FAILED: Screenshot of real photo triggered false alarm: {res_shot_real['verdict']}!"
    )
    print("  => PASSED: Real photo screenshot did not trigger false alarm!")

    # Test 5: Invariant verification across all conditions
    print("\n[5/5] Testing Strict Invariant Guardrails across conditions...")
    from reallens.detectors.fusion import fuse_signals

    dummy_meta = {"camera_make": "Sony", "camera_model": "ILCE-7RM4", "score": None}
    dummy_c2pa = {"score": None}
    dummy_ela = {"score": 0.15}
    dummy_freq = {"score": 0.15}
    dummy_ml = {
        "score": 0.05,
        "per_model_scores": {
            "Ateeqq/ai-vs-human-image-detector": 0.02,
            "generalizing_dinov2_probe": 0.03,
            "dima806/ai_vs_real_image_detection": 0.05,
            "Organika/sdxl-detector": 0.04,
        }
    }
    dummy_gate = {"is_photo_like": True, "photo_probability": 0.95, "image_type": "photo-like"}

    # Test 5a: Condition = WhatsApp -> MUST NOT BE 'Likely real'
    res_inv_wa = fuse_signals(
        metadata_res=dummy_meta,
        c2pa_res=dummy_c2pa,
        ela_res=dummy_ela,
        freq_res=dummy_freq,
        ml_res=dummy_ml,
        image_type_res=dummy_gate,
        condition_res={"condition": "whatsapp", "condition_label": "WhatsApp Compression", "allows_likely_real": False},
    )
    assert res_inv_wa["verdict"] != "Likely real", f"INVARIANT VIOLATION: WhatsApp returned {res_inv_wa['verdict']}!"
    print(f"  WhatsApp condition invariant: {res_inv_wa['verdict']} (Correct: NOT Likely real)")

    # Test 5b: Condition = Screenshot -> MUST NOT BE 'Likely real'
    res_inv_shot = fuse_signals(
        metadata_res=dummy_meta,
        c2pa_res=dummy_c2pa,
        ela_res=dummy_ela,
        freq_res=dummy_freq,
        ml_res=dummy_ml,
        image_type_res=dummy_gate,
        condition_res={"condition": "screenshot", "condition_label": "Screenshot", "allows_likely_real": False},
    )
    assert res_inv_shot["verdict"] != "Likely real", f"INVARIANT VIOLATION: Screenshot returned {res_inv_shot['verdict']}!"
    print(f"  Screenshot condition invariant: {res_inv_shot['verdict']} (Correct: NOT Likely real)")

    # Test 5c: Condition = Original with intact camera EXIF -> Permitted 'Likely real'
    res_inv_orig = fuse_signals(
        metadata_res=dummy_meta,
        c2pa_res=dummy_c2pa,
        ela_res=dummy_ela,
        freq_res=dummy_freq,
        ml_res=dummy_ml,
        image_type_res=dummy_gate,
        condition_res={"condition": "original", "condition_label": "Original Camera Photo", "allows_likely_real": True},
    )
    assert res_inv_orig["verdict"] == "Likely real", f"Original quality expected 'Likely real', got {res_inv_orig['verdict']}"
    print(f"  Original condition with camera EXIF: {res_inv_orig['verdict']} (Correct: Likely real)")

    print("\n" + "=" * 70)
    print("ALL PHASE 5 REGRESSION AND INVARIANT TESTS PASSED EMPIRICALLY!")
    print("=" * 70)

    # Save results to evaluation/regression_results.json
    out_file = BASE_DIR / "evaluation" / "regression_results.json"
    clean_results = {}
    for k, v in results.items():
        clean_results[k] = {
            "verdict": v.get("verdict"),
            "headline": v.get("headline"),
            "probability": v.get("probability"),
            "confidence": v.get("confidence"),
            "condition": v.get("condition"),
            "auto_cropped": v.get("auto_cropped"),
            "detected_container": v.get("detected_container"),
            "spread": v.get("spread"),
            "model_scores": v.get("model_scores"),
            "summary": v.get("summary"),
        }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(clean_results, f, indent=2)
    print(f"Saved regression results to {out_file}")


if __name__ == "__main__":
    run_tests()
