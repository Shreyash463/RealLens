"""Unit tests for Phase 3:
1. Verdict summary counters (verifying all 6 verdict levels).
2. WhatsApp / recompressed image routing to Inconclusive.
3. Cold vs warm SHA-256 caching latency (<5ms on repeat).
"""
import io
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PIL import Image

from reallens.detectors.image_type import classify_image_type
from reallens.detectors.fusion import fuse_signals
from reallens.app import analyze_image_bytes, _ANALYSIS_CACHE


def count_verdicts(results_list):
    """Reference implementation of Phase 3 summary counters matching main.js."""
    counts = {
        "Likely AI": 0,
        "Leaning AI": 0,
        "Inconclusive": 0,
        "Leaning Real": 0,
        "Likely Real": 0,
        "Failed": 0,
    }
    for item in results_list:
        v = (item.get("verdict") or "").strip()
        status = item.get("status")
        if status == "error" or not v:
            counts["Failed"] += 1
            continue

        v_lower = v.lower()
        if v_lower == "likely ai":
            counts["Likely AI"] += 1
        elif v_lower == "leaning ai":
            counts["Leaning AI"] += 1
        elif "inconclusive" in v_lower:
            counts["Inconclusive"] += 1
        elif v_lower in ("leaning real", "leaning real"):
            counts["Leaning Real"] += 1
        elif v_lower in ("likely real", "likely real"):
            counts["Likely Real"] += 1
        elif v_lower == "failed":
            counts["Failed"] += 1
        else:
            counts["Failed"] += 1

    return counts


def test_every_verdict_counter_increments():
    """Verifies that every single verdict level is accounted for and counted correctly."""
    mock_results = [
        {"verdict": "Likely AI", "status": "success"},
        {"verdict": "Likely AI", "status": "success"},
        {"verdict": "Leaning AI", "status": "success"},
        {"verdict": "Inconclusive", "status": "success"},
        {"verdict": "Leaning Real", "status": "success"},
        {"verdict": "Leaning real", "status": "success"},  # case variation
        {"verdict": "Likely Real", "status": "success"},
        {"verdict": "Likely real", "status": "success"},    # case variation
        {"verdict": "Failed", "status": "error"},
        {"verdict": None, "status": "error"},               # network or server error
    ]

    counts = count_verdicts(mock_results)

    assert counts["Likely AI"] == 2, f"Expected 2 Likely AI, got {counts['Likely AI']}"
    assert counts["Leaning AI"] == 1, f"Expected 1 Leaning AI, got {counts['Leaning AI']}"
    assert counts["Inconclusive"] == 1, f"Expected 1 Inconclusive, got {counts['Inconclusive']}"
    assert counts["Leaning Real"] == 2, f"Expected 2 Leaning Real, got {counts['Leaning Real']}"
    assert counts["Likely Real"] == 2, f"Expected 2 Likely Real, got {counts['Likely Real']}"
    assert counts["Failed"] == 2, f"Expected 2 Failed, got {counts['Failed']}"


def test_whatsapp_recompressed_image_is_inconclusive():
    """Verifies that a WhatsApp-compressed 1280x720 image with non-consensus models

    is honestly classified as Inconclusive instead of forcing a guess.
    """
    # Create a 1280x720 JPEG with moderate quality
    img = Image.new("RGB", (1280, 720), (140, 160, 180))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    buf.seek(0)
    recompressed_img = Image.open(buf)

    type_info = classify_image_type(recompressed_img)
    # Check that is_chat_dim is detected
    assert type_info["is_chat_dim"] is True
    assert type_info["is_heavily_recompressed"] is True

    # Simulate model scores where models do not have overwhelming agreement
    # (e.g. median near 0.519 as reported by user)
    ml_res = {
        "score": 0.519,
        "spread": 0.35,
        "min_score": 0.35,
        "max_score": 0.70,
        "models": {
            "dima806/ai_vs_real_image_detection": {"score": 0.35},
            "Organika/sdxl-detector": {"score": 0.70},
            "Ateeqq/ai-vs-human-image-detector": {"score": 0.52},
            "generalizing_dinov2_probe": {"score": 0.50},
        },
    }

    fused = fuse_signals(
        metadata_res={"score": None, "evidence": []},
        c2pa_res={"score": None, "manifest_found": False},
        ela_res={"score": 0.50, "note": "Recompression"},
        freq_res={"score": 0.50, "note": "Slope"},
        ml_res=ml_res,
        gemini_res=None,
        image_type_res=type_info,
    )

    assert fused["verdict"] == "Inconclusive", f"Expected Inconclusive, got {fused['verdict']}"
    assert fused["is_inconclusive"] is True
    assert "low" in fused["confidence"]
    assert "between" in fused["probability_display"] or "51.9%" in fused["probability_display"]


def test_sha256_cache_repeat_speed():
    """Verifies that backend SHA-256 caching returns in <5ms on repeat analysis."""
    img = Image.new("RGB", (256, 256), (100, 150, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw_bytes = buf.getvalue()

    # First run (cold or uncached)
    res1 = analyze_image_bytes(raw_bytes, filename="cache_test.jpg")
    assert res1 is not None

    # Second run (cached)
    t0 = time.perf_counter()
    res2 = analyze_image_bytes(raw_bytes, filename="cache_test.jpg")
    repeat_time_ms = (time.perf_counter() - t0) * 1000.0

    assert res2.get("cached") is True
    assert repeat_time_ms < 5.0, f"Repeat time {repeat_time_ms:.2f}ms exceeded 5ms limit!"


if __name__ == "__main__":
    print("Running test_every_verdict_counter_increments()...")
    test_every_verdict_counter_increments()
    print("PASS: test_every_verdict_counter_increments")

    print("Running test_whatsapp_recompressed_image_is_inconclusive()...")
    test_whatsapp_recompressed_image_is_inconclusive()
    print("PASS: test_whatsapp_recompressed_image_is_inconclusive")

    print("Running test_sha256_cache_repeat_speed()...")
    test_sha256_cache_repeat_speed()
    print("PASS: test_sha256_cache_repeat_speed")

    print("\nALL PHASE 3 UNIT TESTS PASSED SUCCESSFULLY!")
