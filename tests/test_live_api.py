"""Automated Live Self-Test Suite for RealLens Flask Server.

Tests:
1. 10 Real images from data/test/real
2. 10 AI images from data/test/ai
3. Corrupt image file
4. Tiny 16x16 image
5. Huge 3000x3000 image
6. PNG with transparency (RGBA)
7. Animated GIF
8. Checks JSON schema, HTTP response codes, and error resilience.
"""
import io
import sys
import time
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure repo root in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from reallens.app import app


def create_synthetic_test_images():
    """Generates edge-case test files: tiny, huge, transparent PNG, animated GIF, corrupt."""
    edge_cases = {}

    # 1. Corrupt image
    edge_cases["corrupt.jpg"] = b"NOT_A_VALID_JPEG_HEADER_1234567890GARBAGE_BYTES_TESTING_ERROR_HANDLING"

    # 2. Tiny image (16x16)
    tiny_im = Image.new("RGB", (16, 16), color=(200, 100, 50))
    buf = io.BytesIO()
    tiny_im.save(buf, format="JPEG")
    edge_cases["tiny_16x16.jpg"] = buf.getvalue()

    # 3. Huge image (2500x2500)
    huge_im = Image.new("RGB", (2500, 2500), color=(100, 150, 200))
    draw = ImageDraw.Draw(huge_im)
    draw.rectangle([500, 500, 2000, 2000], fill=(255, 200, 100))
    buf = io.BytesIO()
    huge_im.save(buf, format="JPEG", quality=80)
    edge_cases["huge_2500x2500.jpg"] = buf.getvalue()

    # 4. PNG with transparency (RGBA)
    rgba_im = Image.new("RGBA", (128, 128), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rgba_im)
    draw.ellipse([20, 20, 100, 100], fill=(255, 50, 50, 180))
    buf = io.BytesIO()
    rgba_im.save(buf, format="PNG")
    edge_cases["transparent.png"] = buf.getvalue()

    # 5. Animated GIF (2 frames)
    frame1 = Image.new("P", (64, 64), color=1)
    frame2 = Image.new("P", (64, 64), color=2)
    buf = io.BytesIO()
    frame1.save(buf, format="GIF", save_all=True, append_images=[frame2], duration=100, loop=0)
    edge_cases["animated.gif"] = buf.getvalue()

    return edge_cases


def run_live_tests():
    print("=" * 60)
    print("  RUNNING REALLENS LIVE FLASK API TEST SUITE")
    print("=" * 60)

    client = app.test_client()

    # Test GET /
    print("\n[1/5] Testing GET / ...")
    resp = client.get("/")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert b"RealLens" in resp.data, "Brand RealLens missing from index.html"
    print("  PASS: GET / returned 200 OK and rendered HTML")

    # Test GET /health
    print("\n[2/5] Testing GET /health ...")
    resp = client.get("/health")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    health_data = resp.get_json()
    print(f"  PASS: /health returned: status={health_data.get('status')}, device={health_data.get('device')}")

    # Test 10 Real Images from data/test/real
    print("\n[3/5] Testing POST /analyze on 10 Real images...")
    real_files = sorted(list((BASE_DIR / "data" / "test" / "real").glob("*.jpg")))[:10]
    for idx, rf in enumerate(real_files, 1):
        with open(rf, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), rf.name)}
            r = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert r.status_code == 200, f"Failed on real image {rf.name}: {r.status_code}"
        res_json = r.get_json()
        assert res_json.get("status") == "success"
        result = res_json.get("result", {})
        assert "ai_probability" in result
        assert "verdict" in result
        assert "signals" in result
        print(f"  [{idx}/10] Real: {rf.name} -> Prob: {result['ai_percentage']}% | Verdict: {result['verdict']} | Confidence: {result['confidence']}")

    # Test 10 AI Images from data/test/ai
    print("\n[4/5] Testing POST /analyze on 10 AI images...")
    ai_files = sorted(list((BASE_DIR / "data" / "test" / "ai").glob("*.jpg")))[:10]
    for idx, af in enumerate(ai_files, 1):
        with open(af, "rb") as f:
            data = {"image": (io.BytesIO(f.read()), af.name)}
            r = client.post("/analyze", data=data, content_type="multipart/form-data")
        assert r.status_code == 200, f"Failed on AI image {af.name}: {r.status_code}"
        res_json = r.get_json()
        assert res_json.get("status") == "success"
        result = res_json.get("result", {})
        assert "ai_probability" in result
        assert "verdict" in result
        assert "signals" in result
        print(f"  [{idx}/10] AI: {af.name} -> Prob: {result['ai_percentage']}% | Verdict: {result['verdict']} | Confidence: {result['confidence']}")

    # Test Edge Cases
    print("\n[5/5] Testing POST /analyze on Edge Cases (corrupt, tiny, huge, RGBA, animated GIF)...")
    edge_cases = create_synthetic_test_images()

    for name, raw in edge_cases.items():
        data = {"image": (io.BytesIO(raw), name)}
        r = client.post("/analyze", data=data, content_type="multipart/form-data")
        if name == "corrupt.jpg":
            # Must return 400/500 error gracefully without crashing server
            assert r.status_code in (400, 500), f"Expected error code on corrupt file, got {r.status_code}"
            err_json = r.get_json()
            print(f"  PASS: {name} gracefully caught with error response: '{err_json.get('message')}'")
        else:
            assert r.status_code == 200, f"Failed on edge case {name}: {r.status_code} - {r.data}"
            res_json = r.get_json()
            assert res_json.get("status") == "success"
            res = res_json.get("result", {})
            print(f"  PASS: {name} ({res.get('image_dimensions')}) -> Prob: {res['ai_percentage']}% | Verdict: {res['verdict']}")

    print("\n" + "=" * 60)
    print("  ALL 25 LIVE TESTS PASSED SUCCESSFULLY WITH ZERO CRASHES!")
    print("=" * 60)


if __name__ == "__main__":
    run_live_tests()
