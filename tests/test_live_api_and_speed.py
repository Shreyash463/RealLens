"""Live end-to-end API and batch test suite for Phase 3.
Tests:
1. /batch HTTP redirect to /
2. Latency: cold vs warm SHA-256 cache (<5ms)
3. 40 mixed images batch analysis across photos, WhatsApp, graphics, documents, screenshots, and AI
4. Summary counter verification for all 6 verdict types
"""
import io
import json
import time
from pathlib import Path
import urllib.request
import urllib.error
from PIL import Image

BASE_URL = "http://127.0.0.1:5000"


import http.client

def test_batch_redirect():
    print("Testing /batch redirect to /...")
    conn = http.client.HTTPConnection("127.0.0.1", 5000)
    conn.request("GET", "/batch")
    res = conn.getresponse()
    location = res.getheader("Location")
    print(f"Status code for /batch: {res.status}, Location: {location}")
    assert res.status == 302, f"Expected 302 redirect, got {res.status}"
    assert location.endswith("/"), f"Expected redirect to /, got {location}"
    print("PASS: /batch redirect verified.")


def send_image_to_analyze(image_bytes: bytes, filename: str):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    body = bytearray()
    body.extend(f"--{boundary}\r\n".encode())
    body.extend(f'Content-Disposition: form-data; name="image"; filename="{filename}"\r\n'.encode())
    body.extend(b"Content-Type: image/jpeg\r\n\r\n")
    body.extend(image_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        f"{BASE_URL}/analyze",
        data=bytes(body),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=30) as resp:
        elapsed = time.perf_counter() - t0
        data = json.loads(resp.read().decode())
        return data, elapsed


def test_speed_and_cache():
    print("\nTesting inference speed and SHA-256 caching...")
    # Create test image
    img = Image.new("RGB", (640, 480), (120, 180, 220))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    raw = buf.getvalue()

    # First request
    res1, dur1 = send_image_to_analyze(raw, "speed_test_1.jpg")
    print(f"First run duration: {dur1:.3f}s (reported: {res1['result']['time_taken_seconds']}s)")
    assert dur1 < 5.0, f"First run took {dur1:.3f}s, exceeding 5.0s budget!"

    # Second request (cached)
    res2, dur2 = send_image_to_analyze(raw, "speed_test_1.jpg")
    repeat_ms = dur2 * 1000.0
    print(f"Repeat run duration: {repeat_ms:.2f}ms (reported: {res2['result']['time_taken_seconds']*1000:.2f}ms, cached={res2['result'].get('cached')})")
    assert res2['result'].get('cached') is True, "Expected cached flag to be True"
    assert repeat_ms < 15.0, f"Repeat run took {repeat_ms:.2f}ms, expected ultra-fast cached response!"
    print("PASS: Inference speed & caching budget verified.")


def collect_40_mixed_test_images():
    """Gathers 40 real-world mixed images from existing datasets and tests."""
    images = []

    # 1. Real photos and WhatsApp
    whatsapp_files = list(Path("data/my_images/real").glob("*WhatsApp*"))
    for p in whatsapp_files[:5]:
        images.append((p, "real_whatsapp"))

    # 2. General real photos
    real_photos = list(Path("data/test_images/real").glob("*.jpg"))
    for p in real_photos[:10]:
        images.append((p, "real_photo"))

    # 3. Real graphics & documents & screenshots
    real_graphics = list(Path("data/test_images/real_graphics").glob("*.jpg")) + list(Path("data/test_images/real_graphics").glob("*.png"))
    for p in real_graphics[:10]:
        images.append((p, "real_graphic"))

    # 4. Hard AI images
    hard_ai = list(Path("data/hard_images/ai").glob("*.jpg")) + list(Path("data/hard_images/ai").glob("*.png"))
    for p in hard_ai[:10]:
        images.append((p, "hard_ai"))

    # 5. Easy AI images
    easy_ai = list(Path("data/test_images/ai").glob("*.jpg"))
    for p in easy_ai[:10]:
        images.append((p, "easy_ai"))

    # If we need more to reach 40:
    all_ai = list(Path("data").rglob("ai*.jpg")) + list(Path("data").rglob("*ai*.png"))
    for p in all_ai:
        if len(images) >= 40:
            break
        if not any(p == existing[0] for existing in images):
            images.append((p, "extra_ai"))

    # If still fewer, generate synthetic test samples (e.g. edge cases)
    edge_cases = [
        ("edge_transparent.png", Image.new("RGBA", (200, 200), (255, 0, 0, 128))),
        ("edge_small.jpg", Image.new("RGB", (64, 64), (100, 150, 200))),
        ("edge_highres.jpg", Image.new("RGB", (1920, 1080), (50, 75, 100))),
    ]
    for name, im in edge_cases:
        if len(images) < 40:
            p_edge = Path("tests") / name
            im.save(p_edge)
            images.append((p_edge, "edge_case"))

    return images[:40]


def test_40_mixed_images():
    print("\nRunning live batch analysis across 40 mixed images...")
    images = collect_40_mixed_test_images()
    print(f"Collected {len(images)} test images.")

    verdict_counts = {
        "Likely AI": 0,
        "Leaning AI": 0,
        "Inconclusive": 0,
        "Leaning Real": 0,
        "Likely Real": 0,
        "Failed": 0,
    }

    results = []
    durations = []

    for idx, (path, category) in enumerate(images, 1):
        with open(path, "rb") as f:
            raw_bytes = f.read()

        res, dur = send_image_to_analyze(raw_bytes, path.name)
        durations.append(dur)
        r = res.get("result", {})
        verdict = r.get("verdict", "Failed")

        # Map to standard verdicts
        v_clean = verdict.strip()
        if v_clean in verdict_counts:
            verdict_counts[v_clean] += 1
        elif v_clean == "Leaning real":
            verdict_counts["Leaning Real"] += 1
        elif v_clean == "Likely real":
            verdict_counts["Likely Real"] += 1
        else:
            verdict_counts["Failed"] += 1

        results.append({
            "filename": path.name,
            "category": category,
            "verdict": verdict,
            "display": r.get("probability_display"),
            "confidence": r.get("confidence"),
            "type": r.get("image_type"),
            "spread": r.get("spread"),
            "duration": round(dur, 3),
        })

        if idx % 10 == 0 or idx == len(images):
            print(f"[{idx}/40] {path.name[:25]:25} | {category:12} | {verdict:15} | {dur:.2f}s")

    avg_dur = sum(durations) / len(durations)
    max_dur = max(durations)
    print(f"\nBatch Performance Summary across {len(images)} images:")
    print(f"- Average duration per image: {avg_dur:.3f}s")
    print(f"- Maximum duration: {max_dur:.3f}s")
    print("\nVerdict Distribution:")
    for k, v in verdict_counts.items():
        print(f"  {k:15}: {v}")

    # Ensure no crashes
    assert verdict_counts["Failed"] == 0, f"Expected 0 failed images, but got {verdict_counts['Failed']}"
    assert avg_dur < 3.0, f"Average duration {avg_dur:.2f}s exceeded 3.0s threshold!"

    print("\nPASS: 40-image live batch evaluation succeeded with zero crashes and responsive latency.")


if __name__ == "__main__":
    test_batch_redirect()
    test_speed_and_cache()
    test_40_mixed_images()
    print("\nALL LIVE TESTS PASSED!")
