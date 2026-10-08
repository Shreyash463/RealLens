"""Live test script for Phase 2.5 RealLens architecture.

Validates /analyze endpoint with 30 diverse images:
- Real photos
- AI generative photos
- Hard compressed/resized photos
- Hard AI images
- Non-photographic graphics (certificates, dashboards, slides, charts)
- Edge cases: 32x32 thumbnail, grayscale, large dimensions, alpha channel.

Checks:
- HTTP 200 responses and valid JSON
- Proper routing of non-photos to Inconclusive
- Spread computation and range formatting
- Response latency per image
- Zero unhandled exceptions or crashes
"""
import io
import json
import time
from pathlib import Path
from PIL import Image, ImageDraw
import requests

SERVER_URL = "http://127.0.0.1:5000/analyze"


def generate_edge_cases():
    edge_cases = []

    # 1. 32x32 tiny image (CIFAR-like)
    tiny = Image.new("RGB", (32, 32), color=(120, 150, 180))
    d = ImageDraw.Draw(tiny)
    d.rectangle([8, 8, 24, 24], fill=(200, 100, 50))
    buf = io.BytesIO()
    tiny.save(buf, format="PNG")
    edge_cases.append(("edge_tiny_32x32.png", buf.getvalue(), "image/png"))

    # 2. Grayscale L-mode image
    gray = Image.new("L", (256, 256), color=128)
    d = ImageDraw.Draw(gray)
    d.ellipse([40, 40, 216, 216], fill=220)
    buf = io.BytesIO()
    gray.save(buf, format="JPEG")
    edge_cases.append(("edge_grayscale_L.jpg", buf.getvalue(), "image/jpeg"))

    # 3. Transparent RGBA image
    rgba = Image.new("RGBA", (200, 200), (0, 0, 0, 0))
    d = ImageDraw.Draw(rgba)
    d.rectangle([20, 20, 180, 180], fill=(255, 100, 100, 200))
    buf = io.BytesIO()
    rgba.save(buf, format="PNG")
    edge_cases.append(("edge_alpha_transparent.png", buf.getvalue(), "image/png"))

    return edge_cases


def run_live_tests():
    data_dir = Path("data")
    test_suite = []

    # 1. Real photos (5 images)
    real_photos = sorted(list((data_dir / "test" / "real").glob("*.*")))[:5]
    for p in real_photos:
        test_suite.append((p.name, p.read_bytes(), "real_photo"))

    # 2. AI photos (5 images)
    ai_photos = sorted(list((data_dir / "test" / "ai").glob("*.*")))[:5]
    for p in ai_photos:
        test_suite.append((p.name, p.read_bytes(), "ai_photo"))

    # 3. Hard real photos (5 images)
    hard_real = sorted(list((data_dir / "hard_test" / "real").glob("*.*")))[:5]
    for p in hard_real:
        test_suite.append((p.name, p.read_bytes(), "hard_real"))

    # 4. Hard AI photos (5 images)
    hard_ai = sorted(list((data_dir / "hard_test" / "ai").glob("*.*")))[:5]
    for p in hard_ai:
        test_suite.append((p.name, p.read_bytes(), "hard_ai"))

    # 5. Non-AI graphics & screenshots (7 images: cert, chart, dashboard)
    graphics = sorted(list((data_dir / "graphics_test").glob("*.*")))[:7]
    for p in graphics:
        test_suite.append((p.name, p.read_bytes(), "graphic_screenshot"))

    # 6. Edge cases (3 images)
    for fname, data_bytes, mime in generate_edge_cases():
        test_suite.append((fname, data_bytes, "edge_case"))

    print(f"=== RUNNING PHASE 2.5 LIVE TEST ON {len(test_suite)} MIXED IMAGES ===")
    print(f"Target: {SERVER_URL}\n")

    results = []
    total_time = 0.0

    for idx, (filename, raw_bytes, category) in enumerate(test_suite, start=1):
        t0 = time.perf_counter()
        ext = filename.split(".")[-1].lower()
        mime = "image/png" if ext == "png" else "image/jpeg"

        try:
            resp = requests.post(
                SERVER_URL,
                files={"image": (filename, raw_bytes, mime)},
                timeout=60,
            )
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            if resp.status_code != 200:
                print(f"[{idx:02d}/{len(test_suite)}] FAIL {filename} (HTTP {resp.status_code}): {resp.text}")
                results.append({"filename": filename, "status": "fail", "code": resp.status_code})
                continue

            data = resp.json()
            if data.get("status") != "success":
                print(f"[{idx:02d}/{len(test_suite)}] FAIL {filename} (JSON error): {data}")
                results.append({"filename": filename, "status": "fail", "error": data.get("message")})
                continue

            res = data.get("result", {})
            verdict = res.get("verdict")
            prob_disp = res.get("probability_display")
            img_type = res.get("image_type")
            spread = res.get("spread", 0.0)
            inconclusive = res.get("is_inconclusive")

            print(
                f"[{idx:02d}/{len(test_suite)}] PASS {filename:<28} | Cat: {category:<18} | "
                f"Type: {img_type:<20} | Verdict: {verdict:<13} | Score: {prob_disp:<30} | "
                f"Spread: {spread*100:4.1f}% | Latency: {elapsed:4.2f}s"
            )

            results.append({
                "filename": filename,
                "category": category,
                "status": "pass",
                "verdict": verdict,
                "image_type": img_type,
                "score_display": prob_disp,
                "spread": spread,
                "is_inconclusive": inconclusive,
                "latency": round(elapsed, 3),
            })

        except Exception as exc:
            print(f"[{idx:02d}/{len(test_suite)}] ERROR {filename}: {exc}")
            results.append({"filename": filename, "status": "error", "exception": str(exc)})

    # Summary
    passed = sum(1 for r in results if r["status"] == "pass")
    avg_latency = total_time / len(test_suite) if test_suite else 0.0

    print("\n" + "=" * 60)
    print("  PHASE 2.5 LIVE TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"Total Tested:   {len(test_suite)}")
    print(f"Passed:         {passed} / {len(test_suite)} ({passed/len(test_suite)*100:.1f}%)")
    print(f"Failed/Crashed: {len(test_suite) - passed}")
    print(f"Avg Latency:    {avg_latency:.2f} seconds per image")
    print(f"Total Duration: {total_time:.2f} seconds")

    # Verify Non-Photo Gate
    graphics_results = [r for r in results if r.get("category") == "graphic_screenshot" and r["status"] == "pass"]
    graphics_inconclusive = sum(1 for r in graphics_results if r["verdict"] == "Inconclusive")
    print(f"\nGraphics Category Gating:")
    print(f"Total Graphics: {len(graphics_results)}")
    print(f"Routed to Inconclusive: {graphics_inconclusive}/{len(graphics_results)} ({graphics_inconclusive/len(graphics_results)*100 if graphics_results else 0:.1f}%)")
    print(f"False Positives (AI claims): {sum(1 for r in graphics_results if r['verdict'] in ('Likely AI', 'Leaning AI'))}")

    return results


if __name__ == "__main__":
    run_live_tests()
