"""Audit Latency & Output Usability Test for RealLens.

Tests /analyze endpoint with 20 mixed images across:
- 5 user camera WhatsApp images
- 5 held-out real images
- 5 held-out AI images
- 5 messy real images

Measures response time per image, checks for crashes, slow responses, and confusing outputs.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
import requests

BASE_DIR = Path(__file__).resolve().parent.parent

def run_latency_audit():
    url = "http://127.0.0.1:5000/analyze"

    # Assemble 20 mixed images
    user_images = list((BASE_DIR / "data" / "my_images" / "real").glob("*.jpeg"))[:5]
    real_images = list((BASE_DIR / "data" / "test" / "real").glob("*.jpg"))[:5]
    ai_images = list((BASE_DIR / "data" / "test" / "ai").glob("*.png"))[:5]
    messy_images = list((BASE_DIR / "data" / "messy_real").glob("*.jpg"))[:5]

    sample_pool = [
        ("User Camera (WhatsApp)", user_images),
        ("Held-Out Test Real", real_images),
        ("Held-Out Test AI", ai_images),
        ("Messy Distorted Real", messy_images),
    ]

    print("=" * 70)
    print("  AUDIT 4: RUNNING 20-IMAGE LATENCY & USABILITY ASSESSMENT")
    print("=" * 70)

    total_time = 0.0
    records = []

    img_counter = 0
    for group_name, files in sample_pool:
        print(f"\n--- Group: {group_name} ({len(files)} images) ---")
        for fpath in files:
            img_counter += 1
            t0 = time.perf_counter()
            with open(fpath, "rb") as f:
                resp = requests.post(url, files={"image": (fpath.name, f.read())})
            elapsed = time.perf_counter() - t0
            total_time += elapsed

            if resp.status_code != 200:
                print(f"[{img_counter}/20] FAILED: {fpath.name} -> HTTP {resp.status_code}: {resp.text}")
                records.append({"filename": fpath.name, "elapsed": elapsed, "error": resp.text})
                continue

            data = resp.json().get("result", {})
            prob = data.get("ai_percentage")
            verdict = data.get("verdict")
            conf = data.get("confidence")
            disagree = data.get("disagreement_flag")
            dims = data.get("image_dimensions")

            # Check for confusing outputs: e.g. empty summary, null verdict, or inconsistent percentage
            confusing_notes = []
            if not data.get("summary_reason"):
                confusing_notes.append("Empty summary reason")
            if disagree and "Uncertain" not in verdict:
                confusing_notes.append("Disagreement flagged but verdict not marked Uncertain")

            notes_str = f" | CONFUSING: {', '.join(confusing_notes)}" if confusing_notes else ""
            print(f"[{img_counter:02d}/20] {fpath.name[:25]:<25} | {dims:<10} | {elapsed:.2f}s | Prob: {prob:>5.1f}% | Verdict: {verdict:<22} (Conf: {conf}){notes_str}")
            records.append({
                "group": group_name,
                "filename": fpath.name,
                "elapsed": elapsed,
                "prob": prob,
                "verdict": verdict,
                "conf": conf,
                "confusing": confusing_notes,
            })

    latencies = [r["elapsed"] for r in records if "elapsed" in r]
    avg_latency = total_time / len(latencies) if latencies else 0.0
    min_latency = min(latencies) if latencies else 0.0
    max_latency = max(latencies) if latencies else 0.0

    print("\n" + "=" * 70)
    print("  LATENCY SUMMARY")
    print("=" * 70)
    print(f"Total Images Analyzed: {len(records)}")
    print(f"Average Latency:       {avg_latency:.2f}s per image")
    print(f"Min Latency:           {min_latency:.2f}s")
    print(f"Max Latency:           {max_latency:.2f}s")
    print(f"Crashes / Errors:      {sum(1 for r in records if 'error' in r)}")
    print(f"Confusing Outputs:     {sum(len(r.get('confusing', [])) for r in records)}")

if __name__ == "__main__":
    run_latency_audit()
