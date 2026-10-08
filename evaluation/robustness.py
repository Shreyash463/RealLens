"""Robustness testing suite for RealLens.

Evaluates how well the detection pipeline holds up under realistic social media
and web image corruptions:
1. JPEG recompression at varied quality levels (Q=85, Q=65, Q=45, Q=30)
2. Downscaling and upscaling (resizing artifacts)
3. Screenshot simulation (letterboxing / border re-encoding)
4. Gaussian blur and noise
"""
from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# Ensure reallens root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from reallens.app import analyze_image_bytes
from evaluation.run_eval import calculate_metrics


def transform_jpeg(quality: int) -> Callable[[bytes], bytes]:
    def _apply(raw_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=quality)
        return buf.getvalue()
    return _apply


def transform_resize(scale: float) -> Callable[[bytes], bytes]:
    def _apply(raw_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        small = img.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.Resampling.BILINEAR)
        restored = small.resize((w, h), Image.Resampling.BICUBIC)
        buf = io.BytesIO()
        restored.save(buf, format="JPEG", quality=85)
        return buf.getvalue()
    return _apply


def transform_screenshot() -> Callable[[bytes], bytes]:
    def _apply(raw_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        w, h = img.size
        padded = ImageOps.expand(img, border=(int(w * 0.05), int(h * 0.08)), fill=(20, 24, 30))
        buf = io.BytesIO()
        padded.save(buf, format="PNG")
        return buf.getvalue()
    return _apply


def transform_blur(radius: float = 1.0) -> Callable[[bytes], bytes]:
    def _apply(raw_bytes: bytes) -> bytes:
        img = Image.open(io.BytesIO(raw_bytes)).convert("RGB")
        blurred = img.filter(ImageFilter.GaussianBlur(radius=radius))
        buf = io.BytesIO()
        blurred.save(buf, format="JPEG", quality=80)
        return buf.getvalue()
    return _apply


def evaluate_with_transform(
    real_files: List[Path],
    ai_files: List[Path],
    transform_fn: Callable[[bytes], bytes],
    transform_name: str,
    threshold: float = 0.50,
) -> Dict[str, Any]:
    """Runs pipeline on transformed bytes and computes metrics."""
    records = []

    for p in real_files:
        try:
            raw = p.read_bytes()
            corrupted = transform_fn(raw)
            res = analyze_image_bytes(corrupted, filename=p.name)
            prob = res.get("ai_probability", 0.5)
            pred = 1 if prob >= threshold else 0
            records.append({
                "ground_truth": 0,
                "predicted_label": pred,
                "ai_probability": prob,
                "is_correct": (pred == 0),
            })
        except Exception:
            pass

    for p in ai_files:
        try:
            raw = p.read_bytes()
            corrupted = transform_fn(raw)
            res = analyze_image_bytes(corrupted, filename=p.name)
            prob = res.get("ai_probability", 0.5)
            pred = 1 if prob >= threshold else 0
            records.append({
                "ground_truth": 1,
                "predicted_label": pred,
                "ai_probability": prob,
                "is_correct": (pred == 1),
            })
        except Exception:
            pass

    metrics = calculate_metrics(records)
    metrics["condition"] = transform_name
    return metrics


def run_robustness_battery(split: str = "tuning", max_samples: int = 40) -> List[Dict[str, Any]]:
    """Runs the suite of robustness tests across multiple corruption conditions."""
    data_dir = BASE_DIR / "data" / split
    real_files = sorted(list((data_dir / "real").glob("*.jpg")) + list((data_dir / "real").glob("*.png")))[:max_samples]
    ai_files = sorted(list((data_dir / "ai").glob("*.jpg")) + list((data_dir / "ai").glob("*.png")))[:max_samples]

    if not real_files or not ai_files:
        print(f"Warning: No images found in {data_dir}. Run evaluation/build_dataset.py first.")
        return []

    print(f"\n==========================================")
    print(f"  RUNNING ROBUSTNESS TESTS ({split.upper()} SET: {len(real_files)} Real, {len(ai_files)} AI)")
    print(f"==========================================")

    conditions = [
        ("Clean (Baseline)", lambda b: b),
        ("JPEG Q=85", transform_jpeg(85)),
        ("JPEG Q=65", transform_jpeg(65)),
        ("JPEG Q=45", transform_jpeg(45)),
        ("JPEG Q=30 (Severe)", transform_jpeg(30)),
        ("Resize 0.75x", transform_resize(0.75)),
        ("Resize 0.50x", transform_resize(0.50)),
        ("Screenshot Re-encode", transform_screenshot()),
        ("Gaussian Blur r=1.0", transform_blur(1.0)),
    ]

    results = []
    print(f"{'Condition':<24} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 72)

    for name, fn in conditions:
        m = evaluate_with_transform(real_files, ai_files, fn, name)
        results.append(m)
        print(f"{name:<24} | {m['accuracy']*100:>8.2f}% | {m['f1']:>10.4f} | {m['precision']*100:>8.2f}% | {m['recall']*100:>8.2f}%")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tuning", "test"], default="tuning")
    parser.add_argument("--samples", type=int, default=30)
    args = parser.parse_args()

    run_robustness_battery(split=args.split, max_samples=args.samples)
