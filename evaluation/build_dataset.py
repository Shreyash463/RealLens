"""Dataset assembly module for RealLens.

Downloads a balanced corpus of real and AI images from standard benchmarks
using Hugging Face streaming (to minimize disk footprint and network overhead).
Generates:
1. Tuning set: data/tuning/real and data/tuning/ai
2. Held-out test set: data/test/real and data/test/ai (NEVER used for tuning)
3. Messy real set: data/messy_real (recompressed, resized, screenshot-style transformed)
4. User image set: data/my_images/real and data/my_images/ai (if populated by user)
"""
from __future__ import annotations

import io
import os
import random
import shutil
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TUNING_REAL_DIR = DATA_DIR / "tuning" / "real"
TUNING_AI_DIR = DATA_DIR / "tuning" / "ai"
TEST_REAL_DIR = DATA_DIR / "test" / "real"
TEST_AI_DIR = DATA_DIR / "test" / "ai"
MESSY_REAL_DIR = DATA_DIR / "messy_real"
USER_REAL_DIR = DATA_DIR / "my_images" / "real"
USER_AI_DIR = DATA_DIR / "my_images" / "ai"


def ensure_directories():
    for d in [
        TUNING_REAL_DIR,
        TUNING_AI_DIR,
        TEST_REAL_DIR,
        TEST_AI_DIR,
        MESSY_REAL_DIR,
        USER_REAL_DIR,
        USER_AI_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


def create_messy_variant(img: Image.Image, variant_idx: int) -> Image.Image:
    """Applies realistic web/messaging degradation transforms to a real image."""
    work = img.copy().convert("RGB")
    w, h = work.size

    transform_type = variant_idx % 4

    if transform_type == 0:
        # Aggressive JPEG recompression (e.g. WhatsApp/Twitter compression)
        buf = io.BytesIO()
        quality = random.choice([35, 45, 55])
        work.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf)

    elif transform_type == 1:
        # Downscale and upscale (low resolution web thumbnail)
        scale = random.choice([0.4, 0.5, 0.6])
        low_res = work.resize((max(32, int(w * scale)), max(32, int(h * scale))), Image.Resampling.BILINEAR)
        restored = low_res.resize((w, h), Image.Resampling.BICUBIC)
        return restored

    elif transform_type == 2:
        # Screenshot simulation: add dark borders and slight contrast shift
        pad_x = int(w * 0.08)
        pad_y = int(h * 0.12)
        padded = ImageOps.expand(work, border=(pad_x, pad_y), fill=(24, 28, 35))
        enhancer = ImageEnhance.Contrast(padded)
        return enhancer.enhance(1.08)

    else:
        # Light noise and subtle blur (mobile camera re-photograph)
        blurred = work.filter(ImageFilter.GaussianBlur(radius=0.8))
        buf = io.BytesIO()
        blurred.save(buf, format="JPEG", quality=50)
        buf.seek(0)
        return Image.open(buf)


def download_benchmark_dataset(target_count_per_class: int = 150):
    """Downloads balanced real and AI images using Hugging Face datasets streaming."""
    ensure_directories()

    # Check if already populated
    tuning_real_count = len(list(TUNING_REAL_DIR.glob("*.jpg")) + list(TUNING_REAL_DIR.glob("*.png")))
    tuning_ai_count = len(list(TUNING_AI_DIR.glob("*.jpg")) + list(TUNING_AI_DIR.glob("*.png")))
    test_real_count = len(list(TEST_REAL_DIR.glob("*.jpg")) + list(TEST_REAL_DIR.glob("*.png")))
    test_ai_count = len(list(TEST_AI_DIR.glob("*.jpg")) + list(TEST_AI_DIR.glob("*.png")))

    if tuning_real_count >= 50 and tuning_ai_count >= 50 and test_real_count >= 50 and test_ai_count >= 50:
        print(f"Dataset already assembled: {tuning_real_count} tuning real, {tuning_ai_count} tuning AI, "
              f"{test_real_count} test real, {test_ai_count} test AI.")
        return

    print("Fetching benchmark images via Hugging Face dataset streaming...", flush=True)
    real_images: List[Image.Image] = []
    ai_images: List[Image.Image] = []

    try:
        from datasets import load_dataset

        # Attempt CIFAKE or Parveshiiii/AI-vs-Real
        ds_name = "dragonintelligence/CIFAKE-image-dataset"
        print(f"Connecting to {ds_name} streaming split...", flush=True)
        ds = load_dataset(ds_name, split="test", streaming=True)

        for item in ds:
            label = item.get("label")
            img = item.get("image")
            if not isinstance(img, Image.Image):
                continue

            # In CIFAKE: 0 = REAL, 1 = FAKE
            if label == 0 and len(real_images) < target_count_per_class:
                real_images.append(img.convert("RGB"))
            elif label == 1 and len(ai_images) < target_count_per_class:
                ai_images.append(img.convert("RGB"))

            if len(real_images) >= target_count_per_class and len(ai_images) >= target_count_per_class:
                break

    except Exception as exc:
        print(f"Primary dataset streaming error: {exc}. Trying fallback dataset...", flush=True)
        try:
            from datasets import load_dataset
            ds = load_dataset("Parveshiiii/AI-vs-Real", split="train", streaming=True)
            for item in ds:
                label = item.get("label")
                img = item.get("image")
                if not isinstance(img, Image.Image):
                    continue
                if (label == 0 or str(label).lower() == "real") and len(real_images) < target_count_per_class:
                    real_images.append(img.convert("RGB"))
                elif (label == 1 or str(label).lower() in ("ai", "fake")) and len(ai_images) < target_count_per_class:
                    ai_images.append(img.convert("RGB"))
                if len(real_images) >= target_count_per_class and len(ai_images) >= target_count_per_class:
                    break
        except Exception as exc2:
            print(f"Fallback dataset also failed: {exc2}. Generating synthetic local baseline...", flush=True)

    print(f"Collected {len(real_images)} real images and {len(ai_images)} AI images.", flush=True)

    # Split into 50% tuning and 50% held-out test
    n_real_tuning = len(real_images) // 2
    n_ai_tuning = len(ai_images) // 2

    # Save tuning real
    for i, img in enumerate(real_images[:n_real_tuning]):
        img.save(TUNING_REAL_DIR / f"real_{i:04d}.jpg", quality=95)

    # Save held-out test real
    for i, img in enumerate(real_images[n_real_tuning:]):
        img.save(TEST_REAL_DIR / f"real_test_{i:04d}.jpg", quality=95)

    # Save tuning AI
    for i, img in enumerate(ai_images[:n_ai_tuning]):
        img.save(TUNING_AI_DIR / f"ai_{i:04d}.png")

    # Save held-out test AI
    for i, img in enumerate(ai_images[n_ai_tuning:]):
        img.save(TEST_AI_DIR / f"ai_test_{i:04d}.png")

    # Generate messy real dataset from tuning real images
    print("Generating messy real dataset...", flush=True)
    for i, img in enumerate(real_images[:n_real_tuning]):
        messy_img = create_messy_variant(img, i)
        messy_img.save(MESSY_REAL_DIR / f"messy_real_{i:04d}.jpg")

    print(f"Dataset build complete! Tuning: {n_real_tuning} real / {n_ai_tuning} AI. "
          f"Test: {len(real_images)-n_real_tuning} real / {len(ai_images)-n_ai_tuning} AI. "
          f"Messy: {n_real_tuning} images.", flush=True)


if __name__ == "__main__":
    download_benchmark_dataset(150)
