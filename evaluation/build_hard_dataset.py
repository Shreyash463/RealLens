"""Builder for the Hard Dataset in RealLens Phase 2.

Builds realistic 'hard' benchmark sets:
1. Real images: Modern high-res photos transformed with:
   - WhatsApp-style pipeline (resizing max dim <= 1080px, JPEG Q=65-75, 4:2:0 subsampling, stripped metadata).
   - Screenshot / screen-capture pipeline (re-encoding, light scale-down).
   - Incorporates user authentic camera photos from data/my_images/real/.
2. AI images: Modern Midjourney & high-res generative images transformed with:
   - JPEG recompression at quality 60 to 85.
   - Resizing to <= 1080px.
   - Screenshot-like re-encoding.
3. Splits into:
   - data/hard_tuning/ (for probe training and learned fusion tuning)
   - data/hard_test/   (held-out test set evaluated ONCE at the end)
"""
from __future__ import annotations

import io
import os
import random
import shutil
import sys
from pathlib import Path
from PIL import Image, ImageOps, ImageFilter
from datasets import load_dataset

BASE_DIR = Path(__file__).resolve().parent.parent

def apply_whatsapp_pipeline(img: Image.Image) -> Image.Image:
    """Simulates WhatsApp mobile compression & downscaling."""
    img = img.convert("RGB")
    # Resize so max dimension is at most 1080px
    max_dim = 1080
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.Resampling.BILINEAR)

    # WhatsApp JPEG recompression at Q=65..75 with subsampling
    buf = io.BytesIO()
    q = random.randint(65, 75)
    img.save(buf, format="JPEG", quality=q, subsampling=2)
    buf.seek(0)
    return Image.open(buf).copy()

def apply_screenshot_pipeline(img: Image.Image) -> Image.Image:
    """Simulates screenshotting / screen-capture re-encoding."""
    img = img.convert("RGB")
    w, h = img.size
    # Slight downscale as if viewed on mobile/desktop screen
    scale = random.uniform(0.70, 0.95)
    new_w, new_h = max(64, int(w * scale)), max(64, int(h * scale))
    img = img.resize((new_w, new_h), Image.Resampling.BICUBIC)

    # Re-encode as PNG or high-Q JPEG screenshot
    buf = io.BytesIO()
    if random.random() > 0.5:
        img.save(buf, format="PNG")
    else:
        img.save(buf, format="JPEG", quality=random.randint(75, 85))
    buf.seek(0)
    return Image.open(buf).copy()

def apply_hard_ai_pipeline(img: Image.Image) -> Image.Image:
    """Recompresses AI images at Q=60..85, resizes to <=1080px, screenshotted."""
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > 1080:
        scale = 1080 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    elif random.random() > 0.5:
        scale = random.uniform(0.65, 0.90)
        img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    q = random.randint(60, 85)
    img.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).copy()

def build_hard_datasets(num_tuning_per_class: int = 30, num_test_per_class: int = 30):
    random.seed(42)
    data_dir = BASE_DIR / "data"

    hard_tuning_real = data_dir / "hard_tuning" / "real"
    hard_tuning_ai = data_dir / "hard_tuning" / "ai"
    hard_test_real = data_dir / "hard_test" / "real"
    hard_test_ai = data_dir / "hard_test" / "ai"

    for d in [hard_tuning_real, hard_tuning_ai, hard_test_real, hard_test_ai]:
        d.mkdir(parents=True, exist_ok=True)
        # Clear previous contents if re-running
        for f in d.glob("*.*"):
            f.unlink()

    print("=" * 60)
    print("  BUILDING HARD REAL & AI DATASETS")
    print("=" * 60)
    print(f"Target Tuning: {num_tuning_per_class} Real, {num_tuning_per_class} AI")
    print(f"Target Test:   {num_test_per_class} Real, {num_test_per_class} AI")

    # 1. Download Modern Midjourney AI images
    print("\nStreaming modern Midjourney AI images...")
    midjourney_ds = load_dataset("ideepankarsharma2003/AIGeneratedImages_Midjourney", split="train", streaming=True)

    ai_images = []
    for item in midjourney_ds:
        if item["label"] == 0:  # 'ai_gen'
            try:
                raw_img = item["image"].convert("RGB")
                if raw_img.width >= 128 and raw_img.height >= 128:
                    ai_images.append(raw_img)
            except Exception:
                continue
        if len(ai_images) >= (num_tuning_per_class + num_test_per_class + 10):
            break

    print(f"Collected {len(ai_images)} high-res Midjourney AI samples.")

    # 2. Download Modern Real Images
    print("\nStreaming modern real photographic images...")
    hemg_ds = load_dataset("Hemg/AI-Generated-vs-Real-Images-Datasets", split="train", streaming=True)

    real_images = []
    for item in hemg_ds:
        if item["label"] == 1:  # 'RealArt' / real photos
            try:
                raw_img = item["image"].convert("RGB")
                if raw_img.width >= 128 and raw_img.height >= 128:
                    real_images.append(raw_img)
            except Exception:
                continue
        if len(real_images) >= (num_tuning_per_class + num_test_per_class + 10):
            break

    print(f"Collected {len(real_images)} authentic photographic samples.")

    # 3. Distribute AI into hard_tuning and hard_test with realistic distortions
    for idx in range(num_tuning_per_class):
        img = apply_hard_ai_pipeline(ai_images[idx])
        out_path = hard_tuning_ai / f"hard_ai_tune_{idx:04d}.jpg"
        img.save(out_path, format="JPEG", quality=random.randint(60, 85))

    for idx in range(num_test_per_class):
        img = apply_hard_ai_pipeline(ai_images[num_tuning_per_class + idx])
        out_path = hard_test_ai / f"hard_ai_test_{idx:04d}.jpg"
        img.save(out_path, format="JPEG", quality=random.randint(60, 85))

    # 4. Distribute Real into hard_tuning and hard_test with WhatsApp / Screenshot pipelines
    for idx in range(num_tuning_per_class):
        if idx % 2 == 0:
            img = apply_whatsapp_pipeline(real_images[idx])
        else:
            img = apply_screenshot_pipeline(real_images[idx])
        out_path = hard_tuning_real / f"hard_real_tune_{idx:04d}.jpg"
        img.save(out_path, format="JPEG", quality=random.randint(65, 80))

    # For hard_test, also include the 5 user smartphone WhatsApp images!
    user_real_imgs = list((data_dir / "my_images" / "real").glob("*.jpeg"))
    print(f"Found {len(user_real_imgs)} user smartphone images to include in hard test set.")

    for u_idx, u_path in enumerate(user_real_imgs):
        shutil.copy2(u_path, hard_test_real / f"hard_user_real_{u_idx:04d}.jpeg")

    # Remaining real test images
    remaining_needed = num_test_per_class - len(user_real_imgs)
    for idx in range(remaining_needed):
        src_img = real_images[num_tuning_per_class + idx]
        if idx % 2 == 0:
            img = apply_whatsapp_pipeline(src_img)
        else:
            img = apply_screenshot_pipeline(src_img)
        out_path = hard_test_real / f"hard_real_test_{idx:04d}.jpg"
        img.save(out_path, format="JPEG", quality=random.randint(65, 80))

    print("\n[SUCCESS] Hard datasets assembled:")
    print(f"  Hard Tuning: {len(list(hard_tuning_real.glob('*.*')))} Real, {len(list(hard_tuning_ai.glob('*.*')))} AI")
    print(f"  Hard Test:   {len(list(hard_test_real.glob('*.*')))} Real, {len(list(hard_test_ai.glob('*.*')))} AI")

if __name__ == "__main__":
    build_hard_datasets(num_tuning_per_class=30, num_test_per_class=30)
