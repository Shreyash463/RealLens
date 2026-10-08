"""Audit Script for RealLens: Dataset Leakage & Provenance Analysis.

Audits:
1. File hash overlap (SHA-256) between tuning and held-out test splits.
2. Pixel-level difference / identical image checks across splits.
3. Generator provenance & real-world source analysis.
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

def compute_sha256(filepath: Path) -> str:
    h = hashlib.sha256()
    h.update(filepath.read_bytes())
    return h.hexdigest()

def audit_leakage():
    data_dir = BASE_DIR / "data"
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    tuning_real, tuning_ai, test_real, test_ai = [], [], [], []
    for e in exts:
        tuning_real.extend((data_dir / "tuning" / "real").glob(e))
        tuning_ai.extend((data_dir / "tuning" / "ai").glob(e))
        test_real.extend((data_dir / "test" / "real").glob(e))
        test_ai.extend((data_dir / "test" / "ai").glob(e))

    print("=" * 60)
    print("  AUDIT 1: DATASET LEAKAGE & HASH COLLISION CHECK")
    print("=" * 60)
    print(f"Tuning Set: {len(tuning_real)} Real, {len(tuning_ai)} AI")
    print(f"Test Set:   {len(test_real)} Real, {len(test_ai)} AI")

    tuning_hashes = {}
    for p in tuning_real:
        tuning_hashes[compute_sha256(p)] = ("tuning_real", p.name)
    for p in tuning_ai:
        tuning_hashes[compute_sha256(p)] = ("tuning_ai", p.name)

    test_hashes = {}
    for p in test_real:
        test_hashes[compute_sha256(p)] = ("test_real", p.name)
    for p in test_ai:
        test_hashes[compute_sha256(p)] = ("test_ai", p.name)

    overlap = set(tuning_hashes.keys()) & set(test_hashes.keys())
    if overlap:
        print(f"\n[CRITICAL LEAKAGE DETECTED]: {len(overlap)} exact hash collisions between tuning and test sets!")
        for h in overlap:
            print(f"  Hash {h[:12]} appears in {tuning_hashes[h]} and {test_hashes[h]}")
    else:
        print("\n[VERIFIED]: 0 Hash Collisions between Tuning and Test sets.")
        print(f"  All {len(tuning_hashes)} tuning images and {len(test_hashes)} test images are completely distinct files.")

    # Check for identical pixel arrays across splits (even if header bytes differed)
    print("\nAuditing pixel-level identical arrays across splits...")
    tuning_pixel_hashes = {}
    for p in tuning_real + tuning_ai:
        img = Image.open(p).convert("RGB")
        arr_bytes = np.array(img).tobytes()
        ph = hashlib.sha256(arr_bytes).hexdigest()
        tuning_pixel_hashes[ph] = p.name

    test_pixel_overlap = []
    for p in test_real + test_ai:
        img = Image.open(p).convert("RGB")
        arr_bytes = np.array(img).tobytes()
        ph = hashlib.sha256(arr_bytes).hexdigest()
        if ph in tuning_pixel_hashes:
            test_pixel_overlap.append((p.name, tuning_pixel_hashes[ph]))

    if test_pixel_overlap:
        print(f"[PIXEL LEAKAGE]: Found {len(test_pixel_overlap)} identical pixel arrays between splits!")
        for t_name, tr_name in test_pixel_overlap:
            print(f"  Test image {t_name} is pixel-identical to Tuning image {tr_name}")
    else:
        print("[VERIFIED]: 0 identical pixel arrays found across splits. Zero cross-contamination.")

    print("\n" + "=" * 60)
    print("  AUDIT 2: GENERATOR PROVENANCE & DATASET COMPLEXITY")
    print("=" * 60)
    print("Source Dataset: dragonintelligence/CIFAKE-image-dataset")
    print("AI Generator: Latent Diffusion Models (LDM / Stable Diffusion v1.4, synthesized on CIFAR-10 classes)")
    print("Real Image Source: Authentic CIFAR-10 photographic set (subset of Tiny Images / 32x32 natural photos)")
    print("Image Resolution: 32x32 pixels (upscaled or native depending on preprocessing)")
    print("Generators in current benchmark: 1 generator family (Stable Diffusion v1.4 LDM)")
    print("Real Sources in current benchmark: 1 source family (CIFAR-10 web photographic scrapes)")
    print("\n[CRITICAL HONESTY FINDING]:")
    print("  The baseline CIFAKE benchmark is an 'easy/homogeneous' benchmark because:")
    print("  1. 100% of AI images originate from a single generator architecture (SD v1.4 LDM on CIFAR).")
    print("  2. Modern high-resolution diffusion models (Midjourney v6, Flux.1, DALL-E 3, SDXL, Imagen 3, Ideogram)")
    print("     and modern smartphone camera pipelines are NOT covered in the baseline test set.")
    print("  3. Model dima806 was explicitly fine-tuned on the CIFAKE training split! This explains why")
    print("     dima806 has extreme confidence on CIFAKE test images, creating domain bias.")


if __name__ == "__main__":
    audit_leakage()
