"""Generalizing Feature-Based Probe Training for RealLens Phase 2.

Extracts frozen embeddings from general vision backbones (DINOv2-small and CLIP),
augments training images with real-world noise (JPEG compression, downscaling, blur, screenshots),
trains and compares Linear Probe (Logistic Regression) vs small MLP,
and exports the winning probe to models/ for inclusion in ml_ensemble.py.
"""
from __future__ import annotations

import io
import os
import random
import sys
import time
from pathlib import Path
from typing import List, Tuple

import joblib
import numpy as np
import torch
from PIL import Image, ImageFilter
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from transformers import AutoImageProcessor, AutoModel, CLIPProcessor, CLIPVisionModelWithProjection

BASE_DIR = Path(__file__).resolve().parent.parent

def augment_image(img: Image.Image) -> List[Image.Image]:
    """Produces 3 augmented variants per input image to ensure robustness."""
    img = img.convert("RGB")
    variants = [img]

    # Variant 1: Heavy JPEG recompression (Q=45 to Q=65)
    buf1 = io.BytesIO()
    img.save(buf1, format="JPEG", quality=random.randint(45, 65))
    buf1.seek(0)
    variants.append(Image.open(buf1).copy())

    # Variant 2: Downscaling (0.5x to 0.75x) then bicubic back
    w, h = img.size
    scale = random.uniform(0.5, 0.75)
    nw, nh = max(32, int(w * scale)), max(32, int(h * scale))
    down = img.resize((nw, nh), Image.Resampling.BILINEAR)
    up = down.resize((w, h), Image.Resampling.BICUBIC)
    variants.append(up)

    # Variant 3: Slight Gaussian blur or screenshot re-encoding
    if random.random() > 0.5:
        variants.append(img.filter(ImageFilter.GaussianBlur(radius=0.7)))
    else:
        buf3 = io.BytesIO()
        img.save(buf3, format="JPEG", quality=random.randint(60, 75), subsampling=2)
        buf3.seek(0)
        variants.append(Image.open(buf3).copy())

    return variants

def load_training_data() -> Tuple[List[Image.Image], List[int]]:
    """Loads images from data/tuning and data/hard_tuning with augmentation."""
    data_dir = BASE_DIR / "data"
    random.seed(42)

    real_paths = []
    ai_paths = []
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")

    # Collect from tuning
    for e in exts:
        real_paths.extend((data_dir / "tuning" / "real").glob(e))
        ai_paths.extend((data_dir / "tuning" / "ai").glob(e))
        real_paths.extend((data_dir / "hard_tuning" / "real").glob(e))
        ai_paths.extend((data_dir / "hard_tuning" / "ai").glob(e))

    print(f"Loaded {len(real_paths)} base Real images and {len(ai_paths)} base AI images.")

    all_images = []
    all_labels = []

    print("Generating real-world data augmentations (JPEG compression, resizing, blur)...")
    for p in real_paths:
        try:
            im = Image.open(p)
            for aug in augment_image(im):
                all_images.append(aug)
                all_labels.append(0)  # Real
        except Exception:
            continue

    for p in ai_paths:
        try:
            im = Image.open(p)
            for aug in augment_image(im):
                all_images.append(aug)
                all_labels.append(1)  # AI
        except Exception:
            continue

    print(f"Total Augmented Training Samples: {len(all_images)} ({all_labels.count(0)} Real, {all_labels.count(1)} AI)")
    return all_images, all_labels

def extract_dinov2_embeddings(images: List[Image.Image]) -> np.ndarray:
    print("\nExtracting DINOv2-small features (dim=384)...")
    device = "cpu"
    processor = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
    model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
    model.eval()

    embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            # CLS token pooled embedding
            cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
            embeddings.append(cls_emb)
    return np.vstack(embeddings)

def extract_clip_embeddings(images: List[Image.Image]) -> np.ndarray:
    print("\nExtracting CLIP ViT-B/32 features (dim=512)...")
    device = "cpu"
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    model = CLIPVisionModelWithProjection.from_pretrained("openai/clip-vit-base-patch32").to(device)
    model.eval()

    embeddings = []
    batch_size = 16
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            inputs = processor(images=batch, return_tensors="pt").to(device)
            outputs = model(**inputs)
            embeds = outputs.image_embeds.cpu().numpy()
            embeddings.append(embeds)
    return np.vstack(embeddings)

def train_and_evaluate_probe():
    images, labels = load_training_data()
    y = np.array(labels)

    # 1. Compare DINOv2 vs CLIP
    t0 = time.time()
    dinov2_X = extract_dinov2_embeddings(images)
    print(f"DINOv2 extraction completed in {time.time() - t0:.1f}s. Shape: {dinov2_X.shape}")

    t1 = time.time()
    clip_X = extract_clip_embeddings(images)
    print(f"CLIP extraction completed in {time.time() - t1:.1f}s. Shape: {clip_X.shape}")

    # Prepare Hard Tuning Validation Set
    val_images = []
    val_labels = []
    ht_real = list((BASE_DIR / "data" / "hard_tuning" / "real").glob("*.*"))
    ht_ai = list((BASE_DIR / "data" / "hard_tuning" / "ai").glob("*.*"))
    for p in ht_real:
        val_images.append(Image.open(p).convert("RGB"))
        val_labels.append(0)
    for p in ht_ai:
        val_images.append(Image.open(p).convert("RGB"))
        val_labels.append(1)

    val_y = np.array(val_labels)
    val_dinov2_X = extract_dinov2_embeddings(val_images)
    val_clip_X = extract_clip_embeddings(val_images)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON ON HARD TUNING SET")
    print("=" * 60)

    results = {}

    # Experiment 1: DINOv2 + Logistic Regression
    scaler_d = StandardScaler()
    X_d_scaled = scaler_d.fit_transform(dinov2_X)
    val_d_scaled = scaler_d.transform(val_dinov2_X)

    lr_d = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_d.fit(X_d_scaled, y)
    preds_d_lr = lr_d.predict(val_d_scaled)
    acc_d_lr = accuracy_score(val_y, preds_d_lr)
    f1_d_lr = f1_score(val_y, preds_d_lr)
    results["DINOv2_LogReg"] = (acc_d_lr, f1_d_lr, lr_d, scaler_d, "dinov2")
    print(f"1. DINOv2 + Logistic Regression: Accuracy = {acc_d_lr * 100:.2f}%, F1 = {f1_d_lr:.4f}")

    # Experiment 2: DINOv2 + Small MLP
    mlp_d = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, early_stopping=True)
    mlp_d.fit(X_d_scaled, y)
    preds_d_mlp = mlp_d.predict(val_d_scaled)
    acc_d_mlp = accuracy_score(val_y, preds_d_mlp)
    f1_d_mlp = f1_score(val_y, preds_d_mlp)
    results["DINOv2_MLP"] = (acc_d_mlp, f1_d_mlp, mlp_d, scaler_d, "dinov2")
    print(f"2. DINOv2 + Small MLP:           Accuracy = {acc_d_mlp * 100:.2f}%, F1 = {f1_d_mlp:.4f}")

    # Experiment 3: CLIP + Logistic Regression
    scaler_c = StandardScaler()
    X_c_scaled = scaler_c.fit_transform(clip_X)
    val_c_scaled = scaler_c.transform(val_clip_X)

    lr_c = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_c.fit(X_c_scaled, y)
    preds_c_lr = lr_c.predict(val_c_scaled)
    acc_c_lr = accuracy_score(val_y, preds_c_lr)
    f1_c_lr = f1_score(val_y, preds_c_lr)
    results["CLIP_LogReg"] = (acc_c_lr, f1_c_lr, lr_c, scaler_c, "clip")
    print(f"3. CLIP + Logistic Regression:   Accuracy = {acc_c_lr * 100:.2f}%, F1 = {f1_c_lr:.4f}")

    # Experiment 4: CLIP + Small MLP
    mlp_c = MLPClassifier(hidden_layer_sizes=(64,), max_iter=300, random_state=42, early_stopping=True)
    mlp_c.fit(X_c_scaled, y)
    preds_c_mlp = mlp_c.predict(val_c_scaled)
    acc_c_mlp = accuracy_score(val_y, preds_c_mlp)
    f1_c_mlp = f1_score(val_y, preds_c_mlp)
    results["CLIP_MLP"] = (acc_c_mlp, f1_c_mlp, mlp_c, scaler_c, "clip")
    print(f"4. CLIP + Small MLP:             Accuracy = {acc_c_mlp * 100:.2f}%, F1 = {f1_c_mlp:.4f}")

    # Find the winner
    best_name = max(results, key=lambda k: (results[k][0], results[k][1]))
    best_acc, best_f1, best_clf, best_scaler, best_arch = results[best_name]
    print(f"\n[WINNER]: {best_name} with Accuracy = {best_acc * 100:.2f}%, F1 = {best_f1:.4f}")

    # Save to models/
    out_dir = BASE_DIR / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    probe_package = {
        "model_name": best_name,
        "backbone": best_arch,
        "backbone_id": "facebook/dinov2-small" if best_arch == "dinov2" else "openai/clip-vit-base-patch32",
        "classifier": best_clf,
        "scaler": best_scaler,
        "validation_accuracy": best_acc,
        "validation_f1": best_f1,
    }
    joblib.dump(probe_package, out_dir / "generalizing_probe.joblib")
    print(f"Saved trained generalizing probe package to: {out_dir / 'generalizing_probe.joblib'}")

if __name__ == "__main__":
    train_and_evaluate_probe()
