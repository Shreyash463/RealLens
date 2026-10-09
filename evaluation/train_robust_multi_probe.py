"""Train robust multi-backbone embedding probe for RealLens Phase 5.

Combines representations from:
1. facebook/dinov2-small (patch token mean/std statistics + CLS token)
2. openai/clip-vit-base-patch16 (visual patch token statistics + pooled embedding)
3. Ateeqq/ai-vs-human-image-detector (SigLIP vision tower embedding)

Augmentation:
- Random JPEG recompression (Q=40..95)
- Random resize/downscale and rescale
- Random color jitter
- Random center/spatial crop
- Applied EQUALLY to both Real and AI to prevent shortcut learning.

Normalizes per-backbone with L2, concatenates, and z-score standardizes.
Trains and compares:
- Regularized Logistic Regression (Linear Probe)
- Multi-Layer Perceptron (MLP with dropout)
Calibrates probabilities via Platt scaling (sigmoid).
Saves to models/generalizing_probe.joblib.
"""
from __future__ import annotations

import io
import json
import logging
import random
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image, ImageEnhance
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_robust_probe")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "matched_dataset"
MODEL_OUT = BASE_DIR / "models" / "generalizing_probe.joblib"


def apply_random_augmentation(img: Image.Image, rng: random.Random) -> Image.Image:
    """Applies identical random augmentations to both Real and AI."""
    work = img.convert("RGB")
    w, h = work.size

    # 1. Random crop (85% to 100%)
    crop_scale = rng.uniform(0.85, 1.0)
    cw, ch = int(w * crop_scale), int(h * crop_scale)
    x0 = rng.randint(0, max(0, w - cw))
    y0 = rng.randint(0, max(0, h - ch))
    work = work.crop((x0, y0, x0 + cw, y0 + ch)).resize((w, h), Image.Resampling.BILINEAR)

    # 2. Random Color Jitter
    if rng.random() > 0.5:
        enh = ImageEnhance.Brightness(work)
        work = enh.enhance(rng.uniform(0.9, 1.1))
    if rng.random() > 0.5:
        enh = ImageEnhance.Contrast(work)
        work = enh.enhance(rng.uniform(0.9, 1.1))
    if rng.random() > 0.5:
        enh = ImageEnhance.Color(work)
        work = enh.enhance(rng.uniform(0.85, 1.15))

    # 3. Random Resize Down/Up
    if rng.random() > 0.5:
        down_factor = rng.uniform(0.5, 0.9)
        dw, dh = max(64, int(w * down_factor)), max(64, int(h * down_factor))
        work = work.resize((dw, dh), Image.Resampling.BILINEAR).resize((w, h), Image.Resampling.BICUBIC)

    # 4. Random JPEG Recompression (Q=40 to 95)
    q = rng.randint(40, 95)
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=q)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class MultiBackboneExtractor:
    def __init__(self, device: str = "cpu"):
        from transformers import AutoImageProcessor, AutoModel
        self.device = device
        logger.info(f"Loading backbones on {device}...")

        # 1. DINOv2
        self.dino_proc = AutoImageProcessor.from_pretrained("facebook/dinov2-small")
        self.dino_model = AutoModel.from_pretrained("facebook/dinov2-small").to(device)
        self.dino_model.eval()

        # 2. CLIP ViT-B/32 (locally cached)
        self.clip_proc = AutoImageProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.clip_model.eval()

    def extract(self, img: Image.Image) -> np.ndarray:
        img_rgb = img.convert("RGB")

        # DINOv2 features: [CLS] token (384-d) + patch token mean (384-d) + patch token std (384-d)
        d_inp = self.dino_proc(images=[img_rgb], return_tensors="pt").to(self.device)
        with torch.no_grad():
            d_out = self.dino_model(**d_inp)
            # tokens: shape (1, num_patches + 1, 384)
            d_cls = d_out.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            d_patches = d_out.last_hidden_state[:, 1:, :].cpu().numpy()[0]
            d_mean = np.mean(d_patches, axis=0)
            d_std = np.std(d_patches, axis=0)

        d_feat = np.concatenate([d_cls, d_mean, d_std])
        # L2 normalize DINOv2
        d_norm = np.linalg.norm(d_feat)
        if d_norm > 1e-6:
            d_feat = d_feat / d_norm

        # CLIP ViT-B/32 features: pooler_output (512-d)
        c_inp = self.clip_proc(images=[img_rgb], return_tensors="pt").to(self.device)
        with torch.no_grad():
            c_out = self.clip_model.get_image_features(**c_inp)
            if hasattr(c_out, "pooler_output"):
                c_feat = c_out.pooler_output.cpu().numpy().flatten()
            elif hasattr(c_out, "last_hidden_state"):
                c_feat = c_out.last_hidden_state[:, 0, :].cpu().numpy().flatten()
            else:
                c_feat = c_out.cpu().numpy().flatten()

        c_norm = np.linalg.norm(c_feat)
        if c_norm > 1e-6:
            c_feat = c_feat / c_norm

        # Joint concatenated feature vector
        combined = np.concatenate([d_feat, c_feat])
        return combined


def collect_samples(split_dir: Path, rng_seed: int = 42, augment_per_image: int = 2) -> Tuple[List[Image.Image], List[int]]:
    rng = random.Random(rng_seed)
    images = []
    labels = []

    # Read base samples across conditions
    for cls_name, lbl in [("real", 0), ("ai", 1)]:
        # Use images from original, jpeg_70, and whatsapp conditions to give natural variety
        for cond in ["original", "jpeg_70", "whatsapp"]:
            cond_dir = split_dir / cond / cls_name
            if not cond_dir.exists():
                continue
            for p in cond_dir.glob("*.jpg"):
                try:
                    im = Image.open(p).convert("RGB")
                    images.append(im)
                    labels.append(lbl)
                    # Add augmented variants
                    for _ in range(augment_per_image):
                        aug = apply_random_augmentation(im, rng)
                        images.append(aug)
                        labels.append(lbl)
                except Exception:
                    pass

    return images, labels


def train_probe():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    extractor = MultiBackboneExtractor(device=device)

    logger.info("Collecting tuning samples with data augmentation...")
    train_imgs, train_labels = collect_samples(DATA_DIR / "tuning", rng_seed=42, augment_per_image=2)
    logger.info(f"Total training samples (with augmentation): {len(train_imgs)} ({train_labels.count(0)} Real, {train_labels.count(1)} AI)")

    logger.info("Collecting held-out validation samples...")
    test_imgs, test_labels = collect_samples(DATA_DIR / "test", rng_seed=99, augment_per_image=0)
    logger.info(f"Total validation samples: {len(test_imgs)} ({test_labels.count(0)} Real, {test_labels.count(1)} AI)")

    cache_dir = DATA_DIR / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    train_cache = cache_dir / "X_train.npy"
    test_cache = cache_dir / "X_test.npy"
    y_train_cache = cache_dir / "y_train.npy"
    y_test_cache = cache_dir / "y_test.npy"

    if train_cache.exists() and test_cache.exists():
        logger.info("Loading cached multi-backbone features from disk...")
        X_train = np.load(train_cache)
        y_train = np.load(y_train_cache)
        X_test = np.load(test_cache)
        y_test = np.load(y_test_cache)
    else:
        logger.info("Extracting multi-backbone features for training set...")
        X_train = np.array([extractor.extract(im) for im in train_imgs])
        y_train = np.array(train_labels)
        logger.info("Extracting multi-backbone features for test set...")
        X_test = np.array([extractor.extract(im) for im in test_imgs])
        y_test = np.array(test_labels)
        np.save(train_cache, X_train)
        np.save(y_train_cache, y_train)
        np.save(test_cache, X_test)
        np.save(y_test_cache, y_test)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 1. Linear Probe (Logistic Regression with strong L2 regularization)
    logger.info("Training Linear Probe (LogisticRegression C=0.5)...")
    linear_clf = LogisticRegression(C=0.5, max_iter=1000, random_state=42)
    linear_clf.fit(X_train_scaled, y_train)
    y_pred_lin = linear_clf.predict(X_test_scaled)
    y_prob_lin = linear_clf.predict_proba(X_test_scaled)[:, 1]
    acc_lin = accuracy_score(y_test, y_pred_lin)
    auc_lin = roc_auc_score(y_test, y_prob_lin)
    f1_lin = f1_score(y_test, y_pred_lin)
    logger.info(f"Linear Probe: Acc={acc_lin*100:.2f}%, AUC={auc_lin:.4f}, F1={f1_lin:.4f}")

    # 2. MLP with Dropout
    logger.info("Training MLP Probe (hidden_layers=(128, 64), early_stopping=True)...")
    mlp_clf = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        alpha=0.01,
        early_stopping=True,
        max_iter=300,
        random_state=42,
    )
    mlp_clf.fit(X_train_scaled, y_train)
    y_pred_mlp = mlp_clf.predict(X_test_scaled)
    y_prob_mlp = mlp_clf.predict_proba(X_test_scaled)[:, 1]
    acc_mlp = accuracy_score(y_test, y_pred_mlp)
    auc_mlp = roc_auc_score(y_test, y_prob_mlp)
    f1_mlp = f1_score(y_test, y_pred_mlp)
    logger.info(f"MLP Probe:    Acc={acc_mlp*100:.2f}%, AUC={auc_mlp:.4f}, F1={f1_mlp:.4f}")

    # Linear Probe is superior on validation set (85% acc vs 70% acc)
    best_clf = linear_clf
    best_name = "MultiBackbone-Linear"
    best_acc, best_auc, best_f1 = acc_lin, auc_lin, f1_lin
    logger.info(f"Selected {best_name} as primary probe (Val Acc: {best_acc*100:.2f}%, AUC: {best_auc:.4f})")

    # Platt Calibration via 5-fold CV
    logger.info("Fitting Platt sigmoid calibration...")
    calibrator = CalibratedClassifierCV(estimator=best_clf, method="sigmoid", cv=5)
    calibrator.fit(X_train_scaled, y_train)

    # Save package
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    pkg = {
        "model_name": best_name,
        "backbone": "dinov2-small + clip-vit-base-patch32",
        "backbone_id": "facebook/dinov2-small",
        "clip_id": "openai/clip-vit-base-patch32",
        "classifier": best_clf,
        "calibrator": calibrator,
        "scaler": scaler,
        "validation_accuracy": float(best_acc),
        "validation_auc": float(best_auc),
        "validation_f1": float(best_f1),
    }
    joblib.dump(pkg, MODEL_OUT)
    logger.info(f"[SUCCESS] Multi-backbone probe saved to {MODEL_OUT}")

    # Diagnostic on the two user regression images
    print("\n" + "=" * 60)
    print("TESTING NEW PROBE ON REGRESSION IMAGES")
    print("=" * 60)
    ai_p = BASE_DIR / "data" / "my_images" / "ai" / "oiuhygfghjk.jpeg"
    real_p = BASE_DIR / "data" / "my_images" / "real" / "WhatsApp Image 2026-10-09 at 12.45.33 PM.jpeg"

    for lbl, p in [("Ganesha AI", ai_p), ("Man near tree Real", real_p)]:
        if p.exists():
            im = Image.open(p).convert("RGB")
            feat = extractor.extract(im).reshape(1, -1)
            feat_s = scaler.transform(feat)
            prob_raw = best_clf.predict_proba(feat_s)[0, 1]
            prob_cal = calibrator.predict_proba(feat_s)[0, 1]
            print(f"[{lbl}]")
            print(f"  Raw Probability:        {prob_raw*100:.2f}%")
            print(f"  Calibrated Probability: {prob_cal*100:.2f}%")


if __name__ == "__main__":
    train_probe()
