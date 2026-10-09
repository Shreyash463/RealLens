"""Trainer and Evaluator for the Learned Photo-vs-Non-Photo Gate Classifier.

Extracts DINOv2-small (384-d) frozen embeddings plus 8 engineered visual features
(flat patch ratio, white paper ratio, color entropy, noise level, edge sharpness,
block boundary ratio, saturation metrics).

Trains Logistic Regression and small MLP classifiers to output a continuous
calibrated `photo_probability` in [0.0, 1.0].

Evaluates on the held-out validation set:
- Photo recall on WhatsApp-compressed phone photos (Target >= 95%)
- Non-photo recall on authentic certificates, slides, charts, screenshots
- Confusion matrix and error breakdown
Saves model package to `models/photo_vs_non_photo_gate.joblib`.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
from transformers import AutoImageProcessor, AutoModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("train_gate_classifier")

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data" / "gate_dataset"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/dinov2-small"


def extract_engineered_features(img: Image.Image) -> np.ndarray:
    """Extracts 8 normalized continuous visual features (no aspect ratio or resolution!)."""
    # Downsample to standard 256x256 for fast uniform computation
    thumb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(thumb, dtype=np.float32)

    # 1. Flat patch ratio (fraction of 8x8 blocks with near-zero standard deviation)
    blocks = arr.reshape(32, 8, 32, 8, 3).swapaxes(1, 2)  # (32, 32, 8, 8, 3)
    block_stds = np.std(blocks, axis=(2, 3, 4))
    flat_patch_ratio = float(np.mean(block_stds < 3.2))

    # 2. White paper dominance (luminance > 240 and saturation < 0.15)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 1e-4, np.clip((max_c - min_c) / np.maximum(max_c, 1e-4), 0.0, 1.0), 0.0)

    white_paper_ratio = float(np.mean((lum > 240.0) & (sat < 0.15)))

    # 3. Unique quantized color palette ratio (16x16x16 bins)
    quant = (arr // 16).astype(np.int32)
    color_codes = quant[:, :, 0] * 256 + quant[:, :, 1] * 16 + quant[:, :, 2]
    unique_colors = len(np.unique(color_codes))
    unique_color_ratio = float(unique_colors / 4096.0)

    # 4. Mean edge sharpness & Laplacian noise floor
    gray = lum
    lap = np.abs(
        gray[1:-1, 1:-1] * 4
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    mean_edge = float(np.mean(lap))

    # Noise on non-flat regions
    non_flat_mask = np.repeat(np.repeat(block_stds >= 4.0, 8, axis=0), 8, axis=1)[1:-1, 1:-1]
    if np.sum(non_flat_mask) > 100:
        noise_level = float(np.std(lap[non_flat_mask]))
    else:
        noise_level = float(np.std(lap))

    # 5. DCT 8x8 block boundary ratio (detects JPEG grid artifacts vs smooth continuous fill)
    diff_h = np.abs(gray[7:-1:8, :] - gray[8::8, :]) if gray.shape[0] >= 16 else np.array([0.0])
    diff_v = np.abs(gray[:, 7:-1:8] - gray[:, 8::8]) if gray.shape[1] >= 16 else np.array([0.0])
    block_boundary_energy = (float(np.mean(diff_h)) + float(np.mean(diff_v))) / 2.0
    int_diff_h = np.abs(gray[3:-1:8, :] - gray[4::8, :]) if gray.shape[0] >= 16 else np.array([1.0])
    int_diff_v = np.abs(gray[:, 3:-1:8] - gray[:, 4::8]) if gray.shape[1] >= 16 else np.array([1.0])
    block_internal_energy = max(1e-4, (float(np.mean(int_diff_h)) + float(np.mean(int_diff_v))) / 2.0)
    block_ratio = float(block_boundary_energy / block_internal_energy)

    # 6. Saturation metrics
    mean_sat = float(np.mean(sat))
    low_sat_ratio = float(np.mean(sat < 0.10))

    return np.array([
        flat_patch_ratio,
        white_paper_ratio,
        unique_color_ratio,
        mean_edge,
        noise_level,
        block_ratio,
        mean_sat,
        low_sat_ratio,
    ], dtype=np.float32)


def extract_dataset_features(
    dataset_dir: Path,
    processor: Any,
    model: Any,
    device: str,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extracts combined DINOv2 + engineered features for all images in directory."""
    photo_dir = dataset_dir / "photo"
    non_photo_dir = dataset_dir / "non_photo"

    photo_files = sorted([p for p in photo_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])
    non_photo_files = sorted([p for p in non_photo_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png", ".webp"]])

    all_files = [(p, 1) for p in photo_files] + [(p, 0) for p in non_photo_files]

    features_list = []
    labels_list = []
    names_list = []

    logger.info(f"Extracting features from {len(all_files)} images in {dataset_dir.name}...")

    batch_size = 16
    for i in range(0, len(all_files), batch_size):
        batch = all_files[i : i + batch_size]
        pil_images = []
        eng_feats = []
        batch_labels = []
        batch_names = []

        for path, label in batch:
            try:
                with Image.open(path) as img:
                    img_rgb = img.convert("RGB")
                    pil_images.append(img_rgb)
                    eng_feats.append(extract_engineered_features(img_rgb))
                    batch_labels.append(label)
                    batch_names.append(path.name)
            except Exception as e:
                logger.warning(f"Error reading {path.name}: {e}")

        if not pil_images:
            continue

        # Extract DINOv2 embeddings
        inputs = processor(images=pil_images, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            # CLS token embedding: shape (batch_size, 384)
            dinov2_embs = outputs.last_hidden_state[:, 0, :].cpu().numpy()

        eng_array = np.array(eng_feats, dtype=np.float32)
        # Concatenate: shape (batch_size, 384 + 8 = 392)
        combined = np.hstack([dinov2_embs, eng_array])

        features_list.append(combined)
        labels_list.extend(batch_labels)
        names_list.extend(batch_names)

    X = np.vstack(features_list)
    y = np.array(labels_list, dtype=np.int32)
    return X, y, names_list


def train_and_evaluate():
    logger.info("Initializing DINOv2 backbone for gate training...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    model.eval()

    # 1. Extract Training Features
    train_dir = DATA_DIR / "train"
    X_train, y_train, names_train = extract_dataset_features(train_dir, processor, model, device)
    logger.info(f"Extracted X_train shape: {X_train.shape}, y_train class balance: {np.bincount(y_train)}")

    # 2. Extract Validation Features (strictly held out!)
    val_dir = DATA_DIR / "val"
    X_val, y_val, names_val = extract_dataset_features(val_dir, processor, model, device)
    logger.info(f"Extracted X_val shape: {X_val.shape}, y_val class balance: {np.bincount(y_val)}")

    # 3. Fit StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 4. Train Candidates: Logistic Regression vs MLP
    logger.info("Training Model 1: Calibrated Logistic Regression (C=1.0)...")
    base_lr = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42)
    clf_lr = CalibratedClassifierCV(estimator=base_lr, method="sigmoid", cv=5)
    clf_lr.fit(X_train_scaled, y_train)

    logger.info("Training Model 2: Small MLP Classifier (64 hidden units)...")
    base_mlp = MLPClassifier(hidden_layer_sizes=(64,), max_iter=500, random_state=42, early_stopping=True)
    clf_mlp = CalibratedClassifierCV(estimator=base_mlp, method="sigmoid", cv=5)
    clf_mlp.fit(X_train_scaled, y_train)

    # 5. Evaluate on Validation Set
    models_to_test = [("LogisticRegression", clf_lr), ("MLPClassifier", clf_mlp)]

    best_name = None
    best_clf = None
    best_f1 = -1.0
    best_val_report = {}

    for name, clf in models_to_test:
        probs = clf.predict_proba(X_val_scaled)[:, 1]
        preds = (probs >= 0.50).astype(int)

        acc = float(np.mean(preds == y_val))
        prec = float(precision_score(y_val, preds, zero_division=0))
        rec = float(recall_score(y_val, preds, zero_division=0))
        f1 = float(f1_score(y_val, preds, zero_division=0))
        cm = confusion_matrix(y_val, preds)

        # Break down photo recall specifically on WhatsApp compressed photos
        wa_indices = [i for i, n in enumerate(names_val) if "wa" in n.lower() or "whatsapp" in n.lower()]
        if wa_indices:
            wa_preds = preds[wa_indices]
            wa_true = y_val[wa_indices]
            wa_recall = float(np.mean(wa_preds == wa_true))
        else:
            wa_recall = rec

        # Non-photo recall
        non_photo_indices = [i for i, y_i in enumerate(y_val) if y_i == 0]
        non_photo_recall = float(np.mean(preds[non_photo_indices] == 0))

        logger.info(f"--- Results for {name} on Held-Out Validation Set ---")
        logger.info(f"Accuracy: {acc*100:.2f}% | Precision: {prec*100:.2f}% | Recall: {rec*100:.2f}% | F1: {f1:.4f}")
        logger.info(f"Photo Recall on WhatsApp compressed photos: {wa_recall*100:.2f}% (Target: >=95%)")
        logger.info(f"Non-Photo Recall (Certificates, Slides, Screenshots): {non_photo_recall*100:.2f}%")
        logger.info(f"Confusion Matrix [TN FP; FN TP]:\n{cm}")

        misclassified = []
        for i in range(len(y_val)):
            if preds[i] != y_val[i]:
                misclassified.append({
                    "name": names_val[i],
                    "true_label": "photo" if y_val[i] == 1 else "non_photo",
                    "pred_label": "photo" if preds[i] == 1 else "non_photo",
                    "photo_probability": round(float(probs[i]), 4),
                })
        logger.info(f"Misclassified count: {len(misclassified)}")
        for m in misclassified[:10]:
            logger.info(f"  Misclassified: {m['name']} (True: {m['true_label']}, Pred: {m['pred_label']}, Prob: {m['photo_probability']})")

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_clf = clf
            best_val_report = {
                "model_name": name,
                "accuracy": round(acc, 4),
                "precision": round(prec, 4),
                "recall": round(rec, 4),
                "f1": round(f1, 4),
                "wa_photo_recall": round(wa_recall, 4),
                "non_photo_recall": round(non_photo_recall, 4),
                "confusion_matrix": cm.tolist(),
                "misclassified": misclassified,
            }

    # 6. Save Best Model Package
    package_path = MODELS_DIR / "photo_vs_non_photo_gate.joblib"
    gate_package = {
        "model_architecture": best_name,
        "backbone": MODEL_NAME,
        "scaler": scaler,
        "classifier": best_clf,
        "validation_metrics": best_val_report,
        "feature_dim": X_train.shape[1],
        "version": "1.0.0",
    }
    joblib.dump(gate_package, package_path)
    logger.info(f"Successfully saved winning gate model ({best_name}) to {package_path}")

    # Write summary metrics to a temporary json for EVAL_REPORT tracking
    with open(MODELS_DIR / "gate_eval_metrics.json", "w", encoding="utf-8") as f:
        json.dump(best_val_report, f, indent=2)

    return best_val_report


if __name__ == "__main__":
    train_and_evaluate()
