"""Trains a learned Calibrated Logistic Regression Fusion Stacker for RealLens.

Trained strictly on the tuning sets (data/tuning/ and data/hard_tuning/).
Inputs:
- Model scores: dima806, Organika, Ateeqq, DINOv2 probe
- Ensemble median and spread
- ELA compression score
- FFT frequency score
- Metadata camera flag
- C2PA flag

Outputs:
- models/learned_fusion_stacker.joblib
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import numpy as np
from PIL import Image
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from reallens.detectors.c2pa_check import analyze_c2pa
from reallens.detectors.ela import analyze_ela
from reallens.detectors.frequency import analyze_frequency
from reallens.detectors.metadata import analyze_metadata
from reallens.detectors.ml_ensemble import MLEnsemble

def extract_features_for_image(img_path: Path, ensemble: MLEnsemble) -> Tuple[np.ndarray, int]:
    raw_bytes = img_path.read_bytes()
    pil_img = Image.open(img_path).convert("RGB")

    # 1. Metadata & C2PA
    meta = analyze_metadata(raw_bytes)
    c2pa = analyze_c2pa(raw_bytes)
    has_meta_cam = 1.0 if (meta.get("score") is not None and meta["score"] < 0.20) else 0.0
    c2pa_ai = 1.0 if (c2pa.get("score") is not None and c2pa["score"] > 0.80) else 0.0

    # 2. Forensics
    ela = analyze_ela(pil_img)
    freq = analyze_frequency(pil_img)
    ela_s = ela.get("score", 0.50) if ela.get("score") is not None else 0.50
    freq_s = freq.get("score", 0.50) if freq.get("score") is not None else 0.50

    # 3. ML Ensemble
    ml_res = ensemble.analyze(pil_img)
    per_scores = ml_res.get("per_model_scores", {})
    s_dima = per_scores.get("dima806/ai_vs_real_image_detection", 0.50) or 0.50
    s_org = per_scores.get("Organika/sdxl-detector", 0.50) or 0.50
    s_at = per_scores.get("Ateeqq/ai-vs-human-image-detector", 0.50) or 0.50
    s_probe = per_scores.get("generalizing_dinov2_probe", 0.50) or 0.50
    s_med = ml_res.get("score", 0.50) or 0.50
    s_spread = ml_res.get("spread", 0.0) or 0.0

    feat = np.array([
        s_dima,
        s_org,
        s_at,
        s_probe,
        s_med,
        s_spread,
        ela_s,
        freq_s,
        has_meta_cam,
        c2pa_ai,
    ], dtype=np.float32)

    return feat

def train_learned_fusion():
    data_dir = BASE_DIR / "data"
    ensemble = MLEnsemble()

    print("=" * 65)
    print("  EXTRACTING DETECTOR SIGNALS FOR LEARNED FUSION STACKER")
    print("=" * 65)

    # Collect tuning paths
    exts = ("*.jpg", "*.jpeg", "*.png", "*.webp")
    train_paths = []
    # 1. Easy tuning
    for e in exts:
        for p in (data_dir / "tuning" / "real").glob(e):
            train_paths.append((p, 0))
        for p in (data_dir / "tuning" / "ai").glob(e):
            train_paths.append((p, 1))
        # 2. Hard tuning
        for p in (data_dir / "hard_tuning" / "real").glob(e):
            train_paths.append((p, 0))
        for p in (data_dir / "hard_tuning" / "ai").glob(e):
            train_paths.append((p, 1))

    print(f"Total tuning images for fusion training: {len(train_paths)}")

    X_list = []
    y_list = []
    for idx, (p, label) in enumerate(train_paths):
        feat = extract_features_for_image(p, ensemble)
        X_list.append(feat)
        y_list.append(label)
        if (idx + 1) % 50 == 0 or idx == len(train_paths) - 1:
            print(f"  Processed {idx + 1}/{len(train_paths)} tuning images...")

    X = np.array(X_list)
    y = np.array(y_list)

    print(f"Feature matrix shape: {X.shape}, Classes: {np.bincount(y)}")

    # Train Calibrated Logistic Regression Stacker
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    base_lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    calibrated_stacker = CalibratedClassifierCV(estimator=base_lr, cv=3, method="sigmoid")
    calibrated_stacker.fit(X_scaled, y)

    # Evaluate on Hard Tuning subset
    # Last 60 samples in train_paths are hard_tuning
    hard_tune_idx = [i for i, (p, _) in enumerate(train_paths) if "hard_tuning" in str(p)]
    val_X_scaled = X_scaled[hard_tune_idx]
    val_y = y[hard_tune_idx]

    preds = calibrated_stacker.predict(val_X_scaled)
    probs = calibrated_stacker.predict_proba(val_X_scaled)[:, 1]

    acc = accuracy_score(val_y, preds)
    rec = recall_score(val_y, preds)
    prec = precision_score(val_y, preds)
    f1 = f1_score(val_y, preds)

    print("\n" + "=" * 65)
    print("  LEARNED FUSION STACKER PERFORMANCE ON HARD TUNING SET")
    print("=" * 65)
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Recall:    {rec * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}%")
    print(f"F1-Score:  {f1:.4f}")

    # Save stacker package
    out_pkg = {
        "scaler": scaler,
        "model": calibrated_stacker,
        "feature_names": [
            "s_dima", "s_org", "s_at", "s_probe",
            "s_med", "s_spread", "ela_score", "freq_score",
            "has_meta_cam", "c2pa_ai",
        ],
        "hard_tune_accuracy": acc,
        "hard_tune_f1": f1,
    }
    out_path = BASE_DIR / "models" / "learned_fusion_stacker.joblib"
    joblib.dump(out_pkg, out_path)
    print(f"\n[SAVED] Learned fusion stacker saved to: {out_path}")

if __name__ == "__main__":
    train_learned_fusion()
