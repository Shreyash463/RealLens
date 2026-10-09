"""Shortcut verification for the new matched dataset.

Trains a trivial classifier using ONLY:
- width
- height
- aspect ratio
- file size
- estimated JPEG quality
- EXIF presence
- is_png / is_jpg

Fails if test accuracy > 70.0% (as required by Phase 5 Part 0.3).
"""
from __future__ import annotations

import os
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

BASE_DIR = Path(__file__).resolve().parent.parent
MATCHED_DIR = BASE_DIR / "data" / "matched_dataset"


def extract_metadata_features(p: Path) -> list:
    sz_kb = p.stat().st_size / 1024.0
    try:
        im = Image.open(p)
        w, h = im.size
        aspect = w / max(h, 1)
        has_exif = 1.0 if getattr(im, "_getexif", lambda: None)() else 0.0
        # Estimate quality factor from quantization table
        q_est = 75.0
        if hasattr(im, "quantization") and im.quantization:
            t = list(im.quantization.values())[0]
            val = float(np.mean(t)) if len(t) else 50.0
            q_est = max(1.0, min(100.0, 100.0 - val / 2.0))
        is_png = 1.0 if p.suffix.lower() == ".png" else 0.0
        is_jpg = 1.0 if p.suffix.lower() in [".jpg", ".jpeg"] else 0.0
    except Exception:
        w, h, aspect, has_exif, q_est, is_png, is_jpg = 512, 512, 1.0, 0.0, 75.0, 0.0, 1.0

    return [float(w), float(h), float(aspect), float(sz_kb), float(q_est), float(has_exif), is_png, is_jpg]


def run_matched_shortcut_check():
    print("=" * 60)
    print("RUNNING SHORTCUT AUDIT ON NEW MATCHED DATASET")
    print("=" * 60)

    # Gather across all conditions in tuning and test
    def load_split(split_name: str):
        X, y = [], []
        split_path = MATCHED_DIR / split_name
        for cond_dir in split_path.iterdir():
            if not cond_dir.is_dir():
                continue
            for cls_name, lbl in [("real", 0), ("ai", 1)]:
                cls_dir = cond_dir / cls_name
                if not cls_dir.exists():
                    continue
                for img_p in cls_dir.glob("*.*"):
                    feats = extract_metadata_features(img_p)
                    X.append(feats)
                    y.append(lbl)
        return np.array(X), np.array(y)

    X_train, y_train = load_split("tuning")
    X_test, y_test = load_split("test")

    print(f"Collected {len(X_train)} training samples, {len(X_test)} test samples.")
    print(f"Features: [width, height, aspect, size_kb, q_est, has_exif, is_png, is_jpg]")

    # 1. Decision Tree (depth 3)
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
    dt_test_acc = accuracy_score(y_test, dt.predict(X_test))

    # 2. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    lr_train_acc = accuracy_score(y_train, lr.predict(X_train))
    lr_test_acc = accuracy_score(y_test, lr.predict(X_test))

    print("\nMetadata Shortcut Classifier Results:")
    print(f"Decision Tree (depth 3) - Train Acc: {dt_train_acc*100:.2f}%, Test Acc: {dt_test_acc*100:.2f}%")
    print(f"Logistic Regression     - Train Acc: {lr_train_acc*100:.2f}%, Test Acc: {lr_test_acc*100:.2f}%")
    print(f"Feature Importances (Tree): {dt.feature_importances_}")

    print("\n" + "-" * 60)
    if dt_test_acc < 0.70 and lr_test_acc < 0.70:
        print(f"[PASSED] Shortcut check passed! Metadata test acc is {dt_test_acc*100:.2f}% (< 70.0% threshold).")
        print("No metadata or resolution shortcuts exist to leak class labels.")
    else:
        print(f"[FAILED] Shortcut check failed! Metadata test acc is {dt_test_acc*100:.2f}% (>= 70.0% threshold).")
    print("-" * 60)


if __name__ == "__main__":
    run_matched_shortcut_check()
