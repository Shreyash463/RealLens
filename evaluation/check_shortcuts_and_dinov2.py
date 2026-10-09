"""Shortcut check and DINOv2 probe diagnostic script.

Audits whether metadata shortcuts (width, height, aspect ratio, file size,
JPEG quality estimate, format) can separate Real vs AI in current datasets.
Also inspects why DINOv2 probe gave ~1.0 on the real WhatsApp photo.
"""
from pathlib import Path
import os
import sys
import io
import joblib
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))


def estimate_jpeg_quality(img: Image.Image, raw_bytes: bytes) -> float:
    """Roughly estimates JPEG quality from quantization tables if present."""
    try:
        if hasattr(img, "quantization") and img.quantization:
            q_tables = list(img.quantization.values())
            if q_tables:
                avg_q = float(np.mean(q_tables[0]))
                return max(1.0, min(100.0, 100.0 - avg_q))
    except Exception:
        pass
    return 75.0


def extract_metadata_features(img_path: Path):
    raw_bytes = img_path.read_bytes()
    try:
        with Image.open(io.BytesIO(raw_bytes)) as img:
            w, h = img.size
            aspect = w / max(1, h)
            q_est = estimate_jpeg_quality(img, raw_bytes)
            is_png = 1.0 if img_path.suffix.lower() == ".png" else 0.0
            is_jpg = 1.0 if img_path.suffix.lower() in [".jpg", ".jpeg"] else 0.0
            size_kb = len(raw_bytes) / 1024.0
            return [w, h, aspect, size_kb, q_est, is_png, is_jpg]
    except Exception:
        return None


def run_shortcut_check():
    print("=" * 60)
    print("RUNNING SHORTCUT AUDIT ON EXISTING DATASETS")
    print("=" * 60)

    # Collect images from tuning and test sets
    train_paths = []
    train_labels = []
    test_paths = []
    test_labels = []

    # Tuning split
    for label, folder in [(0, "real"), (1, "ai")]:
        p = BASE_DIR / "data" / "tuning" / folder
        if p.exists():
            for f in p.glob("*.*"):
                train_paths.append(f)
                train_labels.append(label)

    # Hard tuning split
    for label, folder in [(0, "real"), (1, "ai")]:
        p = BASE_DIR / "data" / "hard_tuning" / folder
        if p.exists():
            for f in p.glob("*.*"):
                train_paths.append(f)
                train_labels.append(label)

    # Test split
    for label, folder in [(0, "real"), (1, "ai")]:
        p = BASE_DIR / "data" / "test" / folder
        if p.exists():
            for f in p.glob("*.*"):
                test_paths.append(f)
                test_labels.append(label)

    # Hard test split
    for label, folder in [(0, "real"), (1, "ai")]:
        p = BASE_DIR / "data" / "hard_test" / folder
        if p.exists():
            for f in p.glob("*.*"):
                test_paths.append(f)
                test_labels.append(label)

    print(f"Collected {len(train_paths)} training samples, {len(test_paths)} test samples.")

    X_train = []
    y_train = []
    for p, y in zip(train_paths, train_labels):
        feat = extract_metadata_features(p)
        if feat:
            X_train.append(feat)
            y_train.append(y)

    X_test = []
    y_test = []
    for p, y in zip(test_paths, test_labels):
        feat = extract_metadata_features(p)
        if feat:
            X_test.append(feat)
            y_test.append(y)

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_test, y_test = np.array(X_test), np.array(y_test)

    # Train a decision tree and logistic regression on metadata shortcuts
    dt = DecisionTreeClassifier(max_depth=3)
    dt.fit(X_train, y_train)
    train_acc_dt = accuracy_score(y_train, dt.predict(X_train))
    test_acc_dt = accuracy_score(y_test, dt.predict(X_test))

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    train_acc_lr = accuracy_score(y_train, lr.predict(X_train))
    test_acc_lr = accuracy_score(y_test, lr.predict(X_test))

    print(f"\nMetadata Shortcut Classifier Results:")
    print(f"Decision Tree (depth 3) - Train Acc: {train_acc_dt*100:.2f}%, Test Acc: {test_acc_dt*100:.2f}%")
    print(f"Logistic Regression     - Train Acc: {train_acc_lr*100:.2f}%, Test Acc: {test_acc_lr*100:.2f}%")
    print(f"Feature Importances (Tree): {dt.feature_importances_}")
    print("Features: [width, height, aspect, size_kb, q_est, is_png, is_jpg]")


def run_dinov2_diagnostic():
    print("\n" + "=" * 60)
    print("RUNNING DINOv2 PROBE DIAGNOSTIC")
    print("=" * 60)
    probe_path = BASE_DIR / "models" / "generalizing_probe.joblib"
    if not probe_path.exists():
        print(f"Probe file not found at {probe_path}")
        return

    pkg = joblib.load(probe_path)
    print(f"Probe keys: {list(pkg.keys())}")
    print(f"Backbone: {pkg.get('backbone_id')}")
    clf = pkg.get("classifier")
    scaler = pkg.get("scaler")
    print(f"Classifier type: {type(clf)}")
    if hasattr(clf, "coef_"):
        print(f"Coef shape: {clf.coef_.shape}, norm: {np.linalg.norm(clf.coef_):.4f}")
        print(f"Intercept: {clf.intercept_}")

    # Test feature extraction on the two regression images
    from transformers import AutoImageProcessor, AutoModel
    import torch

    device = "cpu"
    proc = AutoImageProcessor.from_pretrained(pkg.get("backbone_id", "facebook/dinov2-small"))
    model = AutoModel.from_pretrained(pkg.get("backbone_id", "facebook/dinov2-small")).to(device)
    model.eval()

    ai_p = BASE_DIR / "data" / "my_images" / "ai" / "oiuhygfghjk.jpeg"
    real_p = BASE_DIR / "data" / "my_images" / "real" / "WhatsApp Image 2026-10-09 at 12.45.33 PM.jpeg"

    for lbl, p in [("Ganesha AI", ai_p), ("Man near tree Real", real_p)]:
        img = Image.open(p).convert("RGB")
        inputs = proc(images=[img], return_tensors="pt")
        with torch.no_grad():
            out = model(**inputs)
            emb = out.last_hidden_state[:, 0, :].cpu().numpy()
        
        emb_scaled = scaler.transform(emb)
        prob = clf.predict_proba(emb_scaled)[0, 1]
        raw_margin = clf.decision_function(emb_scaled)[0]
        print(f"\n[{lbl}]")
        print(f"Raw decision function margin: {raw_margin:.4f}")
        print(f"Calibrated probability: {prob:.4f}")
        print(f"Embedding L2 norm: {np.linalg.norm(emb):.4f}")


if __name__ == "__main__":
    run_shortcut_check()
    run_dinov2_diagnostic()
