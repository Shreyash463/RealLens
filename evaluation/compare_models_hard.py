"""Compares all 4 models individually on the Hard Tuning dataset.

Evaluates:
1. dima806/ai_vs_real_image_detection
2. Organika/sdxl-detector
3. Ateeqq/ai-vs-human-image-detector
4. DINOv2 Generalizing Probe
"""
from __future__ import annotations

import sys
from pathlib import Path
import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

BASE_DIR = Path(__file__).resolve().parent.parent

def run_model_comparison():
    data_dir = BASE_DIR / "data" / "hard_tuning"
    real_paths = list((data_dir / "real").glob("*.*"))
    ai_paths = list((data_dir / "ai").glob("*.*"))

    all_paths = [(p, 0) for p in real_paths] + [(p, 1) for p in ai_paths]
    y_true = [label for _, label in all_paths]
    images = [Image.open(p).convert("RGB") for p, _ in all_paths]

    print("=" * 70)
    print("  MODEL COMPARISON ON HARD TUNING SET (30 Real, 30 AI)")
    print("=" * 70)

    # 1. dima806
    print("\n[1/4] Evaluating dima806/ai_vs_real_image_detection...")
    proc_dima = AutoImageProcessor.from_pretrained("dima806/ai_vs_real_image_detection")
    model_dima = AutoModelForImageClassification.from_pretrained("dima806/ai_vs_real_image_detection")
    model_dima.eval()
    dima_probs = []
    with torch.no_grad():
        for im in images:
            inputs = proc_dima(images=im, return_tensors="pt")
            logits = model_dima(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            dima_probs.append(float(probs[0]))  # Index 0 is AI
    dima_preds = [1 if p >= 0.50 else 0 for p in dima_probs]

    # 2. Organika
    print("\n[2/4] Evaluating Organika/sdxl-detector...")
    proc_org = AutoImageProcessor.from_pretrained("Organika/sdxl-detector")
    model_org = AutoModelForImageClassification.from_pretrained("Organika/sdxl-detector")
    model_org.eval()
    org_probs = []
    with torch.no_grad():
        for im in images:
            inputs = proc_org(images=im, return_tensors="pt")
            logits = model_org(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            org_probs.append(float(probs[1]))  # Index 1 is AI
    org_preds = [1 if p >= 0.50 else 0 for p in org_probs]

    # 3. Ateeqq
    print("\n[3/4] Evaluating Ateeqq/ai-vs-human-image-detector...")
    proc_at = AutoImageProcessor.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    model_at = AutoModelForImageClassification.from_pretrained("Ateeqq/ai-vs-human-image-detector")
    model_at.eval()
    at_probs = []
    with torch.no_grad():
        for im in images:
            inputs = proc_at(images=im, return_tensors="pt")
            logits = model_at(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].numpy()
            at_probs.append(float(probs[0]))  # Index 0 is AI
    at_preds = [1 if p >= 0.50 else 0 for p in at_probs]

    # 4. DINOv2 Probe
    print("\n[4/4] Evaluating DINOv2 Generalizing Probe...")
    probe_pkg = joblib.load(BASE_DIR / "models" / "generalizing_probe.joblib")
    proc_dino = AutoImageProcessor.from_pretrained(probe_pkg["backbone_id"])
    model_dino = AutoModel.from_pretrained(probe_pkg["backbone_id"])
    model_dino.eval()
    clf = probe_pkg["classifier"]
    scaler = probe_pkg["scaler"]

    probe_probs = []
    with torch.no_grad():
        for im in images:
            inputs = proc_dino(images=im, return_tensors="pt")
            outputs = model_dino(**inputs)
            emb = outputs.last_hidden_state[:, 0, :].numpy()
            emb_scaled = scaler.transform(emb)
            prob_ai = float(clf.predict_proba(emb_scaled)[0, 1])
            probe_probs.append(prob_ai)
    probe_preds = [1 if p >= 0.50 else 0 for p in probe_probs]

    # Comparison Table
    models_eval = [
        ("dima806 (CIFAKE ViT)", dima_preds, dima_probs),
        ("Organika (SDXL ViT)", org_preds, org_probs),
        ("Ateeqq (SigLIP)", at_preds, at_probs),
        ("DINOv2 Generalizing Probe (Ours)", probe_preds, probe_probs),
    ]

    print("\n" + "=" * 70)
    print(f"{'Model Name':<35} | {'Accuracy':<10} | {'Recall':<10} | {'Precision':<10} | {'F1-Score':<10}")
    print("-" * 70)
    for name, preds, probs in models_eval:
        acc = accuracy_score(y_true, preds)
        rec = recall_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        print(f"{name:<35} | {acc * 100:>8.2f}% | {rec * 100:>8.2f}% | {prec * 100:>8.2f}% | {f1:>8.4f}")

if __name__ == "__main__":
    run_model_comparison()
