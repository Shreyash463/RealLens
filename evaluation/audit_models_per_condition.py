"""Audit individual models across 8 conditions on the matched test set.

Computes AUC, Accuracy@0.5, TPR, and FPR for each model across conditions:
1. original
2. jpeg_85
3. jpeg_70
4. jpeg_55
5. resized_1080
6. resized_720
7. whatsapp
8. screenshot

Models audited:
- dima806 (ViT-B/16 CIFAKE)
- Organika (SDXL Specialist)
- Ateeqq (SigLIP Generalist)
- DINOv2 Probe (Existing Phase 2/4 probe)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score, accuracy_score
import joblib

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("audit_models")

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_DIR = BASE_DIR / "data" / "matched_dataset" / "test"
REPORT_PATH = BASE_DIR / "evaluation" / "model_audit_results.json"

CONDITIONS = [
    "original",
    "jpeg_85",
    "jpeg_70",
    "jpeg_55",
    "resized_1080",
    "resized_720",
    "whatsapp",
    "screenshot",
]


def load_hf_model(model_id: str):
    from transformers import AutoImageProcessor, AutoModelForImageClassification
    device = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModelForImageClassification.from_pretrained(model_id).to(device)
    model.eval()
    return proc, model, device


def infer_hf(proc, model, device, img: Image.Image, fake_idx: int) -> float:
    img_rgb = img.convert("RGB")
    inputs = proc(images=img_rgb, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(**inputs)
        probs = torch.softmax(out.logits, dim=1).cpu().numpy()[0]
    return float(probs[fake_idx])


def run_audit():
    logger.info("Initializing models for Phase 5 condition audit...")
    from transformers import AutoImageProcessor, AutoModel

    # 1. dima806
    logger.info("Loading dima806/ai_vs_real_image_detection...")
    p_dima, m_dima, dev_dima = load_hf_model("dima806/ai_vs_real_image_detection")
    # In dima806: label 0 is REAL, label 1 is FAKE
    dima_fake_idx = 1

    # 2. Organika
    logger.info("Loading Organika/sdxl-detector...")
    p_org, m_org, dev_org = load_hf_model("Organika/sdxl-detector")
    # In Organika: id2label {0: 'artificial', 1: 'human'} -> fake is 0
    org_fake_idx = 0

    # 3. Ateeqq
    logger.info("Loading Ateeqq/ai-vs-human-image-detector...")
    p_atq, m_atq, dev_atq = load_hf_model("Ateeqq/ai-vs-human-image-detector")
    # In Ateeqq: label 0 is AI, label 1 is Human -> fake is 0
    atq_fake_idx = 0

    # 4. DINOv2 Probe
    logger.info("Loading existing generalizing probe...")
    probe_pkg = joblib.load(BASE_DIR / "models" / "generalizing_probe.joblib")
    p_dino = AutoImageProcessor.from_pretrained(probe_pkg.get("backbone_id", "facebook/dinov2-small"))
    m_dino = AutoModel.from_pretrained(probe_pkg.get("backbone_id", "facebook/dinov2-small")).to(dev_dima)
    m_dino.eval()
    dino_clf = probe_pkg.get("classifier")
    dino_scaler = probe_pkg.get("scaler")

    models_dict = {
        "dima806": (p_dima, m_dima, dev_dima, dima_fake_idx, "hf"),
        "Organika_sdxl": (p_org, m_org, dev_org, org_fake_idx, "hf"),
        "Ateeqq_siglip": (p_atq, m_atq, dev_atq, atq_fake_idx, "hf"),
        "DINOv2_probe": (p_dino, m_dino, dev_dima, None, "probe"),
    }

    results: Dict[str, Dict[str, Dict[str, float]]] = {}

    for model_name, (proc, model, dev, fake_idx, m_type) in models_dict.items():
        results[model_name] = {}
        logger.info(f"\nEvaluating {model_name} across all 8 conditions...")

        for cond in CONDITIONS:
            cond_dir = TEST_DIR / cond
            y_true, y_scores = [], []

            # Real
            real_files = sorted(list((cond_dir / "real").glob("*.jpg")))
            for p in real_files:
                im = Image.open(p).convert("RGB")
                if m_type == "hf":
                    score = infer_hf(proc, model, dev, im, fake_idx)
                else:
                    inputs = proc(images=[im], return_tensors="pt").to(dev)
                    with torch.no_grad():
                        out = model(**inputs)
                        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                    emb_s = dino_scaler.transform(emb)
                    score = float(dino_clf.predict_proba(emb_s)[0, 1])
                y_true.append(0)
                y_scores.append(score)

            # AI
            ai_files = sorted(list((cond_dir / "ai").glob("*.jpg")))
            for p in ai_files:
                im = Image.open(p).convert("RGB")
                if m_type == "hf":
                    score = infer_hf(proc, model, dev, im, fake_idx)
                else:
                    inputs = proc(images=[im], return_tensors="pt").to(dev)
                    with torch.no_grad():
                        out = model(**inputs)
                        emb = out.last_hidden_state[:, 0, :].cpu().numpy()
                    emb_s = dino_scaler.transform(emb)
                    score = float(dino_clf.predict_proba(emb_s)[0, 1])
                y_true.append(1)
                y_scores.append(score)

            y_true = np.array(y_true)
            y_scores = np.array(y_scores)
            y_pred = (y_scores >= 0.5).astype(int)

            try:
                auc = float(roc_auc_score(y_true, y_scores))
            except Exception:
                auc = 0.5

            acc = float(accuracy_score(y_true, y_pred))
            # TPR = TP / P (Recall on AI)
            tp = np.sum((y_pred == 1) & (y_true == 1))
            p_total = np.sum(y_true == 1)
            tpr = float(tp / max(p_total, 1))

            # FPR = FP / N (False positives on Real)
            fp = np.sum((y_pred == 1) & (y_true == 0))
            n_total = np.sum(y_true == 0)
            fpr = float(fp / max(n_total, 1))

            results[model_name][cond] = {
                "auc": round(auc, 4),
                "acc": round(acc, 4),
                "tpr": round(tpr, 4),
                "fpr": round(fpr, 4),
            }
            logger.info(f"  [{cond:13s}] AUC: {auc:.4f} | Acc: {acc*100:.1f}% | TPR: {tpr*100:.1f}% | FPR: {fpr*100:.1f}%")

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nModel condition audit saved to {REPORT_PATH}")
    print("\n" + "=" * 80)
    print("PHASE 5 PART 0: MODEL AUDIT SUMMARY ACROSS CONDITIONS")
    print("=" * 80)
    header = f"{'Model':16s} | {'Cond':12s} | {'AUC':6s} | {'Acc@0.5':7s} | {'TPR':6s} | {'FPR':6s}"
    print(header)
    print("-" * 80)
    for m_name in results:
        for cond in CONDITIONS:
            m_res = results[m_name][cond]
            print(f"{m_name:16s} | {cond:12s} | {m_res['auc']:.4f} | {m_res['acc']*100:5.1f}% | {m_res['tpr']*100:5.1f}% | {m_res['fpr']*100:5.1f}%")
        print("-" * 80)


if __name__ == "__main__":
    run_audit()
