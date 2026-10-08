"""Evaluation engine for RealLens.

Evaluates the detector pipeline on specified splits (tuning, held-out test,
messy real, or user images) and reports honest metrics: Accuracy, Precision,
Recall, F1-score, Confusion Matrix, and detailed misclassification errors.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Ensure reallens root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from reallens.app import analyze_image_bytes


def evaluate_directory(image_dir: Path, ground_truth_label: int, threshold: float = 0.50) -> List[Dict[str, Any]]:
    """Evaluates all images in a single directory.

    ground_truth_label: 0 for Real, 1 for AI.
    """
    results = []
    extensions = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    files: List[Path] = []
    for ext in extensions:
        files.extend(image_dir.glob(ext))

    files.sort()

    for p in files:
        try:
            raw_bytes = p.read_bytes()
            res = analyze_image_bytes(raw_bytes, filename=p.name)
            prob = res.get("ai_probability", 0.5)
            pred_label = 1 if prob >= threshold else 0
            results.append({
                "path": str(p),
                "filename": p.name,
                "ground_truth": ground_truth_label,
                "predicted_label": pred_label,
                "ai_probability": prob,
                "verdict": res.get("verdict"),
                "confidence": res.get("confidence"),
                "summary": res.get("summary_reason"),
                "signals": res.get("signals"),
                "is_correct": (pred_label == ground_truth_label),
            })
        except Exception as exc:
            results.append({
                "path": str(p),
                "filename": p.name,
                "ground_truth": ground_truth_label,
                "predicted_label": -1,
                "ai_probability": 0.5,
                "verdict": "Error",
                "confidence": "low",
                "summary": f"Evaluation error: {exc}",
                "signals": {},
                "is_correct": False,
            })
    return results


def calculate_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes binary classification metrics."""
    valid_records = [r for r in eval_records if r["ground_truth"] in (0, 1) and r["predicted_label"] in (0, 1)]
    if not valid_records:
        return {"total": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    tp = sum(1 for r in valid_records if r["ground_truth"] == 1 and r["predicted_label"] == 1)
    tn = sum(1 for r in valid_records if r["ground_truth"] == 0 and r["predicted_label"] == 0)
    fp = sum(1 for r in valid_records if r["ground_truth"] == 0 and r["predicted_label"] == 1)
    fn = sum(1 for r in valid_records if r["ground_truth"] == 1 and r["predicted_label"] == 0)

    total = len(valid_records)
    acc = (tp + tn) / total if total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    real_probs = [r["ai_probability"] for r in valid_records if r["ground_truth"] == 0]
    ai_probs = [r["ai_probability"] for r in valid_records if r["ground_truth"] == 1]

    mean_real_prob = float(sum(real_probs) / len(real_probs)) if real_probs else 0.0
    mean_ai_prob = float(sum(ai_probs) / len(ai_probs)) if ai_probs else 0.0

    errors = [r for r in valid_records if not r["is_correct"]]

    return {
        "total": total,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_real_prob": round(mean_real_prob, 4),
        "mean_ai_prob": round(mean_ai_prob, 4),
        "error_count": len(errors),
        "errors": errors,
    }


def run_evaluation(split: str = "tuning", threshold: float = 0.50) -> Dict[str, Any]:
    """Runs evaluation on a specified split."""
    data_dir = BASE_DIR / "data"

    if split == "tuning":
        real_dir = data_dir / "tuning" / "real"
        ai_dir = data_dir / "tuning" / "ai"
    elif split == "test":
        real_dir = data_dir / "test" / "real"
        ai_dir = data_dir / "test" / "ai"
    elif split == "user":
        real_dir = data_dir / "my_images" / "real"
        ai_dir = data_dir / "my_images" / "ai"
    elif split == "messy":
        real_dir = data_dir / "messy_real"
        ai_dir = data_dir / "test" / "ai"
    else:
        raise ValueError(f"Unknown split '{split}'")

    print(f"\n==========================================")
    print(f"  RUNNING EVALUATION ON '{split.upper()}' SPLIT")
    print(f"==========================================")
    print(f"Real dir: {real_dir}")
    print(f"AI dir:   {ai_dir}")

    real_records = evaluate_directory(real_dir, ground_truth_label=0, threshold=threshold)
    ai_records = evaluate_directory(ai_dir, ground_truth_label=1, threshold=threshold)
    all_records = real_records + ai_records

    metrics = calculate_metrics(all_records)

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total Evaluated: {metrics['total']} images")
    print(f"Accuracy:        {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision:       {metrics['precision'] * 100:.2f}%")
    print(f"Recall:          {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score:        {metrics['f1']:.4f}")
    print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print(f"Mean AI Prob on Real Images: {metrics['mean_real_prob'] * 100:.1f}%")
    print(f"Mean AI Prob on AI Images:   {metrics['mean_ai_prob'] * 100:.1f}%")

    if metrics["errors"]:
        print(f"\n--- ERROR BREAKDOWN ({len(metrics['errors'])} misclassifications) ---")
        for err in metrics["errors"][:10]:
            gt_str = "AI" if err["ground_truth"] == 1 else "REAL"
            pred_str = "AI" if err["predicted_label"] == 1 else "REAL"
            print(f"  [{err['filename']}] True={gt_str}, Pred={pred_str} (Prob: {err['ai_probability']*100:.1f}%) -> {err['verdict']}")
            print(f"     Reason: {err['summary']}")
        if len(metrics["errors"]) > 10:
            print(f"     ...and {len(metrics['errors']) - 10} more errors.")
    else:
        print("\n--- ZERO MISCLASSIFICATIONS ---")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["tuning", "test", "user", "messy"], default="tuning")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    run_evaluation(split=args.split, threshold=args.threshold)
