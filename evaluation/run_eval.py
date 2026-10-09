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
            verdict = res.get("verdict", "")
            is_inconclusive = res.get("is_inconclusive", False) or (verdict == "Inconclusive")

            # Prediction mapping:
            # - Definite/Leaning AI -> 1
            # - Definite/Leaning Real -> 0
            # - Inconclusive / Withheld -> -1
            v_lower = verdict.strip().lower()
            if v_lower in ("likely ai", "leaning ai"):
                pred_label = 1
            elif v_lower in ("likely real", "leaning real"):
                pred_label = 0
            else:
                pred_label = -1

            results.append({
                "path": str(p),
                "filename": p.name,
                "ground_truth": ground_truth_label,
                "predicted_label": pred_label,
                "ai_probability": prob,
                "verdict": verdict,
                "is_inconclusive": is_inconclusive,
                "confidence": res.get("confidence"),
                "summary": res.get("summary_reason"),
                "signals": res.get("signals"),
                "in_uncertain_zone": is_inconclusive or res.get("in_uncertain_zone", False),
                "is_correct": (pred_label == ground_truth_label),
            })
        except Exception as exc:
            results.append({
                "path": str(p),
                "filename": p.name,
                "ground_truth": ground_truth_label,
                "predicted_label": -2,
                "ai_probability": 0.5,
                "verdict": "Error",
                "is_inconclusive": True,
                "confidence": "low",
                "summary": f"Evaluation error: {exc}",
                "signals": {},
                "in_uncertain_zone": True,
                "is_correct": False,
            })
    return results


def calculate_metrics(eval_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Computes honest classification metrics including inconclusive routing."""
    total = len(eval_records)
    if total == 0:
        return {"total": 0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "fp_rate": 0.0}

    total_real = sum(1 for r in eval_records if r["ground_truth"] == 0)
    total_ai = sum(1 for r in eval_records if r["ground_truth"] == 1)

    inconclusive_count = sum(1 for r in eval_records if r.get("is_inconclusive", False) or r["predicted_label"] == -1)
    inconclusive_pct = (inconclusive_count / total * 100) if total > 0 else 0.0

    # False positives: Ground truth is Real (0), but predicted as AI (1)
    fp = sum(1 for r in eval_records if r["ground_truth"] == 0 and r["predicted_label"] == 1)
    fp_rate = (fp / total_real * 100) if total_real > 0 else 0.0

    # False negatives: Ground truth is AI (1), but predicted as Real (0)
    fn = sum(1 for r in eval_records if r["ground_truth"] == 1 and r["predicted_label"] == 0)
    fn_rate = (fn / total_ai * 100) if total_ai > 0 else 0.0

    # Decided records (where system issued a definitive AI or Real verdict)
    decided_records = [r for r in eval_records if r["predicted_label"] in (0, 1)]
    tp = sum(1 for r in decided_records if r["ground_truth"] == 1 and r["predicted_label"] == 1)
    tn = sum(1 for r in decided_records if r["ground_truth"] == 0 and r["predicted_label"] == 0)

    decided_total = len(decided_records)
    decided_acc = (tp + tn) / decided_total if decided_total > 0 else 0.0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

    real_probs = [r["ai_probability"] for r in eval_records if r["ground_truth"] == 0]
    ai_probs = [r["ai_probability"] for r in eval_records if r["ground_truth"] == 1]

    mean_real_prob = float(sum(real_probs) / len(real_probs)) if real_probs else 0.0
    mean_ai_prob = float(sum(ai_probs) / len(ai_probs)) if ai_probs else 0.0

    errors = [r for r in decided_records if not r["is_correct"]]

    return {
        "total": total,
        "total_real": total_real,
        "total_ai": total_ai,
        "decided_total": decided_total,
        "decided_accuracy": round(decided_acc, 4),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "fp_rate": round(fp_rate, 2),
        "fn_rate": round(fn_rate, 2),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "mean_real_prob": round(mean_real_prob, 4),
        "mean_ai_prob": round(mean_ai_prob, 4),
        "inconclusive_count": inconclusive_count,
        "inconclusive_pct": round(inconclusive_pct, 2),
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
    elif split == "hard_tuning":
        real_dir = data_dir / "hard_tuning" / "real"
        ai_dir = data_dir / "hard_tuning" / "ai"
    elif split == "hard_test":
        real_dir = data_dir / "hard_test" / "real"
        ai_dir = data_dir / "hard_test" / "ai"
    elif split == "user":
        real_dir = data_dir / "my_images" / "real"
        ai_dir = data_dir / "my_images" / "ai"
    elif split == "graphics":
        real_dir = data_dir / "graphics_test"
        ai_dir = None
    elif split == "messy":
        real_dir = data_dir / "messy_real"
        ai_dir = data_dir / "test" / "ai"
    elif split == "whatsapp":
        real_dir = data_dir / "my_images" / "real_whatsapp"
        ai_dir = None
    else:
        raise ValueError(f"Unknown split '{split}'")

    print(f"\n==========================================")
    print(f"  RUNNING EVALUATION ON '{split.upper()}' SPLIT")
    print(f"==========================================")
    print(f"Real dir: {real_dir}")
    print(f"AI dir:   {ai_dir}")

    real_records = evaluate_directory(real_dir, ground_truth_label=0, threshold=threshold)
    user_graphics = data_dir / "my_images" / "real_graphics"
    if split == "graphics" and user_graphics.exists():
        user_gfx_records = evaluate_directory(user_graphics, ground_truth_label=0, threshold=threshold)
        if user_gfx_records:
            print(f"Included {len(user_gfx_records)} custom user images from {user_graphics}")
            real_records.extend(user_gfx_records)

    ai_records = evaluate_directory(ai_dir, ground_truth_label=1, threshold=threshold) if ai_dir and ai_dir.exists() else []
    all_records = real_records + ai_records

    metrics = calculate_metrics(all_records)

    print("\n--- PERFORMANCE METRICS ---")
    print(f"Total Evaluated:        {metrics['total']} images (Real: {metrics['total_real']}, AI: {metrics['total_ai']})")
    print(f"Inconclusive / Withheld:{metrics['inconclusive_count']}/{metrics['total']} ({metrics['inconclusive_pct']}%)")
    print(f"Decided Verdicts:       {metrics['decided_total']}/{metrics['total']}")
    if metrics['decided_total'] > 0:
        print(f"Decided Accuracy:       {metrics['decided_accuracy'] * 100:.2f}%")
        print(f"Precision:              {metrics['precision'] * 100:.2f}%")
        print(f"Recall:                 {metrics['recall'] * 100:.2f}%")
        print(f"F1-Score:               {metrics['f1']:.4f}")
    print(f"False Positive Rate:    {metrics['fp_rate']:.1f}% (FP={metrics['fp']}/{metrics['total_real']})")
    print(f"False Negative Rate:    {metrics['fn_rate']:.1f}% (FN={metrics['fn']}/{metrics['total_ai']})")
    print(f"Mean AI Prob on Real:   {metrics['mean_real_prob'] * 100:.1f}%")
    print(f"Mean AI Prob on AI:     {metrics['mean_ai_prob'] * 100:.1f}%")
    print(f"Inconclusive / Withheld Landings: {metrics['inconclusive_count']}/{metrics['total']} ({metrics['inconclusive_pct']:.1f}%)")

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
    parser.add_argument("--split", choices=["tuning", "test", "hard_tuning", "hard_test", "user", "messy", "graphics", "whatsapp"], default="tuning")
    parser.add_argument("--threshold", type=float, default=0.50)
    args = parser.parse_args()

    run_evaluation(split=args.split, threshold=args.threshold)
