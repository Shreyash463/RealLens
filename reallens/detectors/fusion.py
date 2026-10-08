"""Signal Fusion engine for RealLens.

Combines provenance (C2PA, EXIF/PNG metadata), physical/compression forensics
(ELA, FFT frequency analysis), and deep learning vision models (ML ensemble,
optional Gemini layer) into a calibrated probability and transparent verdict.

Supports both Learned Calibrated Stacker (primary) and Hand-Weighted Fusion (fallback),
along with a designated 'Uncertain' zone guardrail.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import joblib
import numpy as np

logger = logging.getLogger(__name__)

_stacker_pkg: Any = None
_stacker_checked: bool = False


def get_learned_stacker():
    global _stacker_pkg, _stacker_checked
    if not _stacker_checked:
        stacker_path = Path(__file__).resolve().parent.parent.parent / "models" / "learned_fusion_stacker.joblib"
        if stacker_path.exists():
            try:
                _stacker_pkg = joblib.load(stacker_path)
                logger.info("Loaded learned fusion stacker from models/learned_fusion_stacker.joblib")
            except Exception as e:
                logger.warning(f"Could not load learned fusion stacker: {e}")
                _stacker_pkg = None
        else:
            _stacker_pkg = None
        _stacker_checked = True
    return _stacker_pkg


def fuse_signals(
    metadata_res: Dict[str, Any],
    c2pa_res: Dict[str, Any],
    ela_res: Dict[str, Any],
    freq_res: Dict[str, Any],
    ml_res: Dict[str, Any],
    gemini_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combines all detector signals into one calibrated probability and verdict.

    Honesty and safety rules:
    - Verified C2PA manifest or explicit AI metadata trace is strong evidence for AI.
    - Camera hardware EXIF is a mild signal toward real, never absolute proof.
    - Missing metadata is neutral, not suspicious.
    - ELA and FFT frequency analysis are supporting signals with modest weights.
    - High ML ensemble disagreement forces the verdict into 'Uncertain'.
    - Output is always a probability estimate with evidence, never 100% certainty.

    Returns:
        dict:
            ai_probability: float (0.0 to 1.0)
            ai_percentage: float (0.0 to 100.0)
            verdict: "Likely AI" | "Uncertain leaning AI" | "Uncertain leaning real" | "Likely real"
            confidence: "low" | "medium" | "high"
            summary_reason: str
            signals: dict of all individual detector outputs
            disagreement_flag: bool
            in_uncertain_zone: bool
            fusion_mode: "learned_stacker" | "hand_weighted_fallback"
    """
    signals: Dict[str, Any] = {
        "metadata": metadata_res,
        "c2pa": c2pa_res,
        "ela": ela_res,
        "frequency": freq_res,
        "ml_ensemble": ml_res,
        "gemini": gemini_res or {"score": None, "confidence": "low", "evidence": ["Disabled"]},
    }

    # 1. Hard Provenance Overrides
    # A) Verified C2PA with AI action
    if c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.95 and c2pa_res.get("confidence") == "high":
        prob = 0.985
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": "Likely AI",
            "confidence": "high",
            "summary_reason": "Cryptographically authenticated C2PA Content Credentials manifest records generative AI origin.",
            "signals": signals,
            "disagreement_flag": False,
            "in_uncertain_zone": False,
            "fusion_mode": "cryptographic_override",
            "honesty_note": "Estimate based on authenticated provenance. Missing metadata never proves authenticity.",
        }

    # B) Explicit AI generator metadata trace (Prompt, Model hash, Midjourney/SD headers)
    if metadata_res.get("score") is not None and metadata_res["score"] >= 0.95 and metadata_res.get("confidence") == "high":
        prob = 0.970
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": "Likely AI",
            "confidence": "high",
            "summary_reason": "Unambiguous AI generator parameters (prompt/sampler/model headers) detected in image metadata.",
            "signals": signals,
            "disagreement_flag": False,
            "in_uncertain_zone": False,
            "fusion_mode": "metadata_override",
            "honesty_note": "Estimate based on file metadata. Missing metadata never proves authenticity.",
        }

    # C) Verified C2PA camera hardware signing
    if c2pa_res.get("score") is not None and c2pa_res["score"] <= 0.08 and c2pa_res.get("confidence") == "high":
        prob = 0.05
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": "Likely real",
            "confidence": "high",
            "summary_reason": "Cryptographic camera hardware signing (C2PA) confirms authentic capture origin.",
            "signals": signals,
            "disagreement_flag": False,
            "in_uncertain_zone": False,
            "fusion_mode": "cryptographic_override",
            "honesty_note": "Estimate based on cryptographic signature.",
        }

    # 2. Check for ML Disagreement
    ml_score = ml_res.get("score")
    ml_disagreement = False
    per_model_scores = ml_res.get("per_model_scores", {})
    valid_scores = [v for v in per_model_scores.values() if v is not None]
    spread = float(max(valid_scores) - min(valid_scores)) if len(valid_scores) >= 2 else 0.0
    if len(valid_scores) >= 2 and spread > 0.45:
        ml_disagreement = True

    # 3. Fusion Prediction (Learned Stacker vs Hand-Weighted Fallback)
    stacker = get_learned_stacker()
    fused_prob = None
    fusion_mode = "hand_weighted_fallback"

    if stacker is not None:
        try:
            s_dima = per_model_scores.get("dima806/ai_vs_real_image_detection", 0.50) or 0.50
            s_org = per_model_scores.get("Organika/sdxl-detector", 0.50) or 0.50
            s_at = per_model_scores.get("Ateeqq/ai-vs-human-image-detector", 0.50) or 0.50
            s_probe = per_model_scores.get("generalizing_dinov2_probe", 0.50) or 0.50
            s_med = float(ml_score) if ml_score is not None else 0.50
            s_spread = spread
            ela_s = ela_res.get("score", 0.50) if ela_res.get("score") is not None else 0.50
            freq_s = freq_res.get("score", 0.50) if freq_res.get("score") is not None else 0.50
            meta_score = metadata_res.get("score")
            has_meta_cam = 1.0 if (meta_score is not None and meta_score <= 0.20) else 0.0
            c2pa_ai = 1.0 if (c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.80) else 0.0

            feat = np.array([[
                s_dima, s_org, s_at, s_probe,
                s_med, s_spread, ela_s, freq_s,
                has_meta_cam, c2pa_ai,
            ]], dtype=np.float32)

            feat_scaled = stacker["scaler"].transform(feat)
            prob_raw = float(stacker["model"].predict_proba(feat_scaled)[0, 1])
            fused_prob = prob_raw
            fusion_mode = "learned_stacker"
        except Exception as exc:
            logger.warning(f"Learned stacker inference failed, falling back to hand-weighted: {exc}")
            fused_prob = None

    if fused_prob is None:
        # Hand-Weighted Fallback
        weights: Dict[str, float] = {"ml": 0.70}
        values: Dict[str, float] = {"ml": float(ml_score) if ml_score is not None else 0.50}

        meta_score = metadata_res.get("score")
        if meta_score is not None:
            weights["metadata"] = 0.15
            values["metadata"] = float(meta_score)

        ela_score = ela_res.get("score")
        if ela_score is not None:
            weights["ela"] = 0.15
            values["ela"] = float(ela_score)

        freq_score = freq_res.get("score")
        if freq_score is not None:
            weights["frequency"] = 0.15
            values["frequency"] = float(freq_score)

        if gemini_res and gemini_res.get("score") is not None:
            weights["gemini"] = 0.15
            values["gemini"] = float(gemini_res["score"])

        total_weight = sum(weights.values())
        fused_prob = sum(values[k] * weights[k] for k in weights) / total_weight if total_weight > 0 else 0.50
        fusion_mode = "hand_weighted_fallback"

    # Clamp probability to honest boundaries (avoid 0.00 or 1.00 claims without cryptographic proof)
    fused_prob = max(0.04, min(0.96, fused_prob))

    # 4. Determine Verdict Bands & Uncertain Zone
    in_uncertain_zone = (0.35 <= fused_prob <= 0.65) or ml_disagreement

    if ml_disagreement:
        if fused_prob >= 0.50:
            verdict = "Uncertain leaning AI"
        else:
            verdict = "Uncertain leaning real"
        confidence = "low"
        summary = (
            f"Machine learning detectors disagree substantially (spread: {spread:.2f}). "
            f"The system recommends manual inspection."
        )
    else:
        if fused_prob >= 0.65:
            verdict = "Likely AI"
            confidence = "high" if fused_prob >= 0.85 else "medium"
            summary = "Consistent AI visual characteristics and artifact patterns identified across detectors."
        elif fused_prob >= 0.50:
            verdict = "Uncertain leaning AI"
            confidence = "low"
            summary = "Subtle generative cues observed, but insufficient confidence for definitive AI determination."
        elif fused_prob >= 0.35:
            verdict = "Uncertain leaning real"
            confidence = "low"
            summary = "Primarily natural characteristics, though compression or lack of hardware provenance leaves ambiguity."
        else:
            verdict = "Likely real"
            confidence = "high" if fused_prob <= 0.18 else "medium"
            summary = "Photographic noise characteristics, natural spectral decay, and vision models align with authentic origin."

    return {
        "ai_probability": round(fused_prob, 3),
        "ai_percentage": round(fused_prob * 100, 1),
        "verdict": verdict,
        "confidence": confidence,
        "summary_reason": summary,
        "signals": signals,
        "disagreement_flag": ml_disagreement,
        "in_uncertain_zone": in_uncertain_zone,
        "fusion_mode": fusion_mode,
        "honesty_note": "Estimate only, not proof. Missing metadata never proves an image is real.",
    }
