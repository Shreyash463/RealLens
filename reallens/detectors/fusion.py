"""Signal Fusion engine for RealLens.

Combines provenance (C2PA, EXIF/PNG metadata), physical/compression forensics
(ELA, FFT frequency analysis), and deep learning vision models (ML ensemble,
optional Gemini layer) into a calibrated probability and transparent verdict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional


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
    """
    signals: Dict[str, Any] = {
        "metadata": metadata_res,
        "c2pa": c2pa_res,
        "ela": ela_res,
        "frequency": freq_res,
        "ml_ensemble": ml_res,
        "gemini": gemini_res or {"score": None, "confidence": "low", "evidence": ["Disabled"]},
    }

    # 1. Check for hard provenance overrides
    # A) Verified C2PA with AI action
    if c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.95 and c2pa_res.get("confidence") == "high":
        prob = 0.985
        verdict = "Likely AI"
        confidence = "high"
        summary = "Cryptographically authenticated C2PA Content Credentials manifest records generative AI origin."
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": verdict,
            "confidence": confidence,
            "summary_reason": summary,
            "signals": signals,
            "disagreement_flag": False,
            "honesty_note": "Estimate based on authenticated provenance. Missing metadata never proves authenticity.",
        }

    # B) Explicit AI generator metadata trace (Prompt, Model hash, Midjourney/SD headers)
    if metadata_res.get("score") is not None and metadata_res["score"] >= 0.95 and metadata_res.get("confidence") == "high":
        prob = 0.970
        verdict = "Likely AI"
        confidence = "high"
        summary = "Unambiguous AI generator parameters (prompt/sampler/model headers) detected in image metadata."
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": verdict,
            "confidence": confidence,
            "summary_reason": summary,
            "signals": signals,
            "disagreement_flag": False,
            "honesty_note": "Estimate based on file metadata. Missing metadata never proves authenticity.",
        }

    # C) Verified C2PA camera hardware signing
    if c2pa_res.get("score") is not None and c2pa_res["score"] <= 0.08 and c2pa_res.get("confidence") == "high":
        prob = 0.05
        verdict = "Likely real"
        confidence = "high"
        summary = "Cryptographic camera hardware signing (C2PA) confirms authentic capture origin."
        return {
            "ai_probability": prob,
            "ai_percentage": round(prob * 100, 1),
            "verdict": verdict,
            "confidence": confidence,
            "summary_reason": summary,
            "signals": signals,
            "disagreement_flag": False,
            "honesty_note": "Estimate based on cryptographic signature.",
        }

    # 2. Dynamic Weighted Fusion
    # Gather active signals and weights
    weights: Dict[str, float] = {}
    values: Dict[str, float] = {}

    # ML Ensemble (primary visual classifier)
    ml_score = ml_res.get("score")
    ml_disagreement = False
    per_model_scores = ml_res.get("per_model_scores", {})
    if len(per_model_scores) >= 2:
        valid_scores = [v for v in per_model_scores.values() if v is not None]
        if valid_scores and (max(valid_scores) - min(valid_scores)) > 0.45:
            ml_disagreement = True

    # Dynamic Weighted Fusion
    weights["ml"] = 0.70
    values["ml"] = float(ml_score) if ml_score is not None else 0.50

    # Metadata (only if camera EXIF was found; missing is neutral = None)
    meta_score = metadata_res.get("score")
    if meta_score is not None:
        weights["metadata"] = 0.15
        values["metadata"] = float(meta_score)

    # ELA (supporting forensic signal)
    ela_score = ela_res.get("score")
    if ela_score is not None:
        weights["ela"] = 0.15
        values["ela"] = float(ela_score)

    # Frequency analysis (supporting forensic signal)
    freq_score = freq_res.get("score")
    if freq_score is not None:
        weights["frequency"] = 0.15
        values["frequency"] = float(freq_score)

    # Optional Gemini layer
    if gemini_res and gemini_res.get("score") is not None:
        weights["gemini"] = 0.15
        values["gemini"] = float(gemini_res["score"])

    # Compute weighted sum
    total_weight = sum(weights.values())
    if total_weight > 0:
        fused_prob = sum(values[k] * weights[k] for k in weights) / total_weight
    else:
        fused_prob = 0.50  # Default neutral

    # Clamp probability to honest boundaries (avoid 0.00 or 1.00 claims without cryptographic proof)
    fused_prob = max(0.04, min(0.96, fused_prob))

    # 3. Determine Verdict Bands
    # Bands: Likely AI / Uncertain leaning AI / Uncertain leaning real / Likely real
    # If ML models disagree strongly, enforce Uncertainty band!
    if ml_disagreement:
        if fused_prob >= 0.50:
            verdict = "Uncertain leaning AI"
        else:
            verdict = "Uncertain leaning real"
        confidence = "low"
        summary = (
            f"Machine learning detectors disagree substantially (spread: "
            f"{(max(valid_scores)-min(valid_scores)):.2f}). The system recommends manual inspection."
        )
    else:
        if fused_prob >= 0.70:
            verdict = "Likely AI"
            confidence = "high" if fused_prob >= 0.85 else "medium"
            summary = "Consistent AI visual characteristics and artifact patterns identified across detectors."
        elif fused_prob >= 0.50:
            verdict = "Uncertain leaning AI"
            confidence = "medium" if fused_prob >= 0.60 else "low"
            summary = "Subtle generative cues observed, but insufficient confidence for definitive AI determination."
        elif fused_prob >= 0.30:
            verdict = "Uncertain leaning real"
            confidence = "medium" if fused_prob <= 0.40 else "low"
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
        "honesty_note": "Estimate only, not proof. Missing metadata never proves an image is real.",
    }
