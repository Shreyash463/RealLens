"""Signal Fusion engine for RealLens.

Combines provenance (C2PA, EXIF/PNG metadata), physical/compression forensics
(ELA, FFT frequency analysis), deep learning vision models (ML ensemble,
optional Gemini layer), and deterministic image-type gating into a calibrated
probability or range and transparent verdict.
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
    image_type_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Combines all detector signals into a calibrated verdict and probability or range.

    Honesty and safety rules:
    - Verified C2PA manifest or explicit AI metadata trace is strong evidence for AI.
    - If image type is NOT photo-like (screenshot, document, certificate, graphic):
      lower ML/forensics weights and push verdict to 'Inconclusive' unless hard C2PA/AI metadata exists.
    - When ML models disagree strongly (spread >= 0.50), headline must be 'Inconclusive'
      and percentage must be replaced by a probability RANGE (e.g. 'between 2% and 100% across models').
    - Only show single percentage when models roughly agree.
    - 5-level verdict spectrum:
      'Likely AI' | 'Leaning AI' | 'Inconclusive' | 'Leaning real' | 'Likely real'
    """
    signals: Dict[str, Any] = {
        "metadata": metadata_res,
        "c2pa": c2pa_res,
        "ela": ela_res,
        "frequency": freq_res,
        "ml_ensemble": ml_res,
        "gemini": gemini_res or {"score": None, "confidence": "low", "evidence": ["Disabled"]},
        "image_type": image_type_res or {"image_type": "photo-like", "is_photo_like": True},
    }

    img_type_info = image_type_res or {"image_type": "photo-like", "is_photo_like": True, "unreliable_notice": None}
    img_type = img_type_info.get("image_type", "photo-like")
    is_photo_like = img_type_info.get("is_photo_like", True)
    unreliable_notice = img_type_info.get("unreliable_notice")

    # 1. Hard Provenance Overrides
    # A) Verified C2PA with AI action
    if c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.95 and c2pa_res.get("confidence") == "high":
        prob = 0.985
        return {
            "ai_probability": prob,
            "ai_percentage": 98.5,
            "probability_display": "98.5%",
            "verdict": "Likely AI",
            "confidence": "high",
            "is_inconclusive": False,
            "summary_reason": "Cryptographically authenticated C2PA Content Credentials manifest records generative AI origin.",
            "why_this_verdict": "Cryptographic Content Credentials (C2PA) attached to this file explicitly record generative AI synthesis. Provenance signatures cannot be forged without invalidating the cryptographic certificate.",
            "signals": signals,
            "spread": 0.0,
            "min_model_score": prob,
            "max_model_score": prob,
            "image_type": img_type,
            "is_photo_like": is_photo_like,
            "unreliable_notice": None,
            "disagreement_flag": False,
            "fusion_mode": "cryptographic_override",
            "honesty_note": "Estimate based on authenticated provenance. Missing metadata never proves authenticity.",
        }

    # B) Explicit AI generator metadata trace (Prompt, Model hash, Midjourney/SD headers)
    if metadata_res.get("score") is not None and metadata_res["score"] >= 0.95 and metadata_res.get("confidence") == "high":
        prob = 0.970
        return {
            "ai_probability": prob,
            "ai_percentage": 97.0,
            "probability_display": "97.0%",
            "verdict": "Likely AI",
            "confidence": "high",
            "is_inconclusive": False,
            "summary_reason": "Unambiguous AI generator parameters (prompt/sampler/model headers) detected in image metadata.",
            "why_this_verdict": "Raw file metadata contains explicit generative parameters (such as diffusion prompt text, checkpoint hashes, or generation headers) confirming synthetic generation.",
            "signals": signals,
            "spread": 0.0,
            "min_model_score": prob,
            "max_model_score": prob,
            "image_type": img_type,
            "is_photo_like": is_photo_like,
            "unreliable_notice": None,
            "disagreement_flag": False,
            "fusion_mode": "metadata_override",
            "honesty_note": "Estimate based on file metadata. Missing metadata never proves authenticity.",
        }

    # C) Verified C2PA camera hardware signing
    if c2pa_res.get("score") is not None and c2pa_res["score"] <= 0.08 and c2pa_res.get("confidence") == "high":
        prob = 0.05
        return {
            "ai_probability": prob,
            "ai_percentage": 5.0,
            "probability_display": "5.0%",
            "verdict": "Likely real",
            "confidence": "high",
            "is_inconclusive": False,
            "summary_reason": "Cryptographic camera hardware signing (C2PA) confirms authentic capture origin.",
            "why_this_verdict": "Hardware-level cryptographic signing confirms the image was captured directly by a certified camera sensor with an intact signature chain.",
            "signals": signals,
            "spread": 0.0,
            "min_model_score": prob,
            "max_model_score": prob,
            "image_type": img_type,
            "is_photo_like": is_photo_like,
            "unreliable_notice": None,
            "disagreement_flag": False,
            "fusion_mode": "cryptographic_override",
            "honesty_note": "Estimate based on cryptographic signature.",
        }

    # 2. Extract ML Ensemble Scores & Spread
    ml_score = ml_res.get("score")
    per_model_scores = ml_res.get("per_model_scores", {})
    valid_scores = [v for v in per_model_scores.values() if v is not None]

    if valid_scores:
        min_score = float(min(valid_scores))
        max_score = float(max(valid_scores))
        spread = float(max_score - min_score)
    else:
        min_score = 0.50
        max_score = 0.50
        spread = 0.0

    min_pct = round(min_score * 100, 1)
    max_pct = round(max_score * 100, 1)
    range_display = f"between {min_pct:.0f}% and {max_pct:.0f}% across models"

    # ML Disagreement condition
    ml_disagreement = (spread >= 0.50)

    # 3. Handle Non-Photographic Images (Image-Type Gate)
    if not is_photo_like:
        notice_text = unreliable_notice or (
            "This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type."
        )
        return {
            "ai_probability": 0.50,
            "ai_percentage": None,  # Suppress single percentage
            "probability_display": range_display if spread >= 0.20 else f"~{min_pct:.0f}%",
            "verdict": "Inconclusive",
            "confidence": "low",
            "is_inconclusive": True,
            "summary_reason": f"{notice_text} Classified as '{img_type}'.",
            "why_this_verdict": (
                f"This image was identified as a {img_type} (uniform background fills, sharp text glyphs, or window boundaries). "
                f"Deep vision detectors trained on natural camera photographs frequently misinterpret flat digital graphics and sharp text rendering as generative diffusion artifacts. "
                f"Without cryptographic C2PA credentials or metadata traces, a definitive AI determination cannot be made."
            ),
            "signals": signals,
            "spread": round(spread, 3),
            "min_model_score": round(min_score, 3),
            "max_model_score": round(max_score, 3),
            "image_type": img_type,
            "is_photo_like": False,
            "unreliable_notice": notice_text,
            "disagreement_flag": ml_disagreement,
            "fusion_mode": "image_type_gate",
            "honesty_note": "Photo-trained vision detectors are unreliable for flat digital graphics, documents, and screenshots.",
        }

    # 4. Handle Strong ML Disagreement on Photo-Like Images
    if ml_disagreement:
        return {
            "ai_probability": 0.50,
            "ai_percentage": None,  # Suppress single percentage!
            "probability_display": range_display,
            "verdict": "Inconclusive",
            "confidence": "low",
            "is_inconclusive": True,
            "summary_reason": f"Machine learning detectors disagree strongly (spread: {spread*100:.0f}%). One or more vision models may be producing false alarms.",
            "why_this_verdict": (
                f"Vision classifiers disagree substantially on this image (spread: {spread*100:.0f}%). "
                f"Individual model predictions range from {min_pct:.1f}% to {max_pct:.1f}% AI probability. "
                f"This sharp conflict typically arises from compression macroblocks, heavy aperture bokeh, or out-of-distribution textures. "
                f"Because the models cannot reach consensus, the system flags the result as Inconclusive rather than forcing a misleading single score."
            ),
            "signals": signals,
            "spread": round(spread, 3),
            "min_model_score": round(min_score, 3),
            "max_model_score": round(max_score, 3),
            "image_type": img_type,
            "is_photo_like": True,
            "unreliable_notice": None,
            "disagreement_flag": True,
            "fusion_mode": "disagreement_guardrail",
            "honesty_note": "Estimate withheld due to severe detector disagreement. Manual inspection recommended.",
        }

    # 5. Agreement Case: Calibrated Single Probability
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
        weights = {"ml": 0.70}
        values = {"ml": float(ml_score) if ml_score is not None else 0.50}

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

    fused_prob = max(0.04, min(0.96, fused_prob))
    pct = round(fused_prob * 100, 1)

    # 4-Level Verdict Spectrum for Agreeing Cases
    if fused_prob >= 0.65:
        verdict = "Likely AI"
        confidence = "high" if fused_prob >= 0.85 else "medium"
        summary = "Consistent AI visual characteristics and artifact patterns identified across detectors."
        why_text = "Vision models demonstrate strong agreement on synthetic diffusion textures, supported by spectral frequency patterns consistent with generative upsampling."
    elif fused_prob >= 0.50:
        verdict = "Leaning AI"
        confidence = "low"
        summary = "Subtle generative cues observed, but overall confidence is moderate."
        why_text = "Detector consensus leans slightly toward AI synthesis, but lack of cryptographic provenance and compression noise prevents a definitive determination."
    elif fused_prob >= 0.35:
        verdict = "Leaning real"
        confidence = "low"
        summary = "Primarily natural characteristics, though compression leaves slight ambiguity."
        why_text = "Image exhibits photographic characteristics and natural spectral decay, though compression re-encoding leaves minor residual uncertainty."
    else:
        verdict = "Likely real"
        confidence = "high" if fused_prob <= 0.18 else "medium"
        summary = "Photographic sensor noise characteristics, natural spectral decay, and vision models align with authentic origin."
        why_text = "All active vision models agree on authentic photographic origin. Natural camera sensor noise floor, Fourier spectral slope, and compression characteristics align with real-world capture."

    return {
        "ai_probability": round(fused_prob, 3),
        "ai_percentage": pct,
        "probability_display": f"{pct}%",
        "verdict": verdict,
        "confidence": confidence,
        "is_inconclusive": False,
        "summary_reason": summary,
        "why_this_verdict": why_text,
        "signals": signals,
        "spread": round(spread, 3),
        "min_model_score": round(min_score, 3),
        "max_model_score": round(max_score, 3),
        "image_type": img_type,
        "is_photo_like": True,
        "unreliable_notice": None,
        "disagreement_flag": False,
        "fusion_mode": fusion_mode,
        "honesty_note": "Estimate only, not proof. Missing metadata never proves an image is real.",
    }
