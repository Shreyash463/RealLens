"""Signal Fusion engine for RealLens.

Combines provenance (C2PA, EXIF/PNG metadata), physical/compression forensics
(ELA, FFT frequency analysis), deep learning vision models (ML ensemble,
optional Gemini layer), and learned image-type gate into a calibrated
probability or range and transparent verdict.

Honesty and safety rules:
- C2PA manifest or explicit AI metadata trace provides high-confidence provenance.
- Soft image-type gating: scales weights of ML and forensics continuously by `photo_probability`.
- When models disagree (spread >= 0.45) or image type is non-photo (photo_prob < 0.35)
  or compression creates strong ambiguity, verdict is honestly routed to 'Inconclusive'.
- Headline for Inconclusive is always 'Inconclusive' with a short reason, and below it
  a small range indicator plus individual model scores.
- Every result contains:
  1. Evidence pointing to AI
  2. Evidence pointing to real
  3. Missing or unreliable evidence
  4. What to do next practical guidance
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
    """Combines all detector signals into an honest calibrated verdict."""
    img_type_info = image_type_res or {
        "photo_probability": 1.0,
        "image_type": "photo-like",
        "is_photo_like": True,
        "unreliable_notice": None,
        "notice_severity": "none",
        "is_heavily_recompressed": False,
        "block_ratio": 1.0,
    }

    photo_prob = float(img_type_info.get("photo_probability", 1.0))
    img_type = img_type_info.get("image_type", "photo-like")
    is_photo_like = img_type_info.get("is_photo_like", photo_prob >= 0.35)
    unreliable_notice = img_type_info.get("unreliable_notice")
    notice_severity = img_type_info.get("notice_severity", "none")
    is_heavily_recompressed = img_type_info.get("is_heavily_recompressed", False)
    comp_notice = img_type_info.get("compression_notice")

    # 1. Extract and standardize ML ensemble scores
    ml_score = ml_res.get("score")
    per_model_scores: Dict[str, Optional[float]] = {}
    if "per_model_scores" in ml_res and isinstance(ml_res["per_model_scores"], dict):
        per_model_scores = dict(ml_res["per_model_scores"])
    elif "models" in ml_res and isinstance(ml_res["models"], dict):
        for k, v in ml_res["models"].items():
            per_model_scores[k] = v.get("score") if isinstance(v, dict) else v

    valid_scores = [v for v in per_model_scores.values() if v is not None]
    if valid_scores:
        min_score = float(min(valid_scores))
        max_score = float(max(valid_scores))
        spread = float(max_score - min_score)
    else:
        min_score = float(ml_score) if ml_score is not None else 0.50
        max_score = float(ml_score) if ml_score is not None else 0.50
        spread = 0.0

    min_pct = round(min_score * 100, 1)
    max_pct = round(max_score * 100, 1)
    range_indicator = f"Models range: {min_pct:.0f}% – {max_pct:.0f}%" if spread >= 0.15 else None

    # Continuous reliability scaling by photo_probability
    photo_scale = max(0.10, min(1.0, photo_prob / 0.65))

    # Calculate model weights in final decision
    num_models = len(valid_scores) if valid_scores else 1
    base_ml_weight = 0.70 * photo_scale
    per_model_weights = {
        m: round(base_ml_weight / num_models, 3) for m in per_model_scores.keys()
    }

    # Ensure signals dictionary contains everything
    signals: Dict[str, Any] = {
        "metadata": metadata_res,
        "c2pa": c2pa_res,
        "ela": ela_res,
        "frequency": freq_res,
        "ml_ensemble": ml_res,
        "gemini": gemini_res or {"score": None, "confidence": "low", "evidence": ["Disabled"]},
        "image_type": img_type_info,
    }
    # Backward-compat helpers
    signals["ml"] = {
        "score": ml_score,
        "spread": spread,
        "models": {k: {"score": v} for k, v in per_model_scores.items()},
    }

    # 2. Build the three evidence lists
    evidence_ai: List[str] = []
    evidence_real: List[str] = []
    evidence_unreliable: List[str] = []

    # A) Check provenance
    if c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.90:
        evidence_ai.append("Cryptographic C2PA manifest explicitly records generative AI synthesis.")
    elif c2pa_res.get("score") is not None and c2pa_res["score"] <= 0.10:
        evidence_real.append("Cryptographic C2PA hardware signing confirms authentic camera capture.")
    else:
        evidence_unreliable.append("No cryptographic C2PA Content Credentials manifest found (standard for web downloads and chat apps).")

    # B) Check metadata
    if metadata_res.get("score") is not None and metadata_res["score"] >= 0.90:
        evidence_ai.append(f"AI generator metadata detected: {metadata_res.get('ai_tool_detected', 'Generative tool parameters')}.")
    elif metadata_res.get("camera_make") or metadata_res.get("camera_model"):
        cam = f"{metadata_res.get('camera_make', '')} {metadata_res.get('camera_model', '')}".strip()
        evidence_real.append(f"Authentic camera hardware EXIF tags present ({cam}).")
    else:
        evidence_unreliable.append("No camera hardware EXIF metadata found (neutral indicator; stripped by social media / chat apps).")

    # C) Check vision models
    for m_name, m_sc in per_model_scores.items():
        if m_sc is None:
            continue
        short = m_name.split("/")[-1]
        if m_sc >= 0.75:
            evidence_ai.append(f"{short}: {round(m_sc*100):.0f}% synthetic confidence.")
        elif m_sc <= 0.25:
            evidence_real.append(f"{short}: {round((1-m_sc)*100):.0f}% natural photographic confidence.")
    if ml_score is None:
        evidence_unreliable.append("Deep vision models are currently offline or blocked by Windows Application Control.")

    # D) Check physical forensics (ELA & FFT)
    ela_app = ela_res.get("applicable", False)
    if ela_app and ela_res.get("score") is not None:
        if ela_res["score"] >= 0.65:
            evidence_ai.append(f"Elevated ELA mean error ({ela_res['metrics'].get('mean_error', 0):.1f}) deviates from standard camera JPEG equilibrium.")
        elif ela_res["score"] <= 0.35:
            evidence_real.append(f"Low ELA error ({ela_res['metrics'].get('mean_error', 0):.1f}) matches natural JPEG photographic equilibrium.")
    else:
        if not is_photo_like:
            evidence_unreliable.append("ELA is not applicable on flat graphics, screenshots, or documents.")
        elif is_heavily_recompressed:
            evidence_unreliable.append("ELA excluded from fusion: severe compression macroblocks mask subtle error differentials.")

    freq_app = freq_res.get("applicable", False)
    if freq_app and freq_res.get("score") is not None:
        if freq_res["score"] >= 0.65:
            evidence_ai.append(f"2D FFT detected {freq_res['metrics'].get('peak_count', 0)} periodic peaks characteristic of deconvolution upsampling.")
        elif freq_res["score"] <= 0.35:
            evidence_real.append(f"Fourier decay slope ({freq_res['metrics'].get('decay_slope', 0):.2f}) exhibits natural optical dispersion without lattice peaks.")
    else:
        if not is_photo_like:
            evidence_unreliable.append("FFT spectral peak analysis is not applicable on digital graphics or text documents.")

    # E) Check image-type gate & compression
    if photo_prob < 0.35:
        evidence_unreliable.append(f"Non-photographic image type (photo probability: {photo_prob*100:.1f}%). Photo-trained detectors are less reliable.")
    elif photo_prob < 0.65:
        evidence_unreliable.append(f"Borderline image type (photo probability: {photo_prob*100:.1f}%). Reliability is lowered.")

    if is_heavily_recompressed:
        evidence_unreliable.append(f"Heavy lossy compression detected (DCT blockiness: {img_type_info.get('block_ratio', 0):.2f}). Compression artifacts destroy fine camera grain.")

    if spread >= 0.45:
        evidence_unreliable.append(f"Severe disagreement among vision models (spread: {spread*100:.0f}%). One or more detectors may be misclassifying.")

    # 3. Build Actionable "What to do next" Guidance
    what_to_do_next = [
        {
            "label": "Request Original File",
            "description": "Ask the sender to share the original uncompressed photo as a 'Document' or uncompressed file (messaging apps strip EXIF and recompress).",
            "action_type": "ask_source",
        },
        {
            "label": "Reverse Image Search",
            "description": "Check if this image was published elsewhere earlier or belongs to an existing public profile.",
            "action_type": "reverse_search",
            "links": [
                {"title": "Search with Google Lens", "url": "https://lens.google.com/"},
                {"title": "Search with TinEye", "url": "https://tineye.com/"},
            ],
        },
        {
            "label": "Verify Sender & Context",
            "description": "Never send money, credentials, or personal information based solely on an image received online or via chat.",
            "action_type": "caution",
        },
        {
            "label": "Run Deep Check (Gemini)",
            "description": "Use on-demand visual reasoning to inspect physical lighting consistency, anatomical realism, and reflection angles.",
            "action_type": "deep_check",
        },
    ]

    # 4. Hard Provenance Overrides
    if c2pa_res.get("score") is not None and c2pa_res["score"] >= 0.95 and c2pa_res.get("confidence") == "high":
        return _pack_verdict(
            verdict="Likely AI",
            headline="Likely AI",
            prob=0.985,
            confidence="high",
            is_inconclusive=False,
            summary="Cryptographically authenticated C2PA manifest records generative AI origin.",
            why_text="Cryptographic Content Credentials (C2PA) attached to this file explicitly record generative AI synthesis. Provenance signatures cannot be forged without invalidating the certificate.",
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="cryptographic_override",
        )

    if metadata_res.get("score") is not None and metadata_res["score"] >= 0.95 and metadata_res.get("confidence") == "high":
        return _pack_verdict(
            verdict="Likely AI",
            headline="Likely AI",
            prob=0.970,
            confidence="high",
            is_inconclusive=False,
            summary="Unambiguous AI generator parameters (prompt/sampler/model headers) detected in image metadata.",
            why_text="Raw file metadata contains explicit generative parameters (such as diffusion prompt text, checkpoint hashes, or generation headers) confirming synthetic generation.",
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="metadata_override",
        )

    if c2pa_res.get("score") is not None and c2pa_res["score"] <= 0.08 and c2pa_res.get("confidence") == "high":
        return _pack_verdict(
            verdict="Likely Real",
            headline="Likely Real",
            prob=0.05,
            confidence="high",
            is_inconclusive=False,
            summary="Cryptographic camera hardware signing (C2PA) confirms authentic capture origin.",
            why_text="Hardware-level cryptographic signing confirms the image was captured directly by a certified camera sensor with an intact signature chain.",
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="cryptographic_override",
        )

    # 5. Non-Photo Gate Override
    if not is_photo_like:
        return _pack_verdict(
            verdict="Inconclusive",
            headline="Inconclusive",
            prob=0.50,
            confidence="low",
            is_inconclusive=True,
            summary=f"Non-photographic image type ({img_type}). Photo-trained AI detectors are unreliable for this type.",
            why_text=(
                f"This image was classified by the learned vision gate as a {img_type} (photo probability: {photo_prob*100:.1f}%). "
                f"Deep vision detectors trained on camera photographs frequently misinterpret flat digital graphics and sharp text rendering as generative diffusion artifacts. "
                f"Without cryptographic C2PA credentials or metadata traces, a definitive AI determination cannot be made."
            ),
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="image_type_gate",
        )

    # 6. Model Disagreement Guardrail
    if spread >= 0.45:
        return _pack_verdict(
            verdict="Inconclusive",
            headline="Inconclusive",
            prob=0.50,
            confidence="low",
            is_inconclusive=True,
            summary=f"Vision models disagree strongly (spread: {spread*100:.0f}%). Detectors cannot reach consensus.",
            why_text=(
                f"Vision classifiers disagree substantially on this image (spread: {spread*100:.0f}%). "
                f"Individual predictions range from {min_pct:.0f}% to {max_pct:.0f}% AI probability. "
                f"This sharp conflict typically arises from compression macroblocks, heavy aperture bokeh, or out-of-distribution textures. "
                f"Rather than forcing a guess, RealLens issues an honest Inconclusive result."
            ),
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="disagreement_guardrail",
        )

    # 7. Heavy Compression Ambiguity Guardrail
    if is_heavily_recompressed and (spread >= 0.25 or (0.35 <= (ml_score or 0.50) <= 0.65)):
        return _pack_verdict(
            verdict="Inconclusive",
            headline="Inconclusive",
            prob=0.50,
            confidence="low",
            is_inconclusive=True,
            summary="Lossy recompression artifacts obscure subtle sensor noise and textures.",
            why_text=(
                f"This image shows clear evidence of lossy compression (such as WhatsApp/social messaging export, downscaling, or low JPEG quality). "
                f"Compression macroblocks and quantization noise destroy fine camera sensor noise and soften generative artifacts. "
                f"Because model predictions span from {min_pct:.0f}% to {max_pct:.0f}%, RealLens issues an honest Inconclusive verdict."
            ),
            signals=signals,
            model_scores=per_model_scores,
            model_weights=per_model_weights,
            spread=spread,
            min_score=min_score,
            max_score=max_score,
            range_indicator=range_indicator,
            evidence_ai=evidence_ai,
            evidence_real=evidence_real,
            evidence_unreliable=evidence_unreliable,
            what_to_do_next=what_to_do_next,
            image_type_info=img_type_info,
            fusion_mode="compression_awareness_guardrail",
        )

    # 8. Agreeing Photographic Case: Fusion Stacker / Weighted Mean
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
            ela_s = ela_res.get("score", 0.50) if (ela_app and ela_res.get("score") is not None) else 0.50
            freq_s = freq_res.get("score", 0.50) if (freq_app and freq_res.get("score") is not None) else 0.50
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
        weights = {}
        values = {}
        if ml_score is not None:
            weights["ml"] = 0.70 * photo_scale
            values["ml"] = float(ml_score)

        meta_score = metadata_res.get("score")
        if meta_score is not None:
            weights["metadata"] = 0.15
            values["metadata"] = float(meta_score)

        if ela_app and ela_res.get("score") is not None:
            weights["ela"] = 0.15 * photo_scale
            values["ela"] = float(ela_res["score"])

        if freq_app and freq_res.get("score") is not None:
            weights["frequency"] = 0.15 * photo_scale
            values["frequency"] = float(freq_res["score"])

        total_weight = sum(weights.values())
        fused_prob = sum(values[k] * weights[k] for k in weights) / max(1e-4, total_weight)

    fused_prob = max(0.04, min(0.96, float(fused_prob)))
    pct = round(fused_prob * 100, 1)

    # 5-Band Honest Spectrum
    if 0.45 <= fused_prob <= 0.55:
        verdict = "Inconclusive"
        headline = "Inconclusive"
        conf = "low"
        summary = "Model scores hover around 50% with no decisive consensus."
        why_text = f"Forensic signals remain evenly balanced ({pct}% AI probability). RealLens refuses to force a guess."
        is_incon = True
    elif fused_prob >= 0.65:
        verdict = "Likely AI"
        headline = f"Likely AI ({pct:.0f}%)"
        conf = "high" if fused_prob >= 0.82 else "medium"
        summary = "Consistent AI visual characteristics and synthetic artifact patterns identified."
        why_text = "Vision models and physical forensics agree on synthetic generative origin."
        is_incon = False
    elif fused_prob > 0.55:
        verdict = "Leaning AI"
        headline = f"Leaning AI ({pct:.0f}%)"
        conf = "medium" if fused_prob >= 0.60 else "low"
        summary = "Subtle generative cues observed, but confidence is limited."
        why_text = "Signals lean toward AI generation, but residual compression leaves slight ambiguity."
        is_incon = False
    elif fused_prob >= 0.35:
        verdict = "Leaning Real"
        headline = f"Leaning Real ({100-pct:.0f}%)"
        conf = "medium" if fused_prob <= 0.40 else "low"
        summary = "Primarily natural characteristics, though compression leaves minor uncertainty."
        why_text = "Image exhibits photographic characteristics and natural spectral properties."
        is_incon = False
    else:
        verdict = "Likely Real"
        headline = f"Likely Real ({100-pct:.0f}%)"
        conf = "high" if fused_prob <= 0.18 else "medium"
        summary = "Photographic sensor noise and vision models align with authentic origin."
        why_text = "All active vision models agree on authentic photographic origin."
        is_incon = False

    return _pack_verdict(
        verdict=verdict,
        headline=headline,
        prob=fused_prob,
        confidence=conf,
        is_inconclusive=is_incon,
        summary=summary,
        why_text=why_text,
        signals=signals,
        model_scores=per_model_scores,
        model_weights=per_model_weights,
        spread=spread,
        min_score=min_score,
        max_score=max_score,
        range_indicator=range_indicator,
        evidence_ai=evidence_ai,
        evidence_real=evidence_real,
        evidence_unreliable=evidence_unreliable,
        what_to_do_next=what_to_do_next,
        image_type_info=img_type_info,
        fusion_mode=fusion_mode,
    )


def _pack_verdict(
    verdict: str,
    headline: str,
    prob: float,
    confidence: str,
    is_inconclusive: bool,
    summary: str,
    why_text: str,
    signals: Dict[str, Any],
    model_scores: Dict[str, Optional[float]],
    model_weights: Dict[str, float],
    spread: float,
    min_score: float,
    max_score: float,
    range_indicator: Optional[str],
    evidence_ai: List[str],
    evidence_real: List[str],
    evidence_unreliable: List[str],
    what_to_do_next: List[Dict[str, Any]],
    image_type_info: Dict[str, Any],
    fusion_mode: str,
) -> Dict[str, Any]:
    """Helper to assemble standard, comprehensive result dictionary."""
    pct = round(prob * 100, 1)
    prob_display = f"{pct:.1f}%" if not is_inconclusive else (
        range_indicator if range_indicator else "50%"
    )

    return {
        "verdict": verdict,
        "headline": headline,
        "ai_probability": round(prob, 3),
        "ai_percentage": pct if not is_inconclusive else None,
        "probability_display": prob_display,
        "range_indicator": range_indicator,
        "confidence": confidence,
        "is_inconclusive": is_inconclusive,
        "summary_reason": summary,
        "why_this_verdict": why_text,
        "model_scores": model_scores,
        "per_model_weights": model_weights,
        "spread": round(spread, 3),
        "min_model_score": round(min_score, 3),
        "max_model_score": round(max_score, 3),
        "evidence_pointing_to_ai": evidence_ai,
        "evidence_pointing_to_real": evidence_real,
        "missing_or_unreliable_evidence": evidence_unreliable,
        "what_to_do_next": what_to_do_next,
        "signals": signals,
        "image_type": image_type_info.get("image_type", "photo-like"),
        "is_photo_like": image_type_info.get("is_photo_like", True),
        "photo_probability": image_type_info.get("photo_probability", 1.0),
        "unreliable_notice": image_type_info.get("unreliable_notice"),
        "notice_severity": image_type_info.get("notice_severity", "none"),
        "is_heavily_recompressed": image_type_info.get("is_heavily_recompressed", False),
        "block_ratio": image_type_info.get("block_ratio", 1.0),
        "fusion_mode": fusion_mode,
        "honesty_note": "Estimate only, accompanied by detector evidence. Missing metadata never proves authenticity.",
    }
