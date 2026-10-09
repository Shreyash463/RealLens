"""Learned Image-Type Classifier for RealLens.

Replaces hand-set heuristic rules with a learned photo-vs-non-photo model:
1. Frozen DINOv2-small patch/CLS embeddings (384-d).
2. Continuous engineered features: flat patch ratio, white paper dominance,
   color palette entropy, edge sharpness, noise floor, DCT block boundary ratio,
   and saturation distributions. (NO aspect ratio or resolution hard triggers!).
3. Outputs continuous `photo_probability` in [0.0, 1.0].
4. Soft reliability gating:
   - photo_probability >= 0.65: Full photographic confidence.
   - 0.35 <= photo_probability < 0.65: Mild note: "Image type is unclear, so detector reliability is lower."
   - photo_probability < 0.35: Non-photo notice: "This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type."
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import joblib
import numpy as np
from PIL import Image
logger = logging.getLogger(__name__)

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
    TORCH_AVAILABLE = True
except (ImportError, OSError) as e:
    logger.warning(f"PyTorch is unavailable or blocked by system policy: {e}")
    torch = None
    AutoImageProcessor = None
    AutoModel = None
    TORCH_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent.parent
GATE_MODEL_PATH = BASE_DIR / "models" / "photo_vs_non_photo_gate.joblib"

_GATE_PACKAGE: Optional[Dict[str, Any]] = None
_GATE_DINO_MODEL: Optional[Any] = None
_GATE_DINO_PROC: Optional[Any] = None
_DEVICE: str = "cpu"


def _load_gate_resources():
    """Loads the trained gate package and DINOv2 vision model singleton."""
    global _GATE_PACKAGE, _GATE_DINO_MODEL, _GATE_DINO_PROC, _DEVICE
    if not TORCH_AVAILABLE:
        return
    if _GATE_PACKAGE is not None:
        return

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    if GATE_MODEL_PATH.exists():
        try:
            _GATE_PACKAGE = joblib.load(GATE_MODEL_PATH)
            backbone_name = _GATE_PACKAGE.get("backbone", "facebook/dinov2-small")
            _GATE_DINO_PROC = AutoImageProcessor.from_pretrained(backbone_name)
            _GATE_DINO_MODEL = AutoModel.from_pretrained(backbone_name).to(_DEVICE)
            _GATE_DINO_MODEL.eval()
            logger.info(f"Loaded learned image-type gate model ({_GATE_PACKAGE.get('model_architecture')}) on {_DEVICE}.")
        except Exception as e:
            logger.warning(f"Failed to load learned gate model package: {e}")
            _GATE_PACKAGE = None
    else:
        logger.warning(f"Gate model package not found at {GATE_MODEL_PATH}")
        _GATE_PACKAGE = None


def extract_engineered_features(img: Image.Image) -> Tuple[np.ndarray, Dict[str, float]]:
    """Extracts 8 normalized continuous visual features (no aspect ratio or resolution!)."""
    thumb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(thumb, dtype=np.float32)

    # 1. Flat patch ratio (fraction of 8x8 blocks with near-zero standard deviation)
    blocks = arr.reshape(32, 8, 32, 8, 3).swapaxes(1, 2)
    block_stds = np.std(blocks, axis=(2, 3, 4))
    flat_patch_ratio = float(np.mean(block_stds < 3.2))

    # 2. White paper dominance (luminance > 240 and saturation < 0.15)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    sat = np.where(max_c > 1e-4, np.clip((max_c - min_c) / np.maximum(max_c, 1e-4), 0.0, 1.0), 0.0)

    white_paper_ratio = float(np.mean((lum > 240.0) & (sat < 0.15)))

    # 3. Unique quantized color palette ratio (16x16x16 bins)
    quant = (arr // 16).astype(np.int32)
    color_codes = quant[:, :, 0] * 256 + quant[:, :, 1] * 16 + quant[:, :, 2]
    unique_colors = len(np.unique(color_codes))
    unique_color_ratio = float(unique_colors / 4096.0)

    # 4. Mean edge sharpness & Laplacian noise floor
    gray = lum
    lap = np.abs(
        gray[1:-1, 1:-1] * 4
        - gray[:-2, 1:-1]
        - gray[2:, 1:-1]
        - gray[1:-1, :-2]
        - gray[1:-1, 2:]
    )
    mean_edge = float(np.mean(lap))

    non_flat_mask = np.repeat(np.repeat(block_stds >= 4.0, 8, axis=0), 8, axis=1)[1:-1, 1:-1]
    if np.sum(non_flat_mask) > 100:
        noise_level = float(np.std(lap[non_flat_mask]))
    else:
        noise_level = float(np.std(lap))

    # 5. DCT 8x8 block boundary ratio (detects JPEG grid artifacts vs smooth continuous fill)
    diff_h = np.abs(gray[7:-1:8, :] - gray[8::8, :]) if gray.shape[0] >= 16 else np.array([0.0])
    diff_v = np.abs(gray[:, 7:-1:8] - gray[:, 8::8]) if gray.shape[1] >= 16 else np.array([0.0])
    block_boundary_energy = (float(np.mean(diff_h)) + float(np.mean(diff_v))) / 2.0
    int_diff_h = np.abs(gray[3:-1:8, :] - gray[4::8, :]) if gray.shape[0] >= 16 else np.array([1.0])
    int_diff_v = np.abs(gray[:, 3:-1:8] - gray[:, 4::8]) if gray.shape[1] >= 16 else np.array([1.0])
    block_internal_energy = max(1e-4, (float(np.mean(int_diff_h)) + float(np.mean(int_diff_v))) / 2.0)
    block_ratio = float(block_boundary_energy / block_internal_energy)

    # 6. Saturation metrics
    mean_sat = float(np.mean(sat))
    low_sat_ratio = float(np.mean(sat < 0.10))

    feature_vec = np.array([
        flat_patch_ratio,
        white_paper_ratio,
        unique_color_ratio,
        mean_edge,
        noise_level,
        block_ratio,
        mean_sat,
        low_sat_ratio,
    ], dtype=np.float32)

    feat_dict = {
        "flat_patch_ratio": round(flat_patch_ratio, 4),
        "white_paper_ratio": round(white_paper_ratio, 4),
        "unique_color_ratio": round(unique_color_ratio, 4),
        "mean_edge": round(mean_edge, 3),
        "noise_level": round(noise_level, 3),
        "block_ratio": round(block_ratio, 3),
        "mean_sat": round(mean_sat, 3),
        "low_sat_ratio": round(low_sat_ratio, 3),
    }
    return feature_vec, feat_dict


def classify_image_type(img: Image.Image) -> Dict[str, Any]:
    """Classifies image using learned DINOv2-small gate and outputs soft photo_probability.

    Returns:
        dict:
            photo_probability: float (0.0 to 1.0)
            image_type: "photo-like" | "screenshot/UI" | "document/text-heavy" | "flat graphic/illustration" | "unclear / mixed"
            is_photo_like: bool (photo_probability >= 0.35)
            confidence: "low" | "medium" | "high"
            unreliable_notice: str | None
            notice_severity: "none" | "mild" | "warning"
            is_heavily_recompressed: bool
            block_ratio: float
            compression_notice: str | None
            features: dict
    """
    _load_gate_resources()
    feat_vec, feat_dict = extract_engineered_features(img)
    block_ratio = feat_dict["block_ratio"]

    # Compute soft photo_probability via learned classifier
    if _GATE_PACKAGE is not None and _GATE_DINO_MODEL is not None and _GATE_DINO_PROC is not None:
        try:
            inputs = _GATE_DINO_PROC(images=[img.convert("RGB")], return_tensors="pt")
            inputs = {k: v.to(_DEVICE) for k, v in inputs.items()}
            with torch.no_grad():
                out = _GATE_DINO_MODEL(**inputs)
                dinov2_emb = out.last_hidden_state[:, 0, :].cpu().numpy()

            combined = np.hstack([dinov2_emb, feat_vec.reshape(1, -1)])
            scaled = _GATE_PACKAGE["scaler"].transform(combined)
            clf = _GATE_PACKAGE["classifier"]
            photo_prob = float(clf.predict_proba(scaled)[0, 1])
        except Exception as e:
            logger.warning(f"Error during learned gate inference: {e}, falling back to feature estimate")
            photo_prob = 0.50
    else:
        # Graceful fallback using continuous engineered features if DINOv2 / PyTorch is unavailable
        flat_ratio = feat_dict.get("flat_patch_ratio", 0.0)
        white_ratio = feat_dict.get("white_paper_ratio", 0.0)
        noise_var = feat_dict.get("noise_variance", 0.0)
        entropy = feat_dict.get("color_entropy", 0.0)

        if white_ratio > 0.25 or (flat_ratio > 0.40 and entropy < 3.5):
            photo_prob = 0.12
        elif flat_ratio > 0.35:
            photo_prob = 0.25
        elif noise_var > 1.2 or (flat_ratio < 0.22 and entropy > 3.8):
            photo_prob = 0.92
        else:
            photo_prob = 0.65

    photo_prob = max(0.0, min(1.0, float(photo_prob)))

    # Determine sub-type and notices based on soft photo_probability
    is_photo_like = (photo_prob >= 0.35)

    if photo_prob >= 0.65:
        image_type = "photo-like"
        unreliable_notice = None
        notice_severity = "none"
        confidence = "high" if photo_prob >= 0.85 else "medium"
        reason = "Natural photographic scene or photorealistic capture identified by learned vision gate."
    elif photo_prob >= 0.35:
        image_type = "unclear / mixed"
        unreliable_notice = "Image type is unclear, so detector reliability is lower."
        notice_severity = "mild"
        confidence = "low"
        reason = "Learned gate detected borderline characteristics (mix of natural textures and flat/compressed regions)."
    else:
        # Non-photo: identify descriptive subtype for display (not hard gate!)
        if feat_dict["white_paper_ratio"] > 0.30 or feat_dict["flat_patch_ratio"] > 0.60:
            image_type = "document/text-heavy"
        elif feat_dict["mean_edge"] > 4.5 and feat_dict["flat_patch_ratio"] > 0.35:
            image_type = "screenshot/UI"
        else:
            image_type = "flat graphic/illustration"

        unreliable_notice = "This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type."
        notice_severity = "warning"
        confidence = "high" if photo_prob <= 0.15 else "medium"
        reason = f"Learned vision gate classified this as {image_type} (photo probability: {photo_prob*100:.1f}%)."

    # Recompression detection: prominent DCT 8x8 block boundary artifacts
    is_heavily_recompressed = (block_ratio > 3.2 and feat_dict["flat_patch_ratio"] < 0.50)
    compression_notice = None
    if is_heavily_recompressed:
        compression_notice = f"Heavy lossy recompression detected (DCT blockiness: {block_ratio:.2f}). Compression artifacts obscure subtle sensor noise."

    return {
        "photo_probability": round(photo_prob, 4),
        "image_type": image_type,
        "is_photo_like": is_photo_like,
        "confidence": confidence,
        "reason": reason,
        "unreliable_notice": unreliable_notice,
        "notice_severity": notice_severity,
        "is_heavily_recompressed": is_heavily_recompressed,
        "block_ratio": round(block_ratio, 3),
        "compression_notice": compression_notice,
        "features": feat_dict,
    }
