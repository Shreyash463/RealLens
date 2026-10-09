"""Condition Detector for RealLens.

Classifies incoming image into one of the operational conditions:
- 'original': Camera original quality (intact EXIF with camera Make/Model/Exposure, high quality table, no chat compression).
- 'whatsapp': Mobile chat app recompression (WhatsApp naming, 4:2:0 subsampling, Q=65..78, stripped EXIF, max dim <= 1600).
- 'screenshot': Device screenshot (status bar, browser chrome, or letterboxing detected by cropper).
- 'jpeg_compressed': Re-encoded JPEG (Q <= 80, stripped metadata, moderate DCT blockiness).
- 'standard': Standard digital image.

Condition dictates model weighting and precision guardrails in fusion.py.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional
import numpy as np
from PIL import Image

logger = logging.getLogger("condition")


def detect_image_condition(
    image: Image.Image,
    filename: Optional[str] = None,
    metadata_res: Optional[Dict[str, Any]] = None,
    crop_info: Optional[Dict[str, Any]] = None,
    image_type_res: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Detects degradation condition to dynamically calibrate detector weights.

    Returns dict:
        - condition: 'original' | 'whatsapp' | 'screenshot' | 'jpeg_compressed' | 'standard'
        - condition_label: Human-readable badge text
        - confidence: 'high' | 'medium'
        - details: str
        - allows_likely_real: bool (Strictly False for WhatsApp & Screenshot)
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    fname = (filename or "").lower()
    meta = metadata_res or {}
    cropper = crop_info or {}
    gate = image_type_res or {}

    # 1. Screenshot Detection
    if cropper.get("auto_cropped", False):
        container = cropper.get("detected_container", "screenshot")
        return {
            "condition": "screenshot",
            "condition_label": f"Screenshot ({container.replace('_', ' ').title()})",
            "confidence": "high",
            "details": f"Cropped surrounding {container} container prior to pixel analysis.",
            "allows_likely_real": False,  # INVARIANT: Never return 'Likely real' on screenshot
        }

    # 2. WhatsApp / Chat-app Compression
    is_whatsapp_name = "whatsapp" in fname
    has_exif_camera = bool(meta.get("camera_make") or meta.get("camera_model"))
    is_heavy_comp = gate.get("is_heavily_recompressed", False)

    # Estimate JPEG quality from quantization table if available
    q_est = 85.0
    if hasattr(image, "quantization") and image.quantization:
        t = list(image.quantization.values())[0]
        val = float(np.mean(t)) if len(t) else 30.0
        q_est = max(1.0, min(100.0, 100.0 - val / 2.0))

    if is_whatsapp_name or (not has_exif_camera and max(w, h) <= 1600 and (60.0 <= q_est <= 78.0 or is_heavy_comp)):
        return {
            "condition": "whatsapp",
            "condition_label": "WhatsApp / Chat-App Compression",
            "confidence": "high" if is_whatsapp_name else "medium",
            "details": "Image exhibits chat-app DCT re-encoding, stripped metadata, and resolution capping.",
            "allows_likely_real": False,  # INVARIANT: Never return 'Likely real' on chat-compressed image
        }

    # 3. Generic Re-encoded JPEG
    if not has_exif_camera and (q_est < 82.0 or is_heavy_comp):
        return {
            "condition": "jpeg_compressed",
            "condition_label": f"Lossy JPEG Recompression (~Q{int(q_est)})",
            "confidence": "medium",
            "details": "Lossy JPEG quantization detected without camera origin tags.",
            "allows_likely_real": False,  # Recompressed images cannot prove authenticity
        }

    # 4. Authentic Camera Original Quality
    if has_exif_camera and (meta.get("exposure_time") or meta.get("iso") or meta.get("f_number")):
        cam_str = f"{meta.get('camera_make', '')} {meta.get('camera_model', '')}".strip()
        return {
            "condition": "original",
            "condition_label": f"Original Camera Quality ({cam_str or 'Intact EXIF'})",
            "confidence": "high",
            "details": "Original camera metadata, exposure settings, and sensor noise profile intact.",
            "allows_likely_real": True,  # Eligible for 'Likely real' IF all models agree
        }

    # 5. Standard Digital Image
    return {
        "condition": "standard",
        "condition_label": "Standard Web / Digital Image",
        "confidence": "medium",
        "details": "Standard digital resolution without identifiable camera metadata or heavy recompression.",
        "allows_likely_real": False,
    }
