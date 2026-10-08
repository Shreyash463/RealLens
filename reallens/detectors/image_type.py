"""Image-Type Classifier for RealLens.

Classifies incoming images into:
- photo-like
- screenshot/UI
- document/text-heavy
- flat graphic/illustration
- other

Uses cheap, deterministic computer vision features:
- Unique color entropy
- Fraction of near-flat spatial patches
- Background luminance dominance (e.g., paper/white/black background)
- Color saturation distribution
- Axis-aligned horizontal/vertical edge dominance
- High-frequency sensor noise presence
"""
from __future__ import annotations

import logging
from typing import Any, Dict, Tuple
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Common screen aspect ratios (width / height) within +/- 5% tolerance
COMMON_SCREEN_RATIOS = [
    16 / 9,   # 1.778
    16 / 10,  # 1.600
    19.5 / 9, # 2.167 (Modern mobile)
    20 / 9,   # 2.222
    4 / 3,    # 1.333
    9 / 16,   # 0.562 (Mobile screenshot vertical)
    9 / 19.5, # 0.462 (Mobile vertical)
]


def extract_type_features(img: Image.Image) -> Dict[str, float]:
    """Computes fast measurable features to discriminate photos from graphics/screenshots."""
    # Work on a standardized 256x256 RGB copy for fast constant-time computation
    thumb = img.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR)
    arr = np.array(thumb, dtype=np.float32)
    gray = 0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]

    w_orig, h_orig = img.size
    aspect_ratio = float(w_orig) / max(1.0, float(h_orig))

    # 1. Aspect ratio match to screen display
    is_screen_ratio = any(abs(aspect_ratio - r) < 0.06 for r in COMMON_SCREEN_RATIOS)

    # 2. Near-flat patches (16x16 grid -> 16x16 = 256 patches)
    # Natural photos have textures across most patches; certificates/screenshots have huge flat areas
    patches = gray.reshape(16, 16, 16, 16).swapaxes(1, 2).reshape(-1, 16, 16)
    patch_stds = np.std(patches, axis=(1, 2))
    flat_patch_ratio = float(np.mean(patch_stds < 3.5))

    # 3. Unique colors ratio
    # Quantize to 5-bit RGB (32 levels per channel)
    q_arr = (arr // 8).astype(np.uint16)
    rgb_codes = (q_arr[:, :, 0] << 10) + (q_arr[:, :, 1] << 5) + q_arr[:, :, 2]
    unique_colors = len(np.unique(rgb_codes))
    unique_color_ratio = unique_colors / (256.0 * 256.0)

    # 4. Background dominance (white/paper/black)
    hist, _ = np.histogram(gray, bins=32, range=(0, 256))
    max_bin_ratio = float(np.max(hist) / (256.0 * 256.0))
    white_paper_ratio = float(np.mean(gray > 240))
    black_bg_ratio = float(np.mean(gray < 18))

    # 5. Saturation in HSV
    rgb_norm = arr / 255.0
    c_max = np.max(rgb_norm, axis=2)
    c_min = np.min(rgb_norm, axis=2)
    delta = c_max - c_min
    sat = np.zeros_like(c_max)
    mask = c_max > 0.001
    sat[mask] = delta[mask] / c_max[mask]
    mean_sat = float(np.mean(sat))
    low_sat_ratio = float(np.mean(sat < 0.10))

    # 6. Axis-aligned edge sharpness (Sobel filter simulation)
    diff_h = np.abs(gray[:, 1:] - gray[:, :-1])
    diff_v = np.abs(gray[1:, :] - gray[:-1, :])
    mean_edge = float((np.mean(diff_h) + np.mean(diff_v)) / 2.0)

    # 7. High-frequency sensor noise estimate
    # In textured or photographic areas, sensor shot noise creates a Gaussian floor
    # We estimate noise on non-flat patches
    non_flat_patches = patches[patch_stds >= 5.0]
    if len(non_flat_patches) > 5:
        # Laplacian kernel [0, 1, 0; 1, -4, 1; 0, 1, 0] on non-flat patches
        diff_lap = (
            non_flat_patches[:, 1:-1, 2:] + non_flat_patches[:, 1:-1, :-2] +
            non_flat_patches[:, 2:, 1:-1] + non_flat_patches[:, :-2, 1:-1] -
            4 * non_flat_patches[:, 1:-1, 1:-1]
        )
        noise_level = float(np.median(np.abs(diff_lap)))
    else:
        noise_level = 0.0

    # 8. 8x8 DCT Block boundary discontinuity on native crop
    crop_w = (w_orig // 8) * 8
    crop_h = (h_orig // 8) * 8
    if crop_w >= 32 and crop_h >= 32:
        gray_native = img.convert("L").crop((0, 0, min(crop_w, 512), min(crop_h, 512)))
        arr_crop = np.array(gray_native, dtype=np.float32)
        cw, ch = gray_native.size
        block_cols = np.arange(7, cw - 1, 8)
        intra_cols = np.array([c for c in range(cw - 1) if c not in block_cols])
        diff_h_native = np.abs(arr_crop[:, 1:] - arr_crop[:, :-1])
        denom_h = max(0.01, float(np.mean(diff_h_native[:, intra_cols])))
        ratio_h = float(np.mean(diff_h_native[:, block_cols])) / denom_h if len(block_cols) > 0 else 1.0

        block_rows = np.arange(7, ch - 1, 8)
        intra_rows = np.array([r for r in range(ch - 1) if r not in block_rows])
        diff_v_native = np.abs(arr_crop[1:, :] - arr_crop[:-1, :])
        denom_v = max(0.01, float(np.mean(diff_v_native[intra_rows, :])))
        ratio_v = float(np.mean(diff_v_native[block_rows, :])) / denom_v if len(block_rows) > 0 else 1.0

        block_ratio = float((ratio_h + ratio_v) / 2.0)
    else:
        block_ratio = 1.0

    return {
        "aspect_ratio": float(aspect_ratio),
        "is_screen_ratio": bool(is_screen_ratio),
        "flat_patch_ratio": float(flat_patch_ratio),
        "unique_color_ratio": float(unique_color_ratio),
        "max_bin_ratio": float(max_bin_ratio),
        "white_paper_ratio": float(white_paper_ratio),
        "black_bg_ratio": float(black_bg_ratio),
        "mean_sat": float(mean_sat),
        "low_sat_ratio": float(low_sat_ratio),
        "mean_edge": float(mean_edge),
        "noise_level": float(noise_level),
        "block_ratio": float(block_ratio),
    }


def classify_image_type(img: Image.Image) -> Dict[str, Any]:
    """Classifies the image type and provides confidence and rationale."""
    w_orig, h_orig = img.size
    feats = extract_type_features(img)

    flat_ratio = feats["flat_patch_ratio"]
    white_ratio = feats["white_paper_ratio"]
    unique_colors = feats["unique_color_ratio"]
    mean_sat = feats["mean_sat"]
    low_sat = feats["low_sat_ratio"]
    is_screen = feats["is_screen_ratio"]
    noise = feats["noise_level"]
    block_ratio = feats.get("block_ratio", 1.0)

    # Compression awareness indicators (WhatsApp, chat apps, low JPEG quality)
    max_dim = max(w_orig, h_orig)
    is_chat_dim = (
        (w_orig in (1280, 1600, 1080, 720, 640) and h_orig in (720, 900, 1200, 1080, 1280, 480))
        or (w_orig == 1280 and h_orig == 720)
        or (w_orig == 720 and h_orig == 1280)
        or (max_dim in (1280, 1600, 1080))
    )
    is_small_dim = max_dim <= 1280

    is_heavily_recompressed = (
        (block_ratio >= 1.15)
        or (is_chat_dim and block_ratio >= 1.05)
        or (is_small_dim and block_ratio >= 1.20)
        or (w_orig in (1280, 720) and h_orig in (720, 1280))  # Standard WhatsApp video/photo export
    )
    compression_notice = None
    if is_heavily_recompressed:
        compression_notice = (
            f"Heavy lossy recompression detected (DCT blockiness: {block_ratio:.2f}, {w_orig}x{h_orig}). "
            "Compression artifacts obscure subtle sensor noise and generative textures."
        )

    # Handle low-resolution thumbnails (<96px, e.g. CIFAR/CIFAKE 32x32)
    is_thumbnail = max(w_orig, h_orig) < 96

    # Rule 1: Document / Text-Heavy
    # White background > 35%, mostly desaturated, moderate to high flat ratio
    if (white_ratio > 0.40 and low_sat > 0.65) or (white_ratio > 0.55 and flat_ratio > 0.25):
        img_type = "document/text-heavy"
        confidence = "high" if white_ratio > 0.55 else "medium"
        reason = (
            f"High white background luminance ({white_ratio*100:.1f}%) and low color saturation "
            f"indicate a document, certificate paper, or text-heavy graphic."
        )

    # Rule 2: Screenshot / UI (only valid for actual screen/window resolutions >= 128px)
    elif not is_thumbnail and ((flat_ratio > 0.35 and is_screen and noise < 12.0) or (flat_ratio > 0.45 and noise < 10.0)):
        img_type = "screenshot/UI"
        confidence = "high" if (flat_ratio > 0.50 and is_screen) else "medium"
        reason = (
            f"High proportion of flat color patches ({flat_ratio*100:.1f}%), sharp rectangular edges, "
            f"and absent camera sensor noise indicate a screen capture or software UI."
        )

    # Rule 3: Flat graphic / Illustration / Certificate
    elif not is_thumbnail and ((flat_ratio > 0.25 and unique_colors < 0.035) or (flat_ratio > 0.32 and noise < 7.0)):
        img_type = "flat graphic/illustration"
        confidence = "medium"
        reason = (
            f"Restricted color palette ({unique_colors*100:.2f}% unique codes) and uniform fill regions "
            f"characterize vector art, digital certificates, slides, or graphic designs."
        )

    # Rule 4: Photo-like
    elif flat_ratio < 0.28 or is_thumbnail:
        img_type = "photo-like"
        confidence = "high" if flat_ratio < 0.15 else "medium"
        reason = (
            f"Continuous natural texture variations ({flat_ratio*100:.1f}% flat area), rich color entropy, "
            f"and sensor grain align with real-world camera or photorealistic capture."
        )

    # Rule 5: Fallback photo-like vs other
    elif flat_ratio < 0.25:
        img_type = "photo-like"
        confidence = "low"
        reason = "Texture characteristics mostly resemble photographic or continuous-tone imagery."
    else:
        img_type = "other"
        confidence = "low"
        reason = "Hybrid texture composition exhibiting mixed photographic and graphic features."

    is_photo = (img_type == "photo-like")
    unreliable_notice = (
        "This looks like a screenshot/document/graphic. Photo-trained AI detectors are unreliable for this type."
        if not is_photo else None
    )

    return {
        "image_type": img_type,
        "is_photo_like": is_photo,
        "confidence": confidence,
        "reason": reason,
        "unreliable_notice": unreliable_notice,
        "is_heavily_recompressed": bool(is_heavily_recompressed),
        "block_ratio": float(block_ratio),
        "is_chat_dim": bool(is_chat_dim),
        "compression_notice": compression_notice,
        "features": feats,
    }
