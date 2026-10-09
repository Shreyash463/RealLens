"""Screenshot and Letterbox Auto-Cropper for RealLens.

Detects and crops:
1. 'phone_screenshot': Status bar (top ~4-6%) and gesture pill / nav bar (bottom ~3-5%) on mobile screenshots.
2. 'browser_screenshot': Browser header / address bar / tabs (top ~6-10%).
3. 'letterbox': Uniform black, white, or dark-gray padding borders (horizontal or vertical).

Prevents pixel AI detectors from evaluating UI elements instead of genuine image content.
"""
from __future__ import annotations

import logging
from typing import Optional, Tuple, Dict, Any
import numpy as np
from PIL import Image

logger = logging.getLogger("screenshot_cropper")


def detect_and_crop_screenshot(
    image: Image.Image,
    min_content_dim: int = 128
) -> Dict[str, Any]:
    """Inspects image for surrounding screenshot containers and letterboxing.

    Returns dict with:
        - cropped_image: Image.Image
        - auto_cropped: bool
        - detected_container: Optional[str] ('phone_screenshot' | 'browser_screenshot' | 'letterbox')
        - crop_box: Tuple[int, int, int, int] (left, top, right, bottom)
        - original_size: Tuple[int, int]
        - cropped_size: Tuple[int, int]
    """
    img_rgb = image.convert("RGB")
    w, h = img_rgb.size
    orig_box = (0, 0, w, h)

    if w < min_content_dim or h < min_content_dim:
        return {
            "cropped_image": image,
            "auto_cropped": False,
            "detected_container": None,
            "crop_box": orig_box,
            "original_size": (w, h),
            "cropped_size": (w, h),
        }

    arr = np.array(img_rgb)  # shape (H, W, 3)

    # 1. Check for Letterboxing (solid or near-uniform black/white/dark borders)
    top_crop, bottom_crop, left_crop, right_crop = 0, 0, 0, 0

    # Top rows uniformity
    for y in range(min(int(h * 0.35), 300)):
        row = arr[y, :, :]
        row_std = np.std(row)
        if row_std < 8.0:
            top_crop = y + 1
        else:
            break

    # Bottom rows uniformity
    for y in range(h - 1, max(h - int(h * 0.35), h - 300), -1):
        row = arr[y, :, :]
        row_std = np.std(row)
        if row_std < 8.0:
            bottom_crop = h - y
        else:
            break

    # Left cols uniformity
    for x in range(min(int(w * 0.35), 300)):
        col = arr[:, x, :]
        col_std = np.std(col)
        if col_std < 8.0:
            left_crop = x + 1
        else:
            break

    # Right cols uniformity
    for x in range(w - 1, max(w - int(w * 0.35), w - 300), -1):
        col = arr[:, x, :]
        col_std = np.std(col)
        if col_std < 8.0:
            right_crop = w - x
        else:
            break

    if (top_crop + bottom_crop > int(h * 0.05)) or (left_crop + right_crop > int(w * 0.05)):
        new_left = left_crop
        new_top = top_crop
        new_right = w - right_crop
        new_bottom = h - bottom_crop
        if (new_right - new_left >= min_content_dim) and (new_bottom - new_top >= min_content_dim):
            cropped = img_rgb.crop((new_left, new_top, new_right, new_bottom))
            return {
                "cropped_image": cropped,
                "auto_cropped": True,
                "detected_container": "letterbox",
                "crop_box": (new_left, new_top, new_right, new_bottom),
                "original_size": (w, h),
                "cropped_size": cropped.size,
            }

    # 2. Check for Phone Screenshot (aspect ratio + status bar / nav bar)
    aspect = h / max(w, 1)
    # Common smartphone screen aspect ratios: 16:9 (1.78), 18:9 (2.0), 19.5:9 (2.17), 20:9 (2.22)
    is_phone_aspect = 1.65 <= aspect <= 2.35

    if is_phone_aspect and h >= 400:
        # Check top status bar (typically top 3% to 6%)
        # Look for horizontal boundary line or distinct header background
        top_band_h = int(h * 0.05)
        top_band = arr[:top_band_h, :, :]
        # Check if top band is largely solid color with small glyphs
        top_band_median = np.median(top_band, axis=(0, 1))
        diff = np.abs(top_band - top_band_median)
        flat_ratio = np.mean(diff < 15.0)

        # Check bottom navigation bar (typically bottom 3% to 5%)
        bottom_band_h = int(h * 0.04)
        bottom_band = arr[-bottom_band_h:, :, :]
        bottom_band_median = np.median(bottom_band, axis=(0, 1))
        bottom_diff = np.abs(bottom_band - bottom_band_median)
        bottom_flat_ratio = np.mean(bottom_diff < 15.0)

        if flat_ratio > 0.70 or bottom_flat_ratio > 0.70:
            # Phone screenshot confirmed!
            crop_top = top_band_h if flat_ratio > 0.60 else int(h * 0.04)
            crop_bot = h - (bottom_band_h if bottom_flat_ratio > 0.60 else int(h * 0.03))
            cropped = img_rgb.crop((0, crop_top, w, crop_bot))
            return {
                "cropped_image": cropped,
                "auto_cropped": True,
                "detected_container": "phone_screenshot",
                "crop_box": (0, crop_top, w, crop_bot),
                "original_size": (w, h),
                "cropped_size": cropped.size,
            }

    # 3. Check for Browser Screenshot (desktop landscape 16:9 or 16:10 with top address bar)
    if w >= 800 and 1.2 <= (w / max(h, 1)) <= 2.0:
        # Desktop browser window
        browser_bar_h = int(h * 0.08)
        top_browser = arr[:browser_bar_h, :, :]
        # Look for horizontal divider line right below address bar
        divider_row = arr[browser_bar_h - 2 : browser_bar_h + 2, :, :]
        if np.std(divider_row) < 12.0 and np.std(top_browser) < 30.0:
            cropped = img_rgb.crop((0, browser_bar_h, w, h))
            return {
                "cropped_image": cropped,
                "auto_cropped": True,
                "detected_container": "browser_screenshot",
                "crop_box": (0, browser_bar_h, w, h),
                "original_size": (w, h),
                "cropped_size": cropped.size,
            }

    return {
        "cropped_image": image,
        "auto_cropped": False,
        "detected_container": None,
        "crop_box": orig_box,
        "original_size": (w, h),
        "cropped_size": (w, h),
    }
