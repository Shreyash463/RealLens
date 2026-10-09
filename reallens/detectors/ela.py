"""Error Level Analysis (ELA) detector for RealLens.

Performs JPEG Error Level Analysis by computing pixel-wise differences between
the image and a re-compressed JPEG version. Produces both analytical metrics
and a visualization heatmap for forensic inspection.
"""
from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image, ImageChops, ImageEnhance


def analyze_ela(
    image_source: Union[str, Path, bytes, Image.Image],
    quality: int = 90,
    scale: int = 15,
    image_type: str = "photo-like",
) -> Dict[str, Any]:
    """Computes Error Level Analysis (ELA) metrics and heatmap.

    Args:
        image_source: Path, bytes, or PIL Image.
        quality: JPEG compression quality level (default 90).
        scale: Multiplier to amplify subtle compression differences.
        image_type: Detected image category ("photo-like", "screenshot/UI", etc.).

    Returns:
        dict:
            score: float (0.0=real-like JPEG equilibrium, 1.0=elevated synthetic/uncompressed error)
            confidence: "low" | "medium"
            evidence: list of str
            metrics: dict of analytical numbers (mean_error, std_error, max_error, high_error_ratio)
            heatmap_base64: str (base64 data URL for frontend display)
    """
    evidence: List[str] = []
    close_img = False
    img: Optional[Image.Image] = None

    try:
        if isinstance(image_source, Image.Image):
            img = image_source
        elif isinstance(image_source, (str, Path)):
            img = Image.open(image_source)
            close_img = True
        elif isinstance(image_source, bytes):
            img = Image.open(io.BytesIO(image_source))
            close_img = True
        else:
            return {
                "score": None,
                "confidence": "low",
                "evidence": ["Unsupported image input type for ELA."],
                "metrics": {},
                "heatmap_base64": None,
            }

        # Convert to RGB safely (handling RGBA, CMYK, P, L, etc.)
        if img.mode != "RGB":
            # For transparent images, paste onto neutral white background
            if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                rgba = img.convert("RGBA")
                background = Image.new("RGB", rgba.size, (255, 255, 255))
                background.paste(rgba, mask=rgba.split()[3])
                work_img = background
            else:
                work_img = img.convert("RGB")
        else:
            work_img = img.copy()

        # Handle extreme image sizes: downsample large images for fast responsive ELA analysis
        max_dim = 1600
        w, h = work_img.size
        if max(w, h) > max_dim:
            scaling = max_dim / max(w, h)
            new_size = (int(w * scaling), int(h * scaling))
            work_img = work_img.resize(new_size, Image.Resampling.LANCZOS)

        # 1. Re-save as JPEG into in-memory buffer at specified quality
        buf = io.BytesIO()
        work_img.save(buf, format="JPEG", quality=quality)
        buf.seek(0)
        recompressed = Image.open(buf)

        # 2. Compute absolute difference image
        diff = ImageChops.difference(work_img, recompressed)

        # 3. Compute quantitative metrics via NumPy
        diff_arr = np.array(diff, dtype=np.float32)
        # Luminance difference across RGB
        gray_diff = np.mean(diff_arr, axis=2)

        mean_error = float(np.mean(gray_diff))
        std_error = float(np.std(gray_diff))
        max_error = float(np.max(gray_diff))
        high_error_ratio = float(np.mean(gray_diff > 12.0))  # Fraction of pixels with noticeable error

        # 4. Generate enhanced heatmap
        # Amplify difference for visual clarity
        extrema = diff.getextrema()
        max_diff_val = max([ex[1] for ex in extrema]) if extrema else 1
        scale_factor = 255.0 / max(1, max_diff_val) * 0.85
        scale_factor = min(scale_factor, float(scale))

        enhancer = ImageEnhance.Brightness(diff)
        amplified = enhancer.enhance(scale_factor)

        # Encode amplified diff image to Base64 PNG data URL
        out_buf = io.BytesIO()
        amplified.save(out_buf, format="PNG")
        b64_str = base64.b64encode(out_buf.getvalue()).decode("ascii")
        heatmap_base64 = f"data:image/png;base64,{b64_str}"

        metrics = {
            "mean_error": round(mean_error, 3),
            "std_error": round(std_error, 3),
            "max_error": round(max_error, 2),
            "high_error_ratio": round(high_error_ratio, 4),
        }

        # If image is non-photographic, heavily recompressed, or too small, ELA is not applicable
        w_orig, h_orig = work_img.size
        is_too_small = (w_orig < 64 or h_orig < 64)

        if image_type != "photo-like":
            score = None
            applicable = False
            status_text = "Not applicable"
            confidence = "low"
            evidence.append(
                f"Image classified as '{image_type}'. Error Level Analysis is not applicable for non-photographic graphics or documents."
            )
        elif is_too_small:
            score = None
            applicable = False
            status_text = "Not applicable"
            confidence = "low"
            evidence.append("Image resolution is too small (<64px) for reliable Error Level Analysis.")
        elif mean_error > 9.5:
            score = min(0.85, 0.5 + (mean_error - 9.5) * 0.035)
            applicable = True
            status_text = f"Elevated error ({round(score*100):.0f}%)"
            evidence.append(
                f"Elevated ELA mean error ({mean_error:.2f}) indicates image has not undergone standard camera JPEG compression equilibrium."
            )
            confidence = "medium"
        elif mean_error < 4.5:
            score = max(0.15, 0.5 - (4.5 - mean_error) * 0.06)
            applicable = True
            status_text = f"Low error ({round(score*100):.0f}%)"
            evidence.append(
                f"Low ELA error ({mean_error:.2f}) consistent with previously compressed natural JPEG photographic content."
            )
            confidence = "medium"
        else:
            score = None
            applicable = False
            status_text = "Not applicable"
            confidence = "low"
            evidence.append(
                f"Moderate ELA error ({mean_error:.2f}, std: {std_error:.2f}) - neutral compression profile without clear synthetic or camera deviation."
            )

        evidence.append(
            f"High-error pixel ratio: {high_error_ratio*100:.1f}%, max differential: {max_error:.1f}."
        )

        return {
            "score": round(score, 3) if score is not None else None,
            "applicable": applicable,
            "status_text": status_text,
            "confidence": confidence,
            "evidence": evidence,
            "metrics": metrics,
            "heatmap_base64": heatmap_base64,
        }

    except Exception as exc:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [f"Error Level Analysis failed: {str(exc)}"],
            "metrics": {},
            "heatmap_base64": None,
        }
    finally:
        if close_img and img:
            try:
                img.close()
            except Exception:
                pass
