"""Metadata detector for RealLens.

Analyzes EXIF, PNG chunks, IPTC, and XMP metadata for signs of AI generation
or camera authenticity.
"""
from __future__ import annotations

import io
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image, ExifTags

# Known keywords that indicate AI generation in metadata
AI_INDICATORS = [
    r"stable\s*diffusion",
    r"midjourney",
    r"dall[-_ ]?e",
    r"novelai",
    r"comfyui",
    r"automatic1111",
    r"invokeai",
    r"fooocus",
    r"bing\s*image\s*creator",
    r"adobe\s*firefly",
    r"ideogram",
    r"flux(\.1)?",
    r"latent\s*diffusion",
    r"text2image",
    r"txt2img",
    r"prompt:",
    r"negative_prompt:",
    r"sampler_name",
    r"cfg_scale",
    r"model_hash",
    r"clip_skip",
    r"photoshop\s*generative\s*fill",
    r"generative\s*ai",
    r"synthetic\s*media",
]

# Camera tags that suggest authentic photographic provenance
AUTHENTIC_CAMERA_TAGS = [
    "Make",
    "Model",
    "LensModel",
    "LensMake",
    "ExposureTime",
    "FNumber",
    "ISOSpeedRatings",
    "FocalLength",
    "DateTimeOriginal",
    "ShutterSpeedValue",
    "ApertureValue",
]


def extract_exif_dict(img: Image.Image) -> Dict[str, Any]:
    """Safely extracts EXIF as a human-readable dict."""
    exif_data = {}
    try:
        raw_exif = img.getexif()
        if raw_exif:
            for tag_id, value in raw_exif.items():
                tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                exif_data[tag_name] = value

            # Try IFD tags (Exif / GPS)
            try:
                for ifd_id in (ExifTags.IFD.Exif, ExifTags.IFD.GPSInfo):
                    try:
                        ifd = raw_exif.get_ifd(ifd_id)
                        for tag_id, value in ifd.items():
                            tag_name = ExifTags.TAGS.get(tag_id, str(tag_id))
                            exif_data[tag_name] = value
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass
    return exif_data


def analyze_metadata(image_source: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
    """Analyzes image metadata for AI indicators or camera authenticity.

    Returns:
        dict:
            score: float | None (0.0=real, 1.0=AI, None=missing/neutral)
            confidence: "low" | "medium" | "high"
            evidence: list of str
            raw_metadata: dict of extracted key metadata
    """
    evidence: List[str] = []
    raw_info: Dict[str, Any] = {}
    ai_matches: List[str] = []

    # Load image safely
    img: Optional[Image.Image] = None
    close_img = False

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
                "evidence": ["Unsupported image input type."],
                "raw_metadata": {},
            }

        # 1. Inspect PIL info dict (contains PNG chunks, JPEG comments, etc.)
        info = img.info or {}
        for k, v in info.items():
            if isinstance(v, (str, bytes)):
                v_str = v.decode("utf-8", errors="ignore") if isinstance(v, bytes) else str(v)
                raw_info[f"info:{k}"] = v_str[:500]  # truncate long prompts

                for pat in AI_INDICATORS:
                    if re.search(pat, v_str, re.IGNORECASE):
                        ai_matches.append(f"Header '{k}' contains indicator matching '{pat}'")

        # 2. Inspect EXIF
        exif = extract_exif_dict(img)
        camera_signals = 0
        camera_details = []

        if exif:
            for k, v in exif.items():
                v_str = str(v)
                raw_info[f"exif:{k}"] = v_str[:200]

                # Check for AI tags in EXIF values (e.g. Software: Midjourney, ImageDescription: Prompt)
                for pat in AI_INDICATORS:
                    if re.search(pat, v_str, re.IGNORECASE):
                        ai_matches.append(f"EXIF tag '{k}' contains indicator matching '{pat}'")

                # Check authentic camera tags
                if k in AUTHENTIC_CAMERA_TAGS and v_str.strip():
                    camera_signals += 1
                    if k in ("Make", "Model", "LensModel", "ExposureTime", "FNumber", "ISOSpeedRatings"):
                        camera_details.append(f"{k}: {v_str}")

        # Deduplicate AI matches
        ai_matches = list(dict.fromkeys(ai_matches))

        # Decision Logic
        if ai_matches:
            evidence.extend([f"AI generator trace found: {m}" for m in ai_matches[:5]])
            if len(ai_matches) > 5:
                evidence.append(f"...and {len(ai_matches) - 5} more AI metadata matches.")
            return {
                "score": 0.98,
                "confidence": "high",
                "evidence": evidence,
                "raw_metadata": raw_info,
            }

        if camera_signals >= 3:
            # Genuine camera EXIF present: Mild signal toward real, never absolute proof
            evidence.append(
                f"Camera hardware EXIF detected ({camera_signals} tags: {', '.join(camera_details[:4])}). "
                "Consistent with a physical camera capture, though EXIF can be spoofed or transplanted."
            )
            # Score ~0.15 indicates lean toward real
            return {
                "score": 0.15,
                "confidence": "medium",
                "evidence": evidence,
                "raw_metadata": raw_info,
            }

        # If no camera EXIF and no AI tags: Neutral!
        # Remember honesty rule: Missing metadata is neutral, not suspicious.
        evidence.append(
            "No camera EXIF or AI generation metadata detected. "
            "This is neutral: messaging apps, web platforms, and screenshot utilities routinely strip metadata."
        )
        return {
            "score": None,
            "confidence": "low",
            "evidence": evidence,
            "raw_metadata": raw_info,
        }

    except Exception as exc:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [f"Metadata extraction failed or file format not parsed: {str(exc)}"],
            "raw_metadata": {},
        }
    finally:
        if close_img and img:
            try:
                img.close()
            except Exception:
                pass
