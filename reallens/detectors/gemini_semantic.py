"""Gemini Multimodal Semantic & Common-Sense Forensic Inspector for RealLens.

Leverages Gemini 3.8 Flash (with fallback to 2.5 Flash) to detect subtle semantic,
physical, and anatomical AI generation artifacts that pixel-level models miss:
1. Anatomical anomalies (hands, fingers, teeth, ears, eye reflection symmetry).
2. Physics and lighting (impossible shadows, conflicting light sources, inconsistent reflections).
3. Text and logos (garbled text, pseudowords, distorted brand marks).
4. Background coherence (objects melting, repeated faces, impossible architecture).
5. Texture telltales (airbrushed skin, plastic sheen, overly uniform hair, brushstroke-less painting).

Caches results persistently by SHA-256 hash + prompt version in data/gemini_semantic_cache.json.
"""
from __future__ import annotations

import hashlib
import io
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

logger = logging.getLogger("gemini_semantic")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_FILE = BASE_DIR / "data" / "gemini_semantic_cache.json"
PROMPT_VERSION = "v5.0-structured-rubric"


def _get_cache() -> Dict[str, Any]:
    if CACHE_FILE.exists():
        try:
            with open(CACHE_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def _save_cache(cache: Dict[str, Any]):
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save Gemini cache: {e}")


def _compute_image_hash(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=90)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def analyze_gemini_semantic(
    image_source: Union[str, Path, bytes, Image.Image],
    force_refresh: bool = False
) -> Dict[str, Any]:
    """Runs structured multimodal forensic analysis on an image using Gemini.

    Returns:
        dict:
            score: float | None (0.0=authentic real, 1.0=AI synthetic, None=disabled/unavailable)
            confidence: 'high' | 'medium' | 'low'
            findings: list of dict {category, description, region_description, severity}
            counter_evidence: list of str
            cached: bool
            model_used: str
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "score": None,
            "confidence": "low",
            "findings": [],
            "counter_evidence": ["GEMINI_API_KEY not configured. Semantic AI artifact reasoning disabled."],
            "cached": False,
            "model_used": "none",
        }

    # Load image
    if isinstance(image_source, Image.Image):
        img = image_source.convert("RGB")
    elif isinstance(image_source, (str, Path)):
        img = Image.open(image_source).convert("RGB")
    elif isinstance(image_source, bytes):
        img = Image.open(io.BytesIO(image_source)).convert("RGB")
    else:
        return {
            "score": None,
            "confidence": "low",
            "findings": [],
            "counter_evidence": ["Invalid image source type."],
            "cached": False,
            "model_used": "none",
        }

    # Check cache
    img_hash = _compute_image_hash(img)
    cache_key = f"{img_hash}:{PROMPT_VERSION}"
    cache = _get_cache()
    if not force_refresh and cache_key in cache:
        res = dict(cache[cache_key])
        res["cached"] = True
        return res

    # Downscale for API payload if needed
    max_dim = 1024
    w, h = img.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        work_img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
    else:
        work_img = img.copy()

    # Client setup
    try:
        from google import genai
        client = genai.Client(api_key=api_key)
        sdk_v2 = True
    except ImportError:
        try:
            import google.generativeai as legacy_genai
            legacy_genai.configure(api_key=api_key)
            sdk_v2 = False
        except ImportError:
            return {
                "score": None,
                "confidence": "low",
                "findings": [],
                "counter_evidence": ["google-genai SDK not installed."],
                "cached": False,
                "model_used": "none",
            }

    system_instruction = (
        "You are an expert digital forensic image analyst specializing in detecting AI-generated images "
        "(Midjourney, SDXL, Flux, Gemini/Imagen, DALL-E) vs authentic photographic or physical camera captures.\n"
        "Evaluate the image across 5 specific artifact categories:\n"
        "1. Anatomy: hand finger count/joints, teeth symmetry, ear geometry, pupil shape, eye catchlight reflection matching.\n"
        "2. Physics & Lighting: impossible shadows, conflicting light vectors, reflection mismatches in glasses or water.\n"
        "3. Text & Typography: garbled glyphs, pseudowords, distorted logos, inconsistent font weights.\n"
        "4. Background Coherence: objects melting into surroundings, repeated crowd faces, impossible architectural perspective.\n"
        "5. Texture Telltales: unnatural airbrushed plastic skin, lack of camera sensor grain, overly smooth hair clumps, "
        "or digital painting textures lacking physical brush strokes/canvas weave.\n\n"
        "IMPORTANT: Distinguish between real physical paintings/art photographed by a camera vs digital AI paintings. "
        "A photographed physical painting will have real surface canvas texture, frame/edge lighting, or photographic sensor noise. "
        "An AI digital painting will have hyper-smooth synthetic blending, melting ornaments, or illogical details.\n\n"
        "Respond STRICTLY in valid JSON format matching this schema:\n"
        "{\n"
        '  "overall_ai_likelihood": <float 0.0 to 1.0>,\n'
        '  "confidence": "high" | "medium" | "low",\n'
        '  "findings": [\n'
        '    {\n'
        '      "category": "Anatomy" | "Lighting" | "Text" | "Background" | "Texture",\n'
        '      "description": "<detailed observation>",\n'
        '      "region_description": "<approximate location e.g. bottom-right, subject face, background>",\n'
        '      "severity": "high" | "medium" | "low"\n'
        '    }\n'
        "  ],\n"
        '  "counter_evidence": [<list of natural photographic or physical features supporting authenticity>]\n'
        "}"
    )

    models_to_try = ["gemini-3.7-flash", "gemini-flash-latest", "gemini-3.8-flash", "gemini-3.5-flash"]
    response_text = ""
    used_model = "unknown"

    for model_name in models_to_try:
        try:
            if sdk_v2:
                resp = client.models.generate_content(
                    model=model_name,
                    contents=[system_instruction, work_img],
                )
                response_text = resp.text or ""
                used_model = model_name
                break
            else:
                l_model = legacy_genai.GenerativeModel(model_name)
                resp = l_model.generate_content([system_instruction, work_img])
                response_text = resp.text or ""
                used_model = model_name
                break
        except Exception as e:
            logger.warning(f"Gemini model {model_name} failed: {e}")
            continue

    if not response_text:
        return {
            "score": None,
            "confidence": "low",
            "findings": [],
            "counter_evidence": ["Gemini multimodal inference failed across candidate models."],
            "cached": False,
            "model_used": "none",
        }

    # Extract JSON block
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(0))
            score = float(data.get("overall_ai_likelihood", 0.5))
            score = max(0.0, min(1.0, score))
            confidence = str(data.get("confidence", "medium")).lower()
            if confidence not in ("high", "medium", "low"):
                confidence = "medium"
            findings = data.get("findings", [])
            counter_evidence = data.get("counter_evidence", [])

            result = {
                "score": round(score, 3),
                "confidence": confidence,
                "findings": findings,
                "counter_evidence": counter_evidence,
                "cached": False,
                "model_used": used_model,
            }

            # Cache result
            cache[cache_key] = result
            _save_cache(cache)
            return result
        except Exception as parse_err:
            logger.error(f"Failed parsing Gemini JSON: {parse_err}")

    return {
        "score": 0.5,
        "confidence": "low",
        "findings": [],
        "counter_evidence": [f"Unformatted response: {response_text[:200]}"],
        "cached": False,
        "model_used": used_model,
    }
