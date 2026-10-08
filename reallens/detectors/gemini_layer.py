"""Optional Gemini visual-artifact reasoning layer for RealLens.

If GEMINI_API_KEY is configured in the environment or .env, uses multimodal
reasoning to detect subtle semantic and geometric AI artifacts (e.g., anatomical
anomalies, unnatural lighting, impossible perspective). If the key is not
provided, gracefully returns null/disabled without affecting the rest of the pipeline.
"""
from __future__ import annotations

import io
import json
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

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()


def analyze_gemini(
    image_source: Union[str, Path, bytes, Image.Image]
) -> Dict[str, Any]:
    """Analyzes an image using the Gemini API if GEMINI_API_KEY is present.

    Returns:
        dict:
            score: float | None (0.0=real, 1.0=AI, None=disabled or error)
            confidence: "low" | "medium" | "high"
            evidence: list of str
    """
    api_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if not api_key:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [
                "Gemini visual reasoning is inactive (GEMINI_API_KEY not configured in .env). "
                "Local forensic detectors operate fully independently."
            ],
        }

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
                "evidence": ["Unsupported image input type for Gemini analysis."],
            }

        # Downsample if image is excessively large to optimize bandwidth
        max_dim = 1024
        w, h = img.size
        if max(w, h) > max_dim:
            scale = max_dim / max(w, h)
            work_img = img.resize((int(w * scale), int(h * scale)), Image.Resampling.LANCZOS)
        else:
            work_img = img.copy()

        if work_img.mode != "RGB":
            work_img = work_img.convert("RGB")

        # Attempt to import google-genai or google.generativeai
        genai_client = None
        try:
            from google import genai
            genai_client = genai.Client(api_key=api_key)
            use_v2_sdk = True
        except ImportError:
            try:
                import google.generativeai as legacy_genai
                legacy_genai.configure(api_key=api_key)
                use_v2_sdk = False
            except ImportError:
                return {
                    "score": None,
                    "confidence": "low",
                    "evidence": ["google-genai library not installed. Gemini reasoning skipped."],
                }

        prompt = (
            "You are an expert digital forensics investigator specializing in distinguishing AI-generated images "
            "from authentic camera photographs. Carefully inspect this image for visual cues:\n"
            "- AI cues: irregular anatomy (hands, teeth, eyes), melting textures, unnatural symmetry/depth-of-field, "
            "impossible lighting/shadows, glossiness, prompt-leak artifacts, or garbled background text.\n"
            "- Photographic cues: natural optical sensor noise, genuine lens depth, chromatic aberration, coherent texture detail.\n\n"
            "Return ONLY a valid JSON object with this exact structure:\n"
            "{\n"
            '  "ai_probability": <float between 0.0 and 1.0>,\n'
            '  "confidence": "low" | "medium" | "high",\n'
            '  "observations": [<list of 2-4 specific visual observations as strings>]\n'
            "}"
        )

        response_text = ""
        if use_v2_sdk and genai_client:
            # Using current google-genai SDK
            response = genai_client.models.generate_content(
                model="gemini-2.5-flash",
                contents=[prompt, work_img],
            )
            response_text = response.text or ""
        else:
            # Legacy SDK fallback
            model = legacy_genai.GenerativeModel("gemini-1.5-flash")
            response = model.generate_content([prompt, work_img])
            response_text = response.text or ""

        # Extract JSON from markdown fences if present
        json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group(0))
            score = float(data.get("ai_probability", 0.5))
            score = max(0.0, min(1.0, score))
            confidence = str(data.get("confidence", "medium")).lower()
            if confidence not in ("low", "medium", "high"):
                confidence = "medium"
            observations = data.get("observations", [])
            evidence = [f"Gemini visual reasoning: {obs}" for obs in observations]
            if not evidence:
                evidence = [f"Gemini assessed visual cues with score {score:.2f}."]

            return {
                "score": round(score, 3),
                "confidence": confidence,
                "evidence": evidence,
            }
        else:
            return {
                "score": 0.5,
                "confidence": "low",
                "evidence": [f"Gemini response could not be parsed as JSON: {response_text[:200]}"],
            }

    except Exception as exc:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [f"Gemini API analysis skipped or failed: {str(exc)}"],
        }
    finally:
        if close_img and img:
            try:
                img.close()
            except Exception:
                pass
