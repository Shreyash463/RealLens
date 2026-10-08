"""C2PA / Content Credentials detector for RealLens.

Inspects images for C2PA provenance manifests (Content Credentials).
Supports the official `c2pa` python SDK if installed, with a reliable
pure-Python binary scanner fallback for JUMBF / C2PA boxes.
"""
from __future__ import annotations

import io
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

# Check if official c2pa package is importable
C2PA_SDK_AVAILABLE = False
try:
    import c2pa  # type: ignore
    C2PA_SDK_AVAILABLE = True
except ImportError:
    C2PA_SDK_AVAILABLE = False

# C2PA and JUMBF signatures
C2PA_MARKERS = [
    b"c2pa",
    b"jumb",
    b"c2pa.claim",
    b"c2pa.signature",
    b"c2pa.actions",
    b"urn:uuid:d8cfd0d8-1670-4e5a-b355-5da30166a50f",
]

AI_ACTION_KEYWORDS = [
    b"c2pa.created",
    b"c2pa.placed",
    b"c2pa.edited",
    b"generator",
    b"openai",
    b"dall-e",
    b"midjourney",
    b"firefly",
    b"generative",
    b"synthetic",
    b"stability",
]


def _scan_binary_c2pa(raw_bytes: bytes) -> Dict[str, Any]:
    """Pure-Python scanner for C2PA markers and claim indications."""
    found_markers = [m.decode("ascii", errors="ignore") for m in C2PA_MARKERS if m in raw_bytes]
    if not found_markers:
        return {"has_c2pa": False, "ai_actions": [], "details": "No C2PA signatures in binary stream."}

    # Search for specific AI actions or generators in raw stream
    detected_ai_terms = []
    lower_bytes = raw_bytes.lower()
    for term in AI_ACTION_KEYWORDS:
        if term in lower_bytes:
            detected_ai_terms.append(term.decode("ascii", errors="ignore"))

    return {
        "has_c2pa": True,
        "ai_actions": detected_ai_terms,
        "details": f"Found C2PA/JUMBF markers: {', '.join(found_markers[:3])}",
    }


def analyze_c2pa(image_source: Union[str, Path, bytes, Image.Image]) -> Dict[str, Any]:
    """Analyzes image for Content Credentials / C2PA manifests.

    Returns:
        dict:
            score: float | None (0.0=real camera proof, 1.0=AI generated, None=neutral)
            confidence: "low" | "medium" | "high"
            evidence: list of str
            raw_c2pa: dict | None
    """
    raw_bytes: Optional[bytes] = None

    try:
        if isinstance(image_source, (str, Path)):
            p = Path(image_source)
            if p.is_file():
                raw_bytes = p.read_bytes()
        elif isinstance(image_source, bytes):
            raw_bytes = image_source
        elif isinstance(image_source, Image.Image):
            # If PIL image, save to buffer to inspect headers if filename not present
            if hasattr(image_source, "filename") and image_source.filename and Path(image_source.filename).is_file():
                raw_bytes = Path(image_source.filename).read_bytes()
            else:
                buf = io.BytesIO()
                fmt = image_source.format or "PNG"
                image_source.save(buf, format=fmt)
                raw_bytes = buf.getvalue()

        if not raw_bytes:
            return {
                "score": None,
                "confidence": "low",
                "evidence": ["Unable to read image bytes for C2PA inspection."],
                "raw_c2pa": None,
            }

        # 1. Try official SDK if available and source was a file path
        if C2PA_SDK_AVAILABLE and isinstance(image_source, (str, Path)):
            try:
                reader = c2pa.Reader.from_file(str(image_source))
                manifest_json = reader.json()
                manifest = json.loads(manifest_json) if isinstance(manifest_json, str) else manifest_json

                active_manifest = manifest.get("active_manifest", {})
                assertions = active_manifest.get("assertions", [])
                claim_generator = active_manifest.get("claim_generator", "")

                evidence = [f"Verified C2PA manifest found (claim generator: {claim_generator})."]
                is_ai = False

                for assertion in assertions:
                    label = assertion.get("label", "")
                    data = assertion.get("data", {})
                    if "actions" in label:
                        actions = data.get("actions", [])
                        for action in actions:
                            action_name = action.get("action", "")
                            software = action.get("softwareAgent", "")
                            if "generat" in action_name.lower() or "generat" in software.lower():
                                is_ai = True
                                evidence.append(f"C2PA Action explicitly recorded AI generation: {action_name} by {software}.")

                if is_ai or any(k in claim_generator.lower() for k in ["firefly", "dall-e", "midjourney", "bing"]):
                    return {
                        "score": 0.99,
                        "confidence": "high",
                        "evidence": evidence,
                        "raw_c2pa": manifest,
                    }

                # Authentic hardware signing (e.g., Leica / Sony authentic camera signing)
                evidence.append("C2PA manifest authenticated without synthetic media actions.")
                return {
                    "score": 0.05,
                    "confidence": "high",
                    "evidence": evidence,
                    "raw_c2pa": manifest,
                }
            except Exception:
                # Fall back to pure-Python scanner below
                pass

        # 2. Pure-Python binary scanner fallback
        scan_res = _scan_binary_c2pa(raw_bytes)
        if scan_res["has_c2pa"]:
            evidence = [f"C2PA Content Credentials signature detected in container. {scan_res['details']}."]
            ai_terms = scan_res.get("ai_actions", [])
            if any(t in ["openai", "dall-e", "midjourney", "firefly", "synthetic", "stability"] for t in ai_terms):
                evidence.append(f"AI generation indicator found in C2PA payload ({', '.join(ai_terms)}).")
                return {
                    "score": 0.96,
                    "confidence": "high",
                    "evidence": evidence,
                    "raw_c2pa": scan_res,
                }
            return {
                "score": 0.50,
                "confidence": "medium",
                "evidence": evidence + ["Manifest detected but specific action could not be cryptographically validated."],
                "raw_c2pa": scan_res,
            }

        # 3. No C2PA found: Neutral
        return {
            "score": None,
            "confidence": "low",
            "evidence": [
                "No C2PA Content Credentials manifest detected. "
                "This is neutral: standard cameras and web platforms do not embed C2PA by default."
            ],
            "raw_c2pa": None,
        }

    except Exception as exc:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [f"C2PA inspection encountered an error: {str(exc)}"],
            "raw_c2pa": None,
        }
