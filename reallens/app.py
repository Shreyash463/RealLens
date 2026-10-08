"""Flask backend application for RealLens.

Provides REST endpoints for analyzing images with provenance, forensic ELA,
FFT spectral decomposition, and machine learning ensemble classifiers.
"""
from __future__ import annotations

import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, render_template, request
from flask_cors import CORS
from PIL import Image, ImageOps

# Ensure reallens root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reallens.detectors.c2pa_check import analyze_c2pa
from reallens.detectors.ela import analyze_ela
from reallens.detectors.frequency import analyze_frequency
from reallens.detectors.fusion import fuse_signals
from reallens.detectors.gemini_layer import analyze_gemini
from reallens.detectors.image_type import classify_image_type
from reallens.detectors.metadata import analyze_metadata
from reallens.detectors.ml_ensemble import MLEnsemble

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("reallens.app")

# Initialize Flask app
app = Flask(
    __name__,
    static_folder=str(BASE_DIR / "static"),
    template_folder=str(BASE_DIR / "templates"),
)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 25 * 1024 * 1024  # 25 MB max upload

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".gif", ".bmp", ".tiff", ".tif"}

# Lazy or eager ML ensemble singleton
_ensemble_instance: MLEnsemble | None = None


def get_ensemble() -> MLEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        logger.info("Initializing ML Ensemble...")
        _ensemble_instance = MLEnsemble()
    return _ensemble_instance


def sanitize_image_bytes(raw_bytes: bytes) -> tuple[Image.Image, bytes]:
    """Validates and sanitizes image bytes, handling animated files, transparency, and orientations."""
    img = Image.open(io.BytesIO(raw_bytes))

    # For animated GIFs or multi-frame WebPs, seek to the first frame
    if getattr(img, "is_animated", False):
        img.seek(0)

    # Correct EXIF orientation transpose if present
    try:
        img = ImageOps.exif_transpose(img)
    except Exception:
        pass

    # Ensure RGB conversion for downstream models
    if img.mode != "RGB":
        if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
            rgba = img.convert("RGBA")
            bg = Image.new("RGB", rgba.size, (255, 255, 255))
            bg.paste(rgba, mask=rgba.split()[3])
            clean_img = bg
        else:
            clean_img = img.convert("RGB")
    else:
        clean_img = img.copy()

    return clean_img, raw_bytes


def to_serializable(val: Any) -> Any:
    """Recursively converts numpy scalars and non-standard types to JSON-serializable Python types."""
    if isinstance(val, dict):
        return {str(k): to_serializable(v) for k, v in val.items()}
    if isinstance(val, (list, tuple)):
        return [to_serializable(v) for v in val]
    if hasattr(val, "item") and callable(getattr(val, "item")):
        try:
            return val.item()
        except Exception:
            pass
    if isinstance(val, (float, int, str, bool)) or val is None:
        return val
    return str(val)


def analyze_image_bytes(raw_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
    """Runs all detectors and fuses the resulting signals."""
    t0 = time.perf_counter()

    # 1. Metadata check on raw input
    meta_res = analyze_metadata(raw_bytes)

    # 2. C2PA provenance check
    c2pa_res = analyze_c2pa(raw_bytes)

    # 3. Sanitize PIL image for vision detectors
    pil_img, _ = sanitize_image_bytes(raw_bytes)

    # 3b. Image type classification gate
    img_type_res = classify_image_type(pil_img)

    # 4. Error Level Analysis (with image type context)
    ela_res = analyze_ela(pil_img, image_type=img_type_res.get("image_type", "photo-like"))

    # 5. FFT Frequency domain analysis
    freq_res = analyze_frequency(pil_img)

    # 6. ML Ensemble classification
    ensemble = get_ensemble()
    ml_res = ensemble.analyze(pil_img)

    # 7. Optional Gemini multimodal visual reasoning
    gemini_res = analyze_gemini(pil_img)

    # 8. Signal Fusion (with image type gate & spread inspection)
    fused = fuse_signals(
        metadata_res=meta_res,
        c2pa_res=c2pa_res,
        ela_res=ela_res,
        freq_res=freq_res,
        ml_res=ml_res,
        gemini_res=gemini_res,
        image_type_res=img_type_res,
    )

    elapsed = round(time.perf_counter() - t0, 3)
    fused["filename"] = filename
    fused["image_dimensions"] = f"{pil_img.width}x{pil_img.height}"
    fused["time_taken_seconds"] = elapsed
    return to_serializable(fused)


@app.route("/", methods=["GET"])
def index():
    """Renders the main single-page application."""
    return render_template("index.html")


@app.route("/batch", methods=["GET"])
def batch_page():
    """Renders the batch test analysis page."""
    return render_template("batch.html")


@app.route("/health", methods=["GET"])
def health():
    """Returns system status, active device, and model registry."""
    ensemble = get_ensemble()
    return jsonify({
        "status": "healthy",
        "device": ensemble.device_str,
        "models_configured": ensemble.model_names,
        "models_loaded": list(ensemble.loaded_models.keys()),
        "gemini_active": bool(os.environ.get("GEMINI_API_KEY")),
    })


@app.route("/analyze", methods=["POST"])
def analyze_endpoint():
    """Analyzes an uploaded image file."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No 'image' file provided in form data."}), 400

    file = request.files["image"]
    if not file or not file.filename:
        return jsonify({"status": "error", "message": "Uploaded file is empty or missing filename."}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({
            "status": "error",
            "message": f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        }), 400

    try:
        raw_bytes = file.read()
        if len(raw_bytes) < 32:
            return jsonify({"status": "error", "message": "File is empty or corrupted."}), 400

        result = analyze_image_bytes(raw_bytes, filename=file.filename)
        return jsonify({"status": "success", "result": result})

    except Exception as exc:
        logger.exception("Error processing image upload")
        return jsonify({"status": "error", "message": f"Failed to analyze image: {str(exc)}"}), 500


@app.route("/batch_analyze", methods=["POST"])
def batch_analyze_endpoint():
    """Analyzes a batch of uploaded image files."""
    files = request.files.getlist("images")
    if not files:
        # Check single 'image' fallback
        if "image" in request.files:
            files = [request.files["image"]]
        else:
            return jsonify({"status": "error", "message": "No 'images' provided."}), 400

    results = []
    for file in files:
        if not file or not file.filename:
            continue
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": f"Unsupported format '{ext}'",
            })
            continue

        try:
            raw_bytes = file.read()
            if len(raw_bytes) < 32:
                results.append({
                    "filename": file.filename,
                    "status": "error",
                    "message": "File corrupted or empty",
                })
                continue
            res = analyze_image_bytes(raw_bytes, filename=file.filename)
            results.append({
                "filename": file.filename,
                "status": "success",
                "ai_percentage": res.get("ai_percentage"),
                "probability_display": res.get("probability_display"),
                "verdict": res.get("verdict"),
                "confidence": res.get("confidence"),
                "is_inconclusive": res.get("is_inconclusive", False),
                "image_type": res.get("image_type", "photo-like"),
                "is_photo_like": res.get("is_photo_like", True),
                "spread": res.get("spread", 0.0),
                "min_model_score": res.get("min_model_score"),
                "max_model_score": res.get("max_model_score"),
                "time_taken_seconds": res.get("time_taken_seconds"),
                "image_dimensions": res.get("image_dimensions"),
                "summary": res.get("summary_reason"),
                "why_this_verdict": res.get("why_this_verdict"),
                "unreliable_notice": res.get("unreliable_notice"),
                "disagreement_flag": res.get("disagreement_flag"),
                "signals": res.get("signals"),
            })
        except Exception as exc:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(exc),
            })

    return jsonify({"status": "success", "total": len(results), "results": results})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting RealLens server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
