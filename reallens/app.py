"""Flask backend application for RealLens.

Provides REST endpoints for analyzing images with provenance, forensic ELA,
FFT spectral decomposition, and machine learning ensemble classifiers.
"""
from __future__ import annotations

import io
import logging
import os
import sys
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


def analyze_image_bytes(raw_bytes: bytes, filename: str = "upload.jpg") -> Dict[str, Any]:
    """Runs all detectors and fuses the resulting signals."""
    # 1. Metadata check on raw input
    meta_res = analyze_metadata(raw_bytes)

    # 2. C2PA provenance check
    c2pa_res = analyze_c2pa(raw_bytes)

    # 3. Sanitize PIL image for vision detectors
    pil_img, _ = sanitize_image_bytes(raw_bytes)

    # 4. Error Level Analysis
    ela_res = analyze_ela(pil_img)

    # 5. FFT Frequency domain analysis
    freq_res = analyze_frequency(pil_img)

    # 6. ML Ensemble classification
    ensemble = get_ensemble()
    ml_res = ensemble.analyze(pil_img)

    # 7. Optional Gemini multimodal visual reasoning
    gemini_res = analyze_gemini(pil_img)

    # 8. Signal Fusion
    fused = fuse_signals(
        metadata_res=meta_res,
        c2pa_res=c2pa_res,
        ela_res=ela_res,
        freq_res=freq_res,
        ml_res=ml_res,
        gemini_res=gemini_res,
    )

    fused["filename"] = filename
    fused["image_dimensions"] = f"{pil_img.width}x{pil_img.height}"
    return fused


@app.route("/", methods=["GET"])
def index():
    """Renders the main single-page application."""
    return render_template("index.html")


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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting RealLens server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
