"""Flask backend application for RealLens.

Provides REST endpoints for analyzing images with provenance, forensic ELA,
FFT spectral decomposition, and machine learning ensemble classifiers.
"""
from __future__ import annotations

import concurrent.futures
import hashlib
import io
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

from flask import Flask, jsonify, redirect, render_template, request, url_for
from flask_cors import CORS
from PIL import Image, ImageOps

# Ensure reallens root is in sys.path
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from reallens.detectors.c2pa_check import analyze_c2pa
from reallens.detectors.condition import detect_image_condition
from reallens.detectors.ela import analyze_ela
from reallens.detectors.frequency import analyze_frequency
from reallens.detectors.fusion import fuse_signals
from reallens.detectors.gemini_layer import analyze_gemini
from reallens.detectors.gemini_semantic import analyze_gemini_semantic
from reallens.detectors.image_type import classify_image_type
from reallens.detectors.metadata import analyze_metadata
from reallens.detectors.ml_ensemble import MLEnsemble
from reallens.detectors.screenshot_cropper import detect_and_crop_screenshot
from reallens.detectors.watermarks import detect_watermarks

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

# Singleton ensemble instance and SHA-256 result cache
_ensemble_instance: MLEnsemble | None = None
_ANALYSIS_CACHE: Dict[str, Dict[str, Any]] = {}


def get_ensemble() -> MLEnsemble:
    global _ensemble_instance
    if _ensemble_instance is None:
        logger.info("Initializing ML Ensemble...")
        _ensemble_instance = MLEnsemble()
    return _ensemble_instance


def warmup_models() -> None:
    """Eagerly warms up ML ensemble, image-type gate, and fusion pipeline on server startup."""
    logger.info("Eagerly warming up ML models, learned gate, and fusion pipeline...")
    _clean_stale_cache()
    t_start = time.perf_counter()
    try:
        ens = get_ensemble()
        dummy_img = Image.new("RGB", (224, 224), (128, 128, 128))
        _ = ens.analyze(dummy_img)
        dur = round(time.perf_counter() - t_start, 2)
        logger.info(f"Model warmup completed in {dur}s. Pipeline ready for instant inference.")
    except Exception as exc:
        logger.warning(f"Model warmup failed: {exc}")


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


PIPELINE_VERSION = "v5.0.0"


def _clean_stale_cache():
    """Purges any cache entries not matching current PIPELINE_VERSION."""
    keys_to_remove = [k for k in list(_ANALYSIS_CACHE.keys()) if not k.endswith(f":{PIPELINE_VERSION}")]
    for k in keys_to_remove:
        _ANALYSIS_CACHE.pop(k, None)


def analyze_image_bytes(raw_bytes: bytes, filename: str = "upload.jpg", ignore_cache: bool = False) -> Dict[str, Any]:
    """Runs all detectors and fuses the resulting signals with SHA-256 caching and concurrency."""
    return _process_image_full(raw_bytes, filename=filename, ignore_cache=ignore_cache)


def _process_image_full(raw_bytes: bytes, filename: str, ignore_cache: bool = False) -> Dict[str, Any]:
    """Runs the full forensic detection pipeline on raw image bytes."""
    t0 = time.perf_counter()

    # 1. Versioned Cache Check
    file_hash = hashlib.sha256(raw_bytes).hexdigest()
    cache_key = f"{file_hash}:{PIPELINE_VERSION}"

    if not ignore_cache and cache_key in _ANALYSIS_CACHE:
        cached_result = dict(_ANALYSIS_CACHE[cache_key])
        cached_result["filename"] = filename
        cached_result["cached"] = True
        orig_time = cached_result.get("original_time_taken_seconds", cached_result.get("time_taken_seconds", 0.5))
        cached_result["original_time_taken_seconds"] = orig_time
        cached_result["time_taken_seconds"] = orig_time
        cached_result["cache_served_seconds"] = round(time.perf_counter() - t0, 4)
        return cached_result

    # 2. Sanitize PIL image for vision detectors
    pil_img, _ = sanitize_image_bytes(raw_bytes)

    # 2b. Screenshot / Letterbox Cropper (prevents UI frames from corrupting pixel models)
    crop_info = detect_and_crop_screenshot(pil_img)
    vision_img = crop_info["cropped_image"] if crop_info.get("auto_cropped") else pil_img

    # Downscale copy for ELA & FFT if >1024px to keep processing fast
    if max(vision_img.width, vision_img.height) > 1024:
        forensic_img = vision_img.copy()
        forensic_img.thumbnail((1024, 1024), Image.Resampling.BILINEAR)
    else:
        forensic_img = vision_img

    # 3. Learned image type classification gate (fast: ~15ms)
    img_type_res = classify_image_type(vision_img)
    detected_type = img_type_res.get("image_type", "photo-like")

    # 4. Concurrent execution of forensic detectors & ML Ensemble
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        fut_meta = executor.submit(analyze_metadata, raw_bytes)
        fut_c2pa = executor.submit(analyze_c2pa, raw_bytes)
        fut_ela = executor.submit(analyze_ela, forensic_img, image_type=detected_type)
        fut_freq = executor.submit(analyze_frequency, forensic_img, image_type=detected_type)
        fut_watermark = executor.submit(detect_watermarks, vision_img)

        # ML Ensemble runs on main thread (utilizing PyTorch intra-op threads)
        ensemble = get_ensemble()
        ml_res = ensemble.analyze(vision_img)

        meta_res = fut_meta.result()
        c2pa_res = fut_c2pa.result()
        ela_res = fut_ela.result()
        freq_res = fut_freq.result()
        watermark_res = fut_watermark.result()

    # 5. Image Condition Detection (WhatsApp, Screenshot, JPEG recompressed, Original)
    condition_res = detect_image_condition(
        pil_img,
        filename=filename,
        metadata_res=meta_res,
        crop_info=crop_info,
        image_type_res=img_type_res,
    )

    # 6. Gemini semantic analysis (checked persistently via data/gemini_semantic_cache.json)
    gemini_res = None
    if os.environ.get("GEMINI_API_KEY", "").strip():
        try:
            gemini_res = analyze_gemini_semantic(vision_img)
        except Exception as exc:
            logger.warning(f"Gemini semantic analysis failed: {exc}")

    # 7. Signal Fusion
    fused = fuse_signals(
        metadata_res=meta_res,
        c2pa_res=c2pa_res,
        ela_res=ela_res,
        freq_res=freq_res,
        ml_res=ml_res,
        gemini_res=gemini_res,
        image_type_res=img_type_res,
        watermark_res=watermark_res,
        condition_res=condition_res,
        crop_info=crop_info,
    )

    elapsed = round(time.perf_counter() - t0, 3)
    fused["filename"] = filename
    fused["image_dimensions"] = f"{pil_img.width}x{pil_img.height}"
    if crop_info.get("auto_cropped"):
        fused["cropped_dimensions"] = f"{vision_img.width}x{vision_img.height}"
    fused["time_taken_seconds"] = elapsed
    fused["original_time_taken_seconds"] = elapsed
    fused["cached"] = False
    fused["file_hash"] = file_hash
    fused["pipeline_version"] = PIPELINE_VERSION

    # Aliases for UI backwards compatibility & convenience
    fused["image_type"] = img_type_res.get("image_type", "photo-like")
    fused["photo_probability"] = img_type_res.get("photo_probability", 1.0)
    fused["is_photo_like"] = img_type_res.get("is_photo_like", True)
    fused["ai_probability"] = fused["probability"]
    fused["ai_percentage"] = round(fused["probability"] * 100, 1)
    fused["min_model_score"] = fused.get("min_score", fused["probability"])
    fused["max_model_score"] = fused.get("max_score", fused["probability"])
    fused["summary_reason"] = fused["summary"]
    fused["evidence_pointing_to_ai"] = fused["evidence_ai"]
    fused["evidence_pointing_to_real"] = fused["evidence_real"]
    fused["missing_or_unreliable_evidence"] = fused["evidence_unreliable"]
    fused["probability_display"] = f"{fused['ai_percentage']:.1f}%"
    fused["per_model_scores"] = fused["model_scores"]
    fused["per_model_weights"] = fused["model_weights"]

    serializable = to_serializable(fused)
    _ANALYSIS_CACHE[cache_key] = serializable
    return serializable


@app.route("/", methods=["GET"])
def index():
    """Renders the main single-page application."""
    return render_template("index.html")


@app.route("/batch", methods=["GET"])
def batch_redirect():
    """Redirects legacy /batch route to the single unified page / as required by Phase 3."""
    return redirect("/", code=302)


@app.route("/health", methods=["GET"])
def health():
    """Returns system status, active device, and model registry."""
    ensemble = get_ensemble()
    return jsonify({
        "status": "healthy",
        "pipeline_version": PIPELINE_VERSION,
        "device": ensemble.device_str,
        "models_configured": ensemble.model_names,
        "models_loaded": list(ensemble.loaded_models.keys()),
        "gemini_active": bool(os.environ.get("GEMINI_API_KEY", "").strip()),
        "cache_entries": len(_ANALYSIS_CACHE),
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

        ignore_cache = (
            request.args.get("ignore_cache", "").lower() in ("1", "true", "yes")
            or request.form.get("ignore_cache", "").lower() in ("1", "true", "yes")
        )
        result = analyze_image_bytes(raw_bytes, filename=file.filename, ignore_cache=ignore_cache)
        return jsonify({"status": "success", "result": result})

    except Exception as exc:
        logger.exception("Error processing image upload")
        return jsonify({"status": "error", "message": f"Failed to analyze image: {str(exc)}"}), 500


@app.route("/analyze_gemini", methods=["POST"])
def analyze_gemini_endpoint():
    """On-demand endpoint to run Gemini multimodal reasoning on a specific image."""
    if "image" not in request.files:
        return jsonify({"status": "error", "message": "No 'image' provided."}), 400

    file = request.files["image"]
    raw_bytes = file.read()
    if len(raw_bytes) < 32:
        return jsonify({"status": "error", "message": "File is empty or corrupted."}), 400

    api_key_configured = bool(os.environ.get("GEMINI_API_KEY", "").strip())
    if not api_key_configured:
        return jsonify({
            "status": "success",
            "api_key_configured": False,
            "message": "Gemini API key not configured in .env",
            "gemini": {
                "score": None,
                "confidence": "low",
                "evidence": ["Gemini API key not configured in .env. Local detectors run fully independently."],
            },
        })

    try:
        pil_img, _ = sanitize_image_bytes(raw_bytes)
        gemini_res = analyze_gemini_semantic(pil_img, force_refresh=True)
        return jsonify({
            "status": "success",
            "api_key_configured": True,
            "gemini": to_serializable(gemini_res),
        })
    except Exception as exc:
        logger.exception("Error during on-demand Gemini analysis")
        return jsonify({"status": "error", "message": f"Gemini analysis error: {str(exc)}"}), 500


@app.route("/batch_analyze", methods=["POST"])
def batch_analyze_endpoint():
    """Analyzes a batch of uploaded image files, returning full results for each image."""
    files = request.files.getlist("images")
    if not files:
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
                **res,
            })
        except Exception as exc:
            results.append({
                "filename": file.filename,
                "status": "error",
                "message": str(exc),
            })

    return jsonify({"status": "success", "total": len(results), "results": results})


# Eagerly initialize models on module import / startup
warmup_models()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    logger.info(f"Starting RealLens server on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
