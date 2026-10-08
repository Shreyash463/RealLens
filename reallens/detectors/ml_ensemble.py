"""Machine Learning Ensemble detector for RealLens.

Loads a curated ensemble of open Hugging Face image classification models:
1. `umm-maybe/AI-image-detector` (ViT-base)
2. `dima806/ai_vs_real_image_detection` (ViT-base fine-tuned on CIFAKE)
3. `Ateeqq/ai-vs-human-image-detector` (SigLIP architecture)

Supports GPU (CUDA) if available, with automatic CPU fallback, Test-Time
Augmentation (TTA), and robust error handling per model.
"""
from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# List of 4 verified ensemble models (including generalizing DINOv2 probe)
DEFAULT_MODELS = [
    "dima806/ai_vs_real_image_detection",
    "Organika/sdxl-detector",
    "Ateeqq/ai-vs-human-image-detector",
    "generalizing_dinov2_probe",
]


class MLEnsemble:
    """Ensemble manager for Hugging Face AI vs Real classification models."""

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        device: Optional[str] = None,
        use_tta: bool = False,
    ):
        self.model_names = model_names or list(DEFAULT_MODELS)
        self.use_tta = use_tta
        self.loaded_models: Dict[str, Tuple[Any, Any, int]] = {}  # name -> (model, processor, ai_idx)
        self.failed_models: Dict[str, str] = {}

        # Device selection: automatic CUDA if available, else CPU
        if device:
            self.device_str = device
        else:
            try:
                import torch
                self.device_str = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                self.device_str = "cpu"

        logger.info(f"Initialized MLEnsemble on device: {self.device_str}")

    def _determine_ai_index(self, config: Any, model_name: str) -> int:
        """Determines which class index corresponds to AI-generated images."""
        # Empirically verified checkpoints where Hugging Face config id2label is known/inverted
        if "dima806" in model_name:
            return 0  # In dima806 checkpoint, index 0 outputs AI/FAKE probability
        if "sdxl-detector" in model_name or "Organika" in model_name:
            return 1  # In Organika checkpoint, index 1 outputs synthetic probability
        if "Ateeqq" in model_name:
            return 0  # In Ateeqq checkpoint, index 0 outputs AI probability

        id2label = getattr(config, "id2label", None) or {}
        ai_terms = ["artificial", "fake", "ai", "synthetic", "generated"]
        real_terms = ["human", "real", "authentic", "natural", "photo", "hum"]

        for idx, label_raw in id2label.items():
            idx_int = int(idx)
            label = str(label_raw).lower()
            if any(term in label for term in ai_terms):
                return idx_int

        for idx, label_raw in id2label.items():
            idx_int = int(idx)
            label = str(label_raw).lower()
            if any(term in label for term in real_terms):
                return 1 - idx_int  # The other class in binary classification

        return 0  # default assumption

    def load_model(self, model_name: str) -> bool:
        """Loads an individual model and processor, caching it."""
        if model_name in self.loaded_models:
            return True
        if model_name in self.failed_models:
            return False

        try:
            import torch
            from transformers import AutoImageProcessor, AutoModel, AutoModelForImageClassification

            if "probe" in model_name:
                import joblib
                probe_path = Path(__file__).resolve().parent.parent.parent / "models" / "generalizing_probe.joblib"
                if not probe_path.exists():
                    logger.warning(f"Probe weights not found at {probe_path}")
                    return False
                pkg = joblib.load(probe_path)
                backbone_id = pkg.get("backbone_id", "facebook/dinov2-small")
                logger.info(f"Loading generalizing probe backbone '{backbone_id}'...")
                processor = AutoImageProcessor.from_pretrained(backbone_id)
                model = AutoModel.from_pretrained(backbone_id).to(self.device_str)
                model.eval()
                self.loaded_models[model_name] = (model, processor, pkg)
                logger.info(f"Successfully loaded '{model_name}'.")
                return True

            logger.info(f"Loading model '{model_name}' on {self.device_str}...")
            processor = AutoImageProcessor.from_pretrained(model_name)
            model = AutoModelForImageClassification.from_pretrained(model_name)
            model.to(self.device_str)
            model.eval()

            ai_idx = self._determine_ai_index(model.config, model_name)
            self.loaded_models[model_name] = (model, processor, ai_idx)
            logger.info(f"Successfully loaded '{model_name}' (AI index: {ai_idx}).")
            return True
        except Exception as exc:
            logger.error(f"Failed to load model '{model_name}': {exc}")
            self.failed_models[model_name] = str(exc)
            return False

    def load_all(self) -> None:
        """Eagerly loads all configured models."""
        for name in self.model_names:
            self.load_model(name)

    def predict_single_model(
        self,
        model_name: str,
        image: Image.Image,
        use_tta: Optional[bool] = None,
    ) -> Optional[float]:
        """Runs inference for a single model on a PIL image."""
        if model_name not in self.loaded_models:
            if not self.load_model(model_name):
                return None

        tta = self.use_tta if use_tta is None else use_tta

        try:
            import torch

            model, processor, extra = self.loaded_models[model_name]

            images_to_run = [image]
            if tta:
                images_to_run.append(image.transpose(Image.Transpose.FLIP_LEFT_RIGHT))

            inputs = processor(images=images_to_run, return_tensors="pt")
            inputs = {k: v.to(self.device_str) for k, v in inputs.items()}

            if "probe" in model_name:
                pkg = extra
                with torch.no_grad():
                    outputs = model(**inputs)
                    cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                cls_scaled = pkg["scaler"].transform(cls_emb)
                probs = pkg["classifier"].predict_proba(cls_scaled)[:, 1]
                avg_ai_prob = float(np.mean(probs))
                return max(0.0, min(1.0, avg_ai_prob))

            ai_idx = extra
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)

            # Average over TTA passes
            ai_probs = probs[:, ai_idx].cpu().numpy()
            avg_ai_prob = float(np.mean(ai_probs))
            return max(0.0, min(1.0, avg_ai_prob))
        except Exception as exc:
            logger.error(f"Inference error on '{model_name}': {exc}")
            return None

    def analyze(
        self,
        image_source: Union[str, Path, bytes, Image.Image],
        use_tta: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """Runs the ensemble across all available models.

        Returns:
            dict:
                score: float | None (mean AI probability)
                confidence: "low" | "medium" | "high"
                evidence: list of str
                per_model_scores: dict mapping model name to probability
                spread: float (max - min)
                models_active: int
                device: str
        """
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
                    "evidence": ["Unsupported image input type for ML ensemble."],
                    "per_model_scores": {},
                    "spread": 0.0,
                    "models_active": 0,
                    "device": self.device_str,
                }

            # Convert to RGB safely
            if img.mode != "RGB":
                # For transparency, paste on white background
                if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
                    rgba = img.convert("RGBA")
                    bg = Image.new("RGB", rgba.size, (255, 255, 255))
                    bg.paste(rgba, mask=rgba.split()[3])
                    work_img = bg
                else:
                    work_img = img.convert("RGB")
            else:
                work_img = img.copy()

            per_model_scores: Dict[str, Optional[float]] = {}
            valid_scores: List[float] = []
            evidence: List[str] = []

            for name in self.model_names:
                score = self.predict_single_model(name, work_img, use_tta=use_tta)
                per_model_scores[name] = round(score, 4) if score is not None else None
                if score is not None:
                    valid_scores.append(score)
                    short_name = name.split("/")[-1]
                    evidence.append(f"Model '{short_name}': {score * 100:.1f}% AI probability.")
                else:
                    err = self.failed_models.get(name, "Inference failed")
                    evidence.append(f"Model '{name.split('/')[-1]}' unavailable: {err[:60]}.")

            if not valid_scores:
                return {
                    "score": None,
                    "confidence": "low",
                    "evidence": ["All ML models failed to load or run. Local heuristics will be relied upon."],
                    "per_model_scores": per_model_scores,
                    "spread": 0.0,
                    "models_active": 0,
                    "device": self.device_str,
                }

            # Use robust median aggregation when 3 models are present (filters out single-model outlier false alarms)
            if len(valid_scores) >= 3:
                ensemble_score = float(np.median(valid_scores))
            else:
                ensemble_score = float(np.mean(valid_scores))

            spread = float(max(valid_scores) - min(valid_scores)) if len(valid_scores) > 1 else 0.0

            if len(valid_scores) >= 2 and spread > 0.45:
                confidence = "low"
                evidence.append(
                    f"Warning: Disagreement among ML models (spread: {spread:.2f}). Predictions exhibit high uncertainty."
                )
            elif len(valid_scores) >= 2 and spread < 0.18:
                confidence = "high"
                evidence.append(f"Strong consensus among models (spread: {spread:.2f}).")
            else:
                confidence = "medium"

            return {
                "score": round(ensemble_score, 3),
                "confidence": confidence,
                "evidence": evidence,
                "per_model_scores": per_model_scores,
                "spread": round(spread, 3),
                "models_active": len(valid_scores),
                "device": self.device_str,
            }

        except Exception as exc:
            return {
                "score": None,
                "confidence": "low",
                "evidence": [f"ML Ensemble execution error: {str(exc)}"],
                "per_model_scores": {},
                "spread": 0.0,
                "models_active": 0,
                "device": self.device_str,
            }
        finally:
            if close_img and img:
                try:
                    img.close()
                except Exception:
                    pass
