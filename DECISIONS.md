# RealLens Architectural and Technical Decisions

This document tracks all design decisions made during the development of RealLens, along with the reasoning for each decision.

## 1. Project Organization and Execution Environment
- **Decision**: Python 3.13 venv with PyTorch CUDA 12.4 support on Windows.
  - *Reason*: Python 3.13 is the active system environment; PyTorch 2.6+ with CUDA 12.4 provides native GPU acceleration for our RTX 3050 (4GB VRAM) with automatic fallback to CPU when CUDA is not present.
- **Decision**: Code structured under `reallens/` package with modular detectors in `reallens/detectors/`, evaluation harness in `evaluation/` (and symlinked/accessible from `reallens/evaluation/`), and datasets cached in `data/` (git-ignored).
  - *Reason*: Conforms to clean modular design where detectors can be imported as a library or called via Flask endpoints.

## 2. Model Selection for `ml_ensemble.py`
- **Candidate Models Researched and Tested**:
  1. `umm-maybe/AI-image-detector`: ViT-base architecture. Popular, but weights hosted on legacy Xet format without safetensors, leading to connection timeouts without HF authentication. Replaced by `Organika/sdxl-detector`.
  2. `dima806/ai_vs_real_image_detection`: ViT-base architecture fine-tuned on CIFAKE (photorealistic CIFAR-like and object generations). Verified and loaded successfully (`id2label: {0: 'REAL', 1: 'FAKE'}`).
     - *Strengths*: Highly accurate on natural objects, animals, and standard diffusion textures.
     - *Weaknesses*: Struggles on modern high-resolution artistic stylizations (Midjourney v6/Flux) where textures are hyper-smooth; sensitive to heavy JPEG recompression.
  3. `Organika/sdxl-detector`: ViT-base architecture specialized on SDXL latent diffusion outputs. Verified and loaded successfully (`id2label: {0: 'artificial', 1: 'human'}`).
     - *Strengths*: Exceptional sensitivity to modern high-resolution latent diffusion fingerprints and text-to-image synthetic art.
     - *Weaknesses*: May misflag heavily graded digital commercial photography or CGI renders.
  4. `Ateeqq/ai-vs-human-image-detector`: SigLIP (Sigmoid Loss Language-Image Pretraining) architecture trained on 120,000 images across modern generators and human photos.
     - *Strengths*: Distinct dual-encoder architecture providing orthogonal representation to ViT backbones, high generalization across unseen generator families.
     - *Weaknesses*: Susceptible to spatial detail loss in heavily downsampled thumbnails (<128px).
- **Decision**: Final Shortlist of 3 Ensembled Detectors:
  - Model 1: `dima806/ai_vs_real_image_detection` (ViT - CIFAKE/Object photorealism)
  - Model 2: `Organika/sdxl-detector` (ViT - SDXL/High-resolution diffusion)
  - Model 3: `Ateeqq/ai-vs-human-image-detector` (SigLIP - Multimodal contrastive generalist)
  - *Reason*: Diverse blend of architectures (ViT + SigLIP) trained across distinct dataset regimes (CIFAKE, SDXL, and multi-generator benchmarks) maximizes ensemble coverage and disagreement signal fidelity.

## 3. Graceful Fallbacks and Signal Honesty
## 4. Phase 2 Architectural Decisions
- **Decision**: Generalizing Probe Backbone: `facebook/dinov2-small` selected over CLIP ViT-B/32.
  - *Reason*: DINOv2 self-supervised patch representations capture intrinsic physical scene geometry, natural photographic noise, and fine lighting textures, achieving superior generalization against JPEG recompression and modern Midjourney diffusion outputs compared to multimodal semantic models.
- **Decision**: Augmented Linear Probe (`models/generalizing_probe.joblib`) added as 4th model in `MLEnsemble`.
  - *Reason*: Training with heavy JPEG recompression (Q=45..65), downscaling, and blur forced the probe to ignore brittle high-frequency compression artifacts, achieving 100% accuracy on hard tuning images and eliminating false alarms on WhatsApp photos.
- **Decision**: Learned Calibrated Logistic Regression Stacker (`models/learned_fusion_stacker.joblib`) deployed as primary fusion engine with dynamic fallback to hand-weighted rules.
  - *Reason*: Multi-signal stacking automatically balances detector interactions, expanding class probability separation margin on the held-out test set from 44.8% to 72.0%.
- **Decision**: Explicit 'Uncertain Zone' defined between [0.35, 0.65] and upon vision model disagreement (`spread > 0.45`).
  - *Reason*: In accordance with the Non-Negotiable Honesty Rules, an explicit uncertain classification with a manual inspection recommendation is far more trustworthy than a confident false positive on compressed mobile photos.
- **Decision**: Batch Testing Mode implemented with native CSV export (`/batch`).
  - *Reason*: Allows users to process multi-image folders locally and download forensic evaluation logs with execution latency and signal breakdowns.

