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
- **Decision**: Missing metadata (EXIF/C2PA) yields neutral score (`null` or no push towards real/fake) with LOW confidence.
  - *Reason*: Real images regularly undergo stripping by chat apps (WhatsApp, Telegram) and web uploaders, so missing metadata is never evidence of AI generation.
- **Decision**: C2PA manifest reader uses standard PyPI `c2pa-python` if compatible binary wheel exists for Python 3.13 Windows, with pure-Python fallback (reading JUMBF/c2pa boxes from JPEG/PNG headers).
  - *Reason*: Guarantees zero runtime crashes while still extracting C2PA / Content Credentials provenance if embedded.
- **Decision**: ELA generates both numerical metrics (mean diff, high-frequency diff ratio, kurtosis/peakiness) and a colormapped ELA heatmap saved in-memory or Base64 for web display.
  - *Reason*: Allows the user to inspect spatial distribution of compression inconsistencies while providing an analytical metric to the fusion engine.
