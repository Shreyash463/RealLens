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

## 5. Phase 2.5 Architectural Decisions: Image-Type Gating, Spread Inspection & Inconclusive Routing
- **Decision**: Image-Type Gate (`reallens/detectors/image_type.py`) added prior to ML and forensics.
  - *Reason*: Vision detectors trained on photographic scenes suffer a 63.3% false-positive rate on certificates, slides, and screenshots because vector text glyphs, uniform flat fills, and missing sensor noise mimic synthetic diffusion artifacts.
- **Decision**: Automatic Routing to `Inconclusive` with probability range for non-photographic imagery.
  - *Reason*: Missing camera metadata and absence of watermarks never prove an image is AI-generated; warning the user that photo-trained detectors are unreliable on flat graphics prevents confident false accusations.
- **Decision**: Disagreement Guardrail (`spread >= 0.50`) enforces `Inconclusive` and replaces single percentage with a transparent probability range (`between X% and Y% across models`).
  - *Reason*: When model predictions conflict violently (e.g. 2% vs 100%), forcing an average or maximum score creates a deceptive illusion of certainty; exposing the spread and range respects user trust.
- **Decision**: ELA Text Guardrail in `reallens/detectors/ela.py`.
  - *Reason*: ELA is physically meaningful only for photographic continuous-tone JPEG imagery; non-photographic graphics report that ELA is not meaningful.
- **Decision**: Multi-Image Client-Side Batch Queue with Sequential Execution.
  - *Reason*: Sequential asynchronous queue prevents local CPU starvation from concurrent PyTorch inference runs while client-side SHA-256 caching eliminates duplicate re-analysis.

## 6. Phase 3 Architectural Decisions: Single-Page Consolidation, Latency Elimination, and Honesty Guardrails
- **Decision**: Unified Single-Page Application (`/`) inspired by VeilShare dark ink aesthetic; eliminate separate `/batch` page and issue HTTP 302 redirect `/batch` -> `/`.
  - *Reason*: Eliminates duplicate code and context switching between single-upload and batch workflows. A sticky left panel (~35%) handles drag-and-drop, clipboard paste, folder drops, queue management, ETA progress, and CSV/JSON export; the right panel (~65%) renders batch summaries, fixed-width non-overlapping tables, and full multi-signal inspections.
- **Decision**: Eager Startup Model Pre-warming (`warmup_models()`).
  - *Reason*: Eliminates the ~20-second cold-start latency penalty on the user's first upload. PyTorch model weights and tokenizer pipelines are initialized during server boot (~11s), reducing subsequent image analysis times down to ~0.55s per image on CPU.
- **Decision**: In-Memory SHA-256 Result Cache (`_ANALYSIS_CACHE`).
  - *Reason*: Users frequently re-upload or re-inspect images across batches. Caching immutable SHA-256 digests allows instant sub-2 millisecond backend responses (<7ms network roundtrip) with zero redundant computation.
- **Decision**: Forensic Downscaling (>1024px) & ThreadPool Concurrency.
  - *Reason*: Running ELA and FFT on multi-megapixel images consumes excessive CPU time without adding forensic value. Downscaling inputs larger than 1024px for ELA/FFT while executing metadata, C2PA, ELA, and FFT concurrently in `ThreadPoolExecutor(max_workers=4)` alongside model inference keeps end-to-end processing under 0.60 seconds.
- **Decision**: WhatsApp / Social Recompression Honesty Guardrails & 5-Band Verdict Spectrum.
  - *Reason*: Heavy DCT quantization from chat apps (WhatsApp/Telegram 1280x720) strips camera EXIF and creates borderline noise that was previously forced into a misleading "Leaning AI 51.9%". RealLens now measures 8x8 DCT blockiness and chat dimensions, expanding the honest spectrum: ambiguous mid-spectrum cases (0.45 - 0.55) or heavily compressed images with low confidence are routed to `Inconclusive` with probability range display (`between X% and Y% across models`).
- **Decision**: Fixed-Width Non-Overlapping Table Layout with `<colgroup>`.
  - *Reason*: Avoids column squishing, overlapping text, and horizontal runaway on varied screen sizes (1920x1080 down to 390x844 mobile viewport) by using fixed pixel column allocations inside a responsive container.

## 7. Phase 4 Architectural Decisions: Learned Gate, Transparent Model Scores, Evidence Cards & Cache Versioning
- **Decision**: Learned Photo-vs-Non-Photo Classifier (`models/photo_vs_non_photo_gate.joblib`) replacing brittle heuristics.
  - *Reason*: Heuristic rules that penalized portrait phone aspect ratios (720x1600) or low color counts misclassified genuine camera photos (such as WhatsApp mirror selfies) as "flat graphic/illustration", incorrectly suppressing detectors. The learned classifier combines frozen DINOv2-small 384-d patch embeddings with 8 continuous visual features (flatness, noise variance, white paper ratio, edge sharpness, DCT blockiness, saturation, color entropy) trained on 813 samples (528 photos, 285 graphics), achieving 99.43% validation accuracy and 100.0% recall on WhatsApp compressed photos.
- **Decision**: Guaranteed Exposure and UI Rendering of Raw Model Scores and Weights (`r.model_scores`, `r.per_model_weights`, `r.spread`).
  - *Reason*: Resolved frontend key mismatch where the detail card displayed "No individual model scores recorded" despite backend knowing the scores. Raw outputs of all 4 active models are now normalized, stored, and displayed as horizontal progress bars with model weights.
- **Decision**: Elimination of 50% Placeholder Scoring in ELA and FFT (`score: None`, `applicable: False`, `status_text: "Not applicable"`).
  - *Reason*: ELA and FFT now return `None` (not applicable) when applied to non-photographic graphics or heavily recompressed images, displaying "Not applicable" in neutral grey and removing false 50% numbers from user reports.
- **Decision**: Strict "Inconclusive" Headline Policy (Never Range String as Headline).
  - *Reason*: Disagreement strings like "between 0% and 98% across models" must never serve as top-level headlines. The top-level headline for inconclusive results is strictly the word "Inconclusive", followed by an explanatory summary line, with the spread range indicator placed neatly below.
- **Decision**: Versioned SHA-256 Cache (`file_hash:PIPELINE_VERSION`) and Cache Bypass (`ignore_cache=1`).
  - *Reason*: Cache keys are tagged with pipeline version (`v1.0.0`), preventing old results from surviving pipeline updates. Cached responses return the original execution duration ("analyzed in X.X s, served from cache"), and a "Re-run (ignore cache)" button allows on-demand cache bypass.
- **Decision**: Structured Detail View Hierarchy with 3 Evidence Lists and Actionable Next Steps.
  - *Reason*: Conforms to user-requested order: Big Verdict Card -> "Why this verdict" -> Three Evidence Lists ("Evidence pointing to AI", "Evidence pointing to Real", "Missing or Unreliable Evidence") -> "What to do next" (including Google Lens and TinEye reverse search buttons) -> Individual Model Predictions & Spread -> Neutral ELA Inspection Viewport (`max-height: 420px; object-fit: contain; background: var(--bg-elev)`) -> Collapsible Multi-Signal Accordion.

## 8. Resilience against Windows 11 Smart App Control (SAC)
- **Decision**: Graceful fallback to continuous engineered visual features and non-ML forensic pipeline when PyTorch C-extension DLLs are blocked by Windows Defender Smart App Control policy.
  - *Reason*: Windows 11 Smart App Control in enforcement mode blocks unsigned `.pyd` binaries (`torch\_C.cp313-win_amd64.pyd`) from PyPI if cloud reputation is not established. Wrapping imports prevents server boot failure, permitting full operation of EXIF/PNG metadata, C2PA manifest verification, ELA compression heatmaps, 2D FFT spectral decomposition, and Gemini reasoning.



