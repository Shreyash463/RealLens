# RealLens 🔍 (v5.0)

**RealLens** is a local, privacy-first forensic analysis tool that estimates whether an image is AI-generated or an authentic camera photograph / real graphic.

Designed specifically for real-world inspection without sugarcoating, RealLens combines **cryptographic provenance**, **compression physics**, **frequency-domain artifacts**, and a **robust multi-backbone machine learning vision ensemble** with rigorous honesty guardrails and learned image gating.

---

## Non-Negotiable Honesty Guarantees

1. **Probability + Evidence, Never "100% Proof"**: Every output is an estimate accompanied by detector evidence. Missing metadata or watermarks is neutral and never constitutes proof of authenticity.
2. **Empirical Grounding**: All performance metrics reported in [`EVAL_REPORT.md`](EVAL_REPORT.md) stem from automated test runs on real benchmark sets. No simulated or hardcoded metrics.
3. **Strict Invariant on Recompression**: The system **strictly forbids** returning `Likely real` on images that have undergone WhatsApp compression, screenshot capture, or lossy recompression without verified camera EXIF provenance. These images honestly receive `No AI evidence found (not proof of real)`.
4. **Condition-Aware Model Weighting**: Fragile models (e.g. SDXL specialist) that fail on recompression are dynamically downweighted, while invariant models (SigLIP and Multi-Backbone Probe) take precedence.
5. **Screenshot Pre-Processing**: Automatically detects and crops letterboxes, pillarboxes, and OS status bars before vision inspection, preventing UI artifacts from misleading detectors.
6. **Watermark Astroid Star Correlator**: Uses 2D normalized cross-correlation against an astroid star template ($|x|^{0.65} + |y|^{0.65} \le R^{0.65}$), preventing false alarms on high-frequency foliage while reliably catching Gemini watermarks.
7. **Learned Image Gate (DINOv2 + Engineered Features)**: A trained classifier distinguishes camera photos from digital graphics, certificates, and UI screenshots, preventing deep vision models from misclassifying flat digital graphics.
8. **No Synthetic Placeholders**: When physical forensics (ELA / FFT) are not applicable, RealLens reports "Not applicable" in neutral grey and completely excludes them from fusion math.

---

## Architecture Overview

```
                                      Uploaded Image
                                            │
                                  Screenshot Cropper
                         (Removes letterbox / OS status bars)
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
   1. Learned Image Gate              2. Forensics & Provenance          3. Robust Multi-Model
   ─────────────────────────          ─────────────────────────          Vision Ensemble
   • DINOv2 384-d Cls Embedding       • EXIF Hardware Metadata           • ViT CIFAKE (dima806)
   • 8 Continuous Visual Features     • PNG Generation Chunks            • ViT SDXL (Organika)
   • Soft photo_probability           • C2PA Manifest Reader             • SigLIP (Ateeqq)
   • Prevents Graphic False Alarms    • ELA Compression Variance         • Multi-Backbone Probe
   • 100% Recall on Chat Photos       • 2D FFT Hann Spectral Decay         (DINOv2 + CLIP ViT-B/32)
   • Condition Detector               • Astroid Star Watermark NCC       • Condition-Aware Weights
         │                                  │                                  │
         └──────────────────────────────────┼──────────────────────────────────┘
                                            ▼
                              4. Signal Fusion & Guardrails
                              ─────────────────────────────────────────
                              • 5 Precision Categories:
                                - AI evidence found (strong)
                                - Leaning AI
                                - Inconclusive
                                - No AI evidence found (not proof of real)
                                - Likely real (Camera EXIF verified only)
                              • Strict Recompression Invariant Enforced
                              • Dynamic Condition-Aware Stacker
                              • Pipeline-Versioned Cache (v5.0.0)
                                            │
                                            ▼
                          Single-Page VeilShare-Style Dashboard
            ┌───────────────────────────────┴───────────────────────────────┐
            ▼                                                               ▼
      Left Panel (Sticky ~35%)                                Right Panel (~65%)
      • Drag & drop, paste, folder drops                      • Summary counter strip (5 filters)
      • Auto-analyze toggle switch                            • Big verdict banner & reason
      • Queue rows with status dots                           • Auto-crop & condition badges
      • Progress bar & ETA remaining                          • 3 Structured evidence lists
      • CSV & JSON export buttons                             • Gemini semantic reasoning cards
                                                              • Reverse search (Google, TinEye, Yandex)
                                                              • Raw model progress bars & reliability
                                                              • Neutral ELA forensics viewport
```

---

## Empirical Benchmark Performance

### 1. Learned Image Gate (`models/photo_vs_non_photo_gate.joblib`)
- **Validation Accuracy**: **99.43%** (174 / 175 held-out samples correct)
- **F1-Score**: **0.9948**
- **WhatsApp / Compressed Photo Recall**: **100.00%** (128 / 128 correctly flagged photo-like)
- **Digital Graphics / Certificate Recall**: **100.00%** (57 / 57 non-photos caught)
- **WhatsApp Mirror Selfie**: Correctly identified as `photo-like` (`p=0.964`), removing false alarms.

### 2. Held-Out Benchmarks
- **Easy Benchmark (Held-Out CIFAKE Split, 150 images)**:
  - Decided Accuracy: **100.00%** (39 / 39 decided samples correct)
  - Precision: **100.00%** | Recall: **100.00%** | F1: **1.0000**
  - False Positive Rate: **0.00%** (0 / 75)
- **Hard Benchmark (Held-Out Hard Split, 60 images)**:
  - Honest Inconclusive Rate: **96.67%** (58 / 60 transparently flagged)
  - False Positive Rate: **3.33%** (1 / 30)
- **Non-AI Graphics Benchmark (30 images: certificates, slides, charts)**:
  - Inconclusive Gating: **100.00%** (30 / 30 correctly withheld)
  - False Positive Rate: **0.00%** (0 / 30)

Full evaluation tables and confusion matrices are documented in [`EVAL_REPORT.md`](EVAL_REPORT.md).

### 1. Easy Benchmark (Held-Out CIFAKE Split, 150 images: 75 Real, 75 AI)
- **Accuracy**: **92.00%** (+6.67% improvement in Phase 2)
- **Precision**: **93.15%** (+15.26% improvement in Phase 2)
- **Recall (AI Detection Rate)**: **90.67%** (68 of 75 AI caught)
- **F1-Score**: **0.9189**
- **False Positives**: Dropped from 21 down to **5** (-76.2% reduction in false alarms!)
- **Class Separation Margin**: **72.0%** (13.2% on Real vs 85.2% on AI)

### 2. Hard Benchmark (Held-Out Hard Split: Midjourney AI + WhatsApp/Screenshot Real, 60 images: 30 Real, 30 AI)
- **Accuracy**: **86.67%** (52 / 60 correct)
- **Precision**: **86.67%**
- **Recall (AI Detection Rate)**: **86.67%** (26 of 30 hard Midjourney AI caught)
- **F1-Score**: **0.8667**
- **WhatsApp Smartphone Photo Accuracy**: **80.00%** (4 of 5 correctly identified as Real, up from 0% before Phase 2)
- **Uncertain Zone Flagging Rate**: **96.7%** (58 / 60 samples transparently flagged with manual inspection warning)

Full evaluation tables and confusion matrices are documented in [`EVAL_REPORT.md`](EVAL_REPORT.md).

---

## Quickstart & Installation

### Prerequisites
- Python 3.10+ (tested on Python 3.10 to 3.13)
- Windows, macOS, or Linux
- CUDA GPU supported automatically; CPU fallback is built-in and verified.

---

### Windows (PowerShell)

```powershell
# 1. Clone repository and navigate inside
git clone https://github.com/your-username/RealLens.git
cd RealLens

# 2. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# 3. Install PyTorch and dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Launch the Web Application
python reallens/app.py
```

---

### macOS / Linux (Bash)

```bash
# 1. Clone repository and navigate inside
git clone https://github.com/your-username/RealLens.git
cd RealLens

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Launch the Web Application
python reallens/app.py
```

Open your browser to:
- Unified Analysis & Batch Queue Dashboard: **`http://127.0.0.1:5000`**
*(Note: Legacy `/batch` automatically redirects to `/`)*

---

## Running Automated Evaluations & Training

### 1. Run Phase 5 Regression & Invariant Test Suite
Tests Ganesha AI painting, real tree man WhatsApp photo, screenshot crops, and invariant guardrails:
```bash
python evaluation/test_phase5_regression.py
```

### 2. Run Model Condition Degradation Audit
Evaluates each vision backbone independently across 8 degradation conditions:
```bash
python evaluation/audit_models_per_condition.py
```

### 3. Retrain Robust Multi-Backbone Probe (DINOv2 + CLIP)
Trains the multi-backbone probe with balanced multiscale augmentations:
```bash
python evaluation/train_robust_multi_probe.py
```

### 4. Run Phase 4 End-to-End Stress Test Suite
Tests user WhatsApp mirror selfie, cache hits, cache bypass (`ignore_cache=1`), certificate suppression, and 45-image mixed batch:
```bash
python tests/test_phase4_end_to_end.py
```

### 2. Run Gate Regression & Signal Honesty Tests
Verifies photo-vs-non-photo classification and ensures ELA/FFT return "Not applicable" rather than placeholder scores:
```bash
python tests/test_gate_regression.py
python tests/test_verdict_consistency_and_models.py
```

### 3. Retrain Learned Photo-vs-Non-Photo Gate
Extracts DINOv2 cls embeddings + visual features and trains the gate in `models/photo_vs_non_photo_gate.joblib`:
```bash
python evaluation/train_gate.py
```

### 4. Retrain Generalizing DINOv2 Feature Probe
Extracts augmented embeddings and retrains the probe in `models/generalizing_probe.joblib`:
```bash
python evaluation/train_probe.py
```

### 5. Retrain Learned Calibrated Fusion Stacker
Extracts multi-signal features across tuning datasets and retrains the stacker in `models/learned_fusion_stacker.joblib`:
```bash
python evaluation/train_fusion_stacker.py
```

### 6. Evaluate Pipeline on Benchmark Splits
```bash
# Evaluate Held-Out Hard Test Set (30 Real, 30 AI)
python evaluation/run_eval.py --split hard_test

# Evaluate Held-Out Easy Test Set (75 Real, 75 AI)
python evaluation/run_eval.py --split test

# Evaluate Non-AI Graphics (Certificates, Charts, Slides, Web Dashboards)
python evaluation/run_eval.py --split graphics

# Evaluate WhatsApp Recompressed Photos
python evaluation/run_eval.py --split whatsapp

# Compare All 4 Models on the Hard Tuning Set
python evaluation/compare_models_hard.py

# Evaluate User Smartphone / Camera Photos
python evaluation/run_eval.py --split user
```

### 7. Run Transformation Robustness Suite
Tests JPEG recompression (Q=85, 65, 45, 30), resizing (0.75x, 0.50x), screenshotting, and Gaussian blur:
```bash
python evaluation/robustness.py --split tuning --samples 12
```

---

## Known Weaknesses & Limitations

1. **Extreme Low-Quality JPEG Compression (Q < 50)**:
   - Quantization block boundary noise can obscure subtle latent diffusion upsampling signatures.
2. **Smooth Macro Bokeh / Wide-Aperture Blur**:
   - Authentic photographs featuring extreme background defocus (portrait mode or macro lenses) possess smooth grainless patches that vision models can suspect as synthetic smoothing.
3. **Double-Compressed Chat App Thumbnails**:
   - Multiple rounds of social media recompression strip camera EXIF metadata and generate cascading block artifacts.

---

## Project Structure

```
RealLens/
├── reallens/
│   ├── app.py                     # Flask server, versioned cache & REST API
│   ├── detectors/
│   │   ├── condition.py           # Degradation & channel condition detector
│   │   ├── screenshot_cropper.py  # Auto-cropper for status bars & letterboxes
│   │   ├── watermarks.py          # Astroid star normalized cross-correlation watermark detector
│   │   ├── gemini_semantic.py     # Gemini multimodal 6-item structured forensic rubric
│   │   ├── image_type.py          # Learned photo-vs-non-photo gate (DINOv2 + visual features)
│   │   ├── metadata.py            # EXIF, PNG chunks, IPTC & camera tags
│   │   ├── c2pa_check.py          # Content Credentials manifest parser
│   │   ├── ela.py                 # Error Level Analysis & heatmap generator
│   │   ├── frequency.py           # 2D FFT & radial power spectral slope
│   │   ├── ml_ensemble.py         # 4-model ViT, SigLIP & multi-backbone probe
│   │   └── fusion.py              # Condition-aware dynamic stacker & precision guardrails
│   ├── static/                    # Dark ink theme CSS & JS (main.js, style.css)
│   └── templates/                 # UI templates (index.html)
├── models/
│   ├── photo_vs_non_photo_gate.joblib # Trained learned image gate
│   ├── generalizing_probe.joblib  # Trained multi-backbone DINOv2 + CLIP probe
│   └── learned_fusion_stacker.joblib # Trained calibrated fusion stacker
├── evaluation/
│   ├── build_matched_dataset.py   # 960-image matched dataset builder across 8 conditions
│   ├── check_matched_shortcuts.py # Shortcut classifier verifier (56.04% acc)
│   ├── audit_models_per_condition.py # Condition degradation audit (AUC, Acc, TPR, FPR)
│   ├── train_robust_multi_probe.py # Multi-backbone probe trainer (85.0% val acc)
│   ├── test_phase5_regression.py  # Empirical regression test suite (Ganesha AI, Tree Man)
│   ├── train_gate.py              # Learned image gate trainer
│   ├── run_eval.py                # Evaluation engine & error diagnostics
│   └── robustness.py              # Transformation robustness test suite
├── tests/
│   ├── test_gate_regression.py    # WhatsApp selfie & certificate gate regression tests
│   ├── test_verdict_consistency_and_models.py # Model scores & signal honesty verification
│   ├── test_phase4_end_to_end.py  # Cache hit/bypass & 45-image stress test
│   └── capture_viewports.py       # Multi-resolution viewport capture script
├── DECISIONS.md                   # Architectural decision log & model rationale
├── LESSONS.md                     # Self-improvement iterations & tuning logs
├── EVAL_REPORT.md                 # Verified evaluation metrics & confusion matrices
├── requirements.txt               # Pinned dependency specifications
└── README.md                      # User documentation & setup guide
```

---

## License

Personal and research project. Built for local forensic inspection.

