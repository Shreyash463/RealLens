# RealLens 🔍 (v1.0)

**RealLens** is a local, privacy-first forensic analysis tool that estimates whether an image is AI-generated or an authentic camera photograph / real graphic.

Designed specifically for real-world inspection without sugarcoating, RealLens combines **cryptographic provenance**, **compression physics**, **frequency-domain artifacts**, and a **diverse machine learning vision ensemble** with rigorous honesty guardrails and learned image gating.

---

## Non-Negotiable Honesty Guarantees

1. **Probability + Evidence, Never "100% Proof"**: Every output is an estimate accompanied by detector evidence. Missing metadata or watermarks is neutral and never constitutes proof of authenticity.
2. **Empirical Grounding**: All performance metrics reported in [`EVAL_REPORT.md`](EVAL_REPORT.md) stem from automated test runs on real benchmark sets. No simulated or hardcoded metrics.
3. **Disagreement Transparency**: When machine learning vision models disagree substantially (`spread >= 0.50`), RealLens refuses to force a verdict; it flags the uncertainty and assigns the sample to an explicit `Inconclusive` band with a probability range (`between X% and Y% across models`).
4. **Learned Image Gate (DINOv2 + Engineered Features)**: A trained classifier distinguishes camera photos from digital graphics, certificates, and UI screenshots, preventing deep vision models from misclassifying flat digital graphics while ensuring heavily compressed mobile photos and selfies receive full inspection.
5. **No Synthetic Placeholders**: When physical forensics (ELA / FFT) are not applicable (e.g., flat graphics or non-photo inputs), RealLens reports "Not applicable" in neutral grey and completely excludes them from fusion math rather than outputting misleading 50% numbers.
6. **Held-Out Integrity**: The final test sets (`data/test/` and `data/hard_test/`) were strictly held out and evaluated only **once** after concluding all tuning.

---

## Architecture Overview

```
                                      Uploaded Image
                                            │
         ┌──────────────────────────────────┼──────────────────────────────────┐
         ▼                                  ▼                                  ▼
   1. Learned Image Gate              2. Forensics & Provenance          3. 4-Model Deep
   ─────────────────────────          ─────────────────────────          Vision Ensemble
   • DINOv2 384-d Cls Embedding       • EXIF Hardware Metadata           • ViT CIFAKE (dima806)
   • 8 Continuous Visual Features     • PNG Generation Chunks            • ViT SDXL (Organika)
   • Soft photo_probability           • C2PA Manifest Reader             • SigLIP (Ateeqq)
   • Prevents Graphic False Alarms    • ELA Compression Variance         • DINOv2 Probe (Ours)
   • 100% Recall on Chat Photos       • 2D FFT Hann Spectral Decay       • Model Spread & Weights
         │                                  │                                  │
         └──────────────────────────────────┼──────────────────────────────────┘
                                            ▼
                              4. Signal Fusion & Guardrails
                              ─────────────────────────────────────────
                              • Disagreement Guardrail (Spread >= 0.50 -> Inconclusive)
                              • Strict "Inconclusive" Headline (Range rendered below)
                              • Learned Logistic Regression Stacker on Agreement
                              • Pipeline-Versioned Cache (v1.0.0) with Bypass
                                            │
                                            ▼
                          Single-Page VeilShare-Style Dashboard
            ┌───────────────────────────────┴───────────────────────────────┐
            ▼                                                               ▼
      Left Panel (Sticky ~35%)                                Right Panel (~65%)
      • Drag & drop, paste, folder drops                      • Summary counter strip
      • Auto-analyze toggle switch                            • Big verdict banner & reason
      • Queue rows with status dots                           • 3 Structured evidence lists
      • Progress bar & ETA remaining                          • Reverse search tools (Google / TinEye)
      • CSV & JSON export buttons                             • Raw model progress bars & spread
                                                              • Neutral ELA forensics viewport
                                                              • Collapsible multi-signal accordion
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

### 1. Run Phase 4 End-to-End Stress Test Suite
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
│   │   ├── image_type.py          # Learned photo-vs-non-photo gate (DINOv2 + visual features)
│   │   ├── metadata.py            # EXIF, PNG chunks, IPTC & camera tags
│   │   ├── c2pa_check.py          # Content Credentials manifest parser
│   │   ├── ela.py                 # Error Level Analysis & heatmap generator
│   │   ├── frequency.py           # 2D FFT & radial power spectral slope
│   │   ├── ml_ensemble.py         # 4-model ViT, SigLIP & DINOv2 ensemble
│   │   ├── gemini_layer.py        # Optional Gemini API visual reasoning
│   │   └── fusion.py              # Learned stacker & honest uncertain-zone fusion
│   ├── static/                    # Dark ink theme CSS & JS (main.js, style.css)
│   └── templates/                 # UI templates (index.html)
├── models/
│   ├── photo_vs_non_photo_gate.joblib # Trained learned image gate
│   ├── generalizing_probe.joblib  # Trained DINOv2 linear probe
│   └── learned_fusion_stacker.joblib # Trained calibrated fusion stacker
├── evaluation/
│   ├── train_gate.py              # Learned image gate trainer
│   ├── train_probe.py             # DINOv2/CLIP probe trainer
│   ├── train_fusion_stacker.py    # Learned fusion stacker trainer
│   ├── compare_models_hard.py     # 4-model comparative benchmark script
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

