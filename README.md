# RealLens 🔍 (Phase 2)

**RealLens** is a local, privacy-first forensic analysis tool that estimates whether an image is AI-generated or an authentic camera photograph / real graphic.

Designed specifically for real-world inspection without sugarcoating, RealLens combines **cryptographic provenance**, **compression physics**, **frequency-domain artifacts**, and a **diverse machine learning vision ensemble** with rigorous honesty guardrails.

---

## Non-Negotiable Honesty Guarantees

1. **Probability + Evidence, Never "100% Proof"**: Every output is an estimate accompanied by detector evidence. Missing metadata or watermarks is neutral and never constitutes proof of authenticity.
2. **Empirical Grounding**: All performance metrics reported in [`EVAL_REPORT.md`](EVAL_REPORT.md) stem from automated test runs on real benchmark sets. No simulated or hardcoded metrics.
3. **Disagreement Transparency**: When machine learning vision models disagree substantially, RealLens refuses to force a verdict; it flags the uncertainty and assigns the sample to an explicit `Uncertain` band with a manual inspection recommendation.
4. **Held-Out Integrity**: The final test sets (`data/test/` and `data/hard_test/`) were strictly held out and evaluated only **once** after concluding all tuning.

---

## Architecture Overview

```
                              Uploaded Image
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        ▼                           ▼                           ▼
  1. Provenance               2. Compression              3. 4-Model Deep
  & Metadata                  & Spectral                  Vision Ensemble
  ─────────────────────────   ─────────────────────────   ─────────────────────────
  • EXIF Camera Hardware      • Error Level Analysis      • ViT CIFAKE (dima806)
  • PNG Generation Chunks       (Q=90 Heatmap + Stats)    • ViT SDXL (Organika)
  • C2PA Manifest Reader      • 2D FFT Hann Window        • SigLIP (Ateeqq)
  • Missing = Neutral           Spectral Slope Decay      • DINOv2 Probe (Ours)
        │                           │                           │
        └───────────────────────────┼───────────────────────────┘
                                    ▼
                      4. Learned Signal Stacker & Fusion
                      ──────────────────────────────────
                      • Calibrated Logistic Regression Stacker
                      • Dynamic Fallback to Hand-Weighted Fusion
                      • Hard Cryptographic Provenance Overrides
                      • Explicit Uncertain Zone Guardrail [35%-65%]
                                    │
                                    ▼
                   Verdict, Probability, Bar List & Heatmap
```

---

## Empirical Benchmark Performance (Phase 2)

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
- Single Image Analysis: **`http://127.0.0.1:5000`**
- Batch Folder Analysis: **`http://127.0.0.1:5000/batch`**

---

## Running Automated Evaluations & Training

### 1. Run Live API Test Suite (25 automated checks)
Tests 10 real images, 10 AI images, tiny thumbnails, 2500x2500 canvases, transparent PNGs, animated GIFs, and corrupt uploads:
```bash
python tests/test_live_api.py
```

### 2. Retrain Generalizing DINOv2 Feature Probe
Extracts augmented embeddings and retrains the probe in `models/generalizing_probe.joblib`:
```bash
python evaluation/train_probe.py
```

### 3. Retrain Learned Calibrated Fusion Stacker
Extracts multi-signal features across tuning datasets and retrains the stacker in `models/learned_fusion_stacker.joblib`:
```bash
python evaluation/train_fusion_stacker.py
```

### 4. Evaluate Pipeline on Benchmark Splits
```bash
# Evaluate Held-Out Hard Test Set (30 Real, 30 AI)
python evaluation/run_eval.py --split hard_test

# Evaluate Held-Out Easy Test Set (75 Real, 75 AI)
python evaluation/run_eval.py --split test

# Compare All 4 Models on the Hard Tuning Set
python evaluation/compare_models_hard.py

# Evaluate User Smartphone / Camera Photos
python evaluation/run_eval.py --split user
```

### 5. Run Transformation Robustness Suite
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
│   ├── app.py                     # Flask server, upload validation & REST API
│   ├── detectors/
│   │   ├── metadata.py            # EXIF, PNG chunks, IPTC & camera tags
│   │   ├── c2pa_check.py          # Content Credentials manifest parser
│   │   ├── ela.py                 # Error Level Analysis & heatmap generator
│   │   ├── frequency.py           # 2D FFT & radial power spectral slope
│   │   ├── ml_ensemble.py         # 4-model ViT, SigLIP & DINOv2 ensemble
│   │   ├── gemini_layer.py        # Optional Gemini API visual reasoning
│   │   └── fusion.py              # Learned stacker & uncertain-zone fusion
│   ├── static/                    # Dark ink theme CSS & JS (main.js, batch.js)
│   └── templates/                 # UI templates (index.html, batch.html)
├── models/
│   ├── generalizing_probe.joblib  # Trained DINOv2 linear probe
│   └── learned_fusion_stacker.joblib # Trained calibrated fusion stacker
├── evaluation/
│   ├── build_dataset.py           # CIFAKE benchmark dataset builder
│   ├── build_hard_dataset.py      # Hard dataset builder (Midjourney + WhatsApp)
│   ├── train_probe.py             # DINOv2/CLIP probe trainer
│   ├── train_fusion_stacker.py    # Learned fusion stacker trainer
│   ├── compare_models_hard.py     # 4-model comparative benchmark script
│   ├── run_eval.py                # Evaluation engine & error diagnostics
│   └── robustness.py              # Transformation robustness test suite
├── tests/
│   ├── test_live_api.py           # 25-case live verification suite
│   └── test_audit_latency.py      # 20-image latency benchmark script
├── DECISIONS.md                   # Architectural decision log & model rationale
├── LESSONS.md                     # Self-improvement iterations & tuning logs
├── EVAL_REPORT.md                 # Verified evaluation metrics & confusion matrices
├── requirements.txt               # Pinned dependency specifications
└── README.md                      # User documentation & setup guide
```

---

## License

Personal and research project. Built for local forensic inspection.
