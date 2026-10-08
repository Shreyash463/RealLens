# RealLens 🔍

**RealLens** is a local, privacy-first forensic analysis tool that estimates whether an image is AI-generated or an authentic camera photograph / real graphic.

Designed specifically for real-world inspection without sugarcoating, RealLens combines **cryptographic provenance**, **compression physics**, **frequency-domain artifacts**, and a **diverse machine learning vision ensemble** with rigorous honesty guardrails.

---

## Non-Negotiable Honesty Guarantees

1. **Probability + Evidence, Never "100% Proof"**: Every output is an estimate accompanied by detector evidence. Missing metadata or watermarks is neutral and never constitutes proof of authenticity.
2. **Empirical Grounding**: All performance metrics reported in [`EVAL_REPORT.md`](EVAL_REPORT.md) stem from automated test runs on real benchmark sets. No simulated or hardcoded metrics.
3. **Disagreement Transparency**: When machine learning vision models disagree substantially, RealLens refuses to force a verdict; it flags the uncertainty and assigns the sample to an explicit `Uncertain` band with a manual inspection recommendation.
4. **Held-Out Integrity**: The final test set was strictly held out and evaluated only **once** after concluding all tuning.

---

## Architecture Overview

```
                      Uploaded Image
                            │
        ┌───────────────────┼───────────────────┐
        ▼                   ▼                   ▼
  1. Provenance       2. Compression      3. Deep Vision
  & Metadata          & Spectral          Ensemble
  ─────────────────   ─────────────────   ─────────────────
  • EXIF Hardware     • ELA Analysis      • ViT CIFAKE (dima806)
  • PNG Chunks        • 2D FFT Hann       • ViT SDXL (Organika)
  • C2PA Manifests    • Spectral Slope    • SigLIP (Ateeqq)
        │                   │                   │
        └───────────────────┼───────────────────┘
                            ▼
              4. Signal Fusion & Calibration
              ──────────────────────────────
              • Dynamic Weighted Combination
              • Hard Provenance Overrides
              • Disagreement Guardrails
                            │
                            ▼
           Verdict, Probability & Heatmap
```

---

## Empirical Benchmark Performance

Evaluated on the **CIFAKE Benchmark** (150 held-out test images evaluated once):

| Metric | Held-Out Test Set | Clean Robustness Subset |
| :--- | :---: | :---: |
| **Accuracy** | **85.33%** | **91.67%** |
| **Recall (AI Detection Rate)** | **98.67%** (74 / 75 caught) | **100.00%** |
| **Precision** | **77.89%** | **85.71%** |
| **F1-Score** | **0.8706** | **0.9231** |
| **Separation Gap** | **44.8%** (Real: 32.7%, AI: 77.5%) | - |

*Under JPEG recompression down to Q=65 and 0.50x downscaling, detection accuracy remains completely stable with 100% AI recall.* Full details are documented in [`EVAL_REPORT.md`](EVAL_REPORT.md).

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

# 3. Install PyTorch (CPU or CUDA) and dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. (Optional) Configure Gemini API for semantic reasoning
# Copy .env.example to .env and set GEMINI_API_KEY=your_key
# The app works completely offline without it.

# 5. Launch the Web App
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

# 3. Install PyTorch and dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# 4. Launch the Web App
python reallens/app.py
```

Open your browser to:
**`http://127.0.0.1:5000`**

---

## Running Automated Evaluations & Tests

### 1. Run Live API Test Suite (25 automated checks)
Runs 10 real images, 10 AI images, tiny thumbnails, 2500x2500 canvases, transparent PNGs, animated GIFs, and corrupt uploads:
```bash
python tests/test_live_api.py
```

### 2. Download / Rebuild Datasets
Downloads balanced CIFAKE subsets into `data/tuning/` and `data/test/`, and generates `data/messy_real/`:
```bash
python evaluation/build_dataset.py --samples 150
```

### 3. Evaluate Pipeline Splits
```bash
# Evaluate Held-Out Test Set (75 Real, 75 AI)
python evaluation/run_eval.py --split test

# Evaluate Tuning Set (75 Real, 75 AI)
python evaluation/run_eval.py --split tuning

# Evaluate Distorted / Messy Real Images
python evaluation/run_eval.py --split messy

# Evaluate User Smartphone / Camera Photos
python evaluation/run_eval.py --split user
```

### 4. Run Transformation Robustness Suite
Tests JPEG recompression (Q=85, 65, 45, 30), resizing (0.75x, 0.50x), screenshotting, and Gaussian blur:
```bash
python evaluation/robustness.py --split tuning --samples 12
```

---

## Known Weaknesses & Limitations

1. **Messaging App Compression (WhatsApp / Telegram)**:
   - Recompression and metadata stripping in chat apps alter high-frequency sensor noise. In our tests, WhatsApp photos triggered vision model disagreement and were routed to `Uncertain leaning AI` with low confidence.
2. **Heavy Smoothing / Low-Pass Blur**:
   - Heavy Gaussian blur or artistic soft-focus strips camera sensor grain, which vision models can misinterpret as latent diffusion smoothing.
3. **CGI & 3D Render Engines**:
   - Photorealistic 3D game engines (Unreal Engine 5) exhibit clean rendering that can mimic generative synthesis.

---

## Project Structure

```
RealLens/
├── reallens/
│   ├── app.py                # Flask server, upload validation & REST API
│   ├── detectors/
│   │   ├── metadata.py       # EXIF, PNG chunks, IPTC & camera tags
│   │   ├── c2pa_check.py     # Content Credentials manifest parser
│   │   ├── ela.py            # Error Level Analysis & heatmap generator
│   │   ├── frequency.py      # 2D FFT & radial power spectral slope
│   │   ├── ml_ensemble.py    # 3-model ViT & SigLIP vision ensemble
│   │   ├── gemini_layer.py   # Optional Gemini API visual reasoning
│   │   └── fusion.py         # Multi-signal fusion & verdict calibration
│   ├── static/               # Dark ink theme CSS & JS
│   └── templates/            # Single-page UI template
├── evaluation/
│   ├── build_dataset.py      # Dataset streaming & distortion pipeline
│   ├── run_eval.py           # Evaluation engine & error diagnostics
│   └── robustness.py         # Transformation robustness test suite
├── tests/
│   └── test_live_api.py      # 25-case live verification suite
├── DECISIONS.md              # Architectural decision log & model rationale
├── LESSONS.md                # Self-improvement iterations & tuning logs
├── EVAL_REPORT.md            # Verified evaluation metrics & confusion matrices
├── requirements.txt          # Pinned dependency specifications
└── README.md                 # User documentation & setup guide
```

---

## License

Personal and research project. Built for local forensic inspection.
