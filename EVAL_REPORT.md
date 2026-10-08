# RealLens Forensic Evaluation Report

This evaluation report details the rigorous, empirical assessment of the **RealLens** multi-signal forensic detection engine.

In adherence to the **Non-Negotiable Honesty Rules**:
1. All reported metrics derive strictly from automated test executions (`evaluation/run_eval.py` and `evaluation/robustness.py`).
2. The held-out test split (`data/test/`) was strictly preserved and evaluated exactly **once** after concluding the self-improvement loop.
3. No numbers have been fabricated, smoothed, or altered.

---

## 1. Executive Summary

- **Architecture**: Multi-signal fusion combining:
  - Provenance & Metadata (EXIF camera hardware tags, PNG parameter chunks, C2PA Content Credentials)
  - Physical & Compression Forensics (Error Level Analysis at Q=90, 2D FFT azimuthal radial power spectrum slope)
  - Machine Learning Vision Ensemble:
    - `dima806/ai_vs_real_image_detection` (ViT-base)
    - `Organika/sdxl-detector` (ViT-base)
    - `Ateeqq/ai-vs-human-image-detector` (SigLIP)
  - Decision Rule: Median aggregation across ML vision models + dynamically weighted multi-signal fusion.
- **Held-Out Test Accuracy**: **85.33%**
- **Held-Out Test F1-Score**: **0.8706**
- **Held-Out Test Recall (AI Detection Rate)**: **98.67%** (Catching 74 out of 75 AI generated images)
- **Held-Out Test Precision**: **77.89%**
- **Probability Separation Margin**: **44.8%** (Mean AI probability of 32.7% on Real vs 77.5% on AI)

---

## 2. Dataset Methodology & Integrity

Evaluations were conducted on four distinct splits:
1. **Held-In Tuning Split (`data/tuning/`)**: 75 Real, 75 AI (150 images total from CIFAKE benchmark). Used exclusively for error diagnosis, bug fixes, and calibration.
2. **Held-Out Test Split (`data/test/`)**: 75 Real, 75 AI (150 images total from CIFAKE benchmark). Kept completely isolated and never inspected or tuned against.
3. **Messy Real Split (`data/messy_real/`)**: 75 Real images subjected to randomized distortions (JPEG Q=40..70, scaling 0.5x..0.8x, screenshot blit) evaluated against 75 held-out AI images.
4. **User Images Split (`data/my_images/`)**: 5 authentic smartphone photos transmitted through WhatsApp (heavy compression and stripped EXIF).

---

## 3. Held-Out Test Evaluation Results (Evaluated Once)

Command executed: `python evaluation/run_eval.py --split test`

| Metric | Result | Notes |
| :--- | :--- | :--- |
| **Total Images** | 150 | 75 Real, 75 AI |
| **Accuracy** | **85.33%** | 128 / 150 correct classifications |
| **Precision** | **77.89%** | True AI / (True AI + False AI) |
| **Recall** | **98.67%** | **74 of 75 AI images detected** (1 False Negative) |
| **F1-Score** | **0.8706** | Harmonic mean of precision and recall |
| **Mean Prob (Real)** | **32.7%** | Fused probability distribution center for authentic images |
| **Mean Prob (AI)** | **77.5%** | Fused probability distribution center for AI images |
| **Separation Margin**| **44.8%** | Robust decision gap |

### Confusion Matrix
```
                   Predicted Real    Predicted AI
Actual Real (75)         54 (TN)          21 (FP)
Actual AI   (75)          1 (FN)          74 (TP)
```

### Analysis of Held-Out Errors:
- **False Negatives (1 image)**:
  - Only 1 AI image evaded detection: `ai_test_0032.jpg`. The ensemble scored it 0.448 (`Uncertain leaning real`).
  - Analysis: A highly textured animal fur sample where diffusion noise artifacts closely resembled natural sensor grain.
- **False Positives (21 images)**:
  - Real images that scored between 73.9% and 83.1%.
  - Critical Observation: In all 21 false positive instances, the system flagged severe ML disagreement (`spread: 0.99 - 1.00`). Consequently, the system **never** declared them "Likely AI" with high confidence. Instead, every single false positive was classified into the transparent guardrail band:
    `Verdict: Uncertain leaning AI`
    `Reason: Machine learning detectors disagree substantially (spread: 1.00). The system recommends manual inspection.`

---

## 4. Robustness Suite Results

Command executed: `python evaluation/robustness.py --split tuning --samples 12`
Evaluates survival of detection signals under standard real-world transformations:

| Condition | Accuracy | Recall | Precision | F1-Score | Performance Impact |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Clean (Baseline)** | **91.67%** | **100.00%** | 85.71% | 0.9231 | Baseline |
| **JPEG Q=85** | **91.67%** | **100.00%** | 85.71% | 0.9231 | Zero degradation |
| **JPEG Q=65** | **91.67%** | **100.00%** | 85.71% | 0.9231 | Zero degradation |
| **JPEG Q=45** | **87.50%** | **100.00%** | 80.00% | 0.8889 | -4.17% (mild artifact masking) |
| **JPEG Q=30 (Severe)** | **83.33%** | **100.00%** | 75.00% | 0.8571 | -8.34% (blocking artifacts dominate) |
| **Resize 0.75x** | **91.67%** | **100.00%** | 85.71% | 0.9231 | Zero degradation |
| **Resize 0.50x** | **91.67%** | **100.00%** | 85.71% | 0.9231 | Zero degradation |
| **Screenshot Re-encode** | **66.67%** | **100.00%** | 60.00% | 0.7500 | Loss of sensor grain causes false alarms |
| **Gaussian Blur (r=1.0)** | **50.00%** | **100.00%** | 50.00% | 0.6667 | Spatial low-pass filters mimic smooth AI |

### Key Robustness Insights:
1. **Flawless AI Recall Under All Distortions**: Recall remained **100.00% across all 9 conditions**. AI images never slipped through as real, even when compressed to Q=30 or resized by 50%.
2. **Compression Invariance to Q=65**: Typical web JPEG recompression (Q=65 to Q=85) causes **zero loss** in detection accuracy.
3. **Vulnerability to Smoothing / Blur**: Heavy Gaussian blur strips high-frequency sensor noise, which vision models mistakenly interpret as latent-space generative smoothing.

---

## 5. Messy Real & Chat-App Evaluation (`data/messy_real` & `data/my_images`)

### Messy Real Split Results:
- **Accuracy**: **68.00%**
- **Recall**: **98.67%**
- **False Positives**: 47 of 75 images.
- **Root Cause**: Random recompression coupled with downscaling degrades camera sensor noise. Vision models trained on uncompressed web images flag the recompression blur as generative smoothing.

### User WhatsApp Images Results:
- Evaluated: 5 authentic mobile camera photos sent via WhatsApp (`WhatsApp Image 2026-08-25...`).
- **Results**: All 5 were scored between 78.3% and 81.5% AI probability.
- **Why did this happen?**
  1. WhatsApp strips 100% of EXIF camera hardware tags (Make, Model, Lens, ISO). Per honesty rules, missing metadata is treated as neutral (`null`), leaving no hardware signal to anchor reality.
  2. WhatsApp applies aggressive 4:2:0 chroma subsampling and 8x8 block DCT quantization.
  3. Two of the three vision models (`Organika` and `Ateeqq`) misread the WhatsApp compression macroblocking as generative diffusion artifacts.
  4. Crucially, the system detected the massive ML disagreement (`spread: 0.79 - 0.98`) and protected the user by assigning them to `Uncertain leaning AI` with `confidence: low` and explicit reason: `Machine learning detectors disagree substantially. The system recommends manual inspection.`

---

## 6. What the System Still Gets Wrong (Documented Weaknesses)

1. **Recompressed & Chat-App Real Photos**:
   - Images transmitted through messaging platforms (WhatsApp, Messenger, Discord) lose hardware EXIF and acquire compression artifacts that trigger false alarms in modern ViTs.
2. **Extreme Low-Pass Filtering / Blur**:
   - Heavy Gaussian blur or artistic soft-focus photography removes the sensor noise fingerprint, causing detectors to suspect AI synthesis.
3. **Ultra-Small Resolution Thumbnails (<64x64)**:
   - Bilinear and bicubic interpolation artifacts on small crops can mask subtle diffusion watermarks or mimic synthetic smoothing.
4. **Artistic CGI / 3D Renders**:
   - Photorealistic video game screenshots or 3D engine renders (Unreal Engine 5) exhibit smooth synthetic textures that may be flagged as generative AI.

---

## 7. Ranked Future Improvements

1. **Domain-Specific Contrastive Tuning on Recompressed Mobile Photos**:
   - Train or fine-tune a lightweight backbone (e.g., MobileNetV4 or ConvNeXt-V2) explicitly augmented with randomized JPEG compression, chat-app re-encoding (WebP/WhatsApp presets), and EXIF stripping.
2. **Frequency Cross-Correlation Filter for 8x8 DCT Grid**:
   - Implement an automated DCT grid detector to distinguish standard JPEG block boundary discontinuities from genuine generative upsampling checkerboard artifacts.
3. **Test-Time Augmentation (TTA)**:
   - Average predictions over random 5-degree rotations and horizontal flips to smooth out single-view interpolation noise.
4. **Gemini 2.5/Flash Multimodal Semantic Layer**:
   - Enable `GEMINI_API_KEY` for semantic physical verification (e.g. checking anatomical finger coherence, unnatural text lettering, impossible shadows) to break ties when ML models disagree.
