"""Frequency domain / spectral analysis detector for RealLens.

Analyzes 2D Fourier Transform (FFT) power spectra for:
1. Periodic upsampling / deconvolution grid peaks (common in GANs and VAE decoders).
2. Unnatural high-frequency energy decay slopes (deviations from natural 1/f photographic spectra).
"""
from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from PIL import Image


def _compute_radial_profile(log_power: np.ndarray, center: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
    """Computes azimuthally averaged radial power distribution."""
    y, x = np.indices(log_power.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2).astype(np.int32)
    max_r = min(center[0], center[1])

    # Radial binning
    r_flat = r.ravel()
    val_flat = log_power.ravel()
    mask = r_flat <= max_r

    r_masked = r_flat[mask]
    val_masked = val_flat[mask]

    bin_counts = np.bincount(r_masked)
    radial_sum = np.bincount(r_masked, weights=val_masked)
    radial_profile = np.zeros(max_r + 1, dtype=np.float32)

    nonzero = bin_counts > 0
    radial_profile[nonzero] = radial_sum[nonzero] / bin_counts[nonzero]

    freqs = np.arange(len(radial_profile)) / float(max_r)
    return freqs, radial_profile


def analyze_frequency(
    image_source: Union[str, Path, bytes, Image.Image],
    target_dim: int = 512,
    image_type: str = "photo-like",
) -> Dict[str, Any]:
    """Performs FFT spectral analysis on the image.

    Args:
        image_source: Path, bytes, or PIL Image.
        target_dim: Target dimension for normalized spectral analysis (default 512).

    Returns:
        dict:
            score: float (0.0=natural camera spectrum, 1.0=unnatural synthetic spectral artifacts)
            confidence: "low" | "medium"
            evidence: list of str
            metrics: dict of frequency metrics
    """
    evidence: List[str] = []
    close_img = False
    img: Optional[Image.Image] = None

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
                "evidence": ["Unsupported image input type for frequency analysis."],
                "metrics": {},
            }

        # Convert to grayscale
        gray = img.convert("L")

        # Resize to square standard dimension for consistent frequency binning
        resized = gray.resize((target_dim, target_dim), Image.Resampling.BILINEAR)
        arr = np.array(resized, dtype=np.float32)

        # Apply 2D Hann window to reduce boundary spectral leakage
        hann_1d = np.hanning(target_dim)
        hann_2d = np.outer(hann_1d, hann_1d)
        windowed = arr * hann_2d

        # 2D FFT
        fft2 = np.fft.fft2(windowed)
        fft_shift = np.fft.fftshift(fft2)
        power = np.abs(fft_shift) ** 2
        log_power = np.log1p(power)

        center = (target_dim // 2, target_dim // 2)

        # 1. Radial profile and spectral decay slope (1/f^beta)
        freqs, radial_prof = _compute_radial_profile(log_power, center)
        # Fit slope in mid-to-high frequency band (avoid DC and near-Nyquist cutoff)
        fit_mask = (freqs >= 0.15) & (freqs <= 0.85)
        if np.sum(fit_mask) > 10:
            log_f = np.log(freqs[fit_mask])
            log_p = radial_prof[fit_mask]
            # Linear regression: p ~ a + slope * log_f
            coeffs = np.polyfit(log_f, log_p, 1)
            decay_slope = float(-coeffs[0])
        else:
            decay_slope = 1.8

        # 2. Peak artifact detection (identifying grid artifacts from deconvolution)
        # Background subtraction using local Gaussian/box blur approximation
        bg = np.zeros_like(log_power)
        # Uniform block filter approximation
        block_size = 16
        for i in range(0, target_dim, block_size):
            for j in range(0, target_dim, block_size):
                bg[i : i + block_size, j : j + block_size] = np.mean(
                    log_power[i : i + block_size, j : j + block_size]
                )

        diff_spectrum = log_power - bg
        # Zero out center DC and axes (axes often have natural orientation energy)
        diff_spectrum[center[0] - 8 : center[0] + 8, :] = 0
        diff_spectrum[:, center[1] - 8 : center[1] + 8] = 0

        std_diff = np.std(diff_spectrum)
        mean_diff = np.mean(diff_spectrum)
        threshold = mean_diff + 3.8 * std_diff
        peaks = diff_spectrum > threshold
        peak_count = int(np.sum(peaks))

        # 3. High-frequency energy ratio
        high_freq_mask = freqs > 0.70
        high_freq_energy = float(np.sum(radial_prof[high_freq_mask]))
        total_energy = float(np.sum(radial_prof))
        hf_ratio = high_freq_energy / max(1e-6, total_energy)

        metrics = {
            "decay_slope": round(decay_slope, 3),
            "peak_count": peak_count,
            "hf_energy_ratio": round(hf_ratio, 4),
        }

        # Scoring logic:
        # Natural photographic images have smooth decay slopes (typically 1.4 to 2.8) and virtually 0 isolated spectral peaks.
        # GAN and diffusion upsamplers create discrete peaks (peak_count > 15) or unnaturally steep/flat slopes.
        if image_type != "photo-like":
            score = None
            applicable = False
            status_text = "Not applicable"
            conf = "low"
            evidence.append(
                f"Image classified as '{image_type}'. 2D FFT spectral peak analysis is not applicable for non-photographic graphics or documents."
            )
        elif peak_count > 25:
            score = min(0.88, 0.55 + 0.01 * (peak_count - 25))
            applicable = True
            status_text = f"Anomalous peaks ({round(score*100):.0f}%)"
            conf = "medium"
            evidence.append(
                f"Detected {peak_count} anomalous spectral peaks in 2D FFT, characteristic of neural upsampling/deconvolution grid artifacts."
            )
        elif peak_count < 8 and 1.3 <= decay_slope <= 2.6:
            score = max(0.20, 0.45 - 0.15)
            applicable = True
            status_text = f"Natural spectrum ({round(score*100):.0f}%)"
            conf = "medium"
            evidence.append(
                f"Spectral decay slope ({decay_slope:.2f}) and clean Fourier domain are consistent with natural photographic optical dispersion."
            )
        else:
            score = None
            applicable = False
            status_text = "Not applicable"
            conf = "low"
            evidence.append(
                f"Fourier spectrum shows moderate profile (slope: {decay_slope:.2f}, {peak_count} peaks) without definitive synthetic lattice artifacts."
            )

        evidence.append(f"High-frequency octave power ratio: {hf_ratio*100:.1f}%.")

        return {
            "score": round(score, 3) if score is not None else None,
            "applicable": applicable,
            "status_text": status_text,
            "confidence": conf,
            "evidence": evidence,
            "metrics": metrics,
        }

    except Exception as exc:
        return {
            "score": None,
            "confidence": "low",
            "evidence": [f"Spectral FFT analysis failed: {str(exc)}"],
            "metrics": {},
        }
    finally:
        if close_img and img:
            try:
                img.close()
            except Exception:
                pass
