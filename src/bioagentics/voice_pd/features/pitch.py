"""F0 (fundamental frequency) and formant frequency extraction.

F0 variability reduction is a key PD biomarker (monotone speech).
Formant frequencies F1-F4 capture vocal tract configuration changes.
"""

import logging
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from bioagentics.voice_pd.config import SAMPLE_RATE

log = logging.getLogger(__name__)

PITCH_FLOOR = 75.0
PITCH_CEILING = 500.0
MAX_FORMANT_HZ = 5500.0  # for adult speech
N_FORMANTS = 5  # request 5 to reliably get F1-F4


def _f0_stats(pitch_values: np.ndarray) -> dict[str, float | None]:
    """Compute F0 statistics from voiced pitch values."""
    voiced = pitch_values[pitch_values > 0]
    if len(voiced) < 2:
        return {
            "f0_mean": None,
            "f0_std": None,
            "f0_min": None,
            "f0_max": None,
            "f0_range": None,
            "f0_cv": None,
        }
    mean = float(np.mean(voiced))
    std = float(np.std(voiced))
    return {
        "f0_mean": mean,
        "f0_std": std,
        "f0_min": float(np.min(voiced)),
        "f0_max": float(np.max(voiced)),
        "f0_range": float(np.max(voiced) - np.min(voiced)),
        "f0_cv": std / mean if mean > 0 else None,
    }


def _formant_stats(snd: parselmouth.Sound) -> dict[str, float | None]:
    """Extract mean formant frequencies F1-F4 and bandwidths."""
    feats: dict[str, float | None] = {}
    try:
        formant = call(snd, "To Formant (burg)", 0.0, N_FORMANTS, MAX_FORMANT_HZ, 0.025, 50.0)
        duration = call(snd, "Get total duration")

        for i in range(1, 5):  # F1-F4
            try:
                freq = call(formant, "Get mean", i, 0.0, duration, "hertz")
                bw = call(formant, "Get mean", i, 0.0, duration, "bark")
                feats[f"f{i}_mean"] = freq if not np.isnan(freq) else None
                feats[f"f{i}_bandwidth"] = bw if not np.isnan(bw) else None
            except Exception:
                feats[f"f{i}_mean"] = None
                feats[f"f{i}_bandwidth"] = None
    except Exception:
        log.warning("Formant extraction failed")
        for i in range(1, 5):
            feats[f"f{i}_mean"] = None
            feats[f"f{i}_bandwidth"] = None

    return feats


def extract_pitch_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract F0 statistics and formant features from an audio file.

    Returns dict with keys: f0_mean, f0_std, f0_min, f0_max, f0_range, f0_cv,
    f1_mean, f1_bandwidth, f2_mean, f2_bandwidth, f3_mean, f3_bandwidth,
    f4_mean, f4_bandwidth.
    """
    try:
        snd = parselmouth.Sound(str(audio_path))
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_features()

    return _extract_from_sound(snd)


def extract_pitch_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract F0 and formant features from a numpy audio array."""
    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
    except Exception:
        log.exception("Failed to create Sound from array")
        return _empty_features()

    return _extract_from_sound(snd)


def _extract_from_sound(snd: parselmouth.Sound) -> dict[str, float | None]:
    """Core extraction from a Parselmouth Sound object."""
    features: dict[str, float | None] = {}

    # F0 statistics
    try:
        pitch = call(snd, "To Pitch", 0.0, PITCH_FLOOR, PITCH_CEILING)
        pitch_values = pitch.selected_array["frequency"]
        features.update(_f0_stats(pitch_values))
    except Exception:
        log.warning("F0 extraction failed")
        features.update({
            "f0_mean": None, "f0_std": None, "f0_min": None,
            "f0_max": None, "f0_range": None, "f0_cv": None,
        })

    # Formant frequencies
    features.update(_formant_stats(snd))

    return features


def _empty_features() -> dict[str, float | None]:
    """Return feature dict with all None values."""
    keys = [
        "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range", "f0_cv",
        "f1_mean", "f1_bandwidth", "f2_mean", "f2_bandwidth",
        "f3_mean", "f3_bandwidth", "f4_mean", "f4_bandwidth",
    ]
    return {k: None for k in keys}
