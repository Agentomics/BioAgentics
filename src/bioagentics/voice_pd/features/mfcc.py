"""MFCC extraction — 13 coefficients + delta + delta-delta.

MFCCs capture vocal tract shape information relevant to articulation
changes in PD. Per-recording summary statistics yield a 78-dim vector.
"""

import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import N_MFCC, SAMPLE_RATE

log = logging.getLogger(__name__)


def extract_mfcc_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract MFCC summary statistics from an audio file.

    Computes 13 MFCCs + delta + delta-delta, then summarizes each
    coefficient across frames with mean and std -> 78 features.

    Returns dict with keys like mfcc_1_mean, mfcc_1_std, ..., delta2_13_std.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_features()

    return _extract_from_array(y, sr)


def extract_mfcc_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract MFCC summary statistics from a numpy audio array."""
    return _extract_from_array(y, sr)


def _extract_from_array(y: np.ndarray, sr: int) -> dict[str, float | None]:
    """Core MFCC extraction from audio array."""
    import librosa

    if len(y) < sr * 0.05:  # less than 50ms
        log.warning("Audio too short for MFCC extraction (%d samples)", len(y))
        return _empty_features()

    features: dict[str, float | None] = {}

    try:
        # 13 MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        # Delta (first derivative)
        delta = librosa.feature.delta(mfccs)
        # Delta-delta (second derivative)
        delta2 = librosa.feature.delta(mfccs, order=2)

        # Summary statistics per coefficient
        for i in range(N_MFCC):
            coeff = i + 1
            features[f"mfcc_{coeff}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{coeff}_std"] = float(np.std(mfccs[i]))
            features[f"delta_{coeff}_mean"] = float(np.mean(delta[i]))
            features[f"delta_{coeff}_std"] = float(np.std(delta[i]))
            features[f"delta2_{coeff}_mean"] = float(np.mean(delta2[i]))
            features[f"delta2_{coeff}_std"] = float(np.std(delta2[i]))

    except Exception:
        log.exception("MFCC extraction failed")
        return _empty_features()

    return features


def _empty_features() -> dict[str, float | None]:
    """Return feature dict with all None values."""
    features: dict[str, float | None] = {}
    for i in range(1, N_MFCC + 1):
        for prefix in ("mfcc", "delta", "delta2"):
            features[f"{prefix}_{i}_mean"] = None
            features[f"{prefix}_{i}_std"] = None
    return features
