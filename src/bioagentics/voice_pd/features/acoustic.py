"""Acoustic biomarker extraction using Parselmouth (Praat interface).

Extracts per-recording: local jitter, local shimmer, HNR, NHR.
These are core acoustic biomarkers for PD voice analysis.
"""

import logging
from pathlib import Path

import numpy as np
import parselmouth
from parselmouth.praat import call

from bioagentics.voice_pd.config import SAMPLE_RATE

log = logging.getLogger(__name__)

# Pitch analysis defaults (suitable for adult speech)
PITCH_FLOOR = 75.0  # Hz
PITCH_CEILING = 500.0  # Hz


def extract_acoustic_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract jitter, shimmer, HNR, and NHR from an audio file.

    Args:
        audio_path: Path to a WAV file (16kHz mono expected).

    Returns:
        Dict with keys: local_jitter, local_shimmer, hnr_db, nhr.
        Values are None if extraction fails for that feature.
    """
    features: dict[str, float | None] = {
        "local_jitter": None,
        "local_shimmer": None,
        "hnr_db": None,
        "nhr": None,
    }

    try:
        snd = parselmouth.Sound(str(audio_path))
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return features

    # Pitch object needed for jitter/shimmer
    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    except Exception:
        log.exception("Pitch extraction failed: %s", audio_path)
        return features

    # Jitter (local) — cycle-to-cycle pitch perturbation
    try:
        features["local_jitter"] = call(
            point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
        )
    except Exception:
        log.warning("Jitter extraction failed: %s", audio_path)

    # Shimmer (local) — cycle-to-cycle amplitude perturbation
    try:
        features["local_shimmer"] = call(
            [snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        log.warning("Shimmer extraction failed: %s", audio_path)

    # Harmonics-to-noise ratio
    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0.0, 0.0)
        if not np.isnan(hnr):
            features["hnr_db"] = hnr
            # NHR is inverse of HNR in linear scale
            features["nhr"] = 1.0 / (10.0 ** (hnr / 10.0)) if hnr > 0 else None
    except Exception:
        log.warning("HNR extraction failed: %s", audio_path)

    return features


def extract_acoustic_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract acoustic features from a numpy audio array.

    Creates a temporary Parselmouth Sound object from the array.
    """
    features: dict[str, float | None] = {
        "local_jitter": None,
        "local_shimmer": None,
        "hnr_db": None,
        "nhr": None,
    }

    try:
        snd = parselmouth.Sound(y, sampling_frequency=sr)
    except Exception:
        log.exception("Failed to create Sound from array")
        return features

    try:
        point_process = call(snd, "To PointProcess (periodic, cc)", PITCH_FLOOR, PITCH_CEILING)
    except Exception:
        log.exception("Pitch extraction failed from array")
        return features

    try:
        features["local_jitter"] = call(
            point_process, "Get jitter (local)", 0.0, 0.0, 0.0001, 0.02, 1.3
        )
    except Exception:
        log.warning("Jitter extraction failed from array")

    try:
        features["local_shimmer"] = call(
            [snd, point_process], "Get shimmer (local)", 0.0, 0.0, 0.0001, 0.02, 1.3, 1.6
        )
    except Exception:
        log.warning("Shimmer extraction failed from array")

    try:
        harmonicity = call(snd, "To Harmonicity (cc)", 0.01, PITCH_FLOOR, 0.1, 1.0)
        hnr = call(harmonicity, "Get mean", 0.0, 0.0)
        if not np.isnan(hnr):
            features["hnr_db"] = hnr
            features["nhr"] = 1.0 / (10.0 ** (hnr / 10.0)) if hnr > 0 else None
    except Exception:
        log.warning("HNR extraction failed from array")

    return features
