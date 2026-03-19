"""Mel-spectrogram and chromagram feature extraction.

Complements the existing MFCC features (voice_pd) and cough spectral shape
features with mel-spectrogram band statistics and chroma features for
multi-disease acoustic screening.
"""

import logging
from pathlib import Path

import numpy as np

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import (
    HOP_LENGTH,
    N_FFT,
    N_MELS,
    SAMPLE_RATE,
)

log = logging.getLogger(__name__)

# Number of summary statistics per mel band / chroma bin
_STAT_NAMES = ("mean", "std", "skew", "kurtosis")


def _summary_stats(x: np.ndarray) -> tuple[float, float, float, float]:
    """Compute mean, std, skew, kurtosis of a 1-D array."""
    mu = float(np.mean(x))
    sigma = float(np.std(x))
    if sigma < 1e-10 or len(x) < 3:
        return mu, sigma, 0.0, 0.0
    centered = x - mu
    skew = float(np.mean(centered**3) / (sigma**3))
    kurt = float(np.mean(centered**4) / (sigma**4) - 3.0)
    return mu, sigma, skew, kurt


def extract_melspec_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract mel-spectrogram summary statistics from an audio file.

    Computes a 128-band mel-spectrogram, then summarizes across frames
    with mean, std, skew, kurtosis for each band. Also includes global
    statistics across all bands.

    Returns dict with keys like melspec_band_0_mean, ..., melspec_global_mean.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_melspec()

    return _melspec_from_array(y, int(sr))


def extract_melspec_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract mel-spectrogram features from a numpy audio array."""
    return _melspec_from_array(y, sr)


def _melspec_from_array(y: np.ndarray, sr: int) -> dict[str, float | None]:
    """Core mel-spectrogram feature extraction."""
    import librosa

    if len(y) < sr * 0.05:
        return _empty_melspec()

    features: dict[str, float | None] = {}

    try:
        S = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS
        )
        S_db = librosa.power_to_db(S, ref=np.max)

        # Per-band summary statistics (subsample bands to keep feature count manageable)
        # Use 16 evenly-spaced bands out of 128 to get 64 features instead of 512
        band_indices = np.linspace(0, N_MELS - 1, 16, dtype=int)
        for idx in band_indices:
            band = S_db[idx]
            mu, sigma, skew, kurt = _summary_stats(band)
            features[f"melspec_b{idx}_mean"] = mu
            features[f"melspec_b{idx}_std"] = sigma
            features[f"melspec_b{idx}_skew"] = skew
            features[f"melspec_b{idx}_kurt"] = kurt

        # Global statistics across all bands and frames
        flat = S_db.flatten()
        g_mu, g_sigma, g_skew, g_kurt = _summary_stats(flat)
        features["melspec_global_mean"] = g_mu
        features["melspec_global_std"] = g_sigma
        features["melspec_global_skew"] = g_skew
        features["melspec_global_kurt"] = g_kurt

        # Band-energy ratio: low (0-25%) vs high (75-100%) mel bands
        low_energy = float(np.mean(S_db[: N_MELS // 4]))
        high_energy = float(np.mean(S_db[3 * N_MELS // 4 :]))
        features["melspec_low_high_ratio"] = (
            low_energy / high_energy if abs(high_energy) > 1e-10 else None
        )

    except Exception:
        log.exception("Mel-spectrogram extraction failed")
        return _empty_melspec()

    return features


def extract_chroma_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract chromagram features from an audio file.

    Computes 12-bin chroma features and summarizes across frames.

    Returns dict with keys like chroma_0_mean, ..., chroma_deviation.
    """
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        return _empty_chroma()

    return _chroma_from_array(y, int(sr))


def extract_chroma_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract chroma features from a numpy audio array."""
    return _chroma_from_array(y, sr)


def _chroma_from_array(y: np.ndarray, sr: int) -> dict[str, float | None]:
    """Core chromagram feature extraction."""
    import librosa

    if len(y) < sr * 0.05:
        return _empty_chroma()

    features: dict[str, float | None] = {}

    try:
        chroma = librosa.feature.chroma_stft(
            y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH
        )

        # Per-chroma-bin summary statistics (12 bins x 4 stats = 48 features)
        for i in range(12):
            mu, sigma, skew, kurt = _summary_stats(chroma[i])
            features[f"chroma_{i}_mean"] = mu
            features[f"chroma_{i}_std"] = sigma
            features[f"chroma_{i}_skew"] = skew
            features[f"chroma_{i}_kurt"] = kurt

        # Chroma deviation (measure of tonal clarity)
        chroma_means = np.mean(chroma, axis=1)
        features["chroma_deviation"] = float(np.std(chroma_means))

    except Exception:
        log.exception("Chromagram extraction failed")
        return _empty_chroma()

    return features


def extract_spectral_features(audio_path: str | Path) -> dict[str, float | None]:
    """Extract both mel-spectrogram and chromagram features."""
    import librosa

    try:
        y, sr = librosa.load(str(audio_path), sr=SAMPLE_RATE, mono=True)
    except Exception:
        log.exception("Failed to load audio: %s", audio_path)
        feats = _empty_melspec()
        feats.update(_empty_chroma())
        return feats

    feats = _melspec_from_array(y, int(sr))
    feats.update(_chroma_from_array(y, int(sr)))
    return feats


def extract_spectral_features_from_array(
    y: np.ndarray, sr: int = SAMPLE_RATE
) -> dict[str, float | None]:
    """Extract both mel-spectrogram and chromagram features from array."""
    feats = _melspec_from_array(y, sr)
    feats.update(_chroma_from_array(y, sr))
    return feats


def _empty_melspec() -> dict[str, float | None]:
    """Return mel-spectrogram feature dict with all None values."""
    features: dict[str, float | None] = {}
    band_indices = np.linspace(0, N_MELS - 1, 16, dtype=int)
    for idx in band_indices:
        for stat in _STAT_NAMES:
            features[f"melspec_b{idx}_{stat}"] = None
    for stat in _STAT_NAMES:
        features[f"melspec_global_{stat}"] = None
    features["melspec_low_high_ratio"] = None
    return features


def _empty_chroma() -> dict[str, float | None]:
    """Return chroma feature dict with all None values."""
    features: dict[str, float | None] = {}
    for i in range(12):
        for stat in _STAT_NAMES:
            features[f"chroma_{i}_{stat}"] = None
    features["chroma_deviation"] = None
    return features
