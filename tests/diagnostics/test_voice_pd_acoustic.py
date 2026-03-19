"""Tests for acoustic feature extraction (jitter, shimmer, HNR)."""

import numpy as np
import pytest

from bioagentics.voice_pd.config import SAMPLE_RATE
from bioagentics.voice_pd.features.acoustic import extract_acoustic_features_from_array


def _synthetic_vowel(freq: float = 150.0, duration: float = 1.0, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Generate a synthetic sustained vowel-like signal (harmonic + noise)."""
    t = np.arange(int(sr * duration)) / sr
    # Fundamental + harmonics (simulates vocal folds)
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    signal += 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    # Small noise component
    rng = np.random.default_rng(42)
    signal += 0.02 * rng.standard_normal(len(signal))
    return signal.astype(np.float64)


class TestAcousticFeatures:
    def test_returns_expected_keys(self):
        y = _synthetic_vowel()
        feats = extract_acoustic_features_from_array(y)
        assert set(feats.keys()) == {"local_jitter", "local_shimmer", "hnr_db", "nhr"}

    def test_synthetic_vowel_produces_values(self):
        y = _synthetic_vowel()
        feats = extract_acoustic_features_from_array(y)
        # A clean harmonic signal should yield non-None values
        assert feats["local_jitter"] is not None
        assert feats["local_shimmer"] is not None
        assert feats["hnr_db"] is not None

    def test_hnr_positive_for_clean_signal(self):
        y = _synthetic_vowel()
        feats = extract_acoustic_features_from_array(y)
        # Clean signal should have high HNR (positive dB)
        assert feats["hnr_db"] is not None
        assert feats["hnr_db"] > 0

    def test_jitter_small_for_clean_signal(self):
        y = _synthetic_vowel()
        feats = extract_acoustic_features_from_array(y)
        # Clean periodic signal should have very low jitter
        assert feats["local_jitter"] is not None
        assert feats["local_jitter"] < 0.05  # less than 5%

    def test_handles_silence(self):
        y = np.zeros(SAMPLE_RATE)  # 1 second of silence
        feats = extract_acoustic_features_from_array(y)
        # Should not crash, values may be None
        assert isinstance(feats, dict)

    def test_handles_short_audio(self):
        y = _synthetic_vowel(duration=0.05)  # 50ms
        feats = extract_acoustic_features_from_array(y)
        assert isinstance(feats, dict)
