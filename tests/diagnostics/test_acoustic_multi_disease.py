"""Tests for acoustic multi-disease panel feature extraction."""

import csv
import tempfile
from pathlib import Path

import numpy as np

from bioagentics.diagnostics.acoustic_multi_disease_panel.config import SAMPLE_RATE
from bioagentics.diagnostics.acoustic_multi_disease_panel.features.cough import (
    _segment_cough_events,
    extract_cough_features_from_array,
)
from bioagentics.diagnostics.acoustic_multi_disease_panel.features.pipeline import (
    extract_all_features,
    extract_features,
    process_manifest,
)


def _synthetic_vowel(freq: float = 150.0, duration: float = 1.0) -> np.ndarray:
    """Generate a synthetic sustained vowel (harmonic + noise)."""
    t = np.arange(int(SAMPLE_RATE * duration)) / SAMPLE_RATE
    signal = np.sin(2 * np.pi * freq * t)
    signal += 0.5 * np.sin(2 * np.pi * 2 * freq * t)
    signal += 0.25 * np.sin(2 * np.pi * 3 * freq * t)
    rng = np.random.default_rng(42)
    signal += 0.02 * rng.standard_normal(len(signal))
    return signal.astype(np.float64)


def _synthetic_cough(n_coughs: int = 3, duration: float = 2.0) -> np.ndarray:
    """Generate synthetic cough-like bursts (broadband noise bursts)."""
    n_samples = int(SAMPLE_RATE * duration)
    signal = np.zeros(n_samples)
    rng = np.random.default_rng(42)

    cough_len = int(SAMPLE_RATE * 0.15)  # 150ms per cough
    spacing = n_samples // (n_coughs + 1)

    for i in range(n_coughs):
        start = spacing * (i + 1)
        end = min(start + cough_len, n_samples)
        # Broadband noise burst with exponential decay
        burst = rng.standard_normal(end - start) * 0.8
        decay = np.exp(-np.linspace(0, 3, end - start))
        signal[start:end] = burst * decay

    return signal.astype(np.float64)


def _save_wav(y: np.ndarray, path: Path, sr: int = SAMPLE_RATE) -> None:
    """Save array to WAV file."""
    import soundfile as sf

    sf.write(str(path), y, sr)


class TestCoughFeatures:
    def test_returns_expected_keys(self):
        y = _synthetic_cough()
        feats = extract_cough_features_from_array(y)
        expected_keys = {
            "spectral_centroid_mean", "spectral_centroid_std",
            "spectral_rolloff_mean", "spectral_rolloff_std",
            "spectral_bandwidth_mean", "spectral_bandwidth_std",
            "spectral_flatness_mean", "spectral_flatness_std",
            "zcr_mean", "zcr_std",
            "n_cough_events",
            "cough_duration_mean_ms", "cough_duration_std_ms",
            "inter_cough_mean_ms", "inter_cough_std_ms",
            "cough_peak_freq_mean", "cough_peak_freq_std",
            "rms_energy_mean", "rms_energy_std", "rms_energy_max",
        }
        assert set(feats.keys()) == expected_keys

    def test_spectral_features_not_none(self):
        y = _synthetic_cough()
        feats = extract_cough_features_from_array(y)
        assert feats["spectral_centroid_mean"] is not None
        assert feats["spectral_rolloff_mean"] is not None
        assert feats["zcr_mean"] is not None
        assert feats["rms_energy_mean"] is not None

    def test_detects_cough_events(self):
        y = _synthetic_cough(n_coughs=3)
        feats = extract_cough_features_from_array(y)
        # Should detect at least 1 cough event
        assert feats["n_cough_events"] is not None
        assert feats["n_cough_events"] >= 1

    def test_silence_returns_no_coughs(self):
        y = np.zeros(SAMPLE_RATE)
        feats = extract_cough_features_from_array(y)
        assert feats["n_cough_events"] == 0.0

    def test_segment_cough_events(self):
        y = _synthetic_cough(n_coughs=2)
        events = _segment_cough_events(y, SAMPLE_RATE)
        assert len(events) >= 1
        for start, end in events:
            assert end > start

    def test_handles_short_audio(self):
        y = np.zeros(int(SAMPLE_RATE * 0.01))  # 10ms
        feats = extract_cough_features_from_array(y)
        assert isinstance(feats, dict)


class TestMultiDiseasePipeline:
    def test_extract_features_vowel(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "vowel.wav"
            _save_wav(_synthetic_vowel(), wav_path)
            feats = extract_features(wav_path, recording_type="sustained_vowel")
            # Should have acoustic + pitch + mfcc features (no temporal, no cough)
            assert "local_jitter" in feats
            assert "f0_mean" in feats
            assert "mfcc_1_mean" in feats
            assert "spectral_centroid_mean" not in feats
            assert "phonation_ratio" not in feats

    def test_extract_features_cough(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "cough.wav"
            _save_wav(_synthetic_cough(), wav_path)
            feats = extract_features(wav_path, recording_type="cough")
            # Should have cough + mfcc features (no acoustic, no temporal)
            assert "spectral_centroid_mean" in feats
            assert "mfcc_1_mean" in feats
            assert "local_jitter" not in feats

    def test_extract_features_reading(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "reading.wav"
            _save_wav(_synthetic_vowel(duration=3.0), wav_path)
            feats = extract_features(wav_path, recording_type="reading_passage")
            # Should have acoustic + pitch + mfcc + temporal
            assert "local_jitter" in feats
            assert "f0_mean" in feats
            assert "mfcc_1_mean" in feats
            assert "phonation_ratio" in feats

    def test_extract_all_features(self):
        with tempfile.TemporaryDirectory() as tmp:
            wav_path = Path(tmp) / "all.wav"
            _save_wav(_synthetic_vowel(duration=2.0), wav_path)
            feats = extract_all_features(wav_path)
            # Should have everything
            assert "local_jitter" in feats
            assert "spectral_centroid_mean" in feats
            assert "mfcc_1_mean" in feats
            assert "phonation_ratio" in feats

    def test_process_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)

            # Create test audio files
            vowel_path = tmp_path / "v1.wav"
            cough_path = tmp_path / "c1.wav"
            _save_wav(_synthetic_vowel(), vowel_path)
            _save_wav(_synthetic_cough(), cough_path)

            # Create manifest
            manifest_path = tmp_path / "manifest.csv"
            with open(manifest_path, "w", newline="") as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=[
                        "recording_id", "audio_path", "condition", "recording_type",
                    ],
                )
                writer.writeheader()
                writer.writerow({
                    "recording_id": "v1",
                    "audio_path": str(vowel_path),
                    "condition": "parkinsons",
                    "recording_type": "sustained_vowel",
                })
                writer.writerow({
                    "recording_id": "c1",
                    "audio_path": str(cough_path),
                    "condition": "respiratory",
                    "recording_type": "cough",
                })

            output_path = tmp_path / "features.csv"
            result = process_manifest(manifest_path, output_path=output_path)

            assert result.exists()
            with open(result) as f:
                reader = list(csv.DictReader(f))
                assert len(reader) == 2
                ids = {r["recording_id"] for r in reader}
                assert ids == {"v1", "c1"}
