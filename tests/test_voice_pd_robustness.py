"""Tests for voice_pd robustness module (augmentation, quality tiers, cross-dataset)."""

import numpy as np
import pytest


@pytest.fixture
def clean_signal():
    """High-quality broadband synthetic signal (speech-like harmonics)."""
    rng = np.random.default_rng(42)
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    # Harmonics spanning 200-6000 Hz to simulate voiced speech
    freqs = [200, 400, 800, 1200, 2000, 3000, 4500, 6000]
    y = sum(0.3 / (i + 1) * np.sin(2 * np.pi * f * t) for i, f in enumerate(freqs))
    y += rng.normal(0, 0.001, size=y.shape)  # tiny noise floor
    y = y / np.abs(y).max() * 0.7  # normalize away from clipping
    return y.astype(np.float32)


@pytest.fixture
def noisy_signal():
    """Low-quality signal: heavy noise, clipping."""
    rng = np.random.default_rng(99)
    sr = 16_000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False)
    y = 0.3 * np.sin(2 * np.pi * 500 * t) + rng.normal(0, 0.5, size=t.shape)
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)


# ── Augmentation tests ──


class TestAddBackgroundNoise:
    def test_output_shape(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_background_noise

        out = add_background_noise(clean_signal, snr_db=20.0)
        assert out.shape == clean_signal.shape

    def test_noisier_at_lower_snr(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_background_noise

        rng = np.random.default_rng(0)
        loud = add_background_noise(clean_signal, snr_db=5.0, rng=rng)
        rng2 = np.random.default_rng(0)
        quiet = add_background_noise(clean_signal, snr_db=40.0, rng=rng2)
        # More noise -> larger absolute difference from original
        assert np.mean(np.abs(loud - clean_signal)) > np.mean(np.abs(quiet - clean_signal))

    def test_silent_signal_unchanged(self):
        from bioagentics.voice_pd.robustness.augmentation import add_background_noise

        silent = np.zeros(1600, dtype=np.float32)
        out = add_background_noise(silent, snr_db=20.0)
        np.testing.assert_array_equal(out, silent)


class TestAddMicrophoneArtifacts:
    def test_output_shape(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_microphone_artifacts

        out = add_microphone_artifacts(clean_signal)
        assert out.shape == clean_signal.shape

    def test_reduces_bandwidth(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_microphone_artifacts

        filtered = add_microphone_artifacts(clean_signal, low_cutoff_hz=300.0, high_cutoff_hz=3400.0)
        # Narrowband filter should reduce energy outside passband
        assert not np.array_equal(filtered, clean_signal)


class TestAddCompressionArtifacts:
    def test_output_shape(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_compression_artifacts

        out = add_compression_artifacts(clean_signal, bit_depth=8)
        assert out.shape == clean_signal.shape

    def test_quantization_changes_signal(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import add_compression_artifacts

        out = add_compression_artifacts(clean_signal, bit_depth=4)
        assert not np.array_equal(out, clean_signal)


class TestAugmentAudio:
    def test_full_chain(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import augment_audio

        out = augment_audio(clean_signal, noise_snr_db=20.0, compression_bits=8)
        assert out.shape == clean_signal.shape
        assert not np.array_equal(out, clean_signal)

    def test_skip_all(self, clean_signal):
        from bioagentics.voice_pd.robustness.augmentation import augment_audio

        out = augment_audio(
            clean_signal, noise_snr_db=None, mic_low_hz=None,
            mic_high_hz=None, compression_bits=None,
        )
        np.testing.assert_array_equal(out, clean_signal)


# ── Quality tier tests ──


class TestAssignQualityTier:
    def test_clean_signal_is_high(self, clean_signal):
        from bioagentics.voice_pd.robustness.quality_tiers import assign_quality_tier

        tier = assign_quality_tier(clean_signal)
        assert tier in ("high", "medium")

    def test_noisy_signal_is_low(self, noisy_signal):
        from bioagentics.voice_pd.robustness.quality_tiers import assign_quality_tier

        tier = assign_quality_tier(noisy_signal)
        assert tier == "low"

    def test_returns_valid_tier(self, clean_signal):
        from bioagentics.voice_pd.robustness.quality_tiers import assign_quality_tier

        tier = assign_quality_tier(clean_signal)
        assert tier in ("high", "medium", "low")

    def test_short_signal(self):
        from bioagentics.voice_pd.robustness.quality_tiers import assign_quality_tier

        short = np.zeros(100, dtype=np.float32)
        tier = assign_quality_tier(short)
        assert tier in ("high", "medium", "low")


class TestEvaluateByQualityTier:
    def test_per_tier_auc(self, tmp_path):
        from bioagentics.voice_pd.robustness.quality_tiers import evaluate_by_quality_tier

        rng = np.random.default_rng(42)
        y_true = np.array([1] * 20 + [0] * 20, dtype=np.float32)
        y_prob = np.clip(y_true * 0.7 + rng.random(40) * 0.3, 0, 1)
        tiers = (["high"] * 14 + ["medium"] * 14 + ["low"] * 12)

        results = evaluate_by_quality_tier(y_true, y_prob, tiers, output_dir=tmp_path)
        assert "tiers" in results
        assert "overall_auc" in results
        assert results["overall_auc"] is not None

    def test_saves_json(self, tmp_path):
        from bioagentics.voice_pd.robustness.quality_tiers import evaluate_by_quality_tier

        y_true = np.array([1] * 10 + [0] * 10, dtype=np.float32)
        y_prob = np.linspace(0.9, 0.1, 20)
        tiers = ["high"] * 10 + ["medium"] * 10

        evaluate_by_quality_tier(y_true, y_prob, tiers, output_dir=tmp_path)
        assert (tmp_path / "quality_tier_results.json").exists()

    def test_empty_tier_handled(self, tmp_path):
        from bioagentics.voice_pd.robustness.quality_tiers import evaluate_by_quality_tier

        y_true = np.array([1, 0, 1, 0], dtype=np.float32)
        y_prob = np.array([0.9, 0.1, 0.8, 0.2])
        tiers = ["high", "high", "high", "high"]  # no medium/low

        results = evaluate_by_quality_tier(y_true, y_prob, tiers, output_dir=tmp_path)
        assert results["tiers"]["medium"]["auc"] is None
        assert results["tiers"]["low"]["auc"] is None

    def test_single_class_tier(self, tmp_path):
        from bioagentics.voice_pd.robustness.quality_tiers import evaluate_by_quality_tier

        y_true = np.array([1, 1, 0, 0], dtype=np.float32)
        y_prob = np.array([0.9, 0.8, 0.2, 0.1])
        tiers = ["high", "high", "low", "low"]  # each tier has only one class

        results = evaluate_by_quality_tier(y_true, y_prob, tiers, output_dir=tmp_path)
        assert results["tiers"]["high"]["auc"] is None
        assert results["tiers"]["low"]["auc"] is None


# ── Cross-dataset tests ──


class TestCrossDatasetEvaluate:
    def test_basic_evaluation(self, tmp_path):
        from bioagentics.voice_pd.robustness.cross_dataset import cross_dataset_evaluate

        rng = np.random.default_rng(42)
        n_train, n_test, n_feat = 60, 20, 10
        train_X = rng.standard_normal((n_train, n_feat))
        train_y = np.array([1] * 30 + [0] * 30, dtype=np.float64)
        test_X = rng.standard_normal((n_test, n_feat))
        test_y = np.array([1] * 10 + [0] * 10, dtype=np.float64)

        results = cross_dataset_evaluate(
            train_X, train_y,
            {"uci": (test_X, test_y)},
            output_dir=tmp_path,
        )
        assert "datasets" in results
        assert "uci" in results["datasets"]
        assert results["datasets"]["uci"]["auc"] is not None

    def test_saves_json(self, tmp_path):
        from bioagentics.voice_pd.robustness.cross_dataset import cross_dataset_evaluate

        rng = np.random.default_rng(42)
        train_X = rng.standard_normal((40, 5))
        train_y = np.array([1] * 20 + [0] * 20, dtype=np.float64)
        test_X = rng.standard_normal((20, 5))
        test_y = np.array([1] * 10 + [0] * 10, dtype=np.float64)

        cross_dataset_evaluate(
            train_X, train_y,
            {"test_set": (test_X, test_y)},
            output_dir=tmp_path,
        )
        assert (tmp_path / "cross_dataset_results.json").exists()
