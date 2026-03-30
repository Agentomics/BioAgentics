"""Tests for voice_pd composite spectrogram modules."""

import numpy as np
import pytest
import torch


@pytest.fixture
def synthetic_vowel_files(tmp_path):
    """Create synthetic 16kHz WAV files for 5 vowels."""
    import soundfile as sf

    sr = 16_000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    paths = {}
    freqs = {"a": 200, "e": 300, "i": 400, "o": 250, "u": 350}
    for vowel, freq in freqs.items():
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = tmp_path / f"vowel_{vowel}.wav"
        sf.write(str(path), audio, sr)
        paths[vowel] = path
    return paths


@pytest.fixture
def synthetic_composite_spectrograms():
    """Create synthetic composite spectrogram data (5 vowels stacked)."""
    rng = np.random.default_rng(42)
    n_samples = 30
    n_mels = 128
    time_frames = 157
    # Composite: 5 vowels stacked = 640 mel bins
    spectrograms = rng.random((n_samples, 3, n_mels * 5, time_frames)).astype(np.float32)
    labels = np.array([1] * 15 + [0] * 15, dtype=np.float32)
    return spectrograms, labels


# ── Composite Spectrogram Generation ──


class TestCompositeSpectrogram:
    def test_full_vowels(self, synthetic_vowel_files):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            generate_composite_spectrogram,
        )

        composite = generate_composite_spectrogram(synthetic_vowel_files)
        assert composite.ndim == 2
        assert composite.shape[0] == 128 * 5  # 5 vowels stacked
        assert composite.shape[1] > 0
        assert composite.min() >= 0.0
        assert composite.max() <= 1.0

    def test_missing_vowels_zero_padded(self, synthetic_vowel_files):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            generate_composite_spectrogram,
        )

        # Only provide 2 out of 5 vowels
        partial = {"a": synthetic_vowel_files["a"], "u": synthetic_vowel_files["u"]}
        composite = generate_composite_spectrogram(partial)
        assert composite.shape[0] == 128 * 5

        # Vowels e, i, o should be zero-padded (strips 1, 2, 3)
        for strip_idx in [1, 2, 3]:
            strip = composite[strip_idx * 128 : (strip_idx + 1) * 128, :]
            assert np.allclose(strip, 0.0)

        # Vowels a, u should have content
        a_strip = composite[0:128, :]
        u_strip = composite[4 * 128 : 5 * 128, :]
        assert a_strip.max() > 0.0
        assert u_strip.max() > 0.0

    def test_empty_vowels(self):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            generate_composite_spectrogram,
        )

        composite = generate_composite_spectrogram({})
        assert composite.shape[0] == 128 * 5
        assert np.allclose(composite, 0.0)

    def test_rgb_tensor(self, synthetic_vowel_files):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            composite_to_rgb_tensor,
            generate_composite_spectrogram,
        )

        composite = generate_composite_spectrogram(synthetic_vowel_files)
        rgb = composite_to_rgb_tensor(composite)
        assert rgb.shape == (3, 128 * 5, composite.shape[1])
        np.testing.assert_array_equal(rgb[0], rgb[1])

    def test_patient_pipeline(self, synthetic_vowel_files):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            patient_audio_to_composite,
        )

        tensor = patient_audio_to_composite(synthetic_vowel_files)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 3
        assert tensor.shape[1] == 128 * 5

    def test_batch_generate(self, synthetic_vowel_files):
        from bioagentics.voice_pd.deep.composite_spectrogram import (
            batch_generate_composites,
        )

        patients = [synthetic_vowel_files, synthetic_vowel_files]
        labels = np.array([1, 0], dtype=np.float32)
        composites, out_labels = batch_generate_composites(patients, labels)
        assert composites.shape[0] == 2
        assert composites.shape[1] == 3
        assert composites.shape[2] == 128 * 5
        np.testing.assert_array_equal(out_labels, labels)


# ── Composite CNN ──


class TestCompositeCNN:
    def test_forward_pass(self):
        from bioagentics.voice_pd.deep.composite_train import build_composite_model

        model = build_composite_model(pretrained=False)
        x = torch.randn(2, 3, 640, 157)
        out = model(x)
        assert out.shape == (2, 1)

    def test_train_runs(self, synthetic_composite_spectrograms, tmp_path):
        from bioagentics.voice_pd.deep.composite_train import train_composite_model

        specs, labels = synthetic_composite_spectrograms
        results = train_composite_model(
            specs, labels,
            output_dir=tmp_path,
            n_splits=2,
            epochs=2,
            batch_size=8,
            pretrained=False,
        )
        assert "mean_auc" in results
        assert "fold_metrics" in results
        assert len(results["fold_metrics"]) == 2
        assert 0.0 <= results["mean_auc"] <= 1.0
        assert results["class_balanced"] is True
        assert (tmp_path / "composite_cnn_results.json").exists()
        assert (tmp_path / "composite_cnn_model.pt").exists()


# ── Grad-CAM ──


class TestGradCAM:
    def test_heatmap_shape(self):
        from bioagentics.voice_pd.deep.composite_train import build_composite_model
        from bioagentics.voice_pd.deep.gradcam import GradCAM

        model = build_composite_model(pretrained=False)
        cam = GradCAM(model)
        x = torch.randn(1, 3, 640, 157)
        heatmap = cam(x)
        assert heatmap.shape == (640, 157)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0
        cam.release()

    def test_vowel_contributions(self):
        from bioagentics.voice_pd.deep.gradcam import vowel_contributions

        rng = np.random.default_rng(42)
        heatmap = rng.random((640, 157)).astype(np.float32)
        scores = vowel_contributions(heatmap)
        assert set(scores.keys()) == {"a", "e", "i", "o", "u"}
        for v in scores.values():
            assert 0.0 <= v <= 1.0

    def test_save_heatmap_overlay(self, tmp_path):
        from bioagentics.voice_pd.deep.gradcam import save_heatmap_overlay

        rng = np.random.default_rng(42)
        composite = rng.random((640, 157)).astype(np.float32)
        heatmap = rng.random((640, 157)).astype(np.float32)
        out_path = tmp_path / "overlay.png"
        save_heatmap_overlay(composite, heatmap, out_path)
        assert out_path.exists()


# ── Benchmark ──


class TestBenchmark:
    def test_bootstrap_ci(self):
        from bioagentics.voice_pd.deep.composite_benchmark import bootstrap_auroc_ci

        rng = np.random.default_rng(42)
        y_true = np.array([1] * 50 + [0] * 50)
        y_prob = rng.random(100)
        auc, lo, hi = bootstrap_auroc_ci(y_true, y_prob, n_bootstrap=200)
        assert 0.0 <= lo <= auc <= hi <= 1.0

    def test_run_benchmark(self, tmp_path):
        from bioagentics.voice_pd.deep.composite_benchmark import run_benchmark

        rng = np.random.default_rng(42)
        y_true = np.array([1] * 50 + [0] * 50, dtype=np.float32)

        models = {
            "composite_cnn": (y_true, rng.random(100).astype(np.float32)),
            "single_vowel_cnn": (y_true, rng.random(100).astype(np.float32)),
            "gradient_boosting": (y_true, rng.random(100).astype(np.float32)),
        }
        result = run_benchmark(models, output_dir=tmp_path)
        assert "models" in result
        assert "composite_cnn" in result["models"]
        assert "auroc" in result["models"]["composite_cnn"]
        assert "ci_95_lower" in result["models"]["composite_cnn"]
        assert (tmp_path / "composite_benchmark.json").exists()
