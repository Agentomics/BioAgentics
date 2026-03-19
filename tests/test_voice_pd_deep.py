"""Tests for voice_pd deep spectrogram CNN module."""

import numpy as np
import pytest
import torch


@pytest.fixture
def synthetic_audio_file(tmp_path):
    """Create a synthetic 16kHz mono WAV file."""
    import soundfile as sf

    sr = 16_000
    duration = 5.0
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Simple sine wave at 200Hz
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def synthetic_spectrograms():
    """Create synthetic spectrogram data for training tests."""
    rng = np.random.default_rng(42)
    n_samples = 40
    n_mels = 128
    time_frames = 157  # matches 5s audio at 16kHz with hop=512
    spectrograms = rng.random((n_samples, 3, n_mels, time_frames)).astype(np.float32)
    labels = np.array([1] * 20 + [0] * 20, dtype=np.float32)
    return spectrograms, labels


class TestMelSpectrogram:
    def test_shape(self, synthetic_audio_file):
        from bioagentics.voice_pd.deep.spectrogram import audio_to_mel_spectrogram

        mel = audio_to_mel_spectrogram(synthetic_audio_file)
        assert mel.ndim == 2
        assert mel.shape[0] == 128  # n_mels
        assert mel.shape[1] > 0  # time frames

    def test_range(self, synthetic_audio_file):
        from bioagentics.voice_pd.deep.spectrogram import audio_to_mel_spectrogram

        mel = audio_to_mel_spectrogram(synthetic_audio_file)
        assert mel.min() >= 0.0
        assert mel.max() <= 1.0

    def test_rgb_conversion(self, synthetic_audio_file):
        from bioagentics.voice_pd.deep.spectrogram import (
            audio_to_mel_spectrogram,
            mel_to_rgb_tensor,
        )

        mel = audio_to_mel_spectrogram(synthetic_audio_file)
        rgb = mel_to_rgb_tensor(mel)
        assert rgb.shape == (3, mel.shape[0], mel.shape[1])
        # All channels should be identical
        np.testing.assert_array_equal(rgb[0], rgb[1])
        np.testing.assert_array_equal(rgb[1], rgb[2])

    def test_cnn_input_pipeline(self, synthetic_audio_file):
        from bioagentics.voice_pd.deep.spectrogram import audio_to_cnn_input

        tensor = audio_to_cnn_input(synthetic_audio_file)
        assert tensor.ndim == 3
        assert tensor.shape[0] == 3


class TestSpectrogramCNN:
    def test_forward_pass(self):
        from bioagentics.voice_pd.deep.cnn_model import build_model

        model = build_model(pretrained=False)
        x = torch.randn(2, 3, 128, 157)
        out = model(x)
        assert out.shape == (2, 1)

    def test_parameter_count(self):
        from bioagentics.voice_pd.deep.cnn_model import build_model

        model = build_model(pretrained=False)
        n_params = sum(p.numel() for p in model.parameters())
        # MobileNetV2 is ~3.4M params; with our head it should be under 4M
        assert n_params < 5_000_000


class TestSpectrogramDataset:
    def test_dataset(self, synthetic_spectrograms):
        from bioagentics.voice_pd.deep.train import SpectrogramDataset

        specs, labels = synthetic_spectrograms
        ds = SpectrogramDataset(specs, labels)
        assert len(ds) == 40
        x, y = ds[0]
        assert isinstance(x, torch.Tensor)
        assert isinstance(y, torch.Tensor)
        assert x.shape == (3, 128, 157)


class TestTrainDeepModel:
    def test_train_runs(self, synthetic_spectrograms, tmp_path):
        from bioagentics.voice_pd.deep.train import train_deep_model

        specs, labels = synthetic_spectrograms
        results = train_deep_model(
            specs, labels,
            output_dir=tmp_path,
            n_splits=2,
            epochs=2,
            batch_size=8,
            pretrained=False,
        )
        assert "mean_auc" in results
        assert "fold_aucs" in results
        assert len(results["fold_aucs"]) == 2
        assert 0.0 <= results["mean_auc"] <= 1.0
        assert (tmp_path / "deep_cnn_results.json").exists()
        assert (tmp_path / "deep_cnn_model.pt").exists()
