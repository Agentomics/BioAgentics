"""Tests for voice_pd wav2vec 2.0 modules (features, contrastive, fusion)."""

import numpy as np
import pytest
import torch


@pytest.fixture
def synthetic_audio_file(tmp_path):
    """Create a synthetic 16kHz mono WAV file."""
    import soundfile as sf

    sr = 16_000
    duration = 2.0  # shorter for fast tests
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    audio = (0.5 * np.sin(2 * np.pi * 200 * t)).astype(np.float32)
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture
def synthetic_audio_files(tmp_path):
    """Create multiple synthetic audio files with labels."""
    import soundfile as sf

    sr = 16_000
    duration = 1.0
    paths = []
    for i in range(20):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        freq = 200 + i * 10  # slightly different frequencies
        audio = (0.5 * np.sin(2 * np.pi * freq * t)).astype(np.float32)
        path = tmp_path / f"audio_{i:03d}.wav"
        sf.write(str(path), audio, sr)
        paths.append(path)
    labels = np.array([1] * 10 + [0] * 10)
    return paths, labels


@pytest.fixture
def synthetic_embeddings():
    """Create synthetic wav2vec-like embeddings and classical features."""
    rng = np.random.default_rng(42)
    n = 40
    wav2vec_emb = rng.standard_normal((n, 768)).astype(np.float32)
    classical_feats = rng.standard_normal((n, 100)).astype(np.float32)
    labels = np.array([1] * 20 + [0] * 20)
    return wav2vec_emb, classical_feats, labels


# ── wav2vec_features tests ──


class TestWav2VecFeatures:
    def test_embedding_dim_constant(self):
        from bioagentics.voice_pd.deep.wav2vec_features import EMBEDDING_DIM

        assert EMBEDDING_DIM == 768

    def test_extract_embedding_returns_correct_shape(self, synthetic_audio_file):
        """Test single file extraction produces 768-d vector."""
        from bioagentics.voice_pd.deep.wav2vec_features import (
            extract_embedding,
            load_wav2vec_model,
        )

        model, processor, device = load_wav2vec_model()
        emb = extract_embedding(synthetic_audio_file, model, processor, device)
        assert emb.shape == (768,)
        assert emb.dtype == np.float32

    def test_batch_extract_shape(self, synthetic_audio_files):
        """Test batch extraction returns correct shape."""
        from bioagentics.voice_pd.deep.wav2vec_features import (
            batch_extract_embeddings,
        )

        paths, _ = synthetic_audio_files
        # Use first 4 files for speed
        emb = batch_extract_embeddings(paths[:4])
        assert emb.shape == (4, 768)
        assert emb.dtype == np.float32

    def test_batch_save_npy(self, synthetic_audio_files, tmp_path):
        """Test that batch extraction saves .npy file."""
        from bioagentics.voice_pd.deep.wav2vec_features import (
            batch_extract_embeddings,
        )

        paths, _ = synthetic_audio_files
        out = tmp_path / "emb.npy"
        batch_extract_embeddings(paths[:2], output_path=out)
        assert out.exists()
        loaded = np.load(out)
        assert loaded.shape == (2, 768)


# ── wav2vec_contrastive tests ──


class TestSupConLoss:
    def test_loss_output_scalar(self):
        from bioagentics.voice_pd.deep.wav2vec_contrastive import SupConLoss

        loss_fn = SupConLoss(temperature=0.07)
        features = torch.randn(8, 128)
        features = torch.nn.functional.normalize(features, dim=1)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        loss = loss_fn(features, labels)
        assert loss.ndim == 0
        assert loss.item() >= 0.0

    def test_loss_single_sample_returns_zero(self):
        from bioagentics.voice_pd.deep.wav2vec_contrastive import SupConLoss

        loss_fn = SupConLoss()
        features = torch.randn(1, 128)
        labels = torch.tensor([0])
        loss = loss_fn(features, labels)
        assert loss.item() == 0.0

    def test_loss_identical_class_lower(self):
        """Loss should be lower when same-class features are similar."""
        from bioagentics.voice_pd.deep.wav2vec_contrastive import SupConLoss

        loss_fn = SupConLoss(temperature=0.07)
        labels = torch.tensor([0, 0, 1, 1])

        # Clustered features (same class similar)
        clustered = torch.tensor([
            [1.0, 0.0], [0.9, 0.1],  # class 0
            [0.0, 1.0], [0.1, 0.9],  # class 1
        ])
        clustered = torch.nn.functional.normalize(clustered, dim=1)
        loss_good = loss_fn(clustered, labels)

        # Random features
        rng = torch.Generator().manual_seed(42)
        random_feats = torch.randn(4, 2, generator=rng)
        random_feats = torch.nn.functional.normalize(random_feats, dim=1)
        loss_random = loss_fn(random_feats, labels)

        assert loss_good.item() < loss_random.item()


class TestWav2VecContrastiveModel:
    def test_forward_shape(self):
        from bioagentics.voice_pd.deep.wav2vec_contrastive import (
            Wav2VecContrastiveModel,
        )

        model = Wav2VecContrastiveModel(proj_dim=128, freeze_encoder=True)
        model.eval()
        waveform = torch.randn(2, 16000)  # 2 samples, 1 second
        with torch.no_grad():
            out = model(waveform)
        assert out.shape == (2, 128)
        # Should be L2-normalized
        norms = torch.norm(out, dim=1)
        assert torch.allclose(norms, torch.ones(2), atol=1e-5)

    def test_extract_embeddings_shape(self):
        from bioagentics.voice_pd.deep.wav2vec_contrastive import (
            Wav2VecContrastiveModel,
        )

        model = Wav2VecContrastiveModel(freeze_encoder=True)
        model.eval()
        waveform = torch.randn(2, 16000)
        emb = model.extract_embeddings(waveform)
        assert emb.shape == (2, 768)


# ── wav2vec_fusion tests ──


class TestFusionMLP:
    def test_forward_shape(self):
        from bioagentics.voice_pd.deep.wav2vec_fusion import FusionMLP

        model = FusionMLP(wav2vec_dim=768, classical_dim=100, hidden_dim=64)
        w2v = torch.randn(4, 768)
        classical = torch.randn(4, 100)
        out = model(w2v, classical)
        assert out.shape == (4, 1)

    def test_training_step(self):
        from bioagentics.voice_pd.deep.wav2vec_fusion import FusionMLP

        model = FusionMLP(wav2vec_dim=768, classical_dim=100, hidden_dim=64)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        criterion = torch.nn.BCEWithLogitsLoss()

        w2v = torch.randn(8, 768)
        classical = torch.randn(8, 100)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0])

        logits = model(w2v, classical).squeeze(-1)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        assert loss.item() > 0


class TestCompareModalities:
    def test_returns_all_modalities(self, synthetic_embeddings):
        from bioagentics.voice_pd.deep.wav2vec_fusion import compare_modalities

        wav2vec_emb, classical_feats, labels = synthetic_embeddings
        results = compare_modalities(wav2vec_emb, classical_feats, labels, n_splits=3)
        assert "wav2vec_only" in results
        assert "classical_only" in results
        assert "fused" in results
        assert "fusion_improvement" in results

        for key in ["wav2vec_only", "classical_only", "fused"]:
            assert "auc_mean" in results[key]
            assert 0.0 <= results[key]["auc_mean"] <= 1.0
