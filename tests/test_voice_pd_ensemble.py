"""Tests for voice_pd ensemble fusion module."""

import numpy as np
import pytest


@pytest.fixture
def synthetic_data():
    """Create synthetic classical features, spectrograms, and labels."""
    rng = np.random.default_rng(42)
    n_samples = 40
    n_features = 20
    n_mels = 128
    time_frames = 157

    X_classical = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    spectrograms = rng.random((n_samples, 3, n_mels, time_frames)).astype(np.float32)
    labels = np.array([1] * 20 + [0] * 20, dtype=np.float32)
    return X_classical, spectrograms, labels


@pytest.fixture
def synthetic_probs():
    """Create synthetic probability predictions for fusion tests."""
    rng = np.random.default_rng(42)
    n = 40
    labels = np.array([1] * 20 + [0] * 20, dtype=np.float32)
    # Make predictions somewhat correlated with labels
    classical_probs = np.clip(labels * 0.6 + rng.random(n) * 0.4, 0, 1)
    deep_probs = np.clip(labels * 0.5 + rng.random(n) * 0.5, 0, 1)
    return classical_probs, deep_probs, labels


class TestLateFusion:
    def test_equal_weights(self, synthetic_probs):
        from bioagentics.voice_pd.models.ensemble import late_fusion

        classical, deep, _ = synthetic_probs
        fused = late_fusion(classical, deep, weight_classical=0.5)
        expected = 0.5 * classical + 0.5 * deep
        np.testing.assert_array_almost_equal(fused, expected)

    def test_classical_only(self, synthetic_probs):
        from bioagentics.voice_pd.models.ensemble import late_fusion

        classical, deep, _ = synthetic_probs
        fused = late_fusion(classical, deep, weight_classical=1.0)
        np.testing.assert_array_almost_equal(fused, classical)

    def test_deep_only(self, synthetic_probs):
        from bioagentics.voice_pd.models.ensemble import late_fusion

        classical, deep, _ = synthetic_probs
        fused = late_fusion(classical, deep, weight_classical=0.0)
        np.testing.assert_array_almost_equal(fused, deep)

    def test_output_shape(self, synthetic_probs):
        from bioagentics.voice_pd.models.ensemble import late_fusion

        classical, deep, _ = synthetic_probs
        fused = late_fusion(classical, deep)
        assert fused.shape == classical.shape


class TestOptimizeFusionWeight:
    def test_returns_valid_weight(self, synthetic_probs):
        from bioagentics.voice_pd.models.ensemble import optimize_fusion_weight

        classical, deep, labels = synthetic_probs
        weight, auc = optimize_fusion_weight(classical, deep, labels)
        assert 0.0 <= weight <= 1.0
        assert 0.0 <= auc <= 1.0

    def test_auc_at_least_as_good_as_equal_weight(self, synthetic_probs):
        from sklearn.metrics import roc_auc_score

        from bioagentics.voice_pd.models.ensemble import (
            late_fusion,
            optimize_fusion_weight,
        )

        classical, deep, labels = synthetic_probs
        _, best_auc = optimize_fusion_weight(classical, deep, labels)
        equal_auc = roc_auc_score(labels, late_fusion(classical, deep, 0.5))
        assert best_auc >= equal_auc - 1e-9


class TestExtractCnnEmbeddings:
    def test_embedding_shape(self):
        from bioagentics.voice_pd.deep.cnn_model import build_model
        from bioagentics.voice_pd.models.ensemble import extract_cnn_embeddings

        rng = np.random.default_rng(42)
        specs = rng.random((6, 3, 128, 157)).astype(np.float32)
        model = build_model(pretrained=False)
        embeddings = extract_cnn_embeddings(model, specs, batch_size=4)
        assert embeddings.shape == (6, 1280)

    def test_deterministic(self):
        from bioagentics.voice_pd.deep.cnn_model import build_model
        from bioagentics.voice_pd.models.ensemble import extract_cnn_embeddings

        rng = np.random.default_rng(42)
        specs = rng.random((4, 3, 128, 157)).astype(np.float32)
        model = build_model(pretrained=False)
        emb1 = extract_cnn_embeddings(model, specs)
        emb2 = extract_cnn_embeddings(model, specs)
        np.testing.assert_array_equal(emb1, emb2)


class TestTrainEnsemble:
    def test_train_runs(self, synthetic_data, tmp_path):
        from bioagentics.voice_pd.models.ensemble import train_ensemble

        X_classical, spectrograms, labels = synthetic_data
        results = train_ensemble(
            X_classical, spectrograms, labels,
            output_dir=tmp_path,
            n_splits=2,
            cnn_epochs=2,
            cnn_batch_size=8,
            cnn_pretrained=False,
        )
        assert "classical" in results
        assert "deep" in results
        assert "late_fusion" in results
        assert "early_fusion" in results
        assert "stacked" in results
        assert "best_strategy" in results
        assert results["best_strategy"] in ("late", "early", "stacked")

    def test_results_file_saved(self, synthetic_data, tmp_path):
        from bioagentics.voice_pd.models.ensemble import train_ensemble

        X_classical, spectrograms, labels = synthetic_data
        train_ensemble(
            X_classical, spectrograms, labels,
            output_dir=tmp_path,
            n_splits=2,
            cnn_epochs=2,
            cnn_batch_size=8,
            cnn_pretrained=False,
        )
        assert (tmp_path / "ensemble_results.json").exists()

    def test_auc_ranges(self, synthetic_data, tmp_path):
        from bioagentics.voice_pd.models.ensemble import train_ensemble

        X_classical, spectrograms, labels = synthetic_data
        results = train_ensemble(
            X_classical, spectrograms, labels,
            output_dir=tmp_path,
            n_splits=2,
            cnn_epochs=2,
            cnn_batch_size=8,
            cnn_pretrained=False,
        )
        assert 0.0 <= results["classical"]["mean_auc"] <= 1.0
        assert 0.0 <= results["deep"]["mean_auc"] <= 1.0
        assert 0.0 <= results["best_ensemble_auc"] <= 1.0
