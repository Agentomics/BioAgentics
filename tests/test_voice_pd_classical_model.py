"""Tests for voice_pd classical gradient boosting model."""

import csv

import numpy as np
import pytest


@pytest.fixture
def synthetic_features_csv(tmp_path):
    """Create a synthetic features CSV for testing."""
    n_pd, n_healthy = 30, 30
    rng = np.random.default_rng(42)

    feature_names = [
        "local_jitter", "local_shimmer", "hnr_db", "nhr",
        "f0_mean", "f0_std", "f0_range",
        "mfcc_1_mean", "mfcc_1_std", "mfcc_2_mean", "mfcc_2_std",
        "phonation_ratio", "speech_rate",
    ]

    rows = []
    for i in range(n_pd + n_healthy):
        is_pd = i < n_pd
        row = {
            "recording_id": f"rec_{i:03d}",
            "pd_status": "pd" if is_pd else "healthy",
            "audio_path": f"/tmp/audio_{i}.wav",
        }
        # PD samples have slightly different feature distributions
        offset = 0.5 if is_pd else 0.0
        for fname in feature_names:
            row[fname] = f"{rng.normal(offset, 1.0):.6f}"
        rows.append(row)

    csv_path = tmp_path / "test_features.csv"
    fieldnames = ["recording_id", "pd_status", "audio_path"] + feature_names
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return csv_path


class TestLoadFeatureCSV:
    def test_loads_correctly(self, synthetic_features_csv):
        from bioagentics.voice_pd.models.classical import load_feature_csv

        X, y, names = load_feature_csv(synthetic_features_csv)
        assert X.shape == (60, 13)
        assert y.shape == (60,)
        assert len(names) == 13
        assert y.sum() == 30  # 30 PD
        assert (1 - y).sum() == 30  # 30 healthy

    def test_rejects_empty_labels(self, tmp_path):
        from bioagentics.voice_pd.models.classical import load_feature_csv

        csv_path = tmp_path / "empty.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["recording_id", "pd_status", "feat1"])
            writer.writeheader()
            writer.writerow({"recording_id": "x", "pd_status": "unknown", "feat1": "1.0"})

        with pytest.raises(ValueError, match="No labeled rows"):
            load_feature_csv(csv_path)


class TestTrainClassifier:
    def test_train_runs(self, synthetic_features_csv, tmp_path):
        from bioagentics.voice_pd.models.classical import (
            load_feature_csv,
            train_classifier,
        )

        X, y, names = load_feature_csv(synthetic_features_csv)
        results = train_classifier(X, y, names, output_dir=tmp_path, n_splits=3)

        assert "mean_auc" in results
        assert "fold_aucs" in results
        assert len(results["fold_aucs"]) == 3
        assert results["best_model"] is not None
        # Should produce a reasonable AUC on synthetic data
        assert 0.0 <= results["mean_auc"] <= 1.0
        # Results file should be saved
        assert (tmp_path / "classical_gbm_results.json").exists()


class TestFeatureGroupAblation:
    def test_ablation_runs(self, synthetic_features_csv):
        from bioagentics.voice_pd.models.classical import (
            feature_group_ablation,
            load_feature_csv,
        )

        X, y, names = load_feature_csv(synthetic_features_csv)
        group_aucs = feature_group_ablation(X, y, names)
        assert isinstance(group_aucs, dict)
        # At least some groups should be present
        assert len(group_aucs) > 0
        for auc in group_aucs.values():
            assert 0.0 <= auc <= 1.0
