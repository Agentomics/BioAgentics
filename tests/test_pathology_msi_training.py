"""Tests for MIL training pipeline."""

import json
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from bioagentics.models.pathology_msi.training import (
    SlideFeatureDataset,
    TrainConfig,
    collate_variable_length,
    evaluate,
    load_labels_and_features,
    nested_cross_validation,
    train_one_epoch,
    train_single_fold,
)


@pytest.fixture
def tmp_features(tmp_path):
    """Create temporary HDF5 feature files."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    slide_ids = [f"slide_{i:03d}" for i in range(20)]
    for sid in slide_ids:
        n_patches = np.random.randint(10, 50)
        features = np.random.randn(n_patches, 512).astype(np.float32)
        with h5py.File(features_dir / f"{sid}.h5", "w") as f:
            f.create_dataset("features", data=features)

    return features_dir, slide_ids


@pytest.fixture
def tmp_labels(tmp_path):
    """Create temporary labels CSV."""
    labels_path = tmp_path / "labels.csv"
    rows = []
    for i in range(20):
        rows.append(
            {
                "case_id": f"case_{i:03d}",
                "submitter_id": f"slide_{i:03d}",
                "cancer_type": "COAD" if i < 10 else "UCEC",
                "msi_status": "MSI-H" if i % 4 == 0 else "MSS",
            }
        )
    pd.DataFrame(rows).to_csv(labels_path, index=False)
    return labels_path


class TestSlideFeatureDataset:
    def test_loads_features(self, tmp_features):
        features_dir, slide_ids = tmp_features
        labels = {sid: 0 for sid in slide_ids}
        labels[slide_ids[0]] = 1

        ds = SlideFeatureDataset(slide_ids[:5], labels, features_dir)
        assert len(ds) == 5

        features, label, sid = ds[0]
        assert isinstance(features, torch.Tensor)
        assert features.ndim == 2
        assert features.shape[1] == 512

    def test_max_patches_subsampling(self, tmp_features):
        features_dir, slide_ids = tmp_features
        labels = {sid: 0 for sid in slide_ids}

        ds = SlideFeatureDataset(slide_ids[:1], labels, features_dir, max_patches=5)
        features, _, _ = ds[0]
        assert features.shape[0] <= 5


class TestCollate:
    def test_pads_to_max_length(self):
        batch = [
            (torch.randn(10, 512), 0, "s1"),
            (torch.randn(20, 512), 1, "s2"),
        ]
        padded, labels, ids = collate_variable_length(batch)
        assert padded.shape == (2, 20, 512)
        assert labels.shape == (2,)
        assert ids == ["s1", "s2"]


class TestLoadLabelsAndFeatures:
    def test_matches_available_features(self, tmp_features, tmp_labels):
        features_dir, _ = tmp_features
        slide_ids, labels, cancer_types = load_labels_and_features(
            tmp_labels, features_dir
        )
        assert len(slide_ids) > 0
        assert all(sid in labels for sid in slide_ids)
        assert all(sid in cancer_types for sid in slide_ids)

    def test_filters_cancer_types(self, tmp_features, tmp_labels):
        features_dir, _ = tmp_features
        slide_ids, _, _ = load_labels_and_features(
            tmp_labels, features_dir, cancer_types=["COAD"]
        )
        # Should only return COAD cases
        assert len(slide_ids) <= 10  # Only COAD cases


class TestTraining:
    def _get_config(self):
        return TrainConfig(
            model_name="abmil",
            input_dim=512,
            n_classes=2,
            n_outer_folds=2,
            n_inner_folds=2,
            n_epochs=3,
            patience=2,
            batch_size=1,
            seed=42,
        )

    def test_train_single_fold(self, tmp_features):
        features_dir, slide_ids = tmp_features
        labels = {sid: (1 if i % 4 == 0 else 0) for i, sid in enumerate(slide_ids)}
        config = self._get_config()

        train_ids = slide_ids[:15]
        val_ids = slide_ids[15:]

        metrics, model = train_single_fold(
            train_ids, val_ids, labels, features_dir, config
        )
        assert "auroc" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert model is not None

    def test_nested_cv(self, tmp_features, tmp_path):
        features_dir, slide_ids = tmp_features
        labels = {sid: (1 if i % 4 == 0 else 0) for i, sid in enumerate(slide_ids)}
        config = self._get_config()

        output_dir = tmp_path / "results"
        results = nested_cross_validation(
            slide_ids, labels, features_dir, config, output_dir
        )
        assert len(results) == 2  # 2 outer folds
        assert all("auroc" in r for r in results)

        # Check output files
        assert (output_dir / "cv_results.csv").exists()
        assert (output_dir / "cv_summary.json").exists()

        with open(output_dir / "cv_summary.json") as f:
            summary = json.load(f)
        assert summary["model"] == "abmil"

    def test_saves_checkpoints(self, tmp_features, tmp_path):
        features_dir, slide_ids = tmp_features
        labels = {sid: (1 if i % 4 == 0 else 0) for i, sid in enumerate(slide_ids)}
        config = self._get_config()

        checkpoint_dir = tmp_path / "checkpoints"
        train_single_fold(
            slide_ids[:15], slide_ids[15:], labels,
            features_dir, config, checkpoint_dir=checkpoint_dir,
        )
        assert (checkpoint_dir / "best_model.pt").exists()
