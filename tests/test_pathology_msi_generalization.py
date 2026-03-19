"""Tests for leave-one-cancer-out cross-cancer generalization evaluation."""

import json

import h5py
import numpy as np
import pandas as pd
import pytest

from bioagentics.models.pathology_msi.generalization import (
    LOCOResult,
    leave_one_cancer_out,
    pairwise_generalization,
)
from bioagentics.models.pathology_msi.training import TrainConfig


@pytest.fixture
def multi_cancer_data(tmp_path):
    """Create test data with multiple cancer types."""
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    labels_path = tmp_path / "labels.csv"
    rows = []

    cancer_types = ["COAD", "READ", "UCEC", "STAD"]
    slide_idx = 0
    for ct in cancer_types:
        # 8 slides per cancer type, mix of MSI-H and MSS
        for j in range(8):
            sid = f"slide_{slide_idx:03d}"
            n_patches = np.random.randint(10, 30)
            features = np.random.randn(n_patches, 256).astype(np.float32)
            with h5py.File(features_dir / f"{sid}.h5", "w") as f:
                f.create_dataset("features", data=features)

            rows.append({
                "case_id": f"case_{slide_idx:03d}",
                "submitter_id": sid,
                "cancer_type": ct,
                "msi_status": "MSI-H" if j < 3 else "MSS",
            })
            slide_idx += 1

    pd.DataFrame(rows).to_csv(labels_path, index=False)
    return labels_path, features_dir


@pytest.fixture
def small_config():
    """Minimal training config for fast tests."""
    return TrainConfig(
        model_name="abmil",
        input_dim=256,
        n_classes=2,
        n_outer_folds=2,
        n_inner_folds=2,
        n_epochs=2,
        patience=1,
        batch_size=1,
        seed=42,
    )


class TestLOCOResult:
    def test_dataclass_fields(self):
        r = LOCOResult(
            held_out_cancer="COAD",
            train_cancers=["READ", "UCEC", "STAD"],
            n_train=24,
            n_eval=8,
            n_eval_msi_h=3,
            n_eval_mss=5,
            auroc=0.85,
            auprc=0.72,
            balanced_accuracy=0.78,
            sensitivity=0.80,
            specificity=0.76,
        )
        assert r.held_out_cancer == "COAD"
        assert len(r.train_cancers) == 3
        assert r.n_eval == 8


class TestLeaveOneCancerOut:
    def test_runs_all_cancer_types(self, multi_cancer_data, small_config, tmp_path):
        labels_path, features_dir = multi_cancer_data
        output_dir = tmp_path / "loco_output"

        results = leave_one_cancer_out(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
            output_dir=output_dir,
        )

        assert len(results) == 2
        held_out_cancers = {r.held_out_cancer for r in results}
        assert held_out_cancers == {"COAD", "UCEC"}

    def test_train_excludes_held_out(self, multi_cancer_data, small_config):
        labels_path, features_dir = multi_cancer_data

        results = leave_one_cancer_out(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
        )

        for r in results:
            assert r.held_out_cancer not in r.train_cancers

    def test_metrics_present(self, multi_cancer_data, small_config):
        labels_path, features_dir = multi_cancer_data

        results = leave_one_cancer_out(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
        )

        for r in results:
            assert r.n_train > 0
            assert r.n_eval > 0
            assert 0.0 <= r.sensitivity <= 1.0
            assert 0.0 <= r.specificity <= 1.0

    def test_saves_results(self, multi_cancer_data, small_config, tmp_path):
        labels_path, features_dir = multi_cancer_data
        output_dir = tmp_path / "loco_output"

        leave_one_cancer_out(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
            output_dir=output_dir,
        )

        assert (output_dir / "loco_results.csv").exists()
        assert (output_dir / "loco_summary.json").exists()

        df = pd.read_csv(output_dir / "loco_results.csv")
        assert len(df) == 2
        assert "held_out_cancer" in df.columns
        assert "auroc" in df.columns

        with open(output_dir / "loco_summary.json") as f:
            summary = json.load(f)
        assert summary["model"] == "abmil"
        assert summary["n_cancer_types"] == 2

    def test_default_cancer_types(self, multi_cancer_data, small_config):
        labels_path, features_dir = multi_cancer_data

        results = leave_one_cancer_out(
            labels_path, features_dir,
            config=small_config,
        )

        # Should use all 4 default types
        assert len(results) == 4


class TestPairwiseGeneralization:
    def test_generates_matrix(self, multi_cancer_data, small_config, tmp_path):
        labels_path, features_dir = multi_cancer_data
        output_dir = tmp_path / "pairwise_output"

        matrix = pairwise_generalization(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
            output_dir=output_dir,
        )

        assert matrix.shape == (2, 2)
        assert "COAD" in matrix.index
        assert "UCEC" in matrix.columns
        assert (output_dir / "generalization_matrix.csv").exists()

    def test_diagonal_is_self_eval(self, multi_cancer_data, small_config):
        labels_path, features_dir = multi_cancer_data

        matrix = pairwise_generalization(
            labels_path, features_dir,
            cancer_types=["COAD", "UCEC"],
            config=small_config,
        )

        # Diagonal entries should be non-NaN (train and eval on same type)
        for ct in ["COAD", "UCEC"]:
            assert not np.isnan(matrix.loc[ct, ct])
