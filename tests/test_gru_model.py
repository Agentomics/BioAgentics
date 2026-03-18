"""Tests for Phase 4 GRU temporal model."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.classifier import evaluate_results
from bioagentics.crohns.flare_prediction.gru_model import (
    build_sequences,
    gru_lopo_cv,
    compare_gru_vs_xgboost,
    save_gru_results,
    PatientSequence,
)


def _make_windows(n_patients=6, per_patient=4):
    """Create synthetic classification windows."""
    base = pd.Timestamp("2015-01-01")
    windows = []
    for p in range(n_patients):
        sid = f"P{p:03d}"
        for i in range(per_patient):
            idx = p * per_patient + i
            label = "pre_flare" if i < per_patient // 2 else "stable"
            windows.append(
                Window(
                    subject_id=sid,
                    window_start=base + pd.Timedelta(days=idx * 14),
                    window_end=base + pd.Timedelta(days=idx * 14 + 14),
                    label=label,
                    anchor_visit=idx + 1,
                )
            )
    return windows


def _make_features(windows, n_features=15, seed=42):
    """Create synthetic features with signal."""
    rng = np.random.default_rng(seed)
    n = len(windows)
    X = rng.normal(0, 1, (n, n_features))

    # Add signal for pre-flare
    for i, w in enumerate(windows):
        if w.label == "pre_flare":
            X[i, :3] += 2.0

    names = [f"feat_{j}" for j in range(n_features)]
    return pd.DataFrame(X, columns=names)


class TestBuildSequences:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)

        assert len(sequences) == 6  # 6 patients
        for seq in sequences:
            assert seq.seq_len == 4  # 4 windows per patient
            assert seq.features.shape == (4, 15)
            assert len(seq.labels) == 4

    def test_time_ordering(self):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)

        for seq in sequences:
            # Timestamps should be sorted
            for i in range(len(seq.timestamps) - 1):
                assert seq.timestamps[i] <= seq.timestamps[i + 1]

    def test_nan_imputation(self):
        windows = _make_windows()
        features = _make_features(windows)
        # Introduce NaN
        features.iloc[0, 0] = np.nan
        features.iloc[3, 2] = np.nan

        sequences = build_sequences(features, windows)
        for seq in sequences:
            assert not np.any(np.isnan(seq.features))

    def test_empty(self):
        sequences = build_sequences(pd.DataFrame(), [])
        assert len(sequences) == 0


class TestGRULOPOCV:
    def test_basic_cv(self):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)

        results = gru_lopo_cv(
            sequences, hidden_size=8, n_epochs=5, patience=3
        )
        assert results.model_name == "gru"
        assert len(results.folds) > 0

        # Each fold should have predictions
        for fold in results.folds:
            assert len(fold.y_true) > 0
            assert len(fold.y_prob) == len(fold.y_true)
            assert all(0 <= p <= 1 for p in fold.y_prob)

    def test_produces_metrics(self):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)

        results = gru_lopo_cv(
            sequences, hidden_size=8, n_epochs=5, patience=3
        )
        metrics = evaluate_results(results)
        assert "auc" in metrics
        assert "model" in metrics

    def test_empty_sequences(self):
        results = gru_lopo_cv([])
        assert len(results.folds) == 0

    def test_early_stopping(self):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)

        # With patience=1, should stop early
        results = gru_lopo_cv(
            sequences, hidden_size=8, n_epochs=100, patience=1
        )
        assert len(results.folds) > 0


class TestCompareGRUvsXGBoost:
    def test_comparison_table(self):
        gru_metrics = {
            "model": "gru",
            "auc": 0.75,
            "sensitivity": 0.7,
            "specificity": 0.8,
            "ppv": 0.75,
            "npv": 0.76,
            "mean_fold_auc": 0.73,
        }
        xgb_metrics = {
            "model": "xgboost",
            "auc": 0.85,
            "sensitivity": 0.8,
            "specificity": 0.9,
            "ppv": 0.85,
            "npv": 0.86,
            "mean_fold_auc": 0.83,
        }
        comparison = compare_gru_vs_xgboost(gru_metrics, xgb_metrics)
        assert len(comparison) == 2
        assert "auc" in comparison.columns
        assert "mean_fold_auc" in comparison.columns


class TestSaveGRUResults:
    def test_save(self, tmp_path):
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)
        results = gru_lopo_cv(
            sequences, hidden_size=8, n_epochs=3, patience=2
        )
        metrics = evaluate_results(results)

        save_gru_results(results, metrics, None, tmp_path)
        assert (tmp_path / "gru_cv_predictions.csv").exists()
        assert (tmp_path / "gru_cv_metrics.csv").exists()

    def test_save_with_comparison(self, tmp_path):
        comparison = pd.DataFrame(
            [
                {"model": "xgboost", "auc": 0.85},
                {"model": "gru", "auc": 0.75},
            ]
        )
        windows = _make_windows()
        features = _make_features(windows)
        sequences = build_sequences(features, windows)
        results = gru_lopo_cv(
            sequences, hidden_size=8, n_epochs=3, patience=2
        )
        metrics = evaluate_results(results)

        save_gru_results(results, metrics, comparison, tmp_path)
        assert (tmp_path / "gru_vs_xgboost.csv").exists()
