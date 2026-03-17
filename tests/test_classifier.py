"""Tests for Phase 4 classifier module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.classifier import (
    lopo_cv,
    evaluate_results,
    CVResults,
)


def _make_data(n_patients=6, instances_per_patient=4, n_features=10, seed=42):
    """Create synthetic classification data with signal."""
    rng = np.random.default_rng(seed)
    windows = []
    feature_rows = []
    base = pd.Timestamp("2015-01-01")

    for p in range(n_patients):
        sid = f"P{p:03d}"
        for i in range(instances_per_patient):
            idx = p * instances_per_patient + i
            label = "pre_flare" if i < instances_per_patient // 2 else "stable"
            windows.append(
                Window(
                    subject_id=sid,
                    window_start=base + pd.Timedelta(days=idx * 14),
                    window_end=base + pd.Timedelta(days=idx * 14 + 14),
                    label=label,
                    anchor_visit=idx + 1,
                )
            )
            row = {}
            for f in range(n_features):
                if f < 3:
                    # Signal features
                    offset = 2.0 if label == "pre_flare" else 0.0
                    row[f"feat_{f}"] = rng.normal(offset, 0.5)
                else:
                    row[f"feat_{f}"] = rng.normal(0, 1)
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows)
    return features, windows


class TestLOPOCV:
    def test_xgboost_runs(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        assert isinstance(results, CVResults)
        assert len(results.folds) > 0

    def test_logistic_runs(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="logistic", calibrate=False)
        assert isinstance(results, CVResults)
        assert len(results.folds) > 0

    def test_predictions_shape(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        total = sum(f.n_instances for f in results.folds)
        assert total == len(windows)

    def test_probabilities_in_range(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        probs = results.all_y_prob
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_calibrated_probabilities(self):
        features, windows = _make_data(n_patients=8, instances_per_patient=6)
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=True)
        probs = results.all_y_prob
        assert (probs >= 0).all() and (probs <= 1).all()


class TestEvaluateResults:
    def test_metrics_keys(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        metrics = evaluate_results(results)
        assert "auc" in metrics
        assert "sensitivity" in metrics
        assert "specificity" in metrics
        assert "ppv" in metrics
        assert "npv" in metrics

    def test_auc_reasonable(self):
        features, windows = _make_data(n_patients=8, instances_per_patient=6)
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        metrics = evaluate_results(results)
        # With clear signal, AUC should be above chance
        assert metrics["auc"] > 0.5

    def test_calibration_data(self):
        features, windows = _make_data()
        results = lopo_cv(features, windows, model_type="xgboost", calibrate=False)
        metrics = evaluate_results(results)
        assert "calibration" in metrics
        cal = metrics["calibration"]
        assert "predicted_means" in cal
        assert "observed_fracs" in cal
