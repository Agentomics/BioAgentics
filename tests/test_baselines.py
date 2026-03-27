"""Tests for Phase 4 baseline comparison module."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.baselines import (
    extract_single_marker_features,
    run_baseline,
    run_all_baselines,
    compare_models,
)


def _make_windows(n_patients=6, per_patient=4):
    base = pd.Timestamp("2015-01-01")
    windows = []
    for p in range(n_patients):
        sid = f"P{p:03d}"
        for i in range(per_patient):
            idx = p * per_patient + i
            label = "pre_flare" if i < per_patient // 2 else "stable"
            windows.append(Window(
                subject_id=sid,
                window_start=base + pd.Timedelta(days=idx * 14),
                window_end=base + pd.Timedelta(days=idx * 14 + 14),
                label=label,
                anchor_visit=idx + 1,
            ))
    return windows


def _make_marker_data(windows, marker_col="value", n_timepoints=3, seed=42):
    """Create synthetic marker data with signal."""
    rng = np.random.default_rng(seed)
    rows = []
    for w in windows:
        dates = pd.date_range(w.window_start, w.window_end, periods=n_timepoints)
        offset = 2.0 if w.label == "pre_flare" else 0.0
        for d in dates:
            rows.append({
                "subject_id": w.subject_id,
                "date": d,
                marker_col: rng.normal(offset + 5, 1),
            })
    return pd.DataFrame(rows)


class TestExtractSingleMarker:
    def test_basic_output(self):
        windows = _make_windows()
        data = _make_marker_data(windows)
        features = extract_single_marker_features("crp", data, windows)
        assert features.shape[0] == len(windows)
        assert "crp_mean" in features.columns
        assert "crp_slope" in features.columns

    def test_none_data(self):
        windows = _make_windows()
        features = extract_single_marker_features("crp", None, windows)
        assert features.shape[0] == len(windows)
        assert features.isna().all().all()


class TestRunBaseline:
    def test_produces_metrics(self):
        windows = _make_windows()
        data = _make_marker_data(windows)
        features = extract_single_marker_features("crp", data, windows)
        metrics = run_baseline("crp", features, windows)
        assert "auc" in metrics
        assert "model" in metrics

    def test_nan_features_handled(self):
        windows = _make_windows()
        features = extract_single_marker_features("crp", None, windows)
        metrics = run_baseline("crp", features, windows)
        assert metrics["model"] == "baseline_crp"


class TestRunAllBaselines:
    def test_runs_all_three(self):
        windows = _make_windows()
        data = {
            "calprotectin": _make_marker_data(windows, "calprotectin"),
            "crp": _make_marker_data(windows, "crp", seed=43),
            "species": _make_marker_data(windows, "diversity", seed=44),
        }
        results = run_all_baselines(data, windows)
        assert len(results) == 4


class TestCompareModels:
    def test_comparison_table(self):
        primary = {"model": "xgboost", "auc": 0.85, "sensitivity": 0.8, "specificity": 0.9, "ppv": 0.85, "npv": 0.86}
        baselines = [
            {"model": "baseline_calprotectin", "auc": 0.70, "sensitivity": 0.6, "specificity": 0.8, "ppv": 0.7, "npv": 0.72},
            {"model": "baseline_crp", "auc": 0.65, "sensitivity": 0.5, "specificity": 0.8, "ppv": 0.6, "npv": 0.65},
        ]
        comparison = compare_models(primary, baselines)
        assert len(comparison) == 3
        assert "auc_delta_vs_calprotectin" in comparison.columns
        # XGBoost should show +0.15 delta vs calprotectin
        xgb_row = comparison[comparison["model"] == "xgboost"]
        assert abs(xgb_row["auc_delta_vs_calprotectin"].values[0] - 0.15) < 1e-10
