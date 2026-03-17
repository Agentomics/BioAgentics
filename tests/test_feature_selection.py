"""Tests for Phase 3 feature selection module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.feature_selection import (
    paired_feature_analysis,
    stability_selection,
    select_features,
    _bh_fdr,
    _infer_layer,
)


def _make_synthetic_data(n_patients=8, n_features=20, seed=42):
    """Create synthetic feature data with known pre-flare/stable differences."""
    rng = np.random.default_rng(seed)

    windows = []
    feature_rows = []
    base_date = pd.Timestamp("2015-01-01")

    for p in range(n_patients):
        sid = f"P{p:03d}"
        # Each patient gets one pre-flare and one stable window
        for label in ["pre_flare", "stable"]:
            offset = len(windows)
            w = Window(
                subject_id=sid,
                window_start=base_date + pd.Timedelta(days=offset * 14),
                window_end=base_date + pd.Timedelta(days=offset * 14 + 14),
                label=label,
                anchor_visit=offset + 1,
            )
            windows.append(w)

            row = {}
            for f in range(n_features):
                if f < 5:
                    # First 5 features: real signal (higher in pre-flare)
                    if label == "pre_flare":
                        row[f"mb_feat_{f}"] = rng.normal(2.0, 0.5)
                    else:
                        row[f"mb_feat_{f}"] = rng.normal(0.0, 0.5)
                else:
                    # Remaining features: noise
                    row[f"met_feat_{f}"] = rng.normal(0.0, 1.0)
            feature_rows.append(row)

    features = pd.DataFrame(feature_rows, index=range(len(windows)))
    labels = pd.Series([w.label for w in windows], index=range(len(windows)), name="label")
    return features, windows, labels


class TestPairedFeatureAnalysis:
    def test_basic_output(self):
        features, windows, _ = _make_synthetic_data()
        report = paired_feature_analysis(features, windows)
        assert "feature" in report.columns
        assert "p_value" in report.columns
        assert "p_adjusted" in report.columns
        assert "significant" in report.columns
        assert "omic_layer" in report.columns
        assert len(report) == features.shape[1]

    def test_signal_features_detected(self):
        features, windows, _ = _make_synthetic_data(n_patients=15)
        report = paired_feature_analysis(features, windows, fdr_threshold=0.1)
        signal_feats = [f"mb_feat_{i}" for i in range(5)]
        sig_report = report[report["significant"]]
        # At least some signal features should be significant
        detected = set(sig_report["feature"]) & set(signal_feats)
        assert len(detected) > 0

    def test_positive_mean_diff_for_signal(self):
        features, windows, _ = _make_synthetic_data()
        report = paired_feature_analysis(features, windows)
        for i in range(5):
            feat_row = report[report["feature"] == f"mb_feat_{i}"]
            if not feat_row.empty:
                # Pre-flare should be higher
                assert feat_row["mean_diff"].values[0] > 0

    def test_empty_input(self):
        report = paired_feature_analysis(pd.DataFrame(), [])
        assert report.empty

    def test_omic_layer_inference(self):
        features, windows, _ = _make_synthetic_data()
        report = paired_feature_analysis(features, windows)
        mb_rows = report[report["omic_layer"] == "microbiome"]
        met_rows = report[report["omic_layer"] == "metabolomics"]
        assert len(mb_rows) == 5
        assert len(met_rows) == 15


class TestStabilitySelection:
    def test_basic_output(self):
        features, _, labels = _make_synthetic_data()
        result = stability_selection(features, labels, n_bootstrap=50)
        assert "feature" in result.columns
        assert "selection_freq" in result.columns
        assert "stable" in result.columns
        assert len(result) == features.shape[1]

    def test_selection_freq_range(self):
        features, _, labels = _make_synthetic_data()
        result = stability_selection(features, labels, n_bootstrap=50)
        assert (result["selection_freq"] >= 0).all()
        assert (result["selection_freq"] <= 1).all()

    def test_empty_input(self):
        result = stability_selection(pd.DataFrame(), pd.Series(dtype=str))
        assert result.empty


class TestSelectFeatures:
    def test_reduces_features(self):
        features, windows, labels = _make_synthetic_data(n_patients=15)
        paired_report = paired_feature_analysis(features, windows, fdr_threshold=0.5)
        stability_report = stability_selection(
            features, labels, n_bootstrap=50, threshold=0.1
        )
        reduced, combined = select_features(
            features, paired_report, stability_report,
            require_significant=True, require_stable=True,
        )
        # Should have fewer or equal columns
        assert reduced.shape[1] <= features.shape[1]
        assert reduced.shape[0] == features.shape[0]

    def test_fallback_on_empty_intersection(self):
        features, windows, labels = _make_synthetic_data()
        # Very strict thresholds -> no features pass both
        paired_report = paired_feature_analysis(features, windows, fdr_threshold=1e-10)
        stability_report = stability_selection(
            features, labels, n_bootstrap=10, threshold=0.99
        )
        reduced, _ = select_features(features, paired_report, stability_report)
        # Should fall back to stability-only or empty
        assert reduced.shape[0] == features.shape[0]


class TestBHFDR:
    def test_monotonicity(self):
        p = np.array([0.01, 0.03, 0.05, 0.1, 0.5])
        adj = _bh_fdr(p)
        valid = adj[np.isfinite(adj)]
        for i in range(1, len(valid)):
            assert valid[i] >= valid[i - 1] or abs(valid[i] - valid[i-1]) < 1e-10

    def test_nan_handling(self):
        p = np.array([0.01, np.nan, 0.05])
        adj = _bh_fdr(p)
        assert np.isfinite(adj[0])
        assert np.isnan(adj[1])
        assert np.isfinite(adj[2])

    def test_bounds(self):
        p = np.array([0.01, 0.5, 0.99])
        adj = _bh_fdr(p)
        assert (adj[np.isfinite(adj)] >= 0).all()
        assert (adj[np.isfinite(adj)] <= 1).all()


class TestInferLayer:
    def test_known_prefixes(self):
        assert _infer_layer("mb_shannon") == "microbiome"
        assert _infer_layer("met_slope__bile") == "metabolomics"
        assert _infer_layer("pw_mean__butyrate") == "pathways"
        assert _infer_layer("tx_module__tnf") == "transcriptomics"
        assert _infer_layer("sero_slope__ASCA") == "serology"
        assert _infer_layer("unknown_feat") == "unknown"
