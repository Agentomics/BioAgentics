"""Tests for Phase 5 clinical utility analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.classifier import CVFold, CVResults, lopo_cv
from bioagentics.crohns.flare_prediction.clinical_utility import (
    compute_shap_importance,
    lead_time_analysis,
    decision_curve_analysis,
    check_success_criteria,
    run_clinical_utility,
    save_clinical_utility,
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


def _make_features(windows, n_features=20, seed=42):
    """Create synthetic features with signal for pre-flare vs stable."""
    rng = np.random.default_rng(seed)
    n = len(windows)
    X = rng.normal(0, 1, (n, n_features))

    # Add signal: first 5 features elevated in pre-flare
    for i, w in enumerate(windows):
        if w.label == "pre_flare":
            X[i, :5] += 1.5

    # Name features with omic layer prefixes
    prefixes = ["mb_", "met_", "pw_", "tx_", "sero_"]
    names = []
    for j in range(n_features):
        prefix = prefixes[j % len(prefixes)]
        names.append(f"{prefix}feat_{j}")

    return pd.DataFrame(X, columns=names)


def _make_cv_results(windows, features):
    """Create synthetic CV results."""
    return lopo_cv(features, windows, model_type="logistic", calibrate=False)


class TestComputeSHAPImportance:
    def test_basic_shap(self):
        windows = _make_windows()
        features = _make_features(windows)
        result = compute_shap_importance(features, windows, model_type="xgboost")

        assert result.shap_values.shape == (len(windows), features.shape[1])
        assert len(result.mean_abs_shap) == features.shape[1]
        assert len(result.feature_ranking) == features.shape[1]
        assert "mean_abs_shap" in result.feature_ranking.columns
        assert "omic_layer" in result.feature_ranking.columns
        assert "rank" in result.feature_ranking.columns

    def test_layer_importance(self):
        windows = _make_windows()
        features = _make_features(windows)
        result = compute_shap_importance(features, windows)

        assert len(result.layer_importance) > 0
        assert "importance_pct" in result.layer_importance.columns
        # Total importance should sum to ~100%
        total_pct = result.layer_importance["importance_pct"].sum()
        assert abs(total_pct - 100.0) < 1.0

    def test_logistic_model(self):
        windows = _make_windows()
        features = _make_features(windows)
        result = compute_shap_importance(features, windows, model_type="logistic")
        assert result.shap_values.shape[0] == len(windows)

    def test_feature_ranking_sorted(self):
        windows = _make_windows()
        features = _make_features(windows)
        result = compute_shap_importance(features, windows)

        ranking = result.feature_ranking
        # Should be sorted by mean_abs_shap descending
        assert ranking["mean_abs_shap"].is_monotonic_decreasing


class TestDecisionCurveAnalysis:
    def test_basic(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)

        result = decision_curve_analysis(y_true, y_prob)
        assert len(result.thresholds) > 0
        assert len(result.net_benefit_model) == len(result.thresholds)
        assert len(result.net_benefit_all) == len(result.thresholds)
        assert len(result.net_benefit_none) == len(result.thresholds)

    def test_custom_thresholds(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)
        thresholds = np.array([0.05, 0.10, 0.20, 0.30])

        result = decision_curve_analysis(y_true, y_prob, thresholds=thresholds)
        assert len(result.thresholds) == 4

    def test_net_benefit_none_always_zero(self):
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, 100)
        y_prob = rng.uniform(0, 1, 100)

        result = decision_curve_analysis(y_true, y_prob)
        assert np.all(result.net_benefit_none == 0.0)


class TestLeadTimeAnalysis:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv_results = _make_cv_results(windows, features)

        result = lead_time_analysis(cv_results, windows)
        assert len(result.lead_weeks) > 0
        assert len(result.detection_rates) == len(result.lead_weeks)
        assert all(0.0 <= r <= 1.0 for r in result.detection_rates)

    def test_custom_bins(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv_results = _make_cv_results(windows, features)

        result = lead_time_analysis(cv_results, windows, week_bins=[2.0, 4.0])
        assert len(result.lead_weeks) == 2


class TestCheckSuccessCriteria:
    def test_passes_with_signal(self):
        windows = _make_windows()
        features = _make_features(windows)
        shap_result = compute_shap_importance(features, windows)

        criteria = check_success_criteria(shap_result)
        assert "criterion_1_directional_features" in criteria
        assert "criterion_2_multi_omic_layers" in criteria
        assert "overall_pass" in criteria

    def test_criterion_structure(self):
        windows = _make_windows()
        features = _make_features(windows)
        shap_result = compute_shap_importance(features, windows)

        criteria = check_success_criteria(shap_result)
        c1 = criteria["criterion_1_directional_features"]
        assert "pass" in c1
        assert "n_directional" in c1
        assert "required" in c1

        c2 = criteria["criterion_2_multi_omic_layers"]
        assert "pass" in c2
        assert "n_contributing_layers" in c2
        assert "layer_breakdown" in c2


class TestRunClinicalUtility:
    def test_full_pipeline(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv_results = _make_cv_results(windows, features)

        result = run_clinical_utility(features, windows, cv_results)
        assert result.shap_result is not None
        assert result.lead_time is not None
        assert result.decision_curve is not None
        assert result.success_criteria is not None

    def test_without_cv_results(self):
        windows = _make_windows()
        features = _make_features(windows)

        result = run_clinical_utility(features, windows, cv_results=None)
        assert result.shap_result is not None
        assert result.lead_time is None
        assert result.decision_curve is None


class TestSaveClinicalUtility:
    def test_save(self, tmp_path):
        windows = _make_windows()
        features = _make_features(windows)
        cv_results = _make_cv_results(windows, features)

        result = run_clinical_utility(features, windows, cv_results)
        save_clinical_utility(result, tmp_path)

        assert (tmp_path / "shap_feature_ranking.csv").exists()
        assert (tmp_path / "shap_layer_importance.csv").exists()
        assert (tmp_path / "shap_values.npz").exists()
        assert (tmp_path / "lead_time_analysis.csv").exists()
        assert (tmp_path / "decision_curve.csv").exists()
        assert (tmp_path / "success_criteria.json").exists()

    def test_save_partial(self, tmp_path):
        windows = _make_windows()
        features = _make_features(windows)

        result = run_clinical_utility(features, windows, cv_results=None)
        save_clinical_utility(result, tmp_path)

        assert (tmp_path / "shap_feature_ranking.csv").exists()
        assert not (tmp_path / "lead_time_analysis.csv").exists()
