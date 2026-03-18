"""Tests for Phase 5 visualization data generation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bioagentics.crohns.flare_prediction.flare_events import Window
from bioagentics.crohns.flare_prediction.classifier import lopo_cv, CVResults
from bioagentics.crohns.flare_prediction.clinical_utility import compute_shap_importance
from bioagentics.crohns.flare_prediction.visualizations import (
    patient_risk_trajectories,
    select_representative_patients,
    feature_trajectory_heatmap_data,
    model_comparison_roc_data,
    calibration_plot_data,
    generate_all_visualizations,
)


def _make_windows(n_patients=6, per_patient=4):
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
    rng = np.random.default_rng(seed)
    n = len(windows)
    X = rng.normal(0, 1, (n, n_features))
    for i, w in enumerate(windows):
        if w.label == "pre_flare":
            X[i, :5] += 1.5

    prefixes = ["mb_", "met_", "pw_", "tx_", "sero_"]
    names = [f"{prefixes[j % len(prefixes)]}feat_{j}" for j in range(n_features)]
    return pd.DataFrame(X, columns=names)


def _make_cv_results(windows, features):
    return lopo_cv(features, windows, model_type="logistic", calibrate=False)


class TestPatientRiskTrajectories:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        traj = patient_risk_trajectories(cv, windows)
        assert "patient_id" in traj.columns
        assert "y_true" in traj.columns
        assert "y_prob" in traj.columns
        assert "label" in traj.columns
        assert len(traj) > 0

    def test_probabilities_valid(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        traj = patient_risk_trajectories(cv, windows)
        assert traj["y_prob"].between(0, 1).all()


class TestSelectRepresentativePatients:
    def test_categories(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        reps = select_representative_patients(cv, windows)
        assert "true_positive" in reps or "false_negative" in reps or "true_negative" in reps

    def test_returns_dataframes(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        reps = select_representative_patients(cv, windows)
        for category, df in reps.items():
            assert isinstance(df, pd.DataFrame)
            assert len(df) > 0


class TestFeatureTrajectoryHeatmap:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        shap_result = compute_shap_importance(features, windows)

        heatmap = feature_trajectory_heatmap_data(shap_result, features, windows)
        assert "patient_id" in heatmap.columns
        assert "feature" in heatmap.columns
        assert "value" in heatmap.columns
        assert "shap_value" in heatmap.columns
        assert "label" in heatmap.columns

    def test_top_n(self):
        windows = _make_windows()
        features = _make_features(windows)
        shap_result = compute_shap_importance(features, windows)

        heatmap = feature_trajectory_heatmap_data(
            shap_result, features, windows, top_n=5
        )
        unique_features = heatmap["feature"].nunique()
        assert unique_features <= 5


class TestModelComparisonROC:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv1 = _make_cv_results(windows, features)
        cv2 = lopo_cv(features, windows, model_type="xgboost", calibrate=False)

        roc = model_comparison_roc_data({"logistic": cv1, "xgboost": cv2})
        assert "model" in roc.columns
        assert "fpr" in roc.columns
        assert "tpr" in roc.columns
        assert roc["model"].nunique() == 2


class TestCalibrationPlotData:
    def test_basic(self):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        cal = calibration_plot_data({"logistic": cv})
        assert "model" in cal.columns
        assert "predicted_mean" in cal.columns
        assert "observed_frac" in cal.columns


class TestGenerateAllVisualizations:
    def test_full_pipeline(self, tmp_path):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)
        shap_result = compute_shap_importance(features, windows)

        generated = generate_all_visualizations(
            cv, windows,
            shap_result=shap_result,
            features=features,
            output_dir=tmp_path,
        )

        assert "patient_risk_trajectories" in generated
        assert (tmp_path / "viz_patient_risk_trajectories.csv").exists()
        assert (tmp_path / "viz_roc_comparison.csv").exists()
        assert (tmp_path / "viz_calibration.csv").exists()

    def test_minimal(self, tmp_path):
        windows = _make_windows()
        features = _make_features(windows)
        cv = _make_cv_results(windows, features)

        generated = generate_all_visualizations(
            cv, windows, output_dir=tmp_path
        )
        assert "patient_risk_trajectories" in generated
        assert len(generated) >= 2  # trajectories + at least one representative
