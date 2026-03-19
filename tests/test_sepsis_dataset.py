"""Tests for sepsis dataset generator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.sepsis.dataset import (
    extract_prediction_samples,
    generate_datasets,
    get_feature_columns,
)


@pytest.fixture
def synthetic_features_and_labels():
    """Create synthetic engineered features and labels for 10 admissions."""
    rng = np.random.default_rng(42)
    rows = []
    labels_data = []

    for i in range(10):
        sid = i + 1
        hid = (i + 1) * 100
        is_sepsis = i < 3  # First 3 are sepsis
        n_hours = 48

        for h in range(n_hours):
            rows.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "hours_in": h,
                    "heart_rate": 80 + rng.normal(0, 5),
                    "wbc": 10 + rng.normal(0, 2),
                    "creatinine": 1.0 + rng.normal(0, 0.3),
                    "heart_rate_delta_1h": rng.normal(0, 2),
                    "wbc_roll_mean": 10 + rng.normal(0, 1),
                    "heart_rate_missing": 0,
                }
            )

        if is_sepsis:
            labels_data.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "sepsis_label": 1,
                    "sepsis_onset_hour": 24 + rng.integers(0, 12),
                }
            )
        else:
            labels_data.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "sepsis_label": 0,
                    "sepsis_onset_hour": np.nan,
                }
            )

    features = pd.DataFrame(rows)
    labels = pd.DataFrame(labels_data)
    return features, labels


def test_get_feature_columns(synthetic_features_and_labels):
    """Feature columns exclude IDs and metadata."""
    features, _ = synthetic_features_and_labels
    feat_cols = get_feature_columns(features)

    assert "subject_id" not in feat_cols
    assert "hadm_id" not in feat_cols
    assert "hours_in" not in feat_cols
    assert "heart_rate" in feat_cols
    assert "heart_rate_delta_1h" in feat_cols


def test_extract_prediction_samples(synthetic_features_and_labels):
    """Prediction samples extracted at correct lookahead."""
    features, labels = synthetic_features_and_labels
    rng = np.random.default_rng(42)

    X, y = extract_prediction_samples(features, labels, lookahead=6, rng=rng)

    assert len(X) > 0
    assert len(X) == len(y)
    assert y.sum() > 0  # Some positive
    assert (y == 0).sum() > 0  # Some negative
    # All features present
    assert X.shape[1] > 0


def test_extract_prediction_samples_insufficient_lead_time(
    synthetic_features_and_labels,
):
    """Admissions with insufficient lead time are skipped."""
    features, labels = synthetic_features_and_labels

    # Set onset very early so 12h lookahead is impossible
    labels_early = labels.copy()
    labels_early.loc[labels_early["sepsis_label"] == 1, "sepsis_onset_hour"] = 5

    X, y = extract_prediction_samples(features, labels_early, lookahead=12)
    # All sepsis cases should be skipped (onset=5, lookahead=12, need hour -7)
    assert y.sum() == 0


def test_generate_datasets(synthetic_features_and_labels):
    """Full dataset generation produces train/test splits for each lookahead."""
    features, labels = synthetic_features_and_labels

    datasets = generate_datasets(features, labels, lookaheads=[6, 12])

    for lh, data in datasets.items():
        assert "X_train" in data
        assert "X_test" in data
        assert "y_train" in data
        assert "y_test" in data
        assert "feature_names" in data
        assert len(data["X_train"]) > 0
        assert len(data["X_test"]) > 0
        assert data["X_train"].shape[1] == data["X_test"].shape[1]
        assert data["X_train"].shape[1] == len(data["feature_names"])


def test_generate_datasets_preserves_class_balance(synthetic_features_and_labels):
    """Stratified split preserves approximate class balance."""
    features, labels = synthetic_features_and_labels

    datasets = generate_datasets(features, labels, lookaheads=[6])

    if 6 in datasets:
        data = datasets[6]
        # Both train and test should have at least one positive and one negative
        assert data["y_train"].sum() > 0
        assert (data["y_train"] == 0).sum() > 0
        assert data["y_test"].sum() > 0
        assert (data["y_test"] == 0).sum() > 0
