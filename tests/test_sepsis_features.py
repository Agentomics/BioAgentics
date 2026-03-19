"""Tests for sepsis feature engineering module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.sepsis.features import (
    compute_deltas,
    compute_missingness,
    compute_rolling_stats,
    engineer_features_single,
)


@pytest.fixture
def sample_hourly():
    """Single-admission hourly DataFrame with known values."""
    return pd.DataFrame(
        {
            "subject_id": [1] * 12,
            "hadm_id": [100] * 12,
            "hours_in": list(range(12)),
            "heart_rate": [80, 82, 85, 88, 90, 92, 95, 98, 100, 102, 104, 106],
            "wbc": [
                10.0,
                np.nan,
                np.nan,
                np.nan,
                12.0,
                np.nan,
                np.nan,
                np.nan,
                14.0,
                np.nan,
                np.nan,
                np.nan,
            ],
        }
    )


def test_compute_deltas(sample_hourly):
    """Delta values computed correctly."""
    deltas = compute_deltas(sample_hourly, ["heart_rate"], windows=[1, 6])

    # 1h delta at index 1: 82 - 80 = 2
    assert deltas["heart_rate_delta_1h"].iloc[1] == 2.0

    # 6h delta at index 6: 95 - 80 = 15
    assert deltas["heart_rate_delta_6h"].iloc[6] == 15.0

    # 1h delta at index 0 should be NaN (no prior value)
    assert np.isnan(deltas["heart_rate_delta_1h"].iloc[0])


def test_compute_rolling_stats(sample_hourly):
    """Rolling mean, std, slope computed over window."""
    rolling = compute_rolling_stats(sample_hourly, ["heart_rate"], window=6)

    assert "heart_rate_roll_mean" in rolling.columns
    assert "heart_rate_roll_std" in rolling.columns
    assert "heart_rate_roll_slope" in rolling.columns

    # At index 5 (6 values: 80,82,85,88,90,92), mean = 86.17
    mean_at_5 = rolling["heart_rate_roll_mean"].iloc[5]
    expected_mean = np.mean([80, 82, 85, 88, 90, 92])
    assert abs(mean_at_5 - expected_mean) < 0.01

    # Slope should be positive (increasing HR)
    slope_at_5 = rolling["heart_rate_roll_slope"].iloc[5]
    assert not np.isnan(slope_at_5)
    assert slope_at_5 > 0


def test_compute_missingness(sample_hourly):
    """Missingness indicators computed correctly."""
    missing = compute_missingness(sample_hourly, ["heart_rate", "wbc"])

    # heart_rate has no missing values
    assert missing["heart_rate_missing"].sum() == 0

    # wbc: missing at hours 1,2,3,5,6,7,9,10,11 = 9 missing
    assert missing["wbc_missing"].sum() == 9

    # hours_since_obs at hour 0 should be 0 (observed)
    assert missing["wbc_hours_since_obs"].iloc[0] == 0

    # hours_since_obs at hour 3 should be 3 (last obs at hour 0)
    assert missing["wbc_hours_since_obs"].iloc[3] == 3

    # hours_since_obs at hour 4 should be 0 (observed)
    assert missing["wbc_hours_since_obs"].iloc[4] == 0


def test_engineer_features_single(sample_hourly):
    """Full feature engineering produces expected columns."""
    result = engineer_features_single(sample_hourly, features=["heart_rate", "wbc"])

    # Original columns preserved
    assert "heart_rate" in result.columns
    assert "wbc" in result.columns

    # Delta columns
    assert "heart_rate_delta_1h" in result.columns
    assert "wbc_delta_6h" in result.columns

    # Rolling columns
    assert "heart_rate_roll_mean" in result.columns
    assert "heart_rate_roll_slope" in result.columns

    # Missingness columns
    assert "wbc_missing" in result.columns
    assert "wbc_hours_since_obs" in result.columns

    # Same number of rows
    assert len(result) == len(sample_hourly)


def test_missing_feature_handled():
    """Features not in DataFrame produce NaN columns gracefully."""
    df = pd.DataFrame(
        {
            "subject_id": [1] * 5,
            "hadm_id": [100] * 5,
            "hours_in": list(range(5)),
            "heart_rate": [80, 82, 84, 86, 88],
        }
    )
    result = engineer_features_single(df, features=["heart_rate", "nonexistent_feature"])

    # nonexistent_feature missingness should be all 1
    assert "nonexistent_feature_missing" in result.columns
    assert result["nonexistent_feature_missing"].sum() == 5
