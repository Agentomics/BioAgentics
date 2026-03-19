"""Tests for longitudinal outcome tracker."""

import numpy as np
import pandas as pd

from bioagentics.diagnostics.fp_mining.extract import ExtractionResult, OperatingPoint
from bioagentics.diagnostics.fp_mining.longitudinal import (
    build_patient_timeline,
    compute_hazard_ratio,
    compute_time_to_event_stats,
    run_longitudinal_analysis,
    track_fp_outcomes,
)


def _make_admissions() -> pd.DataFrame:
    """Create synthetic admissions data."""
    return pd.DataFrame({
        "subject_id": ["P1", "P1", "P2", "P2", "P3"],
        "hadm_id": ["A1", "A2", "A3", "A4", "A5"],
        "admittime": pd.to_datetime([
            "2024-01-01", "2024-03-15", "2024-02-01", "2024-06-01", "2024-01-15",
        ]),
    })


def _make_labels() -> pd.DataFrame:
    """Create synthetic sepsis labels."""
    return pd.DataFrame({
        "subject_id": ["P1", "P1", "P2", "P2", "P3"],
        "hadm_id": ["A1", "A2", "A3", "A4", "A5"],
        "sepsis_label": [0, 1, 0, 1, 0],
        "sepsis_onset_hour": [np.nan, 12.0, np.nan, 8.0, np.nan],
    })


def _make_fp_result() -> ExtractionResult:
    """Create FP result where P1 and P2 are FPs on their first admissions."""
    op = OperatingPoint("test", 0.5, 0.95, 0.80)
    fp = pd.DataFrame({
        "sample_id": ["P1_A1", "P2_A3"],
        "y_true": [0, 0],
        "y_score": [0.8, 0.7],
    })
    tn = pd.DataFrame({
        "sample_id": ["P3_A5"],
        "y_true": [0],
        "y_score": [0.2],
    })
    return ExtractionResult("sepsis", op, fp, tn, pd.DataFrame(), pd.DataFrame())


class TestBuildPatientTimeline:
    def test_sorted_by_time(self) -> None:
        admissions = _make_admissions()
        labels = _make_labels()
        timeline = build_patient_timeline(admissions, labels)
        for sid in timeline["subject_id"].unique():
            patient = timeline[timeline["subject_id"] == sid]
            times = patient["admittime"].values
            assert all(times[i] <= times[i + 1] for i in range(len(times) - 1))

    def test_labels_merged(self) -> None:
        admissions = _make_admissions()
        labels = _make_labels()
        timeline = build_patient_timeline(admissions, labels)
        assert "sepsis_label" in timeline.columns
        assert timeline["sepsis_label"].sum() == 2  # A2 and A4

    def test_missing_columns_raises(self) -> None:
        import pytest
        bad = pd.DataFrame({"x": [1]})
        with pytest.raises(ValueError, match="missing required columns"):
            build_patient_timeline(bad, pd.DataFrame())


class TestTrackFpOutcomes:
    def test_detects_subsequent_sepsis(self) -> None:
        result = _make_fp_result()
        timeline = build_patient_timeline(_make_admissions(), _make_labels())
        outcomes = track_fp_outcomes(result, timeline)
        assert len(outcomes) == 2

        p1 = outcomes[outcomes["subject_id"] == "P1"].iloc[0]
        assert p1["developed_sepsis"] == True
        assert p1["subsequent_admissions"] == 1
        assert p1["time_to_sepsis_days"] > 0

        p2 = outcomes[outcomes["subject_id"] == "P2"].iloc[0]
        assert p2["developed_sepsis"] == True

    def test_no_subsequent_admissions(self) -> None:
        op = OperatingPoint("test", 0.5, 0.95, 0.80)
        fp = pd.DataFrame({
            "sample_id": ["P3_A5"],
            "y_true": [0],
            "y_score": [0.8],
        })
        result = ExtractionResult("sepsis", op, fp, pd.DataFrame(), pd.DataFrame(), pd.DataFrame())
        timeline = build_patient_timeline(_make_admissions(), _make_labels())
        outcomes = track_fp_outcomes(result, timeline)
        assert len(outcomes) == 1
        assert outcomes.iloc[0]["developed_sepsis"] == False


class TestComputeHazardRatio:
    def test_ratio_calculation(self) -> None:
        fp_out = pd.DataFrame({"developed_sepsis": [True, True, False, False]})
        tn_out = pd.DataFrame({"developed_sepsis": [True, False, False, False, False, False, False, False]})
        result = compute_hazard_ratio(fp_out, tn_out)
        assert result["fp_sepsis_rate"] == 0.5
        assert result["tn_sepsis_rate"] == 0.125
        assert result["incidence_rate_ratio"] == 4.0
        assert result["hypothesis_supported"] == True

    def test_empty_tn(self) -> None:
        fp_out = pd.DataFrame({"developed_sepsis": [True]})
        tn_out = pd.DataFrame({"developed_sepsis": []})
        result = compute_hazard_ratio(fp_out, tn_out)
        assert result["n_tn"] == 0


class TestComputeTimeToEventStats:
    def test_with_events(self) -> None:
        outcomes = pd.DataFrame({
            "developed_sepsis": [True, True, False],
            "time_to_sepsis_days": [30.0, 60.0, np.nan],
        })
        stats = compute_time_to_event_stats(outcomes)
        assert stats["n_events"] == 2
        assert stats["median_days"] == 45.0
        assert stats["mean_days"] == 45.0

    def test_no_events(self) -> None:
        outcomes = pd.DataFrame({
            "developed_sepsis": [False, False],
            "time_to_sepsis_days": [np.nan, np.nan],
        })
        stats = compute_time_to_event_stats(outcomes)
        assert stats["n_events"] == 0


class TestRunLongitudinalAnalysis:
    def test_full_pipeline(self, tmp_path) -> None:
        result = _make_fp_result()
        admissions = _make_admissions()
        labels = _make_labels()

        output = run_longitudinal_analysis(
            result,
            result.true_negatives,
            admissions,
            labels,
            output_dir=tmp_path,
        )
        assert "hazard_ratio" in output
        assert "fp_outcomes" in output
        assert len(output["fp_outcomes"]) == 2
        # Check files saved
        parquets = list(tmp_path.glob("*.parquet"))
        assert len(parquets) == 2
