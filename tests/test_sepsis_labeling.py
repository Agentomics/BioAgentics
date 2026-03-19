"""Tests for Sepsis-3 labeling module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.sepsis.labeling import (
    compute_sofa_coagulation,
    compute_sofa_liver,
    compute_sofa_renal,
    compute_sofa_scores,
    label_sepsis,
)


def test_sofa_coagulation():
    """Platelet-based SOFA scoring."""
    platelets = pd.Series([200, 149, 99, 49, 19, np.nan])
    scores = compute_sofa_coagulation(platelets)
    assert scores.iloc[0] == 0  # >= 150
    assert scores.iloc[1] == 1  # < 150
    assert scores.iloc[2] == 2  # < 100
    assert scores.iloc[3] == 3  # < 50
    assert scores.iloc[4] == 4  # < 20
    assert scores.iloc[5] == 0  # NaN -> 0


def test_sofa_liver():
    """Bilirubin-based SOFA scoring."""
    bilirubin = pd.Series([0.5, 1.2, 2.0, 6.0, 12.0])
    scores = compute_sofa_liver(bilirubin)
    assert scores.iloc[0] == 0
    assert scores.iloc[1] == 1
    assert scores.iloc[2] == 2
    assert scores.iloc[3] == 3
    assert scores.iloc[4] == 4


def test_sofa_renal():
    """Creatinine-based SOFA scoring."""
    creat = pd.Series([0.8, 1.2, 2.0, 3.5, 5.0])
    scores = compute_sofa_renal(creat)
    assert scores.iloc[0] == 0
    assert scores.iloc[1] == 1
    assert scores.iloc[2] == 2
    assert scores.iloc[3] == 3
    assert scores.iloc[4] == 4


def test_compute_sofa_scores_multi_organ():
    """Total SOFA from multiple organ components."""
    df = pd.DataFrame(
        {
            "platelets": [200, 49],
            "bilirubin_total": [0.5, 6.0],
            "creatinine": [0.8, 3.5],
            "map": [80, 60],
        }
    )
    sofa = compute_sofa_scores(df)
    assert sofa.iloc[0] == 0  # all normal
    assert sofa.iloc[1] == 3 + 3 + 3 + 1  # plat=3, bili=3, creat=3, map=1


def test_label_sepsis_positive():
    """Admission with infection + SOFA >= 2 labeled as sepsis."""
    # Hourly features with rising SOFA
    hourly = pd.DataFrame(
        {
            "subject_id": [1] * 20,
            "hadm_id": [100] * 20,
            "hours_in": list(range(20)),
            "platelets": [200] * 10 + [40] * 10,  # Normal then critical
            "creatinine": [0.8] * 10 + [4.0] * 10,
            "bilirubin_total": [0.5] * 20,
            "map": [80] * 20,
        }
    )
    infection_times = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "infection_time": ["2020-06-01 18:00:00"],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "admittime": ["2020-06-01 08:00:00"],
        }
    )

    labels = label_sepsis(hourly, infection_times, admissions)
    assert len(labels) == 1
    assert labels.iloc[0]["sepsis_label"] == 1
    assert labels.iloc[0]["sepsis_onset_hour"] == 10  # SOFA rises at hour 10


def test_label_sepsis_negative_no_infection():
    """Admission without infection labeled as no sepsis."""
    hourly = pd.DataFrame(
        {
            "subject_id": [1] * 10,
            "hadm_id": [100] * 10,
            "hours_in": list(range(10)),
            "platelets": [40] * 10,  # High SOFA but no infection
            "creatinine": [4.0] * 10,
        }
    )
    infection_times = pd.DataFrame(
        columns=["subject_id", "hadm_id", "infection_time"]
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "admittime": ["2020-06-01 08:00:00"],
        }
    )

    labels = label_sepsis(hourly, infection_times, admissions)
    assert labels.iloc[0]["sepsis_label"] == 0


def test_label_sepsis_negative_low_sofa():
    """Admission with infection but SOFA < 2 labeled as no sepsis."""
    hourly = pd.DataFrame(
        {
            "subject_id": [1] * 10,
            "hadm_id": [100] * 10,
            "hours_in": list(range(10)),
            "platelets": [200] * 10,  # Normal
            "creatinine": [0.8] * 10,
            "bilirubin_total": [0.5] * 10,
            "map": [80] * 10,
        }
    )
    infection_times = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "infection_time": ["2020-06-01 12:00:00"],
        }
    )
    admissions = pd.DataFrame(
        {
            "subject_id": [1],
            "hadm_id": [100],
            "admittime": ["2020-06-01 08:00:00"],
        }
    )

    labels = label_sepsis(hourly, infection_times, admissions)
    assert labels.iloc[0]["sepsis_label"] == 0
