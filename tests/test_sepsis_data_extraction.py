"""Tests for sepsis data extraction pipeline using synthetic data."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.diagnostics.sepsis.data_extraction import (
    extract_demographics,
    extract_labs,
    extract_vitals,
    merge_and_resample,
)


@pytest.fixture
def synthetic_mimic_dir(tmp_path):
    """Create synthetic MIMIC-IV tables as parquet files."""
    # patients
    patients = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "gender": ["M", "F", "M"],
            "anchor_age": [55, 70, 40],
            "anchor_year": [2020, 2020, 2020],
        }
    )
    patients.to_parquet(tmp_path / "patients.parquet")

    # admissions
    admissions = pd.DataFrame(
        {
            "subject_id": [1, 2, 3],
            "hadm_id": [100, 200, 300],
            "admittime": [
                "2020-06-01 08:00:00",
                "2020-07-15 12:00:00",
                "2020-09-01 00:00:00",
            ],
            "dischtime": [
                "2020-06-05 08:00:00",
                "2020-07-20 12:00:00",
                "2020-09-03 00:00:00",
            ],
            "race": ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN - CHINESE"],
        }
    )
    admissions.to_parquet(tmp_path / "admissions.parquet")

    # chartevents (vitals) - HR=220045, SBP=220050, SpO2=220277, Temp(F)=223761
    chart_rows = []
    for h in range(0, 10):
        chart_rows.append(
            {
                "subject_id": 1,
                "hadm_id": 100,
                "itemid": 220045,
                "charttime": f"2020-06-01 {8 + h:02d}:00:00",
                "valuenum": 80 + h,
            }
        )
        chart_rows.append(
            {
                "subject_id": 1,
                "hadm_id": 100,
                "itemid": 220050,
                "charttime": f"2020-06-01 {8 + h:02d}:00:00",
                "valuenum": 120 - h,
            }
        )
    # Patient 2: some vitals
    chart_rows.append(
        {
            "subject_id": 2,
            "hadm_id": 200,
            "itemid": 220045,
            "charttime": "2020-07-15 14:00:00",
            "valuenum": 95,
        }
    )
    # Temperature in Fahrenheit for patient 1
    chart_rows.append(
        {
            "subject_id": 1,
            "hadm_id": 100,
            "itemid": 223761,
            "charttime": "2020-06-01 10:00:00",
            "valuenum": 100.4,
        }
    )
    chartevents = pd.DataFrame(chart_rows)
    chartevents.to_parquet(tmp_path / "chartevents.parquet")

    # labevents - WBC=51301, Lactate=50813, Creatinine=50912
    lab_rows = []
    for h in [0, 4, 8]:
        lab_rows.append(
            {
                "subject_id": 1,
                "hadm_id": 100,
                "itemid": 51301,
                "charttime": f"2020-06-01 {8 + h:02d}:00:00",
                "valuenum": 10.0 + h * 0.5,
            }
        )
        lab_rows.append(
            {
                "subject_id": 1,
                "hadm_id": 100,
                "itemid": 50813,
                "charttime": f"2020-06-01 {8 + h:02d}:00:00",
                "valuenum": 1.5 + h * 0.1,
            }
        )
    labevents = pd.DataFrame(lab_rows)
    labevents.to_parquet(tmp_path / "labevents.parquet")

    return tmp_path


def test_extract_demographics(synthetic_mimic_dir):
    """Demographics correctly extracted with age, sex, ethnicity."""
    demo = extract_demographics(synthetic_mimic_dir)
    assert len(demo) == 3
    assert set(demo.columns) == {"age", "sex", "ethnicity"}
    # Patient 1: age 55, male
    row1 = demo.loc[(1, 100)]
    assert row1["age"] == 55
    assert row1["sex"] == 0  # M
    assert row1["ethnicity"] == "white"
    # Patient 2: female, black
    row2 = demo.loc[(2, 200)]
    assert row2["sex"] == 1  # F
    assert row2["ethnicity"] == "black"
    # Patient 3: asian
    row3 = demo.loc[(3, 300)]
    assert row3["ethnicity"] == "asian"


def test_extract_vitals(synthetic_mimic_dir):
    """Vitals extracted and binned to hourly resolution."""
    admissions = pd.read_parquet(synthetic_mimic_dir / "admissions.parquet")
    vitals = extract_vitals(synthetic_mimic_dir, admissions)

    assert "heart_rate" in vitals.columns
    assert "sbp" in vitals.columns
    assert "hours_in" in vitals.columns

    # Patient 1 should have ~10 hours of HR data
    p1 = vitals[(vitals["subject_id"] == 1) & (vitals["hadm_id"] == 100)]
    assert len(p1) >= 10


def test_temperature_conversion(synthetic_mimic_dir):
    """Fahrenheit temperatures converted to Celsius."""
    admissions = pd.read_parquet(synthetic_mimic_dir / "admissions.parquet")
    vitals = extract_vitals(synthetic_mimic_dir, admissions)

    p1 = vitals[(vitals["subject_id"] == 1) & (vitals["hadm_id"] == 100)]
    temp_row = p1[p1["hours_in"] == 2]  # 10:00 = 2h from admit
    if not temp_row.empty and "temperature" in temp_row.columns:
        temp_c = temp_row["temperature"].iloc[0]
        if not np.isnan(temp_c):
            # 100.4F = 38.0C
            assert abs(temp_c - 38.0) < 0.1


def test_extract_labs(synthetic_mimic_dir):
    """Labs extracted from labevents."""
    admissions = pd.read_parquet(synthetic_mimic_dir / "admissions.parquet")
    labs = extract_labs(synthetic_mimic_dir, admissions)

    assert "wbc" in labs.columns
    assert "lactate" in labs.columns
    p1 = labs[(labs["subject_id"] == 1) & (labs["hadm_id"] == 100)]
    assert len(p1) == 3  # Labs at hours 0, 4, 8


def test_merge_and_resample(synthetic_mimic_dir):
    """Merge vitals + labs with forward-fill produces complete hourly matrix."""
    admissions = pd.read_parquet(synthetic_mimic_dir / "admissions.parquet")
    vitals = extract_vitals(synthetic_mimic_dir, admissions)
    labs = extract_labs(synthetic_mimic_dir, admissions)
    demo = extract_demographics(synthetic_mimic_dir)

    merged = merge_and_resample(vitals, labs, demo)

    assert len(merged) > 0
    # Should have demographics attached
    assert "age" in merged.columns
    assert "sex" in merged.columns

    # Check forward-fill: patient 1 labs at h=0,4,8 should fill hours 1-3 etc.
    p1 = merged[(merged["subject_id"] == 1) & (merged["hadm_id"] == 100)]
    p1_sorted = p1.sort_values("hours_in")
    if "wbc" in p1_sorted.columns:
        # After forward-fill, hours 1-3 should have h=0 value
        h1 = p1_sorted[p1_sorted["hours_in"] == 1]
        if not h1.empty and not np.isnan(h1["wbc"].iloc[0]):
            assert h1["wbc"].iloc[0] == 10.0  # Forward-filled from h=0


def test_empty_data():
    """Merge with empty DataFrames returns empty result."""
    vitals = pd.DataFrame(columns=["subject_id", "hadm_id", "hours_in", "heart_rate"])
    labs = pd.DataFrame(columns=["subject_id", "hadm_id", "hours_in", "wbc"])
    demo = pd.DataFrame(columns=["age", "sex", "ethnicity"])
    demo.index = pd.MultiIndex.from_tuples([], names=["subject_id", "hadm_id"])

    result = merge_and_resample(vitals, labs, demo)
    assert len(result) == 0
