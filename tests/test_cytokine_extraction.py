"""Tests for cytokine_extraction module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from bioagentics.cytokine_extraction import (
    ANALYTE_VOCABULARY,
    CIRS_BIOMARKERS,
    BBB_MARKERS,
    CytokineDataset,
    CytokineRecord,
    write_template_csv,
)


# -- Vocabulary tests -------------------------------------------------------


def test_cirs_biomarkers_in_vocabulary():
    for marker in CIRS_BIOMARKERS:
        assert marker in ANALYTE_VOCABULARY


def test_bbb_markers_in_vocabulary():
    for marker in BBB_MARKERS:
        assert marker in ANALYTE_VOCABULARY


def test_il6_in_vocabulary():
    assert "IL-6" in ANALYTE_VOCABULARY


# -- CytokineRecord tests ---------------------------------------------------


def test_valid_record():
    r = CytokineRecord(
        study_id="Frankovich2015",
        pmid="25678901",
        analyte_name="IL-6",
        measurement_method="ELISA",
        sample_type="serum",
        condition="flare",
        sample_size_n=20,
        mean_or_median=15.3,
        sd_or_iqr=4.2,
        p_value=0.003,
    )
    assert r.study_id == "Frankovich2015"
    assert r.analyte_name == "IL-6"


def test_record_optional_fields():
    r = CytokineRecord(
        study_id="Test2020",
        analyte_name="TNF-α",
        measurement_method="Luminex",
        sample_type="plasma",
        condition="remission",
        sample_size_n=10,
        mean_or_median=5.0,
    )
    assert r.pmid is None
    assert r.sd_or_iqr is None
    assert r.p_value is None


def test_record_invalid_sample_size():
    with pytest.raises(Exception):
        CytokineRecord(
            study_id="Bad",
            analyte_name="IL-6",
            measurement_method="ELISA",
            sample_type="serum",
            condition="flare",
            sample_size_n=0,
            mean_or_median=1.0,
        )


def test_record_invalid_p_value():
    with pytest.raises(Exception):
        CytokineRecord(
            study_id="Bad",
            analyte_name="IL-6",
            measurement_method="ELISA",
            sample_type="serum",
            condition="flare",
            sample_size_n=5,
            mean_or_median=1.0,
            p_value=1.5,
        )


# -- CSV round-trip tests ---------------------------------------------------


def _make_test_csv(tmp_path: Path) -> Path:
    csv_path = tmp_path / "test.csv"
    df = pd.DataFrame([
        {
            "study_id": "A2020", "pmid": "123", "analyte_name": "IL-6",
            "measurement_method": "ELISA", "sample_type": "serum",
            "condition": "flare", "sample_size_n": 20,
            "mean_or_median": 15.0, "sd_or_iqr": 3.0, "p_value": 0.01,
            "notes": "",
        },
        {
            "study_id": "A2020", "pmid": "123", "analyte_name": "IL-6",
            "measurement_method": "ELISA", "sample_type": "serum",
            "condition": "remission", "sample_size_n": 20,
            "mean_or_median": 8.0, "sd_or_iqr": 2.5, "p_value": 0.01,
            "notes": "",
        },
        {
            "study_id": "B2021", "pmid": "456", "analyte_name": "IL-6",
            "measurement_method": "Luminex", "sample_type": "plasma",
            "condition": "flare", "sample_size_n": 15,
            "mean_or_median": 18.0, "sd_or_iqr": 5.0, "p_value": 0.005,
            "notes": "",
        },
        {
            "study_id": "B2021", "pmid": "456", "analyte_name": "IL-6",
            "measurement_method": "Luminex", "sample_type": "plasma",
            "condition": "remission", "sample_size_n": 15,
            "mean_or_median": 9.0, "sd_or_iqr": 3.0, "p_value": 0.005,
            "notes": "",
        },
    ])
    df.to_csv(csv_path, index=False)
    return csv_path


def test_csv_load(tmp_path):
    csv_path = _make_test_csv(tmp_path)
    ds = CytokineDataset.from_csv(csv_path)
    assert len(ds) == 4
    assert "IL-6" in ds.analytes()
    assert len(ds.studies()) == 2


def test_paired_effects(tmp_path):
    csv_path = _make_test_csv(tmp_path)
    ds = CytokineDataset.from_csv(csv_path)
    pairs = ds.paired_effects("IL-6")
    assert len(pairs) == 2
    assert "mean_a" in pairs.columns
    assert "mean_b" in pairs.columns


def test_json_roundtrip(tmp_path):
    csv_path = _make_test_csv(tmp_path)
    ds = CytokineDataset.from_csv(csv_path)
    json_path = tmp_path / "out.json"
    ds.to_json(json_path)
    ds2 = CytokineDataset.from_json(json_path)
    assert len(ds2) == len(ds)


def test_summary(tmp_path):
    csv_path = _make_test_csv(tmp_path)
    ds = CytokineDataset.from_csv(csv_path)
    s = ds.summary()
    assert s["n_records"] == 4
    assert s["n_studies"] == 2
    assert s["n_analytes"] == 1


# -- Template CSV test -------------------------------------------------------


def test_write_template_csv(tmp_path):
    out = write_template_csv(tmp_path / "template.csv")
    assert out.exists()
    df = pd.read_csv(out)
    assert "study_id" in df.columns
    assert "analyte_name" in df.columns
    assert "treatment" in df.columns
    assert len(df) == 0


def test_treatment_field():
    r = CytokineRecord(
        study_id="T2024",
        analyte_name="IL-6",
        measurement_method="ELISA",
        sample_type="serum",
        condition="flare",
        sample_size_n=15,
        mean_or_median=12.0,
        treatment="IVIG",
    )
    assert r.treatment == "IVIG"


def test_treatment_field_optional():
    r = CytokineRecord(
        study_id="T2024",
        analyte_name="IL-6",
        measurement_method="ELISA",
        sample_type="serum",
        condition="flare",
        sample_size_n=15,
        mean_or_median=12.0,
    )
    assert r.treatment is None


def test_filter_treatment():
    records = [
        CytokineRecord(
            study_id="A", analyte_name="IL-6", measurement_method="ELISA",
            sample_type="serum", condition="flare", sample_size_n=10,
            mean_or_median=10.0, treatment="IVIG",
        ),
        CytokineRecord(
            study_id="B", analyte_name="IL-6", measurement_method="ELISA",
            sample_type="serum", condition="flare", sample_size_n=10,
            mean_or_median=12.0, treatment="plasmapheresis",
        ),
        CytokineRecord(
            study_id="C", analyte_name="IL-6", measurement_method="ELISA",
            sample_type="serum", condition="flare", sample_size_n=10,
            mean_or_median=14.0,
        ),
    ]
    ds = CytokineDataset(records)
    filtered = ds.filter_treatment("IVIG")
    assert len(filtered) == 1
    assert filtered.records[0].study_id == "A"
