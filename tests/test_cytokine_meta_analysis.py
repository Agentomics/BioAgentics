"""Tests for cytokine_meta_analysis module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.cytokine_extraction import CytokineDataset, CytokineRecord
from bioagentics.cytokine_meta_analysis import (
    MetaAnalysisResult,
    dersimonian_laird,
    forest_plot,
    hedges_g,
    results_to_dataframe,
    run_meta_analysis,
)


# -- hedges_g tests ----------------------------------------------------------


def test_hedges_g_positive_effect():
    """Higher mean in group 1 should give positive g."""
    g, v = hedges_g(30, 20.0, 5.0, 30, 10.0, 5.0)
    assert g > 0
    assert v > 0


def test_hedges_g_zero_effect():
    """Equal means should give g ≈ 0."""
    g, v = hedges_g(30, 10.0, 5.0, 30, 10.0, 5.0)
    assert abs(g) < 0.01


def test_hedges_g_invalid_sd():
    """Zero SD should return NaN."""
    g, v = hedges_g(10, 5.0, 0.0, 10, 3.0, 0.0)
    assert np.isnan(g)


# -- DerSimonian-Laird tests -------------------------------------------------


def test_dsl_single_study():
    """Single study should give same effect as input."""
    effects = np.array([1.0])
    variances = np.array([0.1])
    result = dersimonian_laird(effects, variances)
    assert result.k == 1
    assert abs(result.pooled_effect - 1.0) < 0.01


def test_dsl_homogeneous_studies():
    """Identical effects should yield I² ≈ 0."""
    effects = np.array([0.5, 0.5, 0.5, 0.5])
    variances = np.array([0.1, 0.1, 0.1, 0.1])
    result = dersimonian_laird(effects, variances)
    assert result.i_sq < 1.0
    assert abs(result.pooled_effect - 0.5) < 0.05


def test_dsl_heterogeneous_studies():
    """Very different effects should yield high I²."""
    effects = np.array([0.1, 2.0, 0.2, 1.8])
    variances = np.array([0.05, 0.05, 0.05, 0.05])
    result = dersimonian_laird(effects, variances)
    assert result.i_sq > 50


# -- Integration tests -------------------------------------------------------


def _make_dataset() -> CytokineDataset:
    """Create a test dataset with 4 studies measuring IL-6 (flare vs remission)."""
    records = []
    studies = [
        ("StudyA", 20, 15.0, 3.0, 20, 8.0, 2.5),
        ("StudyB", 25, 18.0, 5.0, 25, 9.0, 3.0),
        ("StudyC", 15, 14.0, 4.0, 15, 7.0, 3.5),
        ("StudyD", 30, 16.0, 3.5, 30, 10.0, 2.0),
    ]
    for sid, n_f, m_f, s_f, n_r, m_r, s_r in studies:
        records.append(CytokineRecord(
            study_id=sid, analyte_name="IL-6", measurement_method="ELISA",
            sample_type="serum", condition="flare", sample_size_n=n_f,
            mean_or_median=m_f, sd_or_iqr=s_f,
        ))
        records.append(CytokineRecord(
            study_id=sid, analyte_name="IL-6", measurement_method="ELISA",
            sample_type="serum", condition="remission", sample_size_n=n_r,
            mean_or_median=m_r, sd_or_iqr=s_r,
        ))
    return CytokineDataset(records)


def test_run_meta_analysis():
    ds = _make_dataset()
    results = run_meta_analysis(ds, min_studies=3)
    assert len(results) == 1
    r = results[0]
    assert r.analyte == "IL-6"
    assert r.k == 4
    assert r.pooled_effect > 0  # IL-6 is elevated in flare
    assert r.p_value < 0.05


def test_forest_plot_generated(tmp_path):
    ds = _make_dataset()
    results = run_meta_analysis(ds, min_studies=3, output_dir=tmp_path)
    assert len(results) == 1
    plots = list(tmp_path.glob("forest_*.png"))
    assert len(plots) == 1


def test_results_to_dataframe():
    ds = _make_dataset()
    results = run_meta_analysis(ds, min_studies=3)
    df = results_to_dataframe(results)
    assert len(df) == 1
    assert "pooled_g" in df.columns
    assert df.iloc[0]["analyte"] == "IL-6"
    assert bool(df.iloc[0]["significant"]) is True


def test_min_studies_filter():
    """With min_studies=5, our 4-study dataset should yield no results."""
    ds = _make_dataset()
    results = run_meta_analysis(ds, min_studies=5)
    assert len(results) == 0
