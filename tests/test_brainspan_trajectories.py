"""Tests for BrainSpan developmental trajectory analysis module."""

import numpy as np
import pandas as pd

from bioagentics.analysis.tourettes.brainspan_trajectories import (
    CSTC_REGIONS,
    DEV_STAGES,
    classify_dev_stage,
    cluster_temporal_patterns,
    identify_peak_windows,
    match_cstc_region,
    parse_age,
)


def _make_trajectory_df() -> pd.DataFrame:
    """Create synthetic trajectory data for testing."""
    rng = np.random.default_rng(42)
    genes = ["FLT3", "HDC", "SLITRK1", "WWC1", "MEIS1"]
    stages = list(DEV_STAGES.keys())
    regions = list(CSTC_REGIONS.keys())
    records = []
    for gene in genes:
        for stage in stages:
            for region in regions:
                rpkm = rng.lognormal(3, 1)
                records.append({
                    "gene_symbol": gene,
                    "dev_stage": stage,
                    "cstc_region": region,
                    "mean_rpkm": rpkm,
                    "mean_log2_rpkm": np.log2(rpkm + 1),
                    "n_samples": 3,
                })
    return pd.DataFrame(records)


def test_parse_age_pcw():
    period, value = parse_age("12 pcw")
    assert period == "prenatal"
    assert value == 12.0


def test_parse_age_months():
    period, value = parse_age("6 mos")
    assert period == "postnatal"
    assert abs(value - 0.5) < 0.01


def test_parse_age_years():
    period, value = parse_age("8 yrs")
    assert period == "postnatal"
    assert value == 8.0


def test_classify_dev_stage_prenatal():
    assert classify_dev_stage("prenatal", 10) == "early_prenatal"
    assert classify_dev_stage("prenatal", 20) == "mid_prenatal"
    assert classify_dev_stage("prenatal", 30) == "late_prenatal"


def test_classify_dev_stage_postnatal():
    assert classify_dev_stage("postnatal", 0.5) == "infancy"
    assert classify_dev_stage("postnatal", 3.0) == "early_childhood"
    assert classify_dev_stage("postnatal", 8.0) == "late_childhood"
    assert classify_dev_stage("postnatal", 15.0) == "adolescence"
    assert classify_dev_stage("postnatal", 30.0) == "adulthood"


def test_match_cstc_region():
    assert match_cstc_region("caudate nucleus") == "striatum"
    assert match_cstc_region("dorsolateral prefrontal cortex") == "cortex"
    assert match_cstc_region("mediodorsal nucleus of thalamus") == "thalamus"
    assert match_cstc_region("cerebellum") is None


def test_identify_peak_windows():
    df = _make_trajectory_df()
    peaks = identify_peak_windows(df)
    assert len(peaks) == 5
    assert "peak_stage" in peaks.columns
    assert "peak_region" in peaks.columns
    assert "peak_expression" in peaks.columns


def test_identify_peak_windows_empty():
    result = identify_peak_windows(pd.DataFrame())
    assert result.empty


def test_cluster_temporal_patterns():
    df = _make_trajectory_df()
    clusters = cluster_temporal_patterns(df, n_clusters=2)
    assert len(clusters) == 5
    assert "cluster" in clusters.columns
    assert set(clusters["cluster"].unique()).issubset({0, 1})


def test_cluster_too_few_genes():
    df = _make_trajectory_df()
    single = df[df["gene_symbol"] == "FLT3"]
    result = cluster_temporal_patterns(single, n_clusters=3)
    assert result.empty


def test_dev_stages_complete():
    """All 8 developmental stages should be defined."""
    assert len(DEV_STAGES) == 8
    prenatal = [s for s in DEV_STAGES if "prenatal" in s]
    postnatal = [s for s in DEV_STAGES if "prenatal" not in s]
    assert len(prenatal) == 3
    assert len(postnatal) == 5
