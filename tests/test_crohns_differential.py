"""Tests for single-omic differential analysis (CD vs control)."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_differential import (
    DifferentialAnalysis,
    _benjamini_hochberg,
    cliffs_delta,
    extract_known_features,
    wilcoxon_ranksum_test,
)


@pytest.fixture
def differential_data():
    """Synthetic data with known differentially abundant features."""
    rng = np.random.default_rng(42)
    n_cd, n_ctrl = 20, 15
    n_features = 10

    # Base data
    data = rng.normal(0, 1, (n_cd + n_ctrl, n_features))
    cols = [f"feature_{i}" for i in range(n_features)]

    # Make feature_0 strongly elevated in CD
    data[:n_cd, 0] += 3.0
    # Make feature_1 depleted in CD
    data[:n_cd, 1] -= 2.5
    # feature_2-9: no real difference

    idx = [f"CD_{i}" for i in range(n_cd)] + [f"ctrl_{i}" for i in range(n_ctrl)]
    df = pd.DataFrame(data, index=idx, columns=cols)

    labels = pd.Series(
        ["CD"] * n_cd + ["nonIBD"] * n_ctrl,
        index=idx,
        name="diagnosis",
    )

    return df, labels


# ── Wilcoxon Test ──


def test_wilcoxon_returns_correct_columns(differential_data):
    df, labels = differential_data
    results = wilcoxon_ranksum_test(df, labels)
    assert "feature" in results.columns
    assert "p_value" in results.columns
    assert "fdr" in results.columns
    assert "log2fc" in results.columns
    assert "statistic" in results.columns


def test_wilcoxon_finds_significant_features(differential_data):
    df, labels = differential_data
    results = wilcoxon_ranksum_test(df, labels)
    # feature_0 (elevated in CD) should be significant
    sig = results[results["fdr"] < 0.05]
    sig_features = sig["feature"].values
    assert "feature_0" in sig_features
    assert "feature_1" in sig_features


def test_wilcoxon_fold_change_direction(differential_data):
    df, labels = differential_data
    results = wilcoxon_ranksum_test(df, labels)
    # feature_0 elevated in CD → positive log2fc
    fc0 = results[results["feature"] == "feature_0"]["log2fc"].values[0]
    assert fc0 > 0
    # feature_1 depleted in CD → negative log2fc
    fc1 = results[results["feature"] == "feature_1"]["log2fc"].values[0]
    assert fc1 < 0


def test_wilcoxon_empty_group():
    df = pd.DataFrame({"a": [1, 2, 3]}, index=["s1", "s2", "s3"])
    labels = pd.Series(["CD", "CD", "CD"], index=["s1", "s2", "s3"])
    with pytest.raises(ValueError, match="No samples"):
        wilcoxon_ranksum_test(df, labels)


# ── BH FDR ──


def test_benjamini_hochberg_monotone():
    pvals = np.array([0.001, 0.01, 0.03, 0.5, 0.9])
    fdr = _benjamini_hochberg(pvals)
    # FDR should be <= 1
    assert all(fdr <= 1.0 + 1e-10)
    # Sorted FDR values should be non-decreasing (monotone)
    order = np.argsort(pvals)
    sorted_fdr = fdr[order]
    assert all(sorted_fdr[i] <= sorted_fdr[i + 1] + 1e-10 for i in range(len(sorted_fdr) - 1))


def test_benjamini_hochberg_single():
    fdr = _benjamini_hochberg(np.array([0.05]))
    assert fdr[0] == pytest.approx(0.05)


# ── Cliff's Delta ──


def test_cliffs_delta_range(differential_data):
    df, labels = differential_data
    deltas = cliffs_delta(df, labels)
    # Cliff's delta should be in [-1, 1]
    assert all(-1 <= d <= 1 for d in deltas.values)


def test_cliffs_delta_large_effect(differential_data):
    df, labels = differential_data
    deltas = cliffs_delta(df, labels)
    # feature_0 has large positive effect
    assert abs(deltas["feature_0"]) > 0.474
    # feature_1 has large negative effect
    assert abs(deltas["feature_1"]) > 0.474


# ── Known Feature Extraction ──


def test_extract_known_features(differential_data):
    df, labels = differential_data
    results = wilcoxon_ranksum_test(df, labels)
    known = extract_known_features(results, ["feature_0", "feature_1", "nonexistent"])
    assert len(known) == 2
    assert "known_feature" in known.columns


def test_extract_known_features_no_match(differential_data):
    df, labels = differential_data
    results = wilcoxon_ranksum_test(df, labels)
    known = extract_known_features(results, ["nonexistent_xyz"])
    assert len(known) == 0


# ── DifferentialAnalysis Pipeline ──


def test_differential_analysis_run(differential_data):
    df, labels = differential_data
    da = DifferentialAnalysis()
    results = da.run(df, labels)
    assert "cliffs_delta" in results.columns
    assert "effect_magnitude" in results.columns
    assert len(results) == 10  # all features tested
    # Should be sorted by FDR
    assert list(results["fdr"]) == sorted(results["fdr"])
