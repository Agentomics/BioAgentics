"""Tests for MOFA2 integration pipeline (subtyping-specific)."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_mofa2 import (
    MOFA2Integration,
    compare_with_cmtf,
)


@pytest.fixture
def synthetic_views():
    """Synthetic multi-view data for MOFA2 testing."""
    rng = np.random.default_rng(42)
    n_samples = 25
    n_species = 12
    n_metab = 18

    # Shared latent structure
    F = rng.standard_normal((n_samples, 3))
    A = rng.standard_normal((n_species, 3))
    B = rng.standard_normal((n_metab, 3))

    species = pd.DataFrame(
        F @ A.T + rng.normal(0, 0.3, (n_samples, n_species)),
        index=[f"P{i}" for i in range(n_samples)],
        columns=[f"sp_{i}" for i in range(n_species)],
    )
    metabolomics = pd.DataFrame(
        F @ B.T + rng.normal(0, 0.3, (n_samples, n_metab)),
        index=[f"P{i}" for i in range(n_samples)],
        columns=[f"mb_{i}" for i in range(n_metab)],
    )
    return species, metabolomics


def test_mofa2_integration_fit(synthetic_views):
    """MOFA2Integration produces correct output structure."""
    species, metabolomics = synthetic_views
    mofa = MOFA2Integration(n_factors=5, seed=42)
    results = mofa.fit(species, metabolomics)

    assert "factors" in results
    assert "weights_species" in results
    assert "weights_metabolomics" in results
    assert "variance_explained" in results
    assert "factor_classification" in results

    # Factor matrix should have participant index
    assert list(results["factors"].index) == list(species.index)
    assert results["factors"].shape[0] == len(species)


def test_mofa2_integration_saves_files(synthetic_views, tmp_path):
    """MOFA2Integration saves output files."""
    species, metabolomics = synthetic_views
    mofa = MOFA2Integration(n_factors=3, seed=42)
    mofa.fit(species, metabolomics, output_dir=tmp_path)

    assert (tmp_path / "mofa2_factors.csv").exists()
    assert (tmp_path / "mofa2_weights_species.csv").exists()
    assert (tmp_path / "mofa2_weights_metabolomics.csv").exists()


def test_mofa2_factor_classification(synthetic_views):
    """Factor classification produces valid categories."""
    species, metabolomics = synthetic_views
    mofa = MOFA2Integration(n_factors=5, seed=42)
    results = mofa.fit(species, metabolomics)

    valid_classes = {"shared", "species-specific", "metabolomics-specific", "weak"}
    for factor_idx, cls in results["factor_classification"].items():
        assert cls in valid_classes


def test_compare_with_cmtf():
    """compare_with_cmtf returns valid correlation metrics."""
    rng = np.random.default_rng(42)
    n = 20

    # Create two similar factor matrices
    shared = rng.standard_normal((n, 3))
    mofa_factors = pd.DataFrame(
        shared + rng.normal(0, 0.1, (n, 3)),
        index=[f"P{i}" for i in range(n)],
        columns=["MOFA_0", "MOFA_1", "MOFA_2"],
    )
    cmtf_factors = pd.DataFrame(
        shared + rng.normal(0, 0.1, (n, 3)),
        index=[f"P{i}" for i in range(n)],
        columns=["CMTF_0", "CMTF_1", "CMTF_2"],
    )

    metrics = compare_with_cmtf(mofa_factors, cmtf_factors)
    assert "mean_best_correlation" in metrics
    assert "max_best_correlation" in metrics
    # With low noise, correlation should be high
    assert metrics["mean_best_correlation"] > 0.5
    assert 0 <= metrics["cmtf_coverage"] <= 1


def test_compare_with_cmtf_different_sizes():
    """compare_with_cmtf handles different number of factors."""
    rng = np.random.default_rng(42)
    n = 15

    mofa_factors = pd.DataFrame(
        rng.standard_normal((n, 5)),
        index=[f"P{i}" for i in range(n)],
        columns=[f"MOFA_{i}" for i in range(5)],
    )
    cmtf_factors = pd.DataFrame(
        rng.standard_normal((n, 3)),
        index=[f"P{i}" for i in range(n)],
        columns=[f"CMTF_{i}" for i in range(3)],
    )

    metrics = compare_with_cmtf(mofa_factors, cmtf_factors)
    assert isinstance(metrics["n_matched_factors_gt_0.5"], int)
