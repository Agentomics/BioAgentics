"""Tests for subtype-specific dependency analysis (synthetic data)."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.subtype_analysis import (
    SubtypeResults,
    analyze_subtype_dependencies,
    save_subtype_results,
)


@pytest.fixture
def synthetic_dep_data():
    """Create synthetic dependency data with known subtype differences."""
    rng = np.random.default_rng(42)
    n_per_group = 30
    n_patients = n_per_group * 4
    subtypes = (["KP"] * n_per_group + ["KL"] * n_per_group +
                ["KOnly"] * n_per_group + ["KRAS-WT"] * n_per_group)
    patient_ids = [f"TCGA-XX-{i:04d}" for i in range(n_patients)]

    # DIFF_GENE: strong difference between KP and KL
    kp_vals = rng.normal(-1.0, 0.3, n_per_group)
    kl_vals = rng.normal(0.5, 0.3, n_per_group)
    ko_vals = rng.normal(0.0, 0.3, n_per_group)
    wt_vals = rng.normal(0.0, 0.3, n_per_group)

    # NOISE_GENE: no difference
    noise = rng.normal(0, 1, n_patients)

    dep_matrix = pd.DataFrame({
        "DIFF_GENE": np.concatenate([kp_vals, kl_vals, ko_vals, wt_vals]),
        "NOISE_GENE": noise,
    }, index=patient_ids)

    subtype_series = pd.Series(subtypes, index=patient_ids, name="subtype")
    return dep_matrix, subtype_series


def test_analyze_basic(synthetic_dep_data):
    """Test basic subtype analysis returns correct structure."""
    dep_matrix, subtypes = synthetic_dep_data
    results = analyze_subtype_dependencies(dep_matrix, subtypes)

    assert isinstance(results, SubtypeResults)
    assert results.n_tested == 2
    assert len(results.kruskal_wallis) == 2
    assert "kw_stat" in results.kruskal_wallis.columns
    assert "kw_fdr" in results.kruskal_wallis.columns


def test_differential_gene_detected(synthetic_dep_data):
    """Test that a gene with known subtype differences is significant."""
    dep_matrix, subtypes = synthetic_dep_data
    results = analyze_subtype_dependencies(dep_matrix, subtypes)

    assert "DIFF_GENE" in results.significant_genes
    assert results.n_significant >= 1


def test_pairwise_comparisons(synthetic_dep_data):
    """Test that pairwise comparisons are generated for significant genes."""
    dep_matrix, subtypes = synthetic_dep_data
    results = analyze_subtype_dependencies(dep_matrix, subtypes)

    if results.n_significant > 0:
        assert len(results.pairwise) > 0
        assert "effect_size" in results.pairwise.columns
        assert "direction" in results.pairwise.columns


def test_kras_allele_axis(synthetic_dep_data):
    """Test KRAS allele-stratified analysis."""
    dep_matrix, subtypes = synthetic_dep_data
    rng = np.random.default_rng(99)

    alleles = []
    for st in subtypes:
        if st == "KRAS-WT":
            alleles.append("WT")
        else:
            alleles.append(rng.choice(["G12C", "G12D", "G12V"]))

    kras_alleles = pd.Series(alleles, index=dep_matrix.index)
    results = analyze_subtype_dependencies(dep_matrix, subtypes, kras_alleles=kras_alleles)

    # Should complete without error; allele results appended to pairwise
    assert isinstance(results, SubtypeResults)


def test_save_results(synthetic_dep_data, tmp_path):
    """Test saving subtype results to disk."""
    dep_matrix, subtypes = synthetic_dep_data
    results = analyze_subtype_dependencies(dep_matrix, subtypes)
    save_subtype_results(results, tmp_path / "subtype_results")

    assert (tmp_path / "subtype_results" / "kruskal_wallis_results.csv").exists()
    assert (tmp_path / "subtype_results" / "pairwise_comparisons.csv").exists()
    assert (tmp_path / "subtype_results" / "significant_genes.txt").exists()


def test_fdr_correction(synthetic_dep_data):
    """Test that FDR values are between 0 and 1."""
    dep_matrix, subtypes = synthetic_dep_data
    results = analyze_subtype_dependencies(dep_matrix, subtypes)

    fdr_vals = results.kruskal_wallis["kw_fdr"].values
    assert np.all(fdr_vals >= 0)
    assert np.all(fdr_vals <= 1)
