"""Tests for RDA baseline and sPLS-Regression integration."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_baseline_integration import (
    RDABaseline,
    SPLSRegression,
    cross_validate_pls_components,
)


@pytest.fixture
def coupled_data():
    """Synthetic coupled species-metabolomics data."""
    rng = np.random.default_rng(42)
    n = 30
    # Shared signal
    F = rng.standard_normal((n, 3))
    species = pd.DataFrame(
        F @ rng.standard_normal((3, 10)) + rng.normal(0, 0.2, (n, 10)),
        index=[f"P{i}" for i in range(n)],
        columns=[f"sp_{i}" for i in range(10)],
    )
    metabolomics = pd.DataFrame(
        F @ rng.standard_normal((3, 15)) + rng.normal(0, 0.2, (n, 15)),
        index=[f"P{i}" for i in range(n)],
        columns=[f"mb_{i}" for i in range(15)],
    )
    return species, metabolomics


# ── RDA ──


def test_rda_returns_both_directions(coupled_data):
    species, metabolomics = coupled_data
    rda = RDABaseline()
    results = rda.fit(species, metabolomics)

    assert "species_explains_metabolomics" in results
    assert "metabolomics_explains_species" in results


def test_rda_variance_explained_range(coupled_data):
    species, metabolomics = coupled_data
    rda = RDABaseline()
    results = rda.fit(species, metabolomics)

    for direction in results.values():
        r2 = direction["variance_explained"]
        assert 0 <= r2 <= 1.0


def test_rda_high_r2_with_shared_signal(coupled_data):
    species, metabolomics = coupled_data
    rda = RDABaseline()
    results = rda.fit(species, metabolomics)

    # With strong shared signal, RDA should capture meaningful variance
    r2 = results["species_explains_metabolomics"]["variance_explained"]
    assert r2 > 0.3


def test_rda_saves_results(coupled_data, tmp_path):
    species, metabolomics = coupled_data
    rda = RDABaseline()
    rda.fit(species, metabolomics, output_dir=tmp_path)
    assert (tmp_path / "rda_baseline.csv").exists()


# ── sPLS ──


def test_spls_returns_all_outputs(coupled_data):
    species, metabolomics = coupled_data
    spls = SPLSRegression(n_components=3)
    results = spls.fit(species, metabolomics)

    assert "x_scores" in results
    assert "y_scores" in results
    assert "x_loadings" in results
    assert "y_loadings" in results
    assert "top_pairs" in results
    assert "variance_explained" in results


def test_spls_scores_shape(coupled_data):
    species, metabolomics = coupled_data
    spls = SPLSRegression(n_components=3)
    results = spls.fit(species, metabolomics)

    assert results["x_scores"].shape == (30, 3)
    assert results["y_scores"].shape == (30, 3)


def test_spls_sparsity(coupled_data):
    species, metabolomics = coupled_data
    spls = SPLSRegression(n_components=3, sparsity_quantile=0.9)
    results = spls.fit(species, metabolomics)

    # With 0.9 quantile, ~90% of loadings should be zeroed out
    x_load = results["x_loadings"]
    zero_frac = (x_load == 0).mean().mean()
    assert zero_frac > 0.5  # At least half should be zero


def test_spls_top_pairs(coupled_data):
    species, metabolomics = coupled_data
    spls = SPLSRegression(n_components=3)
    results = spls.fit(species, metabolomics)

    pairs = results["top_pairs"]
    assert len(pairs) > 0
    assert len(pairs[0]) == 4  # (species, metabolite, component, score)
    # Scores should be positive
    assert all(p[3] > 0 for p in pairs)


def test_spls_saves_files(coupled_data, tmp_path):
    species, metabolomics = coupled_data
    spls = SPLSRegression(n_components=3)
    spls.fit(species, metabolomics, output_dir=tmp_path)

    assert (tmp_path / "spls_species_loadings.csv").exists()
    assert (tmp_path / "spls_metabolite_loadings.csv").exists()
    assert (tmp_path / "spls_top_pairs.csv").exists()


# ── Cross-validation ──


def test_cv_pls_components(coupled_data):
    species, metabolomics = coupled_data
    results = cross_validate_pls_components(
        species, metabolomics, component_range=range(2, 6), n_folds=3
    )
    assert "n_components" in results.columns
    assert "mean_r2" in results.columns
    assert len(results) > 0
