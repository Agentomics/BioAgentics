"""Tests for CMTF integration pipeline."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_cmtf import CMTFIntegration, CMTFModel


@pytest.fixture
def synthetic_data():
    """Synthetic coupled matrices with known shared structure."""
    rng = np.random.default_rng(42)
    n_samples, n_species, n_metab = 30, 15, 20
    n_components = 3

    # True factors and loadings
    F_true = rng.standard_normal((n_samples, n_components))
    A_true = rng.standard_normal((n_species, n_components))
    B_true = rng.standard_normal((n_metab, n_components))

    # Generate coupled matrices with noise
    X_species = F_true @ A_true.T + rng.normal(0, 0.1, (n_samples, n_species))
    X_metab = F_true @ B_true.T + rng.normal(0, 0.1, (n_samples, n_metab))

    species_df = pd.DataFrame(
        X_species,
        index=[f"P{i}" for i in range(n_samples)],
        columns=[f"species_{i}" for i in range(n_species)],
    )
    metab_df = pd.DataFrame(
        X_metab,
        index=[f"P{i}" for i in range(n_samples)],
        columns=[f"metab_{i}" for i in range(n_metab)],
    )
    return species_df, metab_df, X_species, X_metab


# ── CMTFModel ──


def test_cmtf_model_fit(synthetic_data):
    _, _, X_sp, X_mb = synthetic_data
    model = CMTFModel(n_components=3, max_iter=100)
    model.fit(X_sp, X_mb)

    assert model.F_ is not None
    assert model.A_ is not None
    assert model.B_ is not None
    assert model.F_.shape == (30, 3)
    assert model.A_.shape == (15, 3)
    assert model.B_.shape == (20, 3)


def test_cmtf_model_converges(synthetic_data):
    _, _, X_sp, X_mb = synthetic_data
    model = CMTFModel(n_components=3, max_iter=200)
    model.fit(X_sp, X_mb)

    # Loss should decrease
    assert len(model.loss_history_) > 1
    assert model.loss_history_[-1] <= model.loss_history_[0]


def test_cmtf_variance_explained(synthetic_data):
    _, _, X_sp, X_mb = synthetic_data
    model = CMTFModel(n_components=3, max_iter=200)
    model.fit(X_sp, X_mb)

    var_exp = model.variance_explained(X_sp, X_mb)
    # With low noise, should explain most variance
    assert var_exp["species_total"] > 0.5
    assert var_exp["metabolomics_total"] > 0.5
    assert "per_component" in var_exp


def test_cmtf_model_sample_mismatch():
    X1 = np.random.randn(10, 5)
    X2 = np.random.randn(15, 8)  # Different sample count
    model = CMTFModel(n_components=3)
    with pytest.raises(AssertionError):
        model.fit(X1, X2)


def test_cmtf_transform_before_fit():
    model = CMTFModel(n_components=3)
    with pytest.raises(ValueError):
        model.transform()


# ── CMTFIntegration ──


def test_integration_fit(synthetic_data, tmp_path):
    species_df, metab_df, _, _ = synthetic_data
    integration = CMTFIntegration(n_components=3)
    results = integration.fit(species_df, metab_df, output_dir=tmp_path)

    assert "factors" in results
    assert "species_loadings" in results
    assert "metabolite_loadings" in results
    assert "variance_explained" in results
    assert results["factors"].shape == (30, 3)
    assert results["species_loadings"].shape == (15, 3)
    assert results["metabolite_loadings"].shape == (20, 3)


def test_integration_saves_outputs(synthetic_data, tmp_path):
    species_df, metab_df, _, _ = synthetic_data
    integration = CMTFIntegration(n_components=3)
    integration.fit(species_df, metab_df, output_dir=tmp_path)

    assert (tmp_path / "cmtf_factors.csv").exists()
    assert (tmp_path / "cmtf_species_loadings.csv").exists()
    assert (tmp_path / "cmtf_metabolite_loadings.csv").exists()


def test_integration_top_features(synthetic_data, tmp_path):
    species_df, metab_df, _, _ = synthetic_data
    integration = CMTFIntegration(n_components=3)
    results = integration.fit(species_df, metab_df, output_dir=tmp_path)

    assert 0 in results["top_species_per_factor"]
    assert 0 in results["top_metabolites_per_factor"]
    assert len(results["top_species_per_factor"][0]) <= 10


def test_integration_alpha_weighting(synthetic_data, tmp_path):
    species_df, metab_df, _, _ = synthetic_data
    # Higher alpha = more weight on species
    int1 = CMTFIntegration(n_components=3, alpha=0.9)
    r1 = int1.fit(species_df, metab_df, output_dir=tmp_path)
    # Lower alpha = more weight on metabolites
    int2 = CMTFIntegration(n_components=3, alpha=0.1)
    r2 = int2.fit(species_df, metab_df, output_dir=tmp_path)

    # Different alpha should give different factor matrices
    assert not np.allclose(r1["factors"].values, r2["factors"].values)


def test_map_to_metabolic_axes(synthetic_data, tmp_path):
    species_df, metab_df, _, _ = synthetic_data
    # Rename some features to match axis markers
    species_df = species_df.rename(columns={"species_0": "Bifidobacterium_longum"})
    metab_df = metab_df.rename(columns={"metab_0": "tryptophan"})

    integration = CMTFIntegration(n_components=3)
    results = integration.fit(species_df, metab_df, output_dir=tmp_path)

    axes = integration.map_to_metabolic_axes(
        results["species_loadings"], results["metabolite_loadings"]
    )
    # Should return a dict mapping factor indices to axis names
    assert isinstance(axes, dict)
    assert all(isinstance(v, list) for v in axes.values())
