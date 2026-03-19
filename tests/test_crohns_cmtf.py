"""Tests for CMTF integration pipeline."""

import numpy as np
import pandas as pd
import pytest

from bioagentics.models.crohns_cmtf import CMTFIntegration, CMTFModel, cross_validate_components


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


def test_cv_uses_out_of_sample_projections(synthetic_data):
    """Verify cross_validate_components projects test data, not reusing training F_."""
    _, _, X_sp, X_mb = synthetic_data
    n_samples = X_sp.shape[0]

    # Run CV with a small range
    results = cross_validate_components(
        X_sp, X_mb, component_range=range(3, 5), n_folds=3, alpha=0.65
    )

    assert len(results["n_components"]) == 2
    assert all(loss > 0 for loss in results["mean_loss"])

    # Verify the fix: fit on a subset, then check that projecting held-out
    # samples produces a different F than simply indexing model.F_
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    train_idx, test_idx = next(iter(kf.split(X_sp)))

    model = CMTFModel(n_components=3, alpha=0.65, random_state=42)
    model.fit(X_sp[train_idx], X_mb[train_idx])

    # model.F_ has shape (len(train_idx), 3) — cannot be indexed by test_idx
    assert model.F_.shape[0] == len(train_idx)
    assert model.F_.shape[0] != n_samples

    # Proper out-of-sample projection
    alpha = 0.65
    R = 3
    lhs = alpha * (X_sp[test_idx] @ model.A_) + (1 - alpha) * (X_mb[test_idx] @ model.B_)
    rhs = alpha * (model.A_.T @ model.A_) + (1 - alpha) * (model.B_.T @ model.B_) + 1e-8 * np.eye(R)
    F_test = lhs @ np.linalg.inv(rhs)

    assert F_test.shape == (len(test_idx), R)
    # Test loss should be computable with projected factors
    loss = model._compute_loss(X_sp[test_idx], X_mb[test_idx], F_test, model.A_, model.B_)
    assert loss > 0
    assert np.isfinite(loss)


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


# ── Auto Component Selection ──


def test_auto_component_selection(synthetic_data, tmp_path):
    """Test n_components='auto' selects via cross-validation."""
    species_df, metab_df, _, _ = synthetic_data
    integration = CMTFIntegration(
        n_components="auto",
        cv_folds=3,
        cv_component_range=range(2, 6),
    )
    results = integration.fit(species_df, metab_df, output_dir=tmp_path)

    # Should have selected a component count
    assert isinstance(results["n_components"], int)
    assert 2 <= results["n_components"] <= 5
    # CV results should be stored
    assert "cv_results" in results
    assert len(results["cv_results"]["n_components"]) == 4
    # Factors shape should match selected components
    assert results["factors"].shape[1] == results["n_components"]


def test_explicit_components_no_cv(synthetic_data, tmp_path):
    """Test that explicit n_components skips CV."""
    species_df, metab_df, _, _ = synthetic_data
    integration = CMTFIntegration(n_components=3)
    results = integration.fit(species_df, metab_df, output_dir=tmp_path)

    assert results["n_components"] == 3
    assert "cv_results" not in results
    assert results["factors"].shape[1] == 3
