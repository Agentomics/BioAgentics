"""Tests for the Genomic SEM pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.pipelines.genomic_sem.pipeline import (
    FactorModel,
    GenomicSEMResult,
    compute_residual_gwas,
    fit_confirmatory_factor_model,
    fit_genomic_sem,
)

RNG = np.random.default_rng(99)

# Small set of traits for testing
TEST_TRAITS = ["TS", "OCD", "ADHD", "ASD", "MDD", "schizophrenia"]


def _make_synthetic_cov_matrix(traits: list[str], n_factors: int = 2) -> np.ndarray:
    """Generate a synthetic positive-definite genetic covariance matrix.

    Simulates a factor structure with some noise.
    """
    p = len(traits)
    # Random factor loadings
    L = RNG.normal(0.3, 0.15, size=(p, n_factors))
    # Factor correlation
    Phi = np.eye(n_factors)
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            Phi[i, j] = Phi[j, i] = 0.2
    # Residual variance
    Theta = np.diag(RNG.uniform(0.1, 0.5, size=p))
    S = L @ Phi @ L.T + Theta
    # Ensure symmetry and positive definiteness
    S = (S + S.T) / 2
    eigvals = np.linalg.eigvalsh(S)
    if eigvals.min() < 0:
        S += np.eye(p) * (abs(eigvals.min()) + 0.01)
    return S


@pytest.fixture()
def cov_matrix():
    return _make_synthetic_cov_matrix(TEST_TRAITS)


@pytest.fixture()
def simple_model():
    return FactorModel(
        name="test_model",
        factor_structure={
            "compulsive": ["OCD", "TS"],
            "neurodevelopmental": ["ADHD", "ASD", "TS"],
            "internalizing": ["MDD"],
            "psychotic": ["schizophrenia"],
        },
    )


# ---------------------------------------------------------------------------
# Tests: FactorModel
# ---------------------------------------------------------------------------


class TestFactorModel:
    def test_traits(self, simple_model):
        traits = simple_model.traits
        assert "TS" in traits
        assert "OCD" in traits

    def test_factors(self, simple_model):
        factors = simple_model.factors
        assert "compulsive" in factors
        assert "neurodevelopmental" in factors

    def test_loading_template(self, simple_model):
        template = simple_model.loading_matrix_template()
        assert template.loc["TS", "compulsive"] == 1
        assert template.loc["TS", "neurodevelopmental"] == 1
        assert template.loc["MDD", "compulsive"] == 0
        assert template.loc["OCD", "compulsive"] == 1

    def test_loading_template_shape(self, simple_model):
        template = simple_model.loading_matrix_template()
        assert template.shape == (len(simple_model.traits), len(simple_model.factors))


# ---------------------------------------------------------------------------
# Tests: fit_confirmatory_factor_model
# ---------------------------------------------------------------------------


class TestFitCFA:
    def test_returns_result(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        assert isinstance(result, GenomicSEMResult)
        assert result.model_name == "test_model"

    def test_loadings_shape(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        if result.converged:
            assert result.loadings.shape[0] > 0
            assert result.loadings.shape[1] == len(simple_model.factors)

    def test_fit_indices_present(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        assert "AIC" in result.model_fit
        assert "BIC" in result.model_fit
        assert "CFI" in result.model_fit
        assert "RMSEA" in result.model_fit

    def test_residual_variances_positive(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        if result.converged:
            assert (result.residual_variances > 0).all()

    def test_few_shared_traits(self, cov_matrix):
        model = FactorModel(
            name="tiny",
            factor_structure={"f1": ["UNKNOWN1", "UNKNOWN2"]},
        )
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, model, n_obs=500
        )
        assert not result.converged


# ---------------------------------------------------------------------------
# Tests: compute_residual_gwas
# ---------------------------------------------------------------------------


class TestResidualGWAS:
    def test_decomposition(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        if result.converged:
            decomp = compute_residual_gwas(
                cov_matrix, result.loadings, result.factor_correlations, TEST_TRAITS, "TS"
            )
            assert "total_h2" in decomp
            assert "residual_h2" in decomp
            assert "prop_residual" in decomp
            assert 0 <= decomp["prop_residual"] <= 1.5  # allow some estimation noise

    def test_raises_missing_trait(self, cov_matrix, simple_model):
        result = fit_confirmatory_factor_model(
            cov_matrix, None, TEST_TRAITS, simple_model, n_obs=500
        )
        if result.converged:
            with pytest.raises(ValueError, match="NONEXISTENT"):
                compute_residual_gwas(
                    cov_matrix, result.loadings, result.factor_correlations, TEST_TRAITS, "NONEXISTENT"
                )


# ---------------------------------------------------------------------------
# Tests: fit_genomic_sem (multiple models)
# ---------------------------------------------------------------------------


class TestFitGenomicSEM:
    def test_default_models(self):
        traits = [
            "TS", "OCD", "ADHD", "ASD", "MDD", "anxiety", "schizophrenia",
            "bipolar", "anorexia", "PTSD", "alcohol_use_disorder", "cannabis_use_disorder",
        ]
        S = _make_synthetic_cov_matrix(traits, n_factors=3)
        results = fit_genomic_sem(S, None, traits, n_obs=500)
        assert len(results) == 3
        # Results should be sorted by AIC
        aics = [r.model_fit.get("AIC", float("inf")) for r in results]
        assert aics == sorted(aics)

    def test_custom_models(self, cov_matrix):
        models = [
            FactorModel("m1", {"f1": ["TS", "OCD"], "f2": ["ADHD", "ASD"]}),
            FactorModel("m2", {"f1": ["TS", "OCD", "ADHD"], "f2": ["ASD", "MDD"]}),
        ]
        results = fit_genomic_sem(cov_matrix, None, TEST_TRAITS, models=models, n_obs=500)
        assert len(results) == 2


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_with_synthetic_data(self, cov_matrix, tmp_path):
        output_dir = tmp_path / "output"

        # Write S matrix
        s_path = tmp_path / "s_matrix.tsv"
        S_df = pd.DataFrame(cov_matrix, index=TEST_TRAITS, columns=TEST_TRAITS)
        S_df.to_csv(s_path, sep="\t")

        from bioagentics.pipelines.genomic_sem.pipeline import main

        main([
            "--s-matrix", str(s_path),
            "--output-dir", str(output_dir),
            "--n-obs", "500",
        ])

        assert (output_dir / "model_comparison.tsv").exists()
        fit_df = pd.read_csv(output_dir / "model_comparison.tsv", sep="\t")
        assert len(fit_df) == 3
        assert "AIC" in fit_df.columns
