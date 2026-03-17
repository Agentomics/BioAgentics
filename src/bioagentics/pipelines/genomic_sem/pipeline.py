"""Genomic SEM pipeline for cross-disorder factor analysis.

Implements confirmatory factor analysis on genetic covariance matrices
to replicate and extend the 5-factor model from Grotzinger et al. (Nature 2025).
Tests whether TS loads on multiple factors (compulsive + neurodevelopmental)
and extracts TS-specific genetic variance via residual GWAS.

This is a pure-Python implementation of the core Genomic SEM factor analysis
(the R GenomicSEM package wraps lavaan for CFA — we replicate the essential
maximum-likelihood factor analysis using scipy.optimize). This allows the
pipeline to be self-contained and testable without R dependencies.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

# Default 5-factor model structure from Nature 2025
# Factor -> list of disorders that load on it
DEFAULT_FACTOR_STRUCTURE = {
    "compulsive": ["OCD", "anorexia", "TS"],
    "psychotic": ["schizophrenia", "bipolar"],
    "neurodevelopmental": ["ADHD", "ASD", "TS"],
    "internalizing": ["MDD", "anxiety", "PTSD"],
    "externalizing": ["alcohol_use_disorder", "cannabis_use_disorder", "opioid_use_disorder"],
}


@dataclass
class FactorModel:
    """Specification of a confirmatory factor model."""

    name: str
    factor_structure: dict[str, list[str]]

    @property
    def factors(self) -> list[str]:
        return sorted(self.factor_structure.keys())

    @property
    def traits(self) -> list[str]:
        all_traits: set[str] = set()
        for traits in self.factor_structure.values():
            all_traits.update(traits)
        return sorted(all_traits)

    def loading_matrix_template(self) -> pd.DataFrame:
        """Create a binary template showing which loadings are free parameters."""
        traits = self.traits
        factors = self.factors
        template = pd.DataFrame(0, index=traits, columns=factors)
        for factor, factor_traits in self.factor_structure.items():
            for t in factor_traits:
                if t in traits:
                    template.loc[t, factor] = 1
        return template


@dataclass
class GenomicSEMResult:
    """Result of a Genomic SEM factor analysis."""

    model_name: str
    loadings: pd.DataFrame  # traits x factors
    factor_correlations: pd.DataFrame  # factors x factors
    residual_variances: pd.Series  # per-trait
    model_fit: dict[str, float] = field(default_factory=dict)
    n_params: int = 0
    converged: bool = False


def _vec_to_params(
    x: np.ndarray,
    template: np.ndarray,
    n_traits: int,
    n_factors: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unpack parameter vector into loading matrix, factor correlations, residual vars."""
    n_loadings = int(template.sum())
    loadings_flat = x[:n_loadings]
    n_fcorr = n_factors * (n_factors - 1) // 2
    fcorr_flat = x[n_loadings : n_loadings + n_fcorr]
    resid_flat = x[n_loadings + n_fcorr :]

    # Fill loading matrix
    L = np.zeros((n_traits, n_factors))
    idx = 0
    for i in range(n_traits):
        for j in range(n_factors):
            if template[i, j] > 0:
                L[i, j] = loadings_flat[idx]
                idx += 1

    # Factor correlation matrix (symmetric, diag=1)
    Phi = np.eye(n_factors)
    idx = 0
    for i in range(n_factors):
        for j in range(i + 1, n_factors):
            Phi[i, j] = fcorr_flat[idx]
            Phi[j, i] = fcorr_flat[idx]
            idx += 1

    # Residual variances (must be positive)
    Theta = np.exp(resid_flat)  # log-transform to enforce positivity

    return L, Phi, Theta


def _model_implied_cov(
    L: np.ndarray, Phi: np.ndarray, Theta: np.ndarray
) -> np.ndarray:
    """Compute model-implied covariance: Sigma = L @ Phi @ L' + diag(Theta)."""
    return L @ Phi @ L.T + np.diag(Theta)


def _ml_objective(
    x: np.ndarray,
    S: np.ndarray,
    template: np.ndarray,
    n_traits: int,
    n_factors: int,
) -> float:
    """Maximum likelihood discrepancy function for CFA.

    F_ML = log|Sigma| + tr(S @ Sigma^{-1}) - log|S| - p
    """
    L, Phi, Theta = _vec_to_params(x, template, n_traits, n_factors)
    Sigma = _model_implied_cov(L, Phi, Theta)

    try:
        sign, logdet_sigma = np.linalg.slogdet(Sigma)
        if sign <= 0:
            return 1e10
        sign_s, logdet_s = np.linalg.slogdet(S)
        if sign_s <= 0:
            return 1e10
        Sigma_inv = np.linalg.inv(Sigma)
    except np.linalg.LinAlgError:
        return 1e10

    F = logdet_sigma + np.trace(S @ Sigma_inv) - logdet_s - n_traits
    return float(max(F, 0))


def fit_confirmatory_factor_model(
    S: np.ndarray,
    V: np.ndarray | None,
    trait_names: list[str],
    model: FactorModel,
    n_obs: int = 1000,
) -> GenomicSEMResult:
    """Fit a confirmatory factor model to a genetic covariance matrix.

    Parameters
    ----------
    S : np.ndarray
        Genetic covariance matrix (p x p), from LDSC.
    V : np.ndarray | None
        Sampling covariance matrix of S (for weighted estimation). Can be None.
    trait_names : list[str]
        Labels for rows/columns of S.
    model : FactorModel
        Factor model specification.
    n_obs : int
        Effective sample size (for fit indices).

    Returns
    -------
    GenomicSEMResult with estimated loadings, fit statistics, etc.
    """
    template_df = model.loading_matrix_template()

    # Align traits between S matrix and model
    shared_traits = [t for t in trait_names if t in template_df.index]
    if len(shared_traits) < 3:
        logger.warning(
            "Only %d shared traits between S matrix and model %s",
            len(shared_traits),
            model.name,
        )
        return GenomicSEMResult(
            model_name=model.name,
            loadings=pd.DataFrame(),
            factor_correlations=pd.DataFrame(),
            residual_variances=pd.Series(dtype=float),
            converged=False,
        )

    trait_idx = [trait_names.index(t) for t in shared_traits]
    S_sub = S[np.ix_(trait_idx, trait_idx)]
    template_sub = template_df.loc[shared_traits].values
    factors = template_df.columns.tolist()

    n_traits = len(shared_traits)
    n_factors = len(factors)
    n_loadings = int(template_sub.sum())
    n_fcorr = n_factors * (n_factors - 1) // 2
    n_resid = n_traits
    n_params = n_loadings + n_fcorr + n_resid

    # Initial parameter values
    x0 = np.concatenate([
        np.full(n_loadings, 0.5),       # loadings
        np.full(n_fcorr, 0.1),          # factor correlations
        np.full(n_resid, np.log(0.5)),  # log(residual variances)
    ])

    # Optimize
    result = optimize.minimize(
        _ml_objective,
        x0,
        args=(S_sub, template_sub, n_traits, n_factors),
        method="L-BFGS-B",
        options={"maxiter": 5000, "ftol": 1e-10},
    )

    L, Phi, Theta = _vec_to_params(result.x, template_sub, n_traits, n_factors)

    # Compute fit indices
    F_min = result.fun
    df_model = n_traits * (n_traits + 1) // 2 - n_params
    chi2 = max((n_obs - 1) * F_min, 0)
    chi2_p = float(stats.chi2.sf(chi2, max(df_model, 1))) if df_model > 0 else np.nan

    # AIC = chi2 - 2*df
    aic = chi2 - 2 * df_model if df_model > 0 else np.nan
    # BIC = chi2 - df*ln(n)
    bic = chi2 - df_model * np.log(n_obs) if df_model > 0 else np.nan

    # CFI = 1 - max(chi2_model - df_model, 0) / max(chi2_null - df_null, 0)
    # Null model: diagonal (no factors)
    df_null = n_traits * (n_traits - 1) // 2
    diag_s = np.diag(np.diag(S_sub))
    try:
        sign, logdet_null = np.linalg.slogdet(diag_s)
        sign_s, logdet_s = np.linalg.slogdet(S_sub)
        F_null = logdet_null + np.trace(S_sub @ np.linalg.inv(diag_s)) - logdet_s - n_traits
        chi2_null = max((n_obs - 1) * F_null, 0)
    except np.linalg.LinAlgError:
        chi2_null = chi2 * 10

    cfi = 1 - max(chi2 - df_model, 0) / max(chi2_null - df_null, 1) if df_null > 0 else np.nan

    # RMSEA = sqrt(max((chi2/df - 1)/(n-1), 0))
    rmsea = np.sqrt(max((chi2 / max(df_model, 1) - 1) / max(n_obs - 1, 1), 0))

    loadings_df = pd.DataFrame(L, index=shared_traits, columns=factors)
    phi_df = pd.DataFrame(Phi, index=factors, columns=factors)
    resid_s = pd.Series(Theta, index=shared_traits)

    return GenomicSEMResult(
        model_name=model.name,
        loadings=loadings_df,
        factor_correlations=phi_df,
        residual_variances=resid_s,
        model_fit={
            "chi2": chi2,
            "df": df_model,
            "chi2_p": chi2_p,
            "AIC": aic,
            "BIC": bic,
            "CFI": cfi,
            "RMSEA": rmsea,
            "F_min": F_min,
        },
        n_params=n_params,
        converged=result.success,
    )


def compute_residual_gwas(
    S: np.ndarray,
    loadings: pd.DataFrame,
    factor_correlations: pd.DataFrame,
    trait_names: list[str],
    target_trait: str = "TS",
) -> dict[str, float]:
    """Compute residual genetic variance for a target trait.

    Decomposes the genetic variance of the target trait into:
    - Variance explained by each factor
    - Residual (TS-specific) variance

    Returns dict with factor contributions and residual proportion.
    """
    if target_trait not in loadings.index:
        raise ValueError(f"{target_trait} not in loadings matrix")

    trait_idx = trait_names.index(target_trait)
    total_var = S[trait_idx, trait_idx]

    lam = loadings.loc[target_trait].values
    Phi = factor_correlations.values

    # Variance from factors: lam' @ Phi @ lam
    factor_var = float(lam @ Phi @ lam)
    residual_var = max(total_var - factor_var, 0)

    result = {
        "total_h2": float(total_var),
        "factor_h2": factor_var,
        "residual_h2": residual_var,
        "prop_residual": residual_var / total_var if total_var > 0 else 0,
    }

    # Per-factor decomposition
    for i, factor in enumerate(loadings.columns):
        # Variance from this factor alone: lam_i^2 * Phi_ii
        result[f"h2_{factor}"] = float(lam[i] ** 2 * Phi[i, i])

    return result


def fit_genomic_sem(
    S: np.ndarray,
    V: np.ndarray | None,
    trait_names: list[str],
    models: list[FactorModel] | None = None,
    n_obs: int = 1000,
) -> list[GenomicSEMResult]:
    """Fit multiple Genomic SEM models and compare fit.

    If no models provided, fits 3 default models:
    1. TS on compulsive factor only
    2. TS on compulsive + neurodevelopmental
    3. Full 5-factor (TS on all)

    Parameters
    ----------
    S : np.ndarray
        Genetic covariance matrix.
    V : np.ndarray | None
        Sampling covariance of S (optional).
    trait_names : list[str]
        Trait labels.
    models : list[FactorModel] | None
        Models to compare. If None, uses defaults.
    n_obs : int
        Effective sample size for fit indices.

    Returns
    -------
    List of GenomicSEMResult, sorted by AIC (best first).
    """
    if models is None:
        models = [
            FactorModel(
                name="TS_compulsive_only",
                factor_structure={
                    "compulsive": ["OCD", "anorexia", "TS"],
                    "psychotic": ["schizophrenia", "bipolar"],
                    "neurodevelopmental": ["ADHD", "ASD"],
                    "internalizing": ["MDD", "anxiety", "PTSD"],
                    "externalizing": ["alcohol_use_disorder", "cannabis_use_disorder"],
                },
            ),
            FactorModel(
                name="TS_compulsive_neurodev",
                factor_structure={
                    "compulsive": ["OCD", "anorexia", "TS"],
                    "psychotic": ["schizophrenia", "bipolar"],
                    "neurodevelopmental": ["ADHD", "ASD", "TS"],
                    "internalizing": ["MDD", "anxiety", "PTSD"],
                    "externalizing": ["alcohol_use_disorder", "cannabis_use_disorder"],
                },
            ),
            FactorModel(
                name="TS_all_factors",
                factor_structure={
                    "compulsive": ["OCD", "anorexia", "TS"],
                    "psychotic": ["schizophrenia", "bipolar", "TS"],
                    "neurodevelopmental": ["ADHD", "ASD", "TS"],
                    "internalizing": ["MDD", "anxiety", "PTSD", "TS"],
                    "externalizing": ["alcohol_use_disorder", "cannabis_use_disorder", "TS"],
                },
            ),
        ]

    results = []
    for model in models:
        logger.info("Fitting model: %s", model.name)
        result = fit_confirmatory_factor_model(S, V, trait_names, model, n_obs)
        results.append(result)

        if result.converged:
            logger.info(
                "  %s: AIC=%.2f, BIC=%.2f, CFI=%.4f, RMSEA=%.4f",
                model.name,
                result.model_fit.get("AIC", float("nan")),
                result.model_fit.get("BIC", float("nan")),
                result.model_fit.get("CFI", float("nan")),
                result.model_fit.get("RMSEA", float("nan")),
            )
        else:
            logger.warning("  %s: did not converge", model.name)

    # Sort by AIC (lower is better)
    results.sort(key=lambda r: r.model_fit.get("AIC", float("inf")))
    return results


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the Genomic SEM pipeline."""
    parser = argparse.ArgumentParser(
        description="Fit Genomic SEM factor models to genetic covariance matrix."
    )
    parser.add_argument(
        "--s-matrix",
        type=Path,
        required=True,
        help="Path to genetic covariance matrix (S matrix) TSV file.",
    )
    parser.add_argument(
        "--v-matrix",
        type=Path,
        default=None,
        help="Path to sampling covariance matrix (V matrix) TSV file (optional).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "output" / "tourettes" / "ts-comorbidity-genetic-architecture" / "phase2",
        help="Output directory.",
    )
    parser.add_argument(
        "--n-obs",
        type=int,
        default=1000,
        help="Effective sample size for fit indices (default: 1000).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not args.s_matrix.exists():
        logger.error("S matrix file not found: %s", args.s_matrix)
        sys.exit(1)

    # Load matrices
    S_df = pd.read_csv(args.s_matrix, sep="\t", index_col=0)
    trait_names = list(S_df.index)
    S = S_df.values

    V = None
    if args.v_matrix and args.v_matrix.exists():
        V = pd.read_csv(args.v_matrix, sep="\t", index_col=0).values

    # Fit models
    results = fit_genomic_sem(S, V, trait_names, n_obs=args.n_obs)

    # Write output
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model comparison table
    fit_rows = []
    for r in results:
        row = {"model": r.model_name, "converged": r.converged, "n_params": r.n_params}
        row.update(r.model_fit)
        fit_rows.append(row)
    fit_df = pd.DataFrame(fit_rows)
    fit_df.to_csv(output_dir / "model_comparison.tsv", sep="\t", index=False, float_format="%.6f")
    logger.info("Wrote model comparison to %s", output_dir / "model_comparison.tsv")

    # Best model loadings
    best = results[0]
    if best.converged and not best.loadings.empty:
        best.loadings.to_csv(output_dir / "factor_loadings.tsv", sep="\t", float_format="%.6f")
        best.factor_correlations.to_csv(output_dir / "factor_correlations.tsv", sep="\t", float_format="%.6f")
        best.residual_variances.to_csv(output_dir / "residual_variances.tsv", sep="\t", header=["residual_var"])

        # Residual GWAS decomposition for TS
        if "TS" in best.loadings.index:
            decomp = compute_residual_gwas(S, best.loadings, best.factor_correlations, trait_names, "TS")
            decomp_df = pd.DataFrame([decomp])
            decomp_df.to_csv(output_dir / "ts_variance_decomposition.tsv", sep="\t", index=False, float_format="%.6f")
            logger.info("TS variance decomposition: %.1f%% residual", decomp["prop_residual"] * 100)

    logger.info("Best model: %s (AIC=%.2f)", best.model_name, best.model_fit.get("AIC", float("nan")))


if __name__ == "__main__":
    main()
