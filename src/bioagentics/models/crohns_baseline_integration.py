"""RDA baseline and sPLS-Regression for microbiome-metabolome integration.

Provides:
1. RDA (Redundancy Analysis): Most consistent method in the 19-method
   benchmark (avg 52% explained variance). Baseline comparator for CMTF/MOFA2.
2. sPLS-Regression: Multivariate feature selection exploiting inter-omics
   correlations. Identifies species-metabolite pairs driving integration factors.

Usage::

    from bioagentics.models.crohns_baseline_integration import (
        RDABaseline,
        SPLSRegression,
    )

    rda = RDABaseline()
    rda_results = rda.fit(species_qc, metabolomics_qc)

    spls = SPLSRegression(n_components=5)
    spls_results = spls.fit(species_qc, metabolomics_qc)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "microbiome-metabolome-subtyping"


# ── Redundancy Analysis (RDA) ──


class RDABaseline:
    """Redundancy Analysis for inter-omic variance explanation.

    RDA measures how much variance in one omic layer (response) is
    explained by another (predictor). Implemented as constrained PCA
    via fitted values from multivariate regression.
    """

    def __init__(self) -> None:
        self.results_: dict[str, dict] | None = None

    def _rda(
        self,
        X: np.ndarray,
        Y: np.ndarray,
    ) -> dict[str, float]:
        """Compute RDA: variance in Y explained by X.

        Uses linear regression projection: Y_hat = X @ (X^T X)^{-1} X^T @ Y
        Then computes explained variance ratio.
        """
        n = X.shape[0]

        # Center both matrices
        X_c = X - X.mean(axis=0)
        Y_c = Y - Y.mean(axis=0)

        # Projection matrix: H = X(X^TX)^{-1}X^T
        try:
            reg = 1e-8 * np.eye(X_c.shape[1])
            beta = np.linalg.solve(X_c.T @ X_c + reg, X_c.T @ Y_c)
            Y_hat = X_c @ beta
        except np.linalg.LinAlgError:
            logger.warning("RDA: singular matrix, returning 0 variance explained")
            return {"variance_explained": 0.0, "adjusted_r2": 0.0}

        # Total variance in Y
        ss_total = np.sum(Y_c**2)
        ss_explained = np.sum(Y_hat**2)

        r2 = ss_explained / ss_total if ss_total > 0 else 0.0

        # Adjusted R²
        p = X_c.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

        return {
            "variance_explained": float(r2),
            "adjusted_r2": float(adj_r2),
            "n_samples": n,
            "n_predictors": p,
        }

    def fit(
        self,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> dict[str, dict]:
        """Run RDA in both directions.

        Returns dict with:
        - "species_explains_metabolomics": RDA(metabolomics ~ species)
        - "metabolomics_explains_species": RDA(species ~ metabolomics)
        """
        output_dir = output_dir or OUTPUT_DIR
        shared = species.index.intersection(metabolomics.index)
        sp = species.loc[shared].values.astype(float)
        mb = metabolomics.loc[shared].values.astype(float)

        # RDA both directions
        sp_to_mb = self._rda(sp, mb)
        mb_to_sp = self._rda(mb, sp)

        self.results_ = {
            "species_explains_metabolomics": sp_to_mb,
            "metabolomics_explains_species": mb_to_sp,
        }

        logger.info(
            "RDA: species → metabolomics R²=%.3f, metabolomics → species R²=%.3f",
            sp_to_mb["variance_explained"],
            mb_to_sp["variance_explained"],
        )

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        results_df = pd.DataFrame(self.results_).T
        results_df.to_csv(output_dir / "rda_baseline.csv")

        return self.results_


# ── Sparse PLS-Regression ──


class SPLSRegression:
    """Sparse PLS-Regression for species-metabolite pair identification.

    Uses scikit-learn PLSRegression with post-hoc sparsity via loading
    thresholding to identify the most important species-metabolite
    associations.
    """

    def __init__(
        self,
        n_components: int = 5,
        sparsity_quantile: float = 0.9,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        n_components : int
            Number of PLS components.
        sparsity_quantile : float
            Features with loadings below this quantile are zeroed out
            (default: 0.9 = keep top 10%).
        random_state : int
            Random seed for cross-validation.
        """
        self.n_components = n_components
        self.sparsity_quantile = sparsity_quantile
        self.random_state = random_state
        self.model_: PLSRegression | None = None
        self.x_loadings_: pd.DataFrame | None = None
        self.y_loadings_: pd.DataFrame | None = None

    def fit(
        self,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame | list]:
        """Fit sPLS and extract species-metabolite associations.

        Parameters
        ----------
        species : DataFrame
            Predictor matrix (participants × species).
        metabolomics : DataFrame
            Response matrix (participants × metabolites).

        Returns
        -------
        dict with:
            - "x_scores": DataFrame (participants × components)
            - "y_scores": DataFrame (participants × components)
            - "x_loadings": DataFrame (species × components)
            - "y_loadings": DataFrame (metabolites × components)
            - "top_pairs": list of (species, metabolite, component, score) tuples
            - "variance_explained": dict
        """
        output_dir = output_dir or OUTPUT_DIR
        shared = species.index.intersection(metabolomics.index)
        species = species.loc[shared]
        metabolomics = metabolomics.loc[shared]

        # Standardize
        scaler_x = StandardScaler()
        scaler_y = StandardScaler()
        X = scaler_x.fit_transform(species.values.astype(float))
        Y = scaler_y.fit_transform(metabolomics.values.astype(float))

        # Cap components
        n_comp = min(
            self.n_components,
            X.shape[0] - 1,
            X.shape[1],
            Y.shape[1],
        )

        # Fit PLS
        self.model_ = PLSRegression(n_components=n_comp, scale=False)
        self.model_.fit(X, Y)

        x_scores = self.model_.x_scores_
        y_scores = self.model_.y_scores_
        x_loadings = self.model_.x_loadings_
        y_loadings = self.model_.y_loadings_

        comp_names = [f"PLS_{i}" for i in range(n_comp)]

        # Apply sparsity threshold
        x_load_sparse = self._apply_sparsity(x_loadings)
        y_load_sparse = self._apply_sparsity(y_loadings)

        self.x_loadings_ = pd.DataFrame(
            x_load_sparse, index=species.columns, columns=comp_names
        )
        self.y_loadings_ = pd.DataFrame(
            y_load_sparse, index=metabolomics.columns, columns=comp_names
        )

        # Identify top species-metabolite pairs per component
        top_pairs = self._extract_pairs(
            self.x_loadings_, self.y_loadings_, top_n=10
        )

        # Variance explained
        var_exp = self._variance_explained(X, Y, x_scores, y_scores)

        results = {
            "x_scores": pd.DataFrame(x_scores, index=shared, columns=comp_names),
            "y_scores": pd.DataFrame(y_scores, index=shared, columns=comp_names),
            "x_loadings": self.x_loadings_,
            "y_loadings": self.y_loadings_,
            "top_pairs": top_pairs,
            "variance_explained": var_exp,
        }

        # Save
        output_dir.mkdir(parents=True, exist_ok=True)
        self.x_loadings_.to_csv(output_dir / "spls_species_loadings.csv")
        self.y_loadings_.to_csv(output_dir / "spls_metabolite_loadings.csv")

        pairs_df = pd.DataFrame(
            top_pairs, columns=["species", "metabolite", "component", "score"]
        )
        pairs_df.to_csv(output_dir / "spls_top_pairs.csv", index=False)

        logger.info(
            "sPLS: %d components, %d top pairs, X VE=%.1f%%, Y VE=%.1f%%",
            n_comp,
            len(top_pairs),
            var_exp.get("x_total", 0) * 100,
            var_exp.get("y_total", 0) * 100,
        )

        return results

    def _apply_sparsity(self, loadings: np.ndarray) -> np.ndarray:
        """Zero out loadings below the sparsity quantile threshold."""
        result = loadings.copy()
        threshold = np.quantile(np.abs(loadings), self.sparsity_quantile)
        result[np.abs(result) < threshold] = 0.0
        return result

    def _extract_pairs(
        self,
        x_loadings: pd.DataFrame,
        y_loadings: pd.DataFrame,
        top_n: int = 10,
    ) -> list[tuple[str, str, str, float]]:
        """Extract top species-metabolite pairs per component."""
        pairs = []
        for comp in x_loadings.columns:
            x_nonzero = x_loadings[comp][x_loadings[comp] != 0]
            y_nonzero = y_loadings[comp][y_loadings[comp] != 0]

            if len(x_nonzero) == 0 or len(y_nonzero) == 0:
                continue

            # Score each pair by product of absolute loadings
            for sp in x_nonzero.index:
                for mb in y_nonzero.index:
                    score = abs(x_nonzero[sp]) * abs(y_nonzero[mb])
                    pairs.append((sp, mb, comp, float(score)))

        # Sort and take top_n per component
        pairs.sort(key=lambda x: x[3], reverse=True)
        return pairs[:top_n * len(x_loadings.columns)]

    def _variance_explained(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        x_scores: np.ndarray,
        y_scores: np.ndarray,
    ) -> dict[str, float]:
        """Compute variance explained by PLS components."""
        ss_x_total = np.sum(X**2)
        ss_y_total = np.sum(Y**2)

        # Reconstruction via scores and loadings
        x_recon = x_scores @ self.model_.x_loadings_.T
        y_recon = self.model_.predict(X)

        ss_x_resid = np.sum((X - x_recon) ** 2)
        ss_y_resid = np.sum((Y - y_recon) ** 2)

        return {
            "x_total": float(1 - ss_x_resid / ss_x_total) if ss_x_total > 0 else 0.0,
            "y_total": float(1 - ss_y_resid / ss_y_total) if ss_y_total > 0 else 0.0,
        }


def cross_validate_pls_components(
    species: pd.DataFrame,
    metabolomics: pd.DataFrame,
    component_range: range = range(2, 11),
    n_folds: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Cross-validate to find optimal number of PLS components.

    Returns DataFrame with n_components, mean_r2, std_r2.
    """
    shared = species.index.intersection(metabolomics.index)
    X = StandardScaler().fit_transform(species.loc[shared].values.astype(float))
    Y = StandardScaler().fit_transform(metabolomics.loc[shared].values.astype(float))

    results = []
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for n_comp in component_range:
        n_comp_actual = min(n_comp, X.shape[0] - 2, X.shape[1], Y.shape[1])
        if n_comp_actual < 1:
            continue

        fold_r2s = []
        for train_idx, test_idx in kf.split(X):
            pls = PLSRegression(n_components=n_comp_actual, scale=False)
            pls.fit(X[train_idx], Y[train_idx])
            Y_pred = pls.predict(X[test_idx])
            ss_res = np.sum((Y[test_idx] - Y_pred) ** 2)
            ss_tot = np.sum((Y[test_idx] - Y[test_idx].mean(axis=0)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
            fold_r2s.append(r2)

        results.append({
            "n_components": n_comp,
            "mean_r2": float(np.mean(fold_r2s)),
            "std_r2": float(np.std(fold_r2s)),
        })

    return pd.DataFrame(results)
