"""Coupled Matrix-Tensor Factorization (CMTF) for microbiome-metabolome integration.

Implements a CMTF-based approach inspired by MiMeJF (Metabolites 2025) for
joint factorization of metagenomic and metabolomic data. Identifies shared
latent factors capturing cross-omic structure in CD patients.

The CMTF model jointly decomposes:
- X_species ≈ F × A^T  (samples × species)
- X_metab   ≈ F × B^T  (samples × metabolites)

Where F (samples × R) is the shared factor matrix, A (species × R) and
B (metabolites × R) are omic-specific loading matrices.

Usage::

    from bioagentics.models.crohns_cmtf import CMTFIntegration

    cmtf = CMTFIntegration(n_components=10)
    results = cmtf.fit(species_qc, metabolomics_qc)
    factors = results["factors"]  # samples × components
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "microbiome-metabolome-subtyping"


class CMTFModel:
    """Coupled Matrix Factorization using alternating least squares.

    Decomposes two matrices sharing a sample dimension into shared
    latent factors with omic-specific loadings.
    """

    def __init__(
        self,
        n_components: int = 10,
        max_iter: int = 500,
        tol: float = 1e-6,
        alpha: float = 0.5,
        random_state: int = 42,
    ) -> None:
        """
        Parameters
        ----------
        n_components : int
            Number of latent factors (R).
        max_iter : int
            Maximum ALS iterations.
        tol : float
            Convergence tolerance (relative change in loss).
        alpha : float
            Weight balancing species (alpha) vs metabolites (1-alpha).
            0.5 = equal weight. Per importance hierarchy: bacteria 40%,
            metabolites 22% → use alpha = 40/(40+22) ≈ 0.65.
        random_state : int
            Random seed for initialization.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.alpha = alpha
        self.random_state = random_state

        self.F_: np.ndarray | None = None  # Shared factors (samples × R)
        self.A_: np.ndarray | None = None  # Species loadings (species × R)
        self.B_: np.ndarray | None = None  # Metabolite loadings (metab × R)
        self.loss_history_: list[float] = []

    def _compute_loss(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        F: np.ndarray,
        A: np.ndarray,
        B: np.ndarray,
    ) -> float:
        """Weighted reconstruction loss."""
        loss1 = np.sum((X1 - F @ A.T) ** 2)
        loss2 = np.sum((X2 - F @ B.T) ** 2)
        return self.alpha * loss1 + (1 - self.alpha) * loss2

    def fit(
        self,
        X_species: np.ndarray,
        X_metab: np.ndarray,
    ) -> CMTFModel:
        """Fit CMTF model using alternating least squares.

        Parameters
        ----------
        X_species : ndarray
            Species matrix (n_samples × n_species).
        X_metab : ndarray
            Metabolite matrix (n_samples × n_metabolites).

        Returns self.
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X_species.shape[0]
        n_species = X_species.shape[1]
        n_metab = X_metab.shape[1]
        R = self.n_components

        assert X_species.shape[0] == X_metab.shape[0], "Sample count mismatch"

        # Initialize with SVD-based initialization
        U1, S1, Vt1 = np.linalg.svd(X_species, full_matrices=False)
        U2, S2, Vt2 = np.linalg.svd(X_metab, full_matrices=False)

        r1 = min(R, len(S1))
        r2 = min(R, len(S2))

        F = np.zeros((n_samples, R))
        F[:, :r1] += U1[:, :r1] * S1[:r1]
        F[:, :r2] += U2[:, :r2] * S2[:r2]
        F /= 2  # Average initialization

        A = rng.standard_normal((n_species, R)) * 0.01
        A[:r1, :r1] = Vt1[:r1, :r1]

        B = rng.standard_normal((n_metab, R)) * 0.01
        B[:r2, :r2] = Vt2[:r2, :r2]

        self.loss_history_ = []
        prev_loss = float("inf")

        for iteration in range(self.max_iter):
            # Update A: min ||X1 - F @ A^T||^2 → A = (F^T F)^{-1} F^T X1
            FtF = F.T @ F
            reg = 1e-8 * np.eye(R)
            A = np.linalg.solve(FtF + reg, F.T @ X_species).T

            # Update B: min ||X2 - F @ B^T||^2 → B = (F^T F)^{-1} F^T X2
            B = np.linalg.solve(FtF + reg, F.T @ X_metab).T

            # Update F: min alpha||X1 - F A^T||^2 + (1-alpha)||X2 - F B^T||^2
            # F = [alpha * X1 A + (1-alpha) * X2 B] @ [alpha * A^T A + (1-alpha) * B^T B]^{-1}
            lhs = self.alpha * (X_species @ A) + (1 - self.alpha) * (X_metab @ B)
            rhs = self.alpha * (A.T @ A) + (1 - self.alpha) * (B.T @ B) + reg
            F = lhs @ np.linalg.inv(rhs)

            loss = self._compute_loss(X_species, X_metab, F, A, B)
            self.loss_history_.append(loss)

            rel_change = abs(prev_loss - loss) / (abs(prev_loss) + 1e-10)
            if rel_change < self.tol and iteration > 10:
                logger.info("CMTF converged at iteration %d (loss=%.4f)", iteration, loss)
                break
            prev_loss = loss

        else:
            logger.info("CMTF reached max_iter=%d (loss=%.4f)", self.max_iter, loss)

        self.F_ = F
        self.A_ = A
        self.B_ = B

        return self

    def transform(self) -> np.ndarray:
        """Return shared factor matrix F (samples × components)."""
        if self.F_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        return self.F_

    def variance_explained(
        self,
        X_species: np.ndarray,
        X_metab: np.ndarray,
    ) -> dict[str, float]:
        """Compute variance explained per omic layer and per component."""
        if self.F_ is None or self.A_ is None or self.B_ is None:
            raise ValueError("Model not fitted.")

        # Total variance
        total_var_species = np.sum(X_species**2)
        total_var_metab = np.sum(X_metab**2)

        # Reconstruction
        recon_species = self.F_ @ self.A_.T
        recon_metab = self.F_ @ self.B_.T

        # Residual
        resid_species = np.sum((X_species - recon_species) ** 2)
        resid_metab = np.sum((X_metab - recon_metab) ** 2)

        var_explained_species = 1 - resid_species / total_var_species
        var_explained_metab = 1 - resid_metab / total_var_metab

        # Per-component variance explained
        per_component: dict[str, list[float]] = {"species": [], "metabolomics": []}
        for r in range(self.n_components):
            # Contribution of component r
            comp_species = np.outer(self.F_[:, r], self.A_[:, r])
            comp_metab = np.outer(self.F_[:, r], self.B_[:, r])
            per_component["species"].append(
                float(np.sum(comp_species**2) / total_var_species)
            )
            per_component["metabolomics"].append(
                float(np.sum(comp_metab**2) / total_var_metab)
            )

        return {
            "species_total": float(var_explained_species),
            "metabolomics_total": float(var_explained_metab),
            "combined_total": float(
                (var_explained_species + var_explained_metab) / 2
            ),
            "per_component": per_component,
        }


def cross_validate_components(
    X_species: np.ndarray,
    X_metab: np.ndarray,
    component_range: range = range(3, 16),
    n_folds: int = 5,
    alpha: float = 0.65,
    random_state: int = 42,
) -> dict[str, list[float]]:
    """Cross-validate to find optimal number of CMTF components.

    Returns dict with 'n_components', 'mean_loss', 'std_loss'.
    """
    results: dict[str, list[float]] = {
        "n_components": [],
        "mean_loss": [],
        "std_loss": [],
    }

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)

    for n_comp in component_range:
        fold_losses = []
        for train_idx, test_idx in kf.split(X_species):
            model = CMTFModel(
                n_components=n_comp, alpha=alpha, random_state=random_state
            )
            model.fit(X_species[train_idx], X_metab[train_idx])

            # Project test samples into factor space using fitted loadings
            R = n_comp
            lhs = alpha * (X_species[test_idx] @ model.A_) + (1 - alpha) * (X_metab[test_idx] @ model.B_)
            rhs = alpha * (model.A_.T @ model.A_) + (1 - alpha) * (model.B_.T @ model.B_) + 1e-8 * np.eye(R)
            F_test = lhs @ np.linalg.inv(rhs)
            loss = model._compute_loss(
                X_species[test_idx],
                X_metab[test_idx],
                F_test,
                model.A_,
                model.B_,
            )
            fold_losses.append(loss)

        results["n_components"].append(n_comp)
        results["mean_loss"].append(float(np.mean(fold_losses)))
        results["std_loss"].append(float(np.std(fold_losses)))

        logger.info(
            "CMTF CV: R=%d, loss=%.4f ± %.4f",
            n_comp, results["mean_loss"][-1], results["std_loss"][-1],
        )

    return results


# ── High-Level Integration Pipeline ──


class CMTFIntegration:
    """CMTF integration pipeline for microbiome-metabolome subtyping."""

    def __init__(
        self,
        n_components: int | str = 10,
        alpha: float = 0.65,
        random_state: int = 42,
        cv_folds: int = 5,
        cv_component_range: range | None = None,
    ) -> None:
        """
        Parameters
        ----------
        n_components : int or "auto"
            Number of latent factors. Use "auto" to select via cross-validation.
        alpha : float
            Weight for species vs metabolites (default: 0.65).
        cv_folds : int
            Number of CV folds when n_components="auto" (default: 5).
        cv_component_range : range, optional
            Range of components to test in CV (default: range(2, min(n_samples//3, 16))).
        """
        self.n_components = n_components
        self.alpha = alpha
        self.random_state = random_state
        self.cv_folds = cv_folds
        self.cv_component_range = cv_component_range
        self.model: CMTFModel | None = None
        self.cv_results_: dict[str, list[float]] | None = None

    def _select_n_components(
        self,
        X_species: np.ndarray,
        X_metab: np.ndarray,
    ) -> int:
        """Select optimal n_components via cross-validation."""
        n_samples = X_species.shape[0]
        comp_range = self.cv_component_range or range(2, min(n_samples // 3, 16))

        logger.info(
            "CMTF CV: selecting n_components from %s (n=%d, %d folds)",
            list(comp_range), n_samples, self.cv_folds,
        )

        self.cv_results_ = cross_validate_components(
            X_species,
            X_metab,
            component_range=comp_range,
            n_folds=self.cv_folds,
            alpha=self.alpha,
            random_state=self.random_state,
        )

        # Select component count with minimum mean loss
        best_idx = int(np.argmin(self.cv_results_["mean_loss"]))
        best_n = int(self.cv_results_["n_components"][best_idx])
        best_loss = self.cv_results_["mean_loss"][best_idx]

        logger.info("CMTF CV: best n_components=%d (loss=%.4f)", best_n, best_loss)
        return best_n

    def fit(
        self,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame | dict]:
        """Fit CMTF and extract results.

        Parameters
        ----------
        species : DataFrame
            QC-normalized species matrix (participants × species).
        metabolomics : DataFrame
            QC-normalized metabolomics matrix (participants × metabolites).

        Returns
        -------
        dict with keys:
            - "factors": DataFrame (participants × components)
            - "species_loadings": DataFrame (species × components)
            - "metabolite_loadings": DataFrame (metabolites × components)
            - "variance_explained": dict
            - "top_species_per_factor": dict[int, list]
            - "top_metabolites_per_factor": dict[int, list]
        """
        output_dir = output_dir or OUTPUT_DIR

        # Ensure aligned indices
        shared = species.index.intersection(metabolomics.index)
        species = species.loc[shared]
        metabolomics = metabolomics.loc[shared]

        X_sp = species.values.astype(float)
        X_mb = metabolomics.values.astype(float)

        # Select n_components via CV if "auto"
        if self.n_components == "auto":
            self.n_components = self._select_n_components(X_sp, X_mb)

        n_comp = int(self.n_components)

        logger.info(
            "CMTF fit: %d participants, %d species, %d metabolites, %d components",
            len(shared), species.shape[1], metabolomics.shape[1], n_comp,
        )

        # Fit CMTF
        self.model = CMTFModel(
            n_components=n_comp,
            alpha=self.alpha,
            random_state=self.random_state,
        )
        self.model.fit(X_sp, X_mb)

        # Extract results
        factors = pd.DataFrame(
            self.model.F_,
            index=species.index,
            columns=[f"CMTF_{i}" for i in range(n_comp)],
        )
        species_loadings = pd.DataFrame(
            self.model.A_,
            index=species.columns,
            columns=[f"CMTF_{i}" for i in range(n_comp)],
        )
        metabolite_loadings = pd.DataFrame(
            self.model.B_,
            index=metabolomics.columns,
            columns=[f"CMTF_{i}" for i in range(n_comp)],
        )

        var_explained = self.model.variance_explained(X_sp, X_mb)

        # Top features per factor
        top_species = {}
        top_metabolites = {}
        for i in range(n_comp):
            sp_load = species_loadings.iloc[:, i].abs().sort_values(ascending=False)
            mb_load = metabolite_loadings.iloc[:, i].abs().sort_values(ascending=False)
            top_species[i] = list(sp_load.head(10).index)
            top_metabolites[i] = list(mb_load.head(10).index)

        results = {
            "factors": factors,
            "species_loadings": species_loadings,
            "metabolite_loadings": metabolite_loadings,
            "variance_explained": var_explained,
            "top_species_per_factor": top_species,
            "top_metabolites_per_factor": top_metabolites,
            "n_components": n_comp,
        }
        if self.cv_results_ is not None:
            results["cv_results"] = self.cv_results_

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        factors.to_csv(output_dir / "cmtf_factors.csv")
        species_loadings.to_csv(output_dir / "cmtf_species_loadings.csv")
        metabolite_loadings.to_csv(output_dir / "cmtf_metabolite_loadings.csv")

        logger.info(
            "CMTF results: species VE=%.1f%%, metabolomics VE=%.1f%%",
            var_explained["species_total"] * 100,
            var_explained["metabolomics_total"] * 100,
        )

        return results

    def map_to_metabolic_axes(
        self,
        species_loadings: pd.DataFrame,
        metabolite_loadings: pd.DataFrame,
    ) -> dict[int, list[str]]:
        """Map CMTF factors to the three candidate metabolic axes.

        Axes:
        1. Bifidobacterium-TCDCA (bile acid/immune)
        2. Tryptophan-NAD (energy/nitrogen)
        3. BCAA (amino acid synthesis)

        Returns dict mapping factor index → list of matching axes.
        """
        axis_markers = {
            "Bifidobacterium-TCDCA": {
                "species": ["bifidobacterium", "bacteroides"],
                "metabolites": ["tcdca", "taurochenodeoxychol", "bile", "lithochol", "deoxychol"],
            },
            "Tryptophan-NAD": {
                "species": ["tryptophan", "tnaa", "kyna"],
                "metabolites": ["tryptophan", "kynurenine", "quinolinate", "nad", "nicotinamide", "indole"],
            },
            "BCAA": {
                "species": [],
                "metabolites": ["leucine", "valine", "isoleucine", "bcaa"],
            },
        }

        factor_axes: dict[int, list[str]] = {}
        for i in range(len(species_loadings.columns)):
            top_sp = [s.lower() for s in species_loadings.iloc[:, i].abs().nlargest(20).index]
            top_mb = [m.lower() for m in metabolite_loadings.iloc[:, i].abs().nlargest(20).index]

            matches = []
            for axis_name, markers in axis_markers.items():
                sp_match = any(
                    any(marker in sp for marker in markers["species"])
                    for sp in top_sp
                )
                mb_match = any(
                    any(marker in mb for marker in markers["metabolites"])
                    for mb in top_mb
                )
                if sp_match or mb_match:
                    matches.append(axis_name)

            factor_axes[i] = matches

        return factor_axes
