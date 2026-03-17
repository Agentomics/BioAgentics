"""MOFA2 integration pipeline for latent factor discovery in CD subtyping.

Runs MOFA2 (Multi-Omics Factor Analysis v2) on metagenomics and metabolomics
views to discover interpretable latent factors capturing shared and
view-specific variance.

Usage::

    from bioagentics.models.crohns_mofa2 import MOFA2Integration

    mofa = MOFA2Integration(n_factors=10)
    results = mofa.fit(species_qc, metabolomics_qc)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "microbiome-metabolome-subtyping"


def run_mofa2(
    views: dict[str, np.ndarray],
    n_factors: int = 10,
    seed: int = 42,
    convergence_mode: str = "fast",
    max_iter: int = 1000,
) -> dict:
    """Run MOFA2 factor analysis on multi-view data.

    Parameters
    ----------
    views : dict
        View name -> (samples × features) arrays.
    n_factors : int
        Number of latent factors.
    seed : int
        Random seed.
    convergence_mode : str
        "fast" or "medium".
    max_iter : int
        Maximum iterations.

    Returns
    -------
    dict with: factors, weights, variance_explained, view_names, n_factors.
    """
    from mofapy2.run.entry_point import entry_point

    n_samples = None
    for v in views.values():
        if n_samples is None:
            n_samples = v.shape[0]
        elif v.shape[0] != n_samples:
            raise ValueError("All views must have the same number of samples")

    if n_samples is None or n_samples == 0:
        raise ValueError("No valid views provided")

    # Cap factors at min(n_samples-1, min_features)
    min_features = min(v.shape[1] for v in views.values())
    n_factors = min(n_factors, n_samples - 1, min_features)
    n_factors = max(n_factors, 1)

    ent = entry_point()

    view_names = list(views.keys())
    data = [[views[v]] for v in view_names]

    ent.set_data_options(scale_groups=False, scale_views=True)
    ent.set_data_matrix(data, views_names=view_names)
    ent.set_model_options(
        factors=n_factors, spikeslab_weights=True, ard_weights=True
    )
    ent.set_train_options(
        iter=max_iter,
        convergence_mode=convergence_mode,
        seed=seed,
        verbose=False,
        dropR2=0.001,
    )

    ent.build()

    try:
        ent.run()
    except SystemExit:
        logger.warning("MOFA2 dropped all factors — no shared structure found")

    # Extract expectations
    expectations = ent.model.getExpectations()

    Z_raw = expectations["Z"]
    if isinstance(Z_raw, dict):
        factors = Z_raw["E"]
    elif isinstance(Z_raw, list):
        factors = Z_raw[0]
    else:
        factors = Z_raw

    if factors.ndim == 1:
        factors = factors.reshape(-1, 1)
    if factors.shape[1] == 0:
        factors = np.zeros((n_samples, 1))

    W_raw = expectations["W"]
    weights = {}
    for i, vname in enumerate(view_names):
        w_item = W_raw[i]
        weights[vname] = w_item["E"] if isinstance(w_item, dict) else w_item

    # Variance explained
    variance_explained: dict[str, list[float]] = {v: [] for v in view_names}
    try:
        r2 = ent.model.calculate_variance_explained()
        if r2 is not None and len(r2) > 0:
            r2_arr = r2[0]  # single group
            if isinstance(r2_arr, np.ndarray) and r2_arr.ndim == 2:
                for i, vname in enumerate(view_names):
                    variance_explained[vname] = r2_arr[:, i].tolist()
    except Exception:
        pass

    result = {
        "factors": factors,
        "weights": weights,
        "variance_explained": variance_explained,
        "view_names": view_names,
        "n_factors": factors.shape[1] if factors.ndim == 2 else 1,
    }

    logger.info(
        "MOFA2 completed: %d factors across %d views, %d samples",
        result["n_factors"], len(view_names), n_samples,
    )
    return result


class MOFA2Integration:
    """MOFA2 integration pipeline for CD subtyping."""

    def __init__(
        self,
        n_factors: int = 10,
        seed: int = 42,
        convergence_mode: str = "fast",
    ) -> None:
        self.n_factors = n_factors
        self.seed = seed
        self.convergence_mode = convergence_mode
        self.result: dict | None = None

    def fit(
        self,
        species: pd.DataFrame,
        metabolomics: pd.DataFrame,
        pathways: pd.DataFrame | None = None,
        output_dir: Path | None = None,
    ) -> dict[str, pd.DataFrame | dict]:
        """Fit MOFA2 on metagenomics and metabolomics views.

        Parameters
        ----------
        species : DataFrame
            QC-normalized species matrix (participants × species).
        metabolomics : DataFrame
            QC-normalized metabolomics matrix (participants × metabolites).
        pathways : DataFrame, optional
            QC-normalized pathway matrix (participants × pathways).

        Returns
        -------
        dict with:
            - "factors": DataFrame (participants × factors)
            - "weights_species": DataFrame (species × factors)
            - "weights_metabolomics": DataFrame (metabolites × factors)
            - "variance_explained": dict
            - "factor_classification": dict classifying shared vs view-specific
        """
        output_dir = output_dir or OUTPUT_DIR

        # Align indices
        shared = species.index.intersection(metabolomics.index)
        if pathways is not None:
            shared = shared.intersection(pathways.index)
        species = species.loc[shared]
        metabolomics = metabolomics.loc[shared]

        views: dict[str, np.ndarray] = {
            "species": species.values.astype(float),
            "metabolomics": metabolomics.values.astype(float),
        }
        if pathways is not None:
            pathways = pathways.loc[shared]
            views["pathways"] = pathways.values.astype(float)

        logger.info(
            "MOFA2 fit: %d participants, views: %s",
            len(shared),
            {k: v.shape[1] for k, v in views.items()},
        )

        self.result = run_mofa2(
            views,
            n_factors=self.n_factors,
            seed=self.seed,
            convergence_mode=self.convergence_mode,
        )

        actual_n = self.result["n_factors"]

        # Build DataFrames
        factor_names = [f"MOFA_{i}" for i in range(actual_n)]

        factors_df = pd.DataFrame(
            self.result["factors"],
            index=shared,
            columns=factor_names,
        )

        weights_species = pd.DataFrame(
            self.result["weights"]["species"],
            index=species.columns,
            columns=factor_names,
        )
        weights_metab = pd.DataFrame(
            self.result["weights"]["metabolomics"],
            index=metabolomics.columns,
            columns=factor_names,
        )

        # Classify factors as shared vs view-specific
        factor_class = self._classify_factors(self.result["variance_explained"])

        # Variance decomposition
        var_exp = self.result["variance_explained"]
        total_per_view = {}
        for vname, variances in var_exp.items():
            total_per_view[vname] = sum(variances) if variances else 0.0

        output = {
            "factors": factors_df,
            "weights_species": weights_species,
            "weights_metabolomics": weights_metab,
            "variance_explained": var_exp,
            "total_variance_per_view": total_per_view,
            "factor_classification": factor_class,
        }

        if pathways is not None and "pathways" in self.result["weights"]:
            output["weights_pathways"] = pd.DataFrame(
                self.result["weights"]["pathways"],
                index=pathways.columns,
                columns=factor_names,
            )

        # Save outputs
        output_dir.mkdir(parents=True, exist_ok=True)
        factors_df.to_csv(output_dir / "mofa2_factors.csv")
        weights_species.to_csv(output_dir / "mofa2_weights_species.csv")
        weights_metab.to_csv(output_dir / "mofa2_weights_metabolomics.csv")

        logger.info(
            "MOFA2 results: %d factors, species VE=%.1f%%, metabolomics VE=%.1f%%",
            actual_n,
            total_per_view.get("species", 0) * 100,
            total_per_view.get("metabolomics", 0) * 100,
        )

        return output

    def _classify_factors(
        self, variance_explained: dict[str, list[float]]
    ) -> dict[int, str]:
        """Classify each factor as shared, view-specific, or weak.

        A factor is:
        - "shared" if it explains >1% variance in both views
        - "species-specific" if >1% only in species
        - "metabolomics-specific" if >1% only in metabolomics
        - "weak" if <1% in all views
        """
        sp_var = variance_explained.get("species", [])
        mb_var = variance_explained.get("metabolomics", [])
        n_factors = max(len(sp_var), len(mb_var))

        classification: dict[int, str] = {}
        threshold = 0.01  # 1% variance explained

        for i in range(n_factors):
            sp_v = sp_var[i] if i < len(sp_var) else 0.0
            mb_v = mb_var[i] if i < len(mb_var) else 0.0

            if sp_v > threshold and mb_v > threshold:
                classification[i] = "shared"
            elif sp_v > threshold:
                classification[i] = "species-specific"
            elif mb_v > threshold:
                classification[i] = "metabolomics-specific"
            else:
                classification[i] = "weak"

        return classification


def compare_with_cmtf(
    mofa_factors: pd.DataFrame,
    cmtf_factors: pd.DataFrame,
) -> dict[str, float]:
    """Compare MOFA2 and CMTF factor structures.

    Uses canonical correlation analysis (CCA) proxy via correlation
    of factor spaces. High correlation suggests methods converge.

    Returns
    -------
    dict with correlation metrics.
    """
    shared = mofa_factors.index.intersection(cmtf_factors.index)
    mofa = mofa_factors.loc[shared].values
    cmtf = cmtf_factors.loc[shared].values

    # Compute pairwise correlations between factors
    n_mofa = mofa.shape[1]
    n_cmtf = cmtf.shape[1]

    corr_matrix = np.zeros((n_mofa, n_cmtf))
    for i in range(n_mofa):
        for j in range(n_cmtf):
            corr_matrix[i, j] = np.corrcoef(mofa[:, i], cmtf[:, j])[0, 1]

    # Best matching correlations
    max_corr_per_mofa = np.max(np.abs(corr_matrix), axis=1)
    max_corr_per_cmtf = np.max(np.abs(corr_matrix), axis=0)

    return {
        "mean_best_correlation": float(np.mean(max_corr_per_mofa)),
        "max_best_correlation": float(np.max(max_corr_per_mofa)),
        "n_matched_factors_gt_0.5": int(np.sum(max_corr_per_mofa > 0.5)),
        "n_matched_factors_gt_0.7": int(np.sum(max_corr_per_mofa > 0.7)),
        "cmtf_coverage": float(np.mean(max_corr_per_cmtf > 0.5)),
    }
