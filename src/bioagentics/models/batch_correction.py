"""Batch correction for multi-study RNA-seq harmonization using ComBat.

Applies ComBat (via pycombat) to remove technical batch effects across
GEO studies while preserving biological variation. Includes PCA
visualization before/after correction.

Usage:
    uv run python -m bioagentics.models.batch_correction input.h5ad --batch-key dataset [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"


def combat_correct(
    adata: ad.AnnData,
    batch_key: str = "dataset",
    covariate_keys: list[str] | None = None,
) -> ad.AnnData:
    """Apply ComBat batch correction to an AnnData object.

    Parameters
    ----------
    adata : AnnData
        Input data. X should contain normalized expression values (not raw counts).
        Must have batch_key in adata.obs.
    batch_key : str
        Column in adata.obs identifying the batch (e.g., "dataset" for GEO accessions).
    covariate_keys : list[str], optional
        Columns in adata.obs to preserve as biological covariates.

    Returns
    -------
    AnnData with batch-corrected expression in X. Original expression stored in
    adata.layers["pre_combat"].
    """
    from combat.pycombat import pycombat

    if batch_key not in adata.obs.columns:
        raise ValueError(f"Batch key '{batch_key}' not found in adata.obs. "
                         f"Available: {list(adata.obs.columns)}")

    batches = adata.obs[batch_key].values
    unique_batches = pd.Series(batches).unique()
    if len(unique_batches) < 2:
        logger.warning("Only %d batch(es) found — skipping batch correction", len(unique_batches))
        return adata

    logger.info("Running ComBat on %d samples across %d batches: %s",
                adata.n_obs, len(unique_batches), list(unique_batches))

    # pycombat expects genes (rows) x samples (columns) DataFrame
    expr_df = pd.DataFrame(
        adata.X.T if hasattr(adata.X, "T") else np.array(adata.X).T,
        index=adata.var_names,
        columns=adata.obs_names,
    ).astype(float)

    batch_list = list(batches)

    # Build covariate list if provided
    mod = []
    if covariate_keys:
        for key in covariate_keys:
            if key in adata.obs.columns:
                mod.append(list(adata.obs[key].cat.codes if hasattr(adata.obs[key], "cat") else adata.obs[key]))
            else:
                logger.warning("Covariate '%s' not found in adata.obs, skipping", key)

    corrected = pycombat(expr_df, batch_list, mod=mod if mod else [])

    result = adata.copy()
    result.layers["pre_combat"] = result.X.copy()
    result.X = corrected.T.values.astype(np.float32)

    logger.info("ComBat correction complete")
    return result


def plot_pca_comparison(
    adata_before: ad.AnnData,
    adata_after: ad.AnnData,
    batch_key: str = "dataset",
    save_path: Path | None = None,
) -> None:
    """Generate PCA plots before and after batch correction.

    Saves a side-by-side figure showing PCA colored by batch.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, adata, title in [
        (axes[0], adata_before, "Before ComBat"),
        (axes[1], adata_after, "After ComBat"),
    ]:
        X = np.array(adata.X, dtype=float)
        # Handle NaN/inf
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(X)

        batches = adata.obs[batch_key].values
        unique = sorted(set(batches))
        for batch in unique:
            mask = batches == batch
            ax.scatter(coords[mask, 0], coords[mask, 1], label=batch, alpha=0.6, s=20)

        ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%})")
        ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%})")
        ax.set_title(title)
        ax.legend(fontsize=8, loc="best")

    plt.tight_layout()

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info("Saved PCA comparison to %s", save_path)
    plt.close(fig)


def batch_correct_pipeline(
    adata: ad.AnnData,
    batch_key: str = "dataset",
    covariate_keys: list[str] | None = None,
    dest_dir: Path | None = None,
    save_plot: bool = True,
) -> ad.AnnData:
    """Full batch correction pipeline with optional PCA visualization.

    Parameters
    ----------
    adata : AnnData
        Input data with batch labels in obs.
    batch_key : str
        Batch identifier column.
    covariate_keys : list[str], optional
        Biological covariates to preserve.
    dest_dir : Path, optional
        Output directory for plots and corrected data.
    save_plot : bool
        Whether to save PCA before/after plot.

    Returns
    -------
    Batch-corrected AnnData.
    """
    if dest_dir is None:
        dest_dir = OUTPUT_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    corrected = combat_correct(adata, batch_key=batch_key, covariate_keys=covariate_keys)

    if save_plot and "pre_combat" in corrected.layers:
        plot_path = dest_dir / "pca_batch_correction.png"
        # Build a "before" view using the pre-combat layer
        before = corrected.copy()
        before.X = before.layers["pre_combat"]
        plot_pca_comparison(before, corrected, batch_key=batch_key, save_path=plot_path)

    return corrected


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Apply ComBat batch correction to AnnData"
    )
    parser.add_argument("input", type=Path, help="Input h5ad file")
    parser.add_argument("--batch-key", default="dataset", help="Batch column in obs")
    parser.add_argument("--covariates", nargs="*", help="Covariate columns to preserve")
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR, help="Output directory")
    parser.add_argument("--no-plot", action="store_true", help="Skip PCA plot")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    adata = ad.read_h5ad(args.input)
    logger.info("Loaded %d samples x %d genes from %s", adata.n_obs, adata.n_vars, args.input)

    corrected = batch_correct_pipeline(
        adata,
        batch_key=args.batch_key,
        covariate_keys=args.covariates,
        dest_dir=args.dest,
        save_plot=not args.no_plot,
    )

    out_path = args.dest / f"{args.input.stem}_combat.h5ad"
    corrected.write_h5ad(out_path)
    print(f"\nSaved corrected data: {corrected.n_obs} samples x {corrected.n_vars} genes")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
