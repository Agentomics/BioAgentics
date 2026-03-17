"""Cross-study batch correction for anti-TNF response prediction.

Applies ComBat batch correction across multi-study expression matrices
and generates diagnostic PCA/UMAP visualizations.

Usage:
    uv run python -m bioagentics.data.anti_tnf.batch_correction [--input-dir PATH] [--output-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import umap
from combat.pycombat import pycombat
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from bioagentics.config import REPO_ROOT
from bioagentics.data.anti_tnf.processing import USABLE_DATASETS

logger = logging.getLogger(__name__)

DEFAULT_INPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "processed"
DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "batch_correction"


def load_processed_data(
    input_dir: Path,
    accessions: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load processed per-study expression matrices and combined metadata.

    Returns a merged expression matrix (common genes x all samples) and metadata.
    """
    if accessions is None:
        accessions = list(USABLE_DATASETS)

    expr_frames = {}
    for acc in accessions:
        expr_path = input_dir / f"{acc}_expression.csv"
        if not expr_path.exists():
            logger.warning("Missing expression file: %s", expr_path)
            continue
        expr_frames[acc] = pd.read_csv(expr_path, index_col=0)

    if not expr_frames:
        raise FileNotFoundError(f"No expression files found in {input_dir}")

    # Find common genes across all studies
    gene_sets = [set(df.index) for df in expr_frames.values()]
    common_genes = sorted(gene_sets[0].intersection(*gene_sets[1:]))
    logger.info("Common genes across %d studies: %d", len(expr_frames), len(common_genes))

    # Merge into single matrix (genes x samples)
    merged = pd.concat(
        [df.loc[common_genes] for df in expr_frames.values()],
        axis=1,
    )

    # Load combined metadata
    meta_path = input_dir / "combined_metadata.csv"
    metadata = pd.read_csv(meta_path)

    # Align metadata to expression columns in same order
    metadata = metadata[metadata["sample_id"].isin(merged.columns)].copy()
    sample_order = {sid: i for i, sid in enumerate(merged.columns)}
    metadata = metadata.assign(_order=metadata["sample_id"].map(sample_order))
    metadata = metadata.sort_values("_order").drop(columns="_order").reset_index(drop=True)

    return merged, metadata


def run_combat(
    expr: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Apply ComBat batch correction.

    Uses study accession as batch variable and response status as biological
    covariate to preserve.
    """
    meta_indexed = metadata.set_index("sample_id")
    batch = meta_indexed.loc[expr.columns, "study"]

    # Build biological covariate as list-of-lists (pycombat API requirement)
    response = meta_indexed.loc[expr.columns, "response_status"]
    mod = pd.get_dummies(response, drop_first=True, dtype=float).values.T.tolist()

    logger.info(
        "Running ComBat: %d genes x %d samples, %d batches, covariate: response_status",
        expr.shape[0], expr.shape[1], batch.nunique(),
    )

    corrected = pycombat(expr, batch, mod=mod)

    logger.info("ComBat complete: output shape %s", corrected.shape)
    return corrected


def compute_pca(expr: pd.DataFrame, n_components: int = 2) -> np.ndarray:
    """Compute PCA on samples (columns) of expression matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(expr.T.values)
    pca = PCA(n_components=n_components, random_state=42)
    return pca.fit_transform(scaled)


def compute_umap(expr: pd.DataFrame, n_components: int = 2) -> np.ndarray:
    """Compute UMAP on samples (columns) of expression matrix."""
    scaler = StandardScaler()
    scaled = scaler.fit_transform(expr.T.values)
    reducer = umap.UMAP(n_components=n_components, random_state=42)
    return reducer.fit_transform(scaled)


def _scatter(ax, coords, labels, palette, title, xlabel, ylabel):
    """Plot a colored scatter on the given axes."""
    for label in sorted(set(labels)):
        mask = [l == label for l in labels]
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            label=label, c=palette.get(label, "gray"), alpha=0.7, edgecolors="k", linewidths=0.3, s=50,
        )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="best")


STUDY_COLORS = {"GSE16879": "#1f77b4", "GSE12251": "#ff7f0e", "GSE73661": "#2ca02c"}
RESPONSE_COLORS = {"responder": "#2ca02c", "non_responder": "#d62728", "unknown": "#7f7f7f"}


def generate_diagnostic_plots(
    expr_before: pd.DataFrame,
    expr_after: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Generate before/after PCA and UMAP plots colored by study and response."""
    output_dir.mkdir(parents=True, exist_ok=True)

    studies = metadata.set_index("sample_id").loc[expr_before.columns, "study"].values
    responses = metadata.set_index("sample_id").loc[expr_before.columns, "response_status"].values

    # Compute embeddings
    pca_before = compute_pca(expr_before)
    pca_after = compute_pca(expr_after)
    umap_before = compute_umap(expr_before)
    umap_after = compute_umap(expr_after)

    # PCA plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _scatter(axes[0, 0], pca_before, studies, STUDY_COLORS, "PCA Before — Study", "PC1", "PC2")
    _scatter(axes[0, 1], pca_after, studies, STUDY_COLORS, "PCA After — Study", "PC1", "PC2")
    _scatter(axes[1, 0], pca_before, responses, RESPONSE_COLORS, "PCA Before — Response", "PC1", "PC2")
    _scatter(axes[1, 1], pca_after, responses, RESPONSE_COLORS, "PCA After — Response", "PC1", "PC2")
    fig.suptitle("ComBat Batch Correction: PCA Diagnostics", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "pca_diagnostics.png", dpi=150)
    plt.close(fig)
    logger.info("Saved PCA diagnostic plot")

    # UMAP plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    _scatter(axes[0, 0], umap_before, studies, STUDY_COLORS, "UMAP Before — Study", "UMAP1", "UMAP2")
    _scatter(axes[0, 1], umap_after, studies, STUDY_COLORS, "UMAP After — Study", "UMAP1", "UMAP2")
    _scatter(axes[1, 0], umap_before, responses, RESPONSE_COLORS, "UMAP Before — Response", "UMAP1", "UMAP2")
    _scatter(axes[1, 1], umap_after, responses, RESPONSE_COLORS, "UMAP After — Response", "UMAP1", "UMAP2")
    fig.suptitle("ComBat Batch Correction: UMAP Diagnostics", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(output_dir / "umap_diagnostics.png", dpi=150)
    plt.close(fig)
    logger.info("Saved UMAP diagnostic plot")


def run_batch_correction(
    input_dir: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT,
    accessions: list[str] | None = None,
) -> pd.DataFrame:
    """Run the full batch correction pipeline.

    Returns the batch-corrected expression matrix.
    """
    # Load data
    expr, metadata = load_processed_data(input_dir, accessions)

    # Run ComBat
    corrected = run_combat(expr, metadata)

    # Save corrected matrix
    output_dir.mkdir(parents=True, exist_ok=True)
    corrected.to_csv(output_dir / "expression_combat.csv")
    logger.info("Saved batch-corrected expression: %s", output_dir / "expression_combat.csv")

    # Copy metadata for convenience
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    # Generate diagnostic plots
    generate_diagnostic_plots(expr, corrected, metadata, output_dir)

    return corrected


def main():
    parser = argparse.ArgumentParser(
        description="Apply ComBat batch correction to anti-TNF expression data"
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_batch_correction(input_dir=args.input_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
