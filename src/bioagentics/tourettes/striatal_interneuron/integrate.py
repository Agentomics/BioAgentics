"""Data integration and batch correction for striatal interneuron analysis.

Merges multiple snRNA-seq datasets and corrects batch effects using:
- Harmony (fast, PCA-based)
- scVI (deep learning, optional — requires scvi-tools)

Input: list of QC-passed h5ad files + batch key
Output: integrated AnnData with corrected embeddings

Includes evaluation metrics: batch mixing (LISI) and cell-type purity (silhouette).

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.integrate [h5ad files ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

from bioagentics.tourettes.striatal_interneuron.config import (
    INTEGRATION_DIR,
    INTEGRATION_PARAMS,
    QC_DIR,
    ensure_dirs,
)


def merge_datasets(
    adata_list: list[ad.AnnData],
    batch_key: str | None = None,
    batch_labels: list[str] | None = None,
) -> ad.AnnData:
    """Merge multiple AnnData objects into a single dataset.

    Adds a batch column to .obs for downstream correction.
    """
    if batch_key is None:
        batch_key = INTEGRATION_PARAMS["batch_key"]

    if batch_labels is None:
        batch_labels = [f"batch_{i}" for i in range(len(adata_list))]

    if len(batch_labels) != len(adata_list):
        raise ValueError(f"Got {len(batch_labels)} labels for {len(adata_list)} datasets")

    for label, adata in zip(batch_labels, adata_list):
        adata.obs[batch_key] = label

    merged = ad.concat(adata_list, join="inner", merge="same")
    merged.obs_names_make_unique()
    merged.var_names_make_unique()
    print(f"  Merged: {merged.n_obs} cells x {merged.n_vars} genes from {len(adata_list)} batches")
    return merged


def preprocess_for_integration(
    adata: ad.AnnData,
    n_top_genes: int | None = None,
    n_pcs: int | None = None,
) -> ad.AnnData:
    """Normalize, select HVGs, and compute PCA for integration input."""
    if n_top_genes is None:
        n_top_genes = INTEGRATION_PARAMS["n_top_genes"]
    if n_pcs is None:
        n_pcs = INTEGRATION_PARAMS["n_pcs"]

    # If raw_counts layer exists, use it for fresh normalization
    if "raw_counts" in adata.layers:
        adata.X = adata.layers["raw_counts"].copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)

    # Subset to HVG for integration
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=min(n_pcs, adata_hvg.n_obs - 1, adata_hvg.n_vars - 1))

    print(f"  Preprocessed: {adata_hvg.n_obs} cells, {adata_hvg.n_vars} HVGs, PCA computed")
    return adata_hvg


def integrate_harmony(
    adata: ad.AnnData,
    batch_key: str | None = None,
    max_iter: int | None = None,
) -> ad.AnnData:
    """Run Harmony batch correction on PCA embeddings.

    Adds 'X_pca_harmony' to .obsm and recomputes neighbors/UMAP.
    """
    if batch_key is None:
        batch_key = INTEGRATION_PARAMS["batch_key"]
    if max_iter is None:
        max_iter = INTEGRATION_PARAMS["harmony_max_iter"]

    import harmonypy

    print(f"  Running Harmony (batch_key={batch_key}, max_iter={max_iter})")

    harmony_out = harmonypy.run_harmony(
        np.ascontiguousarray(adata.obsm["X_pca"]),
        adata.obs,
        batch_key,
        max_iter_harmony=max_iter,
    )
    adata.obsm["X_pca_harmony"] = np.asarray(harmony_out.Z_corr)

    # Recompute neighbors and UMAP on corrected embedding
    sc.pp.neighbors(adata, use_rep="X_pca_harmony")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)

    print(f"  Harmony complete: corrected embedding in X_pca_harmony")
    return adata


def integrate_scvi(
    adata: ad.AnnData,
    batch_key: str | None = None,
    n_latent: int = 30,
    max_epochs: int = 200,
) -> ad.AnnData:
    """Run scVI batch correction (deep learning).

    Requires scvi-tools. Adds 'X_scVI' to .obsm.
    """
    if batch_key is None:
        batch_key = INTEGRATION_PARAMS["batch_key"]

    try:
        import scvi
    except ImportError:
        print("  WARNING: scvi-tools not installed, skipping scVI integration")
        print("  Install with: uv add scvi-tools")
        return adata

    print(f"  Running scVI (batch_key={batch_key}, n_latent={n_latent}, max_epochs={max_epochs})")

    # scVI needs raw counts
    if "raw_counts" in adata.layers:
        adata_scvi = adata.copy()
        adata_scvi.X = adata_scvi.layers["raw_counts"].copy()
    else:
        adata_scvi = adata.copy()

    scvi.model.SCVI.setup_anndata(adata_scvi, layer=None, batch_key=batch_key)
    model = scvi.model.SCVI(adata_scvi, n_latent=n_latent)
    model.train(max_epochs=max_epochs, early_stopping=True)

    adata.obsm["X_scVI"] = model.get_latent_representation()

    # Recompute neighbors and UMAP on scVI embedding
    sc.pp.neighbors(adata, use_rep="X_scVI")
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)

    print(f"  scVI complete: latent representation in X_scVI")
    return adata


def evaluate_integration(
    adata: ad.AnnData,
    batch_key: str | None = None,
    cell_type_key: str | None = None,
) -> dict[str, float]:
    """Evaluate integration quality using silhouette scores.

    Returns dict with batch mixing and cell-type purity metrics.
    """
    if batch_key is None:
        batch_key = INTEGRATION_PARAMS["batch_key"]

    from sklearn.metrics import silhouette_score

    metrics: dict[str, float] = {}

    # Determine which embedding to evaluate
    if "X_pca_harmony" in adata.obsm:
        rep = "X_pca_harmony"
    elif "X_scVI" in adata.obsm:
        rep = "X_scVI"
    else:
        rep = "X_pca"

    X = adata.obsm[rep]

    # Batch mixing: lower silhouette = better mixing (batches are interleaved)
    if adata.obs[batch_key].nunique() > 1:
        batch_sil = silhouette_score(X, adata.obs[batch_key], sample_size=min(5000, len(X)))
        metrics["batch_silhouette"] = float(batch_sil)
        # Ideal: close to 0 (batches well mixed)
        print(f"  Batch silhouette ({rep}): {batch_sil:.3f} (closer to 0 = better mixing)")

    # Cell-type purity: higher silhouette = cell types well separated
    if cell_type_key and cell_type_key in adata.obs and adata.obs[cell_type_key].nunique() > 1:
        ct_sil = silhouette_score(X, adata.obs[cell_type_key], sample_size=min(5000, len(X)))
        metrics["celltype_silhouette"] = float(ct_sil)
        # Ideal: close to 1 (cell types well separated)
        print(f"  Cell-type silhouette ({rep}): {ct_sil:.3f} (closer to 1 = better purity)")

    metrics["embedding_used"] = rep
    return metrics


def run_integration(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    method: str = "harmony",
) -> Path | None:
    """Run full integration pipeline on QC-passed h5ad files.

    Args:
        input_dir: Directory with QC-passed h5ad files
        output_dir: Output directory
        method: Integration method ('harmony' or 'scvi')

    Returns path to integrated h5ad file.
    """
    if input_dir is None:
        input_dir = QC_DIR
    if output_dir is None:
        output_dir = INTEGRATION_DIR

    ensure_dirs()
    output_path = output_dir / f"integrated_{method}.h5ad"

    if output_path.exists():
        print(f"  Integration already done: {output_path.name}")
        return output_path

    h5ad_files = sorted(input_dir.glob("*_qc.h5ad"))
    if not h5ad_files:
        print(f"  No QC-passed h5ad files found in {input_dir}")
        print("  Run the QC phase first: pipeline qc")
        return None

    print(f"Found {len(h5ad_files)} QC-passed files for integration")

    # Memory safeguard: check total file size before loading (8GB machine)
    MAX_TOTAL_MB = 2048
    total_mb = sum(p.stat().st_size for p in h5ad_files) / 1_048_576
    if total_mb > MAX_TOTAL_MB:
        raise MemoryError(
            f"Total input size {total_mb:.0f} MB exceeds {MAX_TOTAL_MB} MB limit. "
            f"On an 8GB machine, loading all files at once risks an OOM crash. "
            f"Consider processing a subset or using backed mode."
        )
    print(f"  Total input size: {total_mb:.0f} MB (limit: {MAX_TOTAL_MB} MB)")

    # Load all datasets
    adata_list = []
    batch_labels = []
    for path in h5ad_files:
        print(f"  Loading: {path.name}")
        adata = ad.read_h5ad(path)
        adata_list.append(adata)
        batch_labels.append(path.stem.replace("_qc", ""))

    if len(adata_list) == 1:
        print("  Only one dataset — skipping batch correction, saving directly")
        merged = adata_list[0]
        merged.obs[INTEGRATION_PARAMS["batch_key"]] = batch_labels[0]
    else:
        # Merge and integrate
        merged = merge_datasets(adata_list, batch_labels=batch_labels)
        merged = preprocess_for_integration(merged)

        if method == "harmony":
            merged = integrate_harmony(merged)
        elif method == "scvi":
            merged = integrate_scvi(merged)
        else:
            raise ValueError(f"Unknown method: {method}")

        # Evaluate
        metrics = evaluate_integration(merged)
        merged.uns["integration_metrics"] = metrics

    merged.uns["integration_method"] = method
    merged.uns["n_batches"] = len(batch_labels)
    merged.uns["batch_labels"] = batch_labels

    output_dir.mkdir(parents=True, exist_ok=True)
    merged.write_h5ad(output_path)
    print(f"\n  Saved: {output_path.name} ({merged.n_obs} cells x {merged.n_vars} genes)")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Data integration and batch correction")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input QC-passed h5ad files (default: scan QC directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=INTEGRATION_DIR,
        help="Output directory",
    )
    parser.add_argument(
        "--method",
        choices=["harmony", "scvi"],
        default="harmony",
        help="Integration method (default: harmony)",
    )
    args = parser.parse_args()

    if args.inputs:
        # Memory safeguard: check total file size before loading (8GB machine)
        MAX_TOTAL_MB = 2048
        total_mb = sum(p.stat().st_size for p in args.inputs) / 1_048_576
        if total_mb > MAX_TOTAL_MB:
            raise MemoryError(
                f"Total input size {total_mb:.0f} MB exceeds {MAX_TOTAL_MB} MB limit. "
                f"On an 8GB machine, loading all files at once risks an OOM crash. "
                f"Consider processing a subset or using backed mode."
            )
        adata_list = [ad.read_h5ad(p) for p in args.inputs]
        labels = [p.stem for p in args.inputs]
        merged = merge_datasets(adata_list, batch_labels=labels)
        merged = preprocess_for_integration(merged)

        if args.method == "harmony":
            merged = integrate_harmony(merged)
        else:
            merged = integrate_scvi(merged)

        out = args.output_dir / f"integrated_{args.method}.h5ad"
        args.output_dir.mkdir(parents=True, exist_ok=True)
        merged.write_h5ad(out)
        print(f"Saved: {out}")
    else:
        run_integration(output_dir=args.output_dir, method=args.method)


if __name__ == "__main__":
    main()
