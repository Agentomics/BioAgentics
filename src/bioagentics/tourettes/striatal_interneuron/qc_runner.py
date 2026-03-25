"""snRNA-seq QC and preprocessing pipeline for striatal interneuron analysis.

Reusable QC pipeline:
1. Cell filtering by n_genes, n_counts, pct_mito
2. Doublet detection with scrublet
3. Gene filtering (min_cells)
4. Normalization (normalize_total + log1p)
5. Highly variable gene selection
6. PCA + neighbors + UMAP embedding

Accepts any h5ad input, writes QC-passed h5ad with metrics in .obs.

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.qc_runner [input.h5ad ...]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc

ad.settings.allow_write_nullable_strings = True

from bioagentics.tourettes.striatal_interneuron.config import (
    INTEGRATION_PARAMS,
    QC_DIR,
    QC_PARAMS,
    REFERENCE_DIR,
    ensure_dirs,
)


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Compute standard QC metrics: n_genes, n_counts, pct_mito."""
    # Mitochondrial genes
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True)
    return adata


def filter_cells(
    adata: ad.AnnData,
    min_genes: int | None = None,
    max_pct_mito: float | None = None,
) -> ad.AnnData:
    """Filter cells by minimum genes detected and max mitochondrial fraction."""
    if min_genes is None:
        min_genes = QC_PARAMS["min_genes"]
    if max_pct_mito is None:
        max_pct_mito = QC_PARAMS["max_pct_mito"]

    n_before = adata.n_obs
    sc.pp.filter_cells(adata, min_genes=min_genes)
    adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()
    n_after = adata.n_obs
    print(f"  Cell filter: {n_before} -> {n_after} ({n_before - n_after} removed)")
    return adata


def detect_doublets(
    adata: ad.AnnData,
    expected_doublet_rate: float | None = None,
) -> ad.AnnData:
    """Detect and remove doublets using scrublet."""
    if expected_doublet_rate is None:
        expected_doublet_rate = QC_PARAMS["expected_doublet_rate"]

    import scrublet as scr

    n_before = adata.n_obs

    # scrublet needs raw counts (not normalized)
    scrub = scr.Scrublet(adata.X, expected_doublet_rate=expected_doublet_rate)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(verbose=False)

    adata.obs["doublet_score"] = doublet_scores

    if predicted_doublets is None:
        # Scrublet couldn't determine a threshold — use score > 0.25 as fallback
        predicted_doublets = doublet_scores > 0.25
        print("  Doublet detection: auto-threshold failed, using score > 0.25")

    adata.obs["predicted_doublet"] = predicted_doublets
    n_doublets = int(np.sum(predicted_doublets))
    adata = adata[~adata.obs["predicted_doublet"]].copy()
    print(f"  Doublet detection: {n_doublets} doublets removed ({n_before} -> {adata.n_obs})")
    return adata


def filter_genes(adata: ad.AnnData, min_cells: int | None = None) -> ad.AnnData:
    """Filter genes expressed in fewer than min_cells."""
    if min_cells is None:
        min_cells = QC_PARAMS["min_cells"]

    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  Gene filter: {n_before} -> {adata.n_vars} genes ({n_before - adata.n_vars} removed)")
    return adata


def normalize_and_log(adata: ad.AnnData) -> ad.AnnData:
    """Normalize counts per cell and log-transform."""
    # Store raw counts for DE analysis later
    adata.layers["raw_counts"] = adata.X.copy()

    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    print(f"  Normalized and log-transformed ({adata.n_obs} cells)")
    return adata


def select_hvg(adata: ad.AnnData, n_top_genes: int | None = None) -> ad.AnnData:
    """Select highly variable genes."""
    if n_top_genes is None:
        n_top_genes = INTEGRATION_PARAMS["n_top_genes"]

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top_genes)
    n_hvg = adata.var["highly_variable"].sum()
    print(f"  HVG selection: {n_hvg} highly variable genes")
    return adata


def compute_embedding(adata: ad.AnnData, n_pcs: int | None = None) -> ad.AnnData:
    """Compute PCA, neighbors graph, and UMAP embedding."""
    if n_pcs is None:
        n_pcs = INTEGRATION_PARAMS["n_pcs"]

    sc.pp.scale(adata, max_value=10)
    actual_pcs = min(n_pcs, adata.n_obs - 1, adata.n_vars - 1)
    sc.tl.pca(adata, n_comps=actual_pcs)
    sc.pp.neighbors(adata, n_pcs=actual_pcs)
    sc.tl.umap(adata)
    sc.tl.leiden(adata, flavor="igraph", n_iterations=2, directed=False)
    print(f"  Embedding: PCA ({n_pcs} PCs) + UMAP + Leiden clustering")
    return adata


def run_qc_single(input_path: Path, output_dir: Path | None = None) -> Path:
    """Run full QC pipeline on a single h5ad file.

    Returns path to QC-passed output h5ad.
    """
    if output_dir is None:
        output_dir = QC_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{input_path.stem}_qc.h5ad"

    if output_path.exists():
        print(f"  QC already done: {output_path.name}")
        return output_path

    print(f"\n--- QC: {input_path.name} ---")
    adata = ad.read_h5ad(input_path)
    print(f"  Input: {adata.n_obs} cells x {adata.n_vars} genes")

    # Ensure X is in a workable format
    if hasattr(adata.X, "toarray"):
        # Sparse matrix — keep as sparse for efficiency
        pass
    else:
        # Dense — convert to float32 for memory efficiency
        adata.X = np.asarray(adata.X, dtype=np.float32)

    adata = compute_qc_metrics(adata)
    adata = filter_cells(adata)
    adata = detect_doublets(adata)
    adata = filter_genes(adata)
    adata = normalize_and_log(adata)
    adata = select_hvg(adata)
    adata = compute_embedding(adata)

    # Record QC provenance
    adata.uns["qc_params"] = {
        "min_genes": QC_PARAMS["min_genes"],
        "max_pct_mito": QC_PARAMS["max_pct_mito"],
        "min_cells": QC_PARAMS["min_cells"],
        "expected_doublet_rate": QC_PARAMS["expected_doublet_rate"],
    }
    adata.uns["source_file"] = input_path.name

    adata.write_h5ad(output_path)
    print(f"  Output: {output_path.name} ({adata.n_obs} cells x {adata.n_vars} genes)")
    return output_path


def run_qc_pipeline(
    input_dir: Path | None = None,
    output_dir: Path | None = None,
) -> list[Path]:
    """Run QC on all h5ad files in the reference directory.

    Returns list of QC-passed output paths.
    """
    if input_dir is None:
        input_dir = REFERENCE_DIR
    if output_dir is None:
        output_dir = QC_DIR

    ensure_dirs()
    h5ad_files = sorted(input_dir.rglob("*.h5ad"))

    if not h5ad_files:
        print(f"  No h5ad files found in {input_dir}")
        print("  Run the download phase first: pipeline download")
        return []

    print(f"Found {len(h5ad_files)} h5ad files for QC")
    results: list[Path] = []
    for path in h5ad_files:
        out = run_qc_single(path, output_dir)
        results.append(out)

    print(f"\n=== QC complete: {len(results)} files processed ===")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="snRNA-seq QC and preprocessing")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Input h5ad files (default: scan reference directory)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=QC_DIR,
        help="Output directory for QC-passed files",
    )
    args = parser.parse_args()

    if args.inputs:
        for path in args.inputs:
            run_qc_single(path, args.output_dir)
    else:
        run_qc_pipeline(output_dir=args.output_dir)


if __name__ == "__main__":
    main()
