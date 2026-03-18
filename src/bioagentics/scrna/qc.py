"""scRNA-seq quality control: cell/gene filtering and doublet removal.

Implements the QC pipeline for the IL-23/Th17 atlas:
1. Cell filtering: remove cells with <500 detected genes or >20% mito reads
2. Gene filtering: remove genes expressed in <10 cells
3. Doublet detection and removal using Scrublet
4. QC summary statistics

Usage:
    from bioagentics.scrna.qc import run_qc
    adata_filtered, qc_stats = run_qc(adata)
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import anndata as ad
import numpy as np
import scanpy as sc
import scrublet as scr
import scipy.sparse as sp


@dataclass
class QCStats:
    """QC summary statistics."""

    cells_before: int
    cells_after: int
    cells_removed: int
    pct_cells_removed: float
    genes_before: int
    genes_after: int
    genes_removed: int
    doublets_detected: int
    pct_doublets: float
    median_genes_per_cell: float
    median_counts_per_cell: float
    median_mito_pct: float

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        return (
            f"QC Summary:\n"
            f"  Cells: {self.cells_before:,} -> {self.cells_after:,} "
            f"(removed {self.cells_removed:,}, {self.pct_cells_removed:.1f}%)\n"
            f"  Genes: {self.genes_before:,} -> {self.genes_after:,} "
            f"(removed {self.genes_removed:,})\n"
            f"  Doublets: {self.doublets_detected:,} ({self.pct_doublets:.1f}%)\n"
            f"  Median genes/cell: {self.median_genes_per_cell:.0f}\n"
            f"  Median counts/cell: {self.median_counts_per_cell:.0f}\n"
            f"  Median mito%: {self.median_mito_pct:.2f}%"
        )


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Compute standard QC metrics: n_genes, n_counts, pct_mito."""
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    return adata


def filter_cells(
    adata: ad.AnnData,
    min_genes: int = 500,
    max_pct_mito: float = 20.0,
) -> ad.AnnData:
    """Filter cells by minimum gene count and maximum mitochondrial percentage."""
    n_before = adata.n_obs

    # Filter by min genes
    sc.pp.filter_cells(adata, min_genes=min_genes)

    # Filter by max mito percentage
    if "pct_counts_mt" in adata.obs.columns:
        adata = adata[adata.obs["pct_counts_mt"] < max_pct_mito].copy()

    print(f"  Cell filtering: {n_before:,} -> {adata.n_obs:,} cells")
    return adata


def filter_genes(adata: ad.AnnData, min_cells: int = 10) -> ad.AnnData:
    """Filter genes expressed in fewer than min_cells cells."""
    n_before = adata.n_vars
    sc.pp.filter_genes(adata, min_cells=min_cells)
    print(f"  Gene filtering: {n_before:,} -> {adata.n_vars:,} genes")
    return adata


def detect_doublets(
    adata: ad.AnnData,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Detect doublets using Scrublet.

    Returns (predicted_doublets, doublet_scores) arrays.
    """
    X = adata.X
    if sp.issparse(X):
        X = np.asarray(X.todense())

    scrub = scr.Scrublet(X, expected_doublet_rate=expected_doublet_rate, random_state=random_state)
    min_gene_variability_pctl = 85
    # Estimate genes remaining after scrublet's variability filter
    n_variable_genes = max(1, int(X.shape[1] * (100 - min_gene_variability_pctl) / 100))
    n_prin_comps = min(30, n_variable_genes - 1, X.shape[0] - 1)
    doublet_scores, predicted_doublets = scrub.scrub_doublets(
        min_counts=2, min_cells=3, min_gene_variability_pctl=min_gene_variability_pctl,
        n_prin_comps=n_prin_comps,
    )

    # scrublet returns predicted_doublets=None when it cannot find a threshold
    if predicted_doublets is None:
        predicted_doublets = scrub.call_doublets(threshold=0.25)

    return predicted_doublets, doublet_scores


def remove_doublets(
    adata: ad.AnnData,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> ad.AnnData:
    """Detect and remove doublets using Scrublet."""
    n_before = adata.n_obs

    predicted_doublets, doublet_scores = detect_doublets(
        adata, expected_doublet_rate=expected_doublet_rate, random_state=random_state
    )

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets

    n_doublets = int(predicted_doublets.sum())
    adata = adata[~adata.obs["predicted_doublet"]].copy()

    print(f"  Doublet removal: {n_before:,} -> {adata.n_obs:,} cells ({n_doublets} doublets)")
    return adata


def run_qc(
    adata: ad.AnnData,
    min_genes: int = 500,
    max_pct_mito: float = 20.0,
    min_cells: int = 10,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> tuple[ad.AnnData, QCStats]:
    """Run the full QC pipeline.

    Steps:
    1. Compute QC metrics (n_genes, n_counts, pct_mito)
    2. Filter cells by min_genes and max_pct_mito
    3. Filter genes by min_cells
    4. Detect and remove doublets with Scrublet

    Returns:
        Tuple of (filtered AnnData, QCStats)
    """
    cells_before = adata.n_obs
    genes_before = adata.n_vars

    print(f"Running QC on {cells_before:,} cells x {genes_before:,} genes...")

    # Ensure we work with raw counts
    if sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # Step 1: Compute QC metrics
    adata = compute_qc_metrics(adata)

    # Step 2: Filter cells
    adata = filter_cells(adata, min_genes=min_genes, max_pct_mito=max_pct_mito)

    # Step 3: Filter genes
    adata = filter_genes(adata, min_cells=min_cells)

    # Step 4: Detect and remove doublets
    if adata.n_obs > 100:  # Scrublet needs enough cells
        adata = remove_doublets(
            adata,
            expected_doublet_rate=expected_doublet_rate,
            random_state=random_state,
        )
    else:
        print("  Skipping doublet detection (too few cells)")

    # Recompute after all filtering
    stats = QCStats(
        cells_before=cells_before,
        cells_after=adata.n_obs,
        cells_removed=cells_before - adata.n_obs,
        pct_cells_removed=(cells_before - adata.n_obs) / cells_before * 100 if cells_before > 0 else 0,
        genes_before=genes_before,
        genes_after=adata.n_vars,
        genes_removed=genes_before - adata.n_vars,
        doublets_detected=cells_before - adata.n_obs,
        pct_doublets=(cells_before - adata.n_obs) / cells_before * 100 if cells_before > 0 else 0,
        median_genes_per_cell=float(np.median(np.asarray(adata.obs["n_genes_by_counts"]))) if "n_genes_by_counts" in adata.obs.columns else 0,
        median_counts_per_cell=float(np.median(np.asarray(adata.obs["total_counts"]))) if "total_counts" in adata.obs.columns else 0,
        median_mito_pct=float(np.median(np.asarray(adata.obs["pct_counts_mt"]))) if "pct_counts_mt" in adata.obs.columns else 0,
    )

    print(f"\n{stats.summary()}")
    return adata, stats
