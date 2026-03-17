"""scRNA-seq QC pipeline for IVIG mechanism single-cell analysis.

Implements quality control for Han VX et al. PANS IVIG scRNA-seq dataset
(144,470 cells, 5 PANS + 4 controls, pre/post-IVIG timepoints).

Pipeline steps:
1. Cell filtering: nFeature_RNA, nCount_RNA, mitochondrial %
2. Doublet detection and removal (Scrublet)
3. Ambient RNA estimation and correction
4. Output clean AnnData with QC metrics

Usage:
    from bioagentics.pandas_pans.ivig_scrna_qc import run_ivig_qc
    adata_clean, stats = run_ivig_qc(adata)

    # Or run as CLI:
    uv run python -m bioagentics.pandas_pans.ivig_scrna_qc path/to/raw.h5ad
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

import anndata as ad
import numpy as np
import scanpy as sc
import scipy.sparse as sp

from bioagentics.config import DATA_DIR

OUTPUT_DIR = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis"


@dataclass
class QCThresholds:
    """Configurable QC thresholds for PBMC scRNA-seq."""

    min_genes: int = 200
    max_genes: int = 5000
    min_counts: int = 500
    max_counts: int = 50000
    max_pct_mito: float = 15.0
    min_cells_per_gene: int = 10
    expected_doublet_rate: float = 0.06
    ambient_correction: bool = True
    random_state: int = 0


@dataclass
class QCStats:
    """QC summary statistics per step."""

    cells_input: int = 0
    genes_input: int = 0
    cells_after_filter: int = 0
    genes_after_filter: int = 0
    cells_low_genes: int = 0
    cells_high_genes: int = 0
    cells_low_counts: int = 0
    cells_high_counts: int = 0
    cells_high_mito: int = 0
    doublets_detected: int = 0
    cells_after_doublets: int = 0
    ambient_corrected: bool = False
    cells_final: int = 0
    genes_final: int = 0
    median_genes_per_cell: float = 0.0
    median_counts_per_cell: float = 0.0
    median_mito_pct: float = 0.0
    per_sample_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    def summary(self) -> str:
        lines = [
            "IVIG scRNA-seq QC Summary:",
            f"  Input: {self.cells_input:,} cells x {self.genes_input:,} genes",
            f"  Cell filtering:",
            f"    Low genes (<min): {self.cells_low_genes:,}",
            f"    High genes (>max): {self.cells_high_genes:,}",
            f"    Low counts (<min): {self.cells_low_counts:,}",
            f"    High counts (>max): {self.cells_high_counts:,}",
            f"    High mito%: {self.cells_high_mito:,}",
            f"    After filtering: {self.cells_after_filter:,} cells",
            f"  Gene filtering: {self.genes_input:,} -> {self.genes_after_filter:,} genes",
            f"  Doublets: {self.doublets_detected:,} detected, {self.cells_after_doublets:,} cells remain",
            f"  Ambient RNA corrected: {self.ambient_corrected}",
            f"  Final: {self.cells_final:,} cells x {self.genes_final:,} genes",
            f"  Median genes/cell: {self.median_genes_per_cell:.0f}",
            f"  Median counts/cell: {self.median_counts_per_cell:.0f}",
            f"  Median mito%: {self.median_mito_pct:.2f}%",
        ]
        if self.per_sample_counts:
            lines.append("  Per-sample cell counts:")
            for sample, count in sorted(self.per_sample_counts.items()):
                lines.append(f"    {sample}: {count:,}")
        return "\n".join(lines)


def compute_qc_metrics(adata: ad.AnnData) -> ad.AnnData:
    """Compute QC metrics: n_genes_by_counts, total_counts, pct_counts_mt."""
    adata.var["mt"] = adata.var_names.str.upper().str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )
    return adata


def filter_cells_by_qc(
    adata: ad.AnnData, thresholds: QCThresholds
) -> tuple[ad.AnnData, dict[str, int]]:
    """Filter cells by gene count, UMI count, and mitochondrial percentage.

    Returns filtered AnnData and dict of removal counts per criterion.
    """
    n_before = adata.n_obs
    removal_counts: dict[str, int] = {}

    # Tag cells failing each criterion (before any removal, for accurate counts)
    low_genes = adata.obs["n_genes_by_counts"] < thresholds.min_genes
    high_genes = adata.obs["n_genes_by_counts"] > thresholds.max_genes
    low_counts = adata.obs["total_counts"] < thresholds.min_counts
    high_counts = adata.obs["total_counts"] > thresholds.max_counts
    high_mito = adata.obs["pct_counts_mt"] > thresholds.max_pct_mito

    removal_counts["low_genes"] = int(low_genes.sum())
    removal_counts["high_genes"] = int(high_genes.sum())
    removal_counts["low_counts"] = int(low_counts.sum())
    removal_counts["high_counts"] = int(high_counts.sum())
    removal_counts["high_mito"] = int(high_mito.sum())

    keep = ~(low_genes | high_genes | low_counts | high_counts | high_mito)
    adata = adata[keep].copy()

    print(f"  Cell filtering: {n_before:,} -> {adata.n_obs:,} cells")
    return adata, removal_counts


def filter_genes(adata: ad.AnnData, min_cells: int = 10) -> ad.AnnData:
    """Filter genes detected in fewer than min_cells."""
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

    Returns (predicted_doublets, doublet_scores) boolean and float arrays.
    """
    import scrublet as scr

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()

    scrub = scr.Scrublet(
        X, expected_doublet_rate=expected_doublet_rate, random_state=random_state
    )
    n_prin_comps = min(30, X.shape[1] - 1)
    if n_prin_comps < 2:
        return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0], dtype=float)
    try:
        doublet_scores, predicted_doublets = scrub.scrub_doublets(
            min_counts=2, min_cells=3, min_gene_variability_pctl=85,
            n_prin_comps=n_prin_comps,
        )
    except ValueError:
        # Scrublet internally filters to variable genes which may reduce
        # dimensions below n_prin_comps; fall back with fewer components
        for n_pc in (20, 15, 10, 5):
            try:
                doublet_scores, predicted_doublets = scrub.scrub_doublets(
                    min_counts=2, min_cells=3, min_gene_variability_pctl=85,
                    n_prin_comps=n_pc,
                )
                break
            except ValueError:
                continue
        else:
            print("  Warning: doublet detection failed (too few variable genes)")
            return np.zeros(X.shape[0], dtype=bool), np.zeros(X.shape[0], dtype=float)
    return predicted_doublets, doublet_scores


def remove_doublets(
    adata: ad.AnnData,
    expected_doublet_rate: float = 0.06,
    random_state: int = 0,
) -> tuple[ad.AnnData, int]:
    """Detect and remove doublets. Returns (filtered adata, n_doublets)."""
    if adata.n_obs < 100:
        print("  Skipping doublet detection (too few cells)")
        return adata, 0

    predicted_doublets, doublet_scores = detect_doublets(
        adata, expected_doublet_rate=expected_doublet_rate, random_state=random_state
    )

    adata.obs["doublet_score"] = doublet_scores
    adata.obs["predicted_doublet"] = predicted_doublets

    n_doublets = int(predicted_doublets.sum())
    adata = adata[~adata.obs["predicted_doublet"]].copy()

    print(
        f"  Doublet removal: {n_doublets:,} doublets removed, "
        f"{adata.n_obs:,} cells remain"
    )
    return adata, n_doublets


def estimate_ambient_profile(adata: ad.AnnData) -> np.ndarray | None:
    """Estimate ambient RNA profile from low-count droplets.

    Uses droplets with very low total counts (bottom 10th percentile)
    as proxy for empty droplets to estimate the ambient RNA profile.
    Returns normalized gene expression vector representing ambient contamination,
    or None if estimation is not possible.
    """
    total_counts = np.asarray(adata.obs["total_counts"])
    threshold = np.percentile(total_counts, 10)
    empty_mask = total_counts <= threshold

    if empty_mask.sum() < 50:
        return None

    X_empty = adata[empty_mask].X
    if sp.issparse(X_empty):
        X_empty = X_empty.toarray()

    ambient = np.asarray(X_empty.sum(axis=0)).ravel()
    total = ambient.sum()
    if total == 0:
        return None

    return ambient / total


def correct_ambient_rna(
    adata: ad.AnnData, contamination_fraction: float = 0.1
) -> ad.AnnData:
    """Subtract estimated ambient RNA contamination from counts.

    Simplified SoupX-inspired approach:
    - Estimate ambient profile from low-count droplets
    - Subtract fraction * total_counts * ambient_profile from each cell
    - Floor at zero (counts can't be negative)
    """
    ambient_profile = estimate_ambient_profile(adata)
    if ambient_profile is None:
        print("  Ambient RNA: could not estimate profile, skipping correction")
        return adata

    print(
        f"  Ambient RNA correction: contamination_fraction={contamination_fraction}"
    )

    X = adata.X
    is_sparse = sp.issparse(X)
    if is_sparse:
        X = X.toarray()

    total_counts = np.asarray(X.sum(axis=1)).ravel()
    ambient_contribution = (
        contamination_fraction * total_counts[:, np.newaxis] * ambient_profile[np.newaxis, :]
    )
    X_corrected = np.maximum(X - ambient_contribution, 0)

    if is_sparse:
        adata.X = sp.csr_matrix(X_corrected)
    else:
        adata.X = X_corrected

    adata.uns["ambient_correction"] = {
        "method": "soupx_simplified",
        "contamination_fraction": contamination_fraction,
        "top_ambient_genes": list(
            adata.var_names[np.argsort(ambient_profile)[-20:]]
        ),
    }
    return adata


def run_ivig_qc(
    adata: ad.AnnData,
    thresholds: QCThresholds | None = None,
    sample_key: str | None = None,
) -> tuple[ad.AnnData, QCStats]:
    """Run full QC pipeline for IVIG scRNA-seq data.

    Args:
        adata: Raw count matrix as AnnData.
        thresholds: QC thresholds (uses PBMC-tuned defaults if None).
        sample_key: Column in adata.obs identifying samples (for per-sample stats).

    Returns:
        Tuple of (filtered AnnData, QCStats).
    """
    if thresholds is None:
        thresholds = QCThresholds()

    stats = QCStats(cells_input=adata.n_obs, genes_input=adata.n_vars)

    print(f"Running IVIG scRNA-seq QC on {adata.n_obs:,} cells x {adata.n_vars:,} genes")

    # Ensure sparse CSR format
    if sp.issparse(adata.X):
        adata.X = sp.csr_matrix(adata.X)

    # Store raw counts
    adata.layers["raw_counts"] = adata.X.copy()

    # Step 1: Compute QC metrics
    adata = compute_qc_metrics(adata)

    # Step 2: Filter cells
    adata, removal_counts = filter_cells_by_qc(adata, thresholds)
    stats.cells_low_genes = removal_counts["low_genes"]
    stats.cells_high_genes = removal_counts["high_genes"]
    stats.cells_low_counts = removal_counts["low_counts"]
    stats.cells_high_counts = removal_counts["high_counts"]
    stats.cells_high_mito = removal_counts["high_mito"]
    stats.cells_after_filter = adata.n_obs

    # Step 3: Filter genes
    adata = filter_genes(adata, min_cells=thresholds.min_cells_per_gene)
    stats.genes_after_filter = adata.n_vars

    # Step 4: Ambient RNA correction (before doublet detection)
    if thresholds.ambient_correction:
        adata = correct_ambient_rna(adata)
        stats.ambient_corrected = True

    # Step 5: Doublet removal
    adata, n_doublets = remove_doublets(
        adata,
        expected_doublet_rate=thresholds.expected_doublet_rate,
        random_state=thresholds.random_state,
    )
    stats.doublets_detected = n_doublets
    stats.cells_after_doublets = adata.n_obs

    # Final stats
    stats.cells_final = adata.n_obs
    stats.genes_final = adata.n_vars

    if "n_genes_by_counts" in adata.obs.columns:
        stats.median_genes_per_cell = float(
            np.median(np.asarray(adata.obs["n_genes_by_counts"]))
        )
    if "total_counts" in adata.obs.columns:
        stats.median_counts_per_cell = float(
            np.median(np.asarray(adata.obs["total_counts"]))
        )
    if "pct_counts_mt" in adata.obs.columns:
        stats.median_mito_pct = float(
            np.median(np.asarray(adata.obs["pct_counts_mt"]))
        )

    # Per-sample breakdown
    if sample_key and sample_key in adata.obs.columns:
        stats.per_sample_counts = dict(adata.obs[sample_key].value_counts())

    # Store thresholds in adata
    adata.uns["qc_thresholds"] = asdict(thresholds)
    adata.uns["qc_stats"] = stats.to_dict()

    print(f"\n{stats.summary()}")
    return adata, stats


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for QC pipeline."""
    parser = argparse.ArgumentParser(
        description="Run scRNA-seq QC for IVIG mechanism analysis"
    )
    parser.add_argument("input", type=Path, help="Input h5ad file (raw counts)")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output h5ad path (default: output/qc/<input_stem>_qc.h5ad)",
    )
    parser.add_argument("--min-genes", type=int, default=200)
    parser.add_argument("--max-genes", type=int, default=5000)
    parser.add_argument("--min-counts", type=int, default=500)
    parser.add_argument("--max-counts", type=int, default=50000)
    parser.add_argument("--max-pct-mito", type=float, default=15.0)
    parser.add_argument("--min-cells-per-gene", type=int, default=10)
    parser.add_argument("--doublet-rate", type=float, default=0.06)
    parser.add_argument("--no-ambient", action="store_true", help="Skip ambient RNA correction")
    parser.add_argument("--sample-key", type=str, default=None, help="obs column for sample ID")
    args = parser.parse_args(argv)

    # Load
    print(f"Loading {args.input}...")
    adata = ad.read_h5ad(args.input)

    # Configure thresholds
    thresholds = QCThresholds(
        min_genes=args.min_genes,
        max_genes=args.max_genes,
        min_counts=args.min_counts,
        max_counts=args.max_counts,
        max_pct_mito=args.max_pct_mito,
        min_cells_per_gene=args.min_cells_per_gene,
        expected_doublet_rate=args.doublet_rate,
        ambient_correction=not args.no_ambient,
    )

    # Run QC
    adata_clean, stats = run_ivig_qc(adata, thresholds=thresholds, sample_key=args.sample_key)

    # Save
    output_path = args.output
    if output_path is None:
        output_dir = OUTPUT_DIR / "qc"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{args.input.stem}_qc.h5ad"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    adata_clean.write_h5ad(output_path)
    print(f"\nSaved QC'd data to {output_path}")


if __name__ == "__main__":
    main()
