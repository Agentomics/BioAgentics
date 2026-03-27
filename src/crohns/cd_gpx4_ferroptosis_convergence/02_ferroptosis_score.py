"""Build ferroptosis gene signature scorer and apply to GSE134809 scRNA-seq.

Phase 1, Task 2 of cd-gpx4-ferroptosis-convergence.
Gene set: GPX4, SLC7A11, ACSL4, LPCAT3, ALOX15, TFRC, FTH1, FTL, NFS1.
Uses z-score-based gene set scoring (equivalent to scanpy.tl.score_genes).
Memory-safe: extracts only the 9 target genes via chunked h5py.
"""

import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/crohns/il23-atlas/GSE134809_annotated.h5ad")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FERROPTOSIS_GENES = ["GPX4", "SLC7A11", "ACSL4", "LPCAT3", "ALOX15", "TFRC", "FTH1", "FTL", "NFS1"]


def extract_genes(adata_path: Path, genes: list[str]) -> tuple[pd.DataFrame, np.ndarray]:
    """Extract multiple genes from h5ad using chunked h5py reading.

    Returns obs metadata DataFrame and (n_cells, n_genes) expression matrix.
    """
    adata = ad.read_h5ad(adata_path, backed="r")
    obs_df = adata.obs[["cell_type", "sample", "il23_high"]].copy()
    var_names = list(adata.var_names)
    n_cells = adata.n_obs
    adata.file.close()

    gene_indices = {g: var_names.index(g) for g in genes}
    gene_idx_set = set(gene_indices.values())

    expr_matrix = np.zeros((n_cells, len(genes)), dtype=np.float32)
    gene_col_map = {idx: col for col, (g, idx) in enumerate(gene_indices.items())}

    chunk_size = 5000
    with h5py.File(adata_path, "r") as f:
        indptr = f["X"]["indptr"][:]
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            data_start = int(indptr[chunk_start])
            data_end = int(indptr[chunk_end])
            chunk_indices = f["X"]["indices"][data_start:data_end]
            chunk_data = f["X"]["data"][data_start:data_end]
            for i in range(chunk_start, chunk_end):
                rs = int(indptr[i]) - data_start
                re = int(indptr[i + 1]) - data_start
                row_idx = chunk_indices[rs:re]
                row_data = chunk_data[rs:re]
                for gi in gene_idx_set:
                    mask = row_idx == gi
                    if mask.any():
                        expr_matrix[i, gene_col_map[gi]] = row_data[mask][0]

    return obs_df, expr_matrix


def score_gene_set(expr_matrix: np.ndarray) -> np.ndarray:
    """Score each cell using mean z-score across the gene set.

    Equivalent to scanpy.tl.score_genes without requiring full adata in memory.
    Each gene is z-scored across all cells, then the mean z-score is taken per cell.
    """
    z_matrix = np.zeros_like(expr_matrix)
    for j in range(expr_matrix.shape[1]):
        col = expr_matrix[:, j]
        mu = col.mean()
        sd = col.std()
        if sd > 0:
            z_matrix[:, j] = (col - mu) / sd
        else:
            z_matrix[:, j] = 0.0
    return z_matrix.mean(axis=1)


def per_celltype_summary(obs_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Mean ferroptosis score per cell type."""
    df = obs_df.copy()
    df["ferroptosis_score"] = scores
    rows = []
    for ct, grp in df.groupby("cell_type"):
        rows.append({
            "cell_type": ct,
            "n_cells": len(grp),
            "mean_score": grp["ferroptosis_score"].mean(),
            "median_score": grp["ferroptosis_score"].median(),
            "std_score": grp["ferroptosis_score"].std(),
        })
    return pd.DataFrame(rows).sort_values("mean_score", ascending=False).reset_index(drop=True)


def il23_ferroptosis_comparison(obs_df: pd.DataFrame, scores: np.ndarray) -> pd.DataFrame:
    """Test IL-23-high vs IL-23-low ferroptosis score per cell type."""
    df = obs_df.copy()
    df["ferroptosis_score"] = scores
    rows = []
    for ct, grp in df.groupby("cell_type"):
        high = grp.loc[grp["il23_high"] == "high", "ferroptosis_score"]
        low = grp.loc[grp["il23_high"] == "low", "ferroptosis_score"]
        if len(high) < 5 or len(low) < 5:
            continue
        stat, pval = sp_stats.mannwhitneyu(high, low, alternative="two-sided")
        pooled_std = np.sqrt(
            ((len(high) - 1) * high.std() ** 2 + (len(low) - 1) * low.std() ** 2)
            / (len(high) + len(low) - 2)
        )
        cohens_d = (high.mean() - low.mean()) / pooled_std if pooled_std > 0 else 0.0
        rows.append({
            "cell_type": ct,
            "n_high": len(high),
            "n_low": len(low),
            "mean_high": high.mean(),
            "mean_low": low.mean(),
            "delta": high.mean() - low.mean(),
            "cohens_d": cohens_d,
            "pvalue": pval,
        })
    result = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)
    n = len(result)
    if n > 0:
        sorted_p = result["pvalue"].sort_values()
        fdr = sorted_p * n / (np.arange(1, n + 1))
        fdr = fdr.clip(upper=1.0)
        fdr_vals = fdr.values.copy()
        for i in range(len(fdr_vals) - 2, -1, -1):
            fdr_vals[i] = min(fdr_vals[i], fdr_vals[i + 1])
        result.loc[sorted_p.index, "FDR"] = fdr_vals
    return result


def main():
    print("Extracting ferroptosis gene set from GSE134809...")
    print(f"  Genes: {', '.join(FERROPTOSIS_GENES)}")
    obs_df, expr_matrix = extract_genes(DATA_PATH, FERROPTOSIS_GENES)
    print(f"  Extracted {expr_matrix.shape[1]} genes for {expr_matrix.shape[0]} cells")

    # Score
    scores = score_gene_set(expr_matrix)
    print(f"  Score range: [{scores.min():.3f}, {scores.max():.3f}]")

    # Per-gene expression summary
    gene_summary = pd.DataFrame({
        "gene": FERROPTOSIS_GENES,
        "mean_expr": expr_matrix.mean(axis=0),
        "pct_expressing": (expr_matrix > 0).mean(axis=0) * 100,
    })
    gene_summary.to_csv(OUTPUT_DIR / "ferroptosis_gene_summary.csv", index=False)
    print("\n--- Per-gene expression summary ---")
    print(gene_summary.to_string(index=False))

    # Per-cell-type scores
    ct_summary = per_celltype_summary(obs_df, scores)
    ct_summary.to_csv(OUTPUT_DIR / "ferroptosis_score_per_celltype.csv", index=False)
    print("\n--- Ferroptosis score per cell type ---")
    print(ct_summary.to_string(index=False))

    # IL-23 comparison
    il23_df = il23_ferroptosis_comparison(obs_df, scores)
    il23_df.to_csv(OUTPUT_DIR / "ferroptosis_score_il23_comparison.csv", index=False)
    print("\n--- Ferroptosis score: IL-23-high vs IL-23-low ---")
    print(il23_df.to_string(index=False))

    # Highlight significant cell types
    sig = il23_df[il23_df["FDR"] < 0.05]
    if len(sig) > 0:
        print(f"\n--- Significant (FDR < 0.05): {len(sig)} cell types ---")
        print(sig[["cell_type", "mean_high", "mean_low", "cohens_d", "FDR"]].to_string(index=False))
    else:
        print("\nNo cell types reach FDR < 0.05 for ferroptosis score.")

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
