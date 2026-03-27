"""Extract GPX4 expression from GSE134809 scRNA-seq and compare across cell types and IL-23 conditions.

Phase 1, Task 1 of cd-gpx4-ferroptosis-convergence.
Loads the annotated h5ad in backed mode (memory-safe for 8GB machine).
Outputs per-cell-type GPX4 stats and IL-23-high vs IL-23-low comparisons.
"""

import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse, stats

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/crohns/il23-atlas/GSE134809_annotated.h5ad")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FOCUS_TYPES = ["Epithelial", "ILC3", "Inflammatory_Mac"]


def extract_gene_from_h5ad(adata_path: Path, gene: str) -> pd.DataFrame:
    """Extract a single gene's expression plus metadata from h5ad using h5py.

    Reads the CSR sparse matrix directly to avoid loading the full dataset.
    Memory-safe for 8GB machine.
    """
    # Read obs metadata via anndata (backed — no X loaded)
    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        df = adata.obs[["cell_type", "sample", "il23_high"]].copy()
        var_names = list(adata.var_names)
        n_cells = adata.n_obs
    finally:
        adata.file.close()

    gene_idx = var_names.index(gene)

    # Read CSR sparse matrix via h5py in chunks (memory-safe)
    gene_expr = np.zeros(n_cells, dtype=np.float32)
    chunk_size = 5000
    with h5py.File(adata_path, "r") as f:
        indptr = f["X"]["indptr"][:]  # small: (n_cells+1,) int64
        for chunk_start in range(0, n_cells, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_cells)
            data_start = int(indptr[chunk_start])
            data_end = int(indptr[chunk_end])
            chunk_indices = f["X"]["indices"][data_start:data_end]
            chunk_data = f["X"]["data"][data_start:data_end]
            for i in range(chunk_start, chunk_end):
                rs = int(indptr[i]) - data_start
                re = int(indptr[i + 1]) - data_start
                mask = chunk_indices[rs:re] == gene_idx
                if mask.any():
                    gene_expr[i] = chunk_data[rs:re][mask][0]

    df[gene] = gene_expr
    return df


def per_celltype_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute GPX4 stats per cell type."""
    rows = []
    for ct, grp in df.groupby("cell_type"):
        rows.append({
            "cell_type": ct,
            "n_cells": len(grp),
            "mean_GPX4": grp["GPX4"].mean(),
            "median_GPX4": grp["GPX4"].median(),
            "std_GPX4": grp["GPX4"].std(),
            "pct_expressing": (grp["GPX4"] > 0).mean() * 100,
        })
    return pd.DataFrame(rows).sort_values("mean_GPX4", ascending=False).reset_index(drop=True)


def il23_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """Compare GPX4 between IL-23-high and IL-23-low per cell type."""
    rows = []
    for ct, grp in df.groupby("cell_type"):
        high = grp.loc[grp["il23_high"] == "high", "GPX4"]
        low = grp.loc[grp["il23_high"] == "low", "GPX4"]

        if len(high) < 5 or len(low) < 5:
            continue

        stat, pval = stats.mannwhitneyu(high, low, alternative="two-sided")

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
            "log2FC": np.log2((high.mean() + 1e-6) / (low.mean() + 1e-6)),
            "cohens_d": cohens_d,
            "mann_whitney_U": stat,
            "pvalue": pval,
            "focus": ct in FOCUS_TYPES,
        })

    result = pd.DataFrame(rows).sort_values("pvalue").reset_index(drop=True)

    # BH FDR correction
    n = len(result)
    if n > 0:
        sorted_p = result["pvalue"].sort_values()
        fdr = sorted_p * n / (np.arange(1, n + 1))
        fdr = fdr.clip(upper=1.0)
        # Enforce monotonicity
        fdr_vals = fdr.values.copy()
        for i in range(len(fdr_vals) - 2, -1, -1):
            fdr_vals[i] = min(fdr_vals[i], fdr_vals[i + 1])
        result.loc[sorted_p.index, "FDR"] = fdr_vals

    return result


def main():
    print("Loading GSE134809 annotated h5ad (backed mode)...")
    df = extract_gene_from_h5ad(DATA_PATH, "GPX4")
    print(f"  Extracted GPX4 for {len(df)} cells, {df['cell_type'].nunique()} cell types")

    # Per-cell-type stats
    ct_stats = per_celltype_stats(df)
    ct_stats.to_csv(OUTPUT_DIR / "gpx4_per_celltype_stats.csv", index=False)
    print("\n--- GPX4 expression per cell type ---")
    print(ct_stats.to_string(index=False))

    # IL-23 high vs low comparison
    il23_df = il23_comparison(df)
    il23_df.to_csv(OUTPUT_DIR / "gpx4_il23_comparison.csv", index=False)
    print("\n--- GPX4: IL-23-high vs IL-23-low per cell type ---")
    print(il23_df.to_string(index=False))

    # Focus types summary
    focus = il23_df[il23_df["focus"]].copy()
    if len(focus) > 0:
        print("\n--- FOCUS TYPES (Epithelial, ILC3, Inflammatory_Mac) ---")
        print(focus[["cell_type", "mean_high", "mean_low", "log2FC", "cohens_d", "pvalue", "FDR"]].to_string(index=False))

    print(f"\nOutputs saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
