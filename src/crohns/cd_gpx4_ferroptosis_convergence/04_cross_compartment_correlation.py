"""Phase 2 Task 4: Cross-compartment correlation of GPX4/ferroptosis across cell types.

Correlates epithelial GPX4 with ILC3 ferroptosis score, fibrosis gene module,
and inflammatory gene module at per-sample pseudo-bulk level in GSE134809.

Note: ILC2 cells are not annotated in this dataset. ILC3 is used as the
closest available ILC population.
"""

import warnings
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/crohns/il23-atlas/GSE134809_annotated.h5ad")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Gene modules
FERROPTOSIS_GENES = ["GPX4", "SLC7A11", "ACSL4", "LPCAT3", "ALOX15", "TFRC", "FTH1", "FTL", "NFS1"]
FIBROSIS_GENES = ["ILK", "COL1A1", "ACTA2", "FAP"]
INFLAMMATORY_GENES = ["TNF", "IL1B", "IL6", "S100A8", "S100A9", "CXCL8", "CCL2", "CCL3"]
EXTRA_GENES = ["GSDMC", "ACAD9"]

ALL_GENES = sorted(set(FERROPTOSIS_GENES + FIBROSIS_GENES + INFLAMMATORY_GENES + EXTRA_GENES))

# Cell types for each compartment
EPITHELIAL_TYPES = ["Epithelial"]
ILC_TYPES = ["ILC3"]  # ILC2 not annotated in GSE134809
FIBROSIS_TYPES = ["Fibroblast", "Myofibroblast"]
INFLAMMATORY_TYPES = ["Inflammatory_Mac", "Macrophage", "Monocyte"]

MIN_CELLS_PER_SAMPLE = 3


def extract_genes_from_h5ad(adata_path: Path, genes: list[str]) -> pd.DataFrame:
    """Extract multiple genes' expression plus metadata from h5ad.

    Memory-safe chunked h5py reading for 8GB machine.
    """
    adata = ad.read_h5ad(adata_path, backed="r")
    try:
        df = adata.obs[["cell_type", "sample", "il23_high"]].copy()
        var_names = list(adata.var_names)
        n_cells = adata.n_obs
    finally:
        adata.file.close()

    gene_indices = {}
    for g in genes:
        if g in var_names:
            gene_indices[g] = var_names.index(g)
        else:
            print(f"  WARNING: {g} not found in var_names")

    gene_expr = {g: np.zeros(n_cells, dtype=np.float32) for g in gene_indices}

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
                for g, gidx in gene_indices.items():
                    mask = row_idx == gidx
                    if mask.any():
                        gene_expr[g][i] = row_data[mask][0]

    for g, vals in gene_expr.items():
        df[g] = vals
    return df


def compute_module_score(df: pd.DataFrame, genes: list[str]) -> pd.Series:
    """Z-score-based module score (mean of z-scored genes)."""
    available = [g for g in genes if g in df.columns]
    if not available:
        return pd.Series(np.nan, index=df.index)
    zscores = df[available].apply(lambda x: (x - x.mean()) / (x.std() + 1e-10))
    return zscores.mean(axis=1)


def pseudo_bulk_by_sample(df: pd.DataFrame, cell_types: list[str],
                          value_cols: list[str]) -> pd.DataFrame:
    """Compute per-sample means for given cell types, filtering low-count samples."""
    mask = df["cell_type"].isin(cell_types)
    sub = df.loc[mask].copy()
    counts = sub.groupby("sample").size()
    valid_samples = counts[counts >= MIN_CELLS_PER_SAMPLE].index
    sub = sub[sub["sample"].isin(valid_samples)]
    result = sub.groupby("sample")[value_cols].mean()
    result["n_cells"] = sub.groupby("sample").size()
    result["il23_high"] = sub.groupby("sample")["il23_high"].first()
    return result


def main():
    print("Extracting genes from GSE134809...")
    df = extract_genes_from_h5ad(DATA_PATH, ALL_GENES)
    print(f"  {len(df)} cells, {df['sample'].nunique()} samples\n")

    # Compute module scores at cell level
    df["ferroptosis_score"] = compute_module_score(df, FERROPTOSIS_GENES)
    df["fibrosis_score"] = compute_module_score(df, FIBROSIS_GENES)
    df["inflammatory_score"] = compute_module_score(df, INFLAMMATORY_GENES)

    # Pseudo-bulk per compartment
    epi = pseudo_bulk_by_sample(df, EPITHELIAL_TYPES, ["GPX4", "ferroptosis_score"])
    epi = epi.rename(columns={"GPX4": "epi_GPX4", "ferroptosis_score": "epi_ferroptosis",
                               "n_cells": "epi_n_cells"})

    ilc = pseudo_bulk_by_sample(df, ILC_TYPES, ["ferroptosis_score", "GPX4"])
    ilc = ilc.rename(columns={"ferroptosis_score": "ilc3_ferroptosis", "GPX4": "ilc3_GPX4",
                               "n_cells": "ilc3_n_cells"})

    fib = pseudo_bulk_by_sample(df, FIBROSIS_TYPES, ["fibrosis_score"] + FIBROSIS_GENES)
    fib = fib.rename(columns={"fibrosis_score": "fib_fibrosis_score", "n_cells": "fib_n_cells"})
    for g in FIBROSIS_GENES:
        if g in fib.columns:
            fib = fib.rename(columns={g: f"fib_{g}"})

    inf = pseudo_bulk_by_sample(df, INFLAMMATORY_TYPES, ["inflammatory_score"] + INFLAMMATORY_GENES)
    inf = inf.rename(columns={"inflammatory_score": "inf_inflammatory_score", "n_cells": "inf_n_cells"})
    for g in INFLAMMATORY_GENES:
        if g in inf.columns:
            inf = inf.rename(columns={g: f"inf_{g}"})

    # Merge across compartments (inner join on sample)
    merged = epi[["epi_GPX4", "epi_ferroptosis", "epi_n_cells", "il23_high"]].join(
        ilc[["ilc3_ferroptosis", "ilc3_GPX4", "ilc3_n_cells"]], how="inner"
    ).join(
        fib[["fib_fibrosis_score", "fib_n_cells"] + [f"fib_{g}" for g in FIBROSIS_GENES if f"fib_{g}" in fib.columns]],
        how="inner"
    ).join(
        inf[["inf_inflammatory_score", "inf_n_cells"] + [f"inf_{g}" for g in INFLAMMATORY_GENES if f"inf_{g}" in inf.columns]],
        how="inner"
    )

    print(f"Merged {len(merged)} samples with all compartments represented\n")
    merged.to_csv(OUTPUT_DIR / "cross_compartment_pseudobulk.csv")

    # Pairwise Spearman correlations
    pairs = [
        ("epi_GPX4", "ilc3_ferroptosis", "Epithelial GPX4 vs ILC3 ferroptosis score"),
        ("epi_GPX4", "ilc3_GPX4", "Epithelial GPX4 vs ILC3 GPX4"),
        ("epi_GPX4", "fib_fibrosis_score", "Epithelial GPX4 vs Fibrosis module score"),
        ("epi_GPX4", "inf_inflammatory_score", "Epithelial GPX4 vs Inflammatory module score"),
        ("epi_ferroptosis", "fib_fibrosis_score", "Epithelial ferroptosis vs Fibrosis module"),
        ("epi_ferroptosis", "inf_inflammatory_score", "Epithelial ferroptosis vs Inflammatory module"),
        ("ilc3_ferroptosis", "fib_fibrosis_score", "ILC3 ferroptosis vs Fibrosis module"),
        ("ilc3_ferroptosis", "inf_inflammatory_score", "ILC3 ferroptosis vs Inflammatory module"),
        ("epi_GPX4", "fib_COL1A1", "Epithelial GPX4 vs Fibroblast COL1A1"),
        ("epi_GPX4", "fib_FAP", "Epithelial GPX4 vs Fibroblast FAP"),
        ("epi_GPX4", "inf_TNF", "Epithelial GPX4 vs Inflammatory TNF"),
        ("epi_GPX4", "inf_IL1B", "Epithelial GPX4 vs Inflammatory IL1B"),
    ]

    corr_rows = []
    for x_col, y_col, desc in pairs:
        if x_col not in merged.columns or y_col not in merged.columns:
            print(f"  SKIP: {desc} — column missing")
            continue
        valid = merged[[x_col, y_col]].dropna()
        if len(valid) < 5:
            print(f"  SKIP: {desc} — too few samples ({len(valid)})")
            continue
        rho, pval = stats.spearmanr(valid[x_col], valid[y_col])
        corr_rows.append({
            "comparison": desc,
            "x": x_col,
            "y": y_col,
            "n_samples": len(valid),
            "spearman_rho": rho,
            "pvalue": pval,
            "significant": pval < 0.05,
            "meets_criterion": abs(rho) > 0.3 and pval < 0.05,
        })
        marker = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
        print(f"  {desc}: rho={rho:.3f}, p={pval:.4f} {marker}")

    corr_df = pd.DataFrame(corr_rows)
    # BH FDR correction
    if len(corr_df) > 0:
        n = len(corr_df)
        sorted_idx = corr_df["pvalue"].sort_values().index
        ranks = range(1, n + 1)
        fdr = corr_df.loc[sorted_idx, "pvalue"].values * n / np.arange(1, n + 1)
        fdr = np.minimum.accumulate(fdr[::-1])[::-1]
        fdr = np.clip(fdr, 0, 1)
        corr_df.loc[sorted_idx, "FDR"] = fdr

    corr_df.to_csv(OUTPUT_DIR / "cross_compartment_correlations.csv", index=False)
    print(f"\nResults saved to {OUTPUT_DIR}/")

    # Summary
    n_sig = (corr_df["meets_criterion"]).sum()
    print(f"\n=== SUMMARY ===")
    print(f"Correlations meeting criterion (|rho|>0.3, p<0.05): {n_sig}/{len(corr_df)}")
    if n_sig > 0:
        print("Meeting criterion:")
        for _, row in corr_df[corr_df["meets_criterion"]].iterrows():
            print(f"  {row['comparison']}: rho={row['spearman_rho']:.3f}, p={row['pvalue']:.4f}, FDR={row['FDR']:.4f}")


if __name__ == "__main__":
    main()
