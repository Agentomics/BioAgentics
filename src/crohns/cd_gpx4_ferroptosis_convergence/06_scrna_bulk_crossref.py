"""Phase 2 Task 6: Cross-reference scRNA-seq and bulk immune profiles.

Tests whether samples with low epithelial GPX4 in GSE134809 scRNA-seq show
an innate-dominant, Th17-absent profile matching anti-TNF non-responders.
Uses per-sample immune cell proportions and gene module scoring.
"""

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore", category=FutureWarning)

DATA_PATH = Path("data/crohns/il23-atlas/GSE134809_annotated.h5ad")
OUTPUT_DIR = Path("output/crohns/cd-gpx4-ferroptosis-convergence")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# The non-responder profile from anti-TNF literature:
# innate-dominant (high macrophage/monocyte/neutrophil), Th17-absent
INNATE_TYPES = ["Macrophage", "Inflammatory_Mac", "Monocyte", "Dendritic_cell"]
TH17_TYPES = ["Th17"]
ADAPTIVE_TYPES = ["Th1", "Th17", "Treg", "CD4_T_naive", "CD8_T"]


def main():
    print("Loading GSE134809 metadata (backed mode)...")
    adata = ad.read_h5ad(DATA_PATH, backed="r")
    try:
        obs = adata.obs[["cell_type", "sample", "il23_high"]].copy()
    finally:
        adata.file.close()

    print(f"  {len(obs)} cells, {obs['sample'].nunique()} samples")

    # Load per-sample epithelial GPX4 from our pseudo-bulk data
    pb = pd.read_csv(OUTPUT_DIR / "cross_compartment_pseudobulk.csv", index_col=0)
    if "epi_GPX4" not in pb.columns:
        print("ERROR: epi_GPX4 not found in pseudo-bulk data")
        return

    # Compute per-sample cell type proportions
    obs["sample"] = obs["sample"].astype(int)
    ct_counts = obs.groupby(["sample", "cell_type"]).size().unstack(fill_value=0)
    ct_props = ct_counts.div(ct_counts.sum(axis=1), axis=0)

    # Innate and Th17 proportions
    innate_cols = [c for c in INNATE_TYPES if c in ct_props.columns]
    th17_cols = [c for c in TH17_TYPES if c in ct_props.columns]
    adaptive_cols = [c for c in ADAPTIVE_TYPES if c in ct_props.columns]

    ct_props["innate_proportion"] = ct_props[innate_cols].sum(axis=1)
    ct_props["th17_proportion"] = ct_props[th17_cols].sum(axis=1)
    ct_props["adaptive_proportion"] = ct_props[adaptive_cols].sum(axis=1)
    ct_props["innate_adaptive_ratio"] = ct_props["innate_proportion"] / (ct_props["adaptive_proportion"] + 1e-6)

    # Merge with epithelial GPX4
    merged = pb[["epi_GPX4", "il23_high"]].join(
        ct_props[["innate_proportion", "th17_proportion", "adaptive_proportion", "innate_adaptive_ratio"]],
        how="inner"
    )
    merged = merged.dropna()
    print(f"\n  Merged {len(merged)} samples")

    # Stratify by GPX4: low vs high (median split)
    gpx4_median = merged["epi_GPX4"].median()
    merged["gpx4_group"] = np.where(merged["epi_GPX4"] < gpx4_median, "low", "high")

    print(f"\n  GPX4 median: {gpx4_median:.4f}")
    print(f"  Low GPX4: {(merged['gpx4_group']=='low').sum()} samples")
    print(f"  High GPX4: {(merged['gpx4_group']=='high').sum()} samples")

    # Compare immune profiles: low-GPX4 vs high-GPX4
    print("\n--- Immune Profile: Low vs High Epithelial GPX4 ---")
    comparisons = []
    for col in ["innate_proportion", "th17_proportion", "adaptive_proportion", "innate_adaptive_ratio"]:
        low_vals = merged.loc[merged["gpx4_group"] == "low", col]
        high_vals = merged.loc[merged["gpx4_group"] == "high", col]
        stat, pval = stats.mannwhitneyu(low_vals, high_vals, alternative="two-sided")
        pooled_std = np.sqrt(
            ((len(low_vals) - 1) * low_vals.std() ** 2 + (len(high_vals) - 1) * high_vals.std() ** 2)
            / (len(low_vals) + len(high_vals) - 2)
        )
        d = (low_vals.mean() - high_vals.mean()) / pooled_std if pooled_std > 0 else 0
        comparisons.append({
            "metric": col,
            "low_gpx4_mean": low_vals.mean(),
            "high_gpx4_mean": high_vals.mean(),
            "cohens_d": d,
            "pvalue": pval,
        })
        marker = "*" if pval < 0.05 else ""
        print(f"  {col}: low={low_vals.mean():.4f}, high={high_vals.mean():.4f}, d={d:.3f}, p={pval:.4f} {marker}")

    comp_df = pd.DataFrame(comparisons)
    comp_df.to_csv(OUTPUT_DIR / "gpx4_immune_profile_comparison.csv", index=False)

    # Continuous correlations: epithelial GPX4 vs immune proportions
    print("\n--- Spearman Correlations: epi_GPX4 vs immune proportions ---")
    corr_rows = []
    for col in ["innate_proportion", "th17_proportion", "adaptive_proportion", "innate_adaptive_ratio"]:
        rho, pval = stats.spearmanr(merged["epi_GPX4"], merged[col])
        corr_rows.append({
            "metric": col,
            "spearman_rho": rho,
            "pvalue": pval,
            "consistent_with_hypothesis": (
                (col == "innate_proportion" and rho < 0)
                or (col == "th17_proportion" and rho > 0)
                or (col == "innate_adaptive_ratio" and rho < 0)
            ),
        })
        marker = "*" if pval < 0.05 else ""
        print(f"  epi_GPX4 vs {col}: rho={rho:.3f}, p={pval:.4f} {marker}")

    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(OUTPUT_DIR / "gpx4_immune_correlations.csv", index=False)

    # Summary
    print("\n=== SUMMARY ===")
    n_sig_comp = (comp_df["pvalue"] < 0.05).sum()
    n_sig_corr = (corr_df["pvalue"] < 0.05).sum()
    n_consistent = corr_df["consistent_with_hypothesis"].sum()
    print(f"Significant group comparisons: {n_sig_comp}/{len(comp_df)}")
    print(f"Significant correlations: {n_sig_corr}/{len(corr_df)}")
    print(f"Correlations consistent with innate-dominant/Th17-absent hypothesis: {n_consistent}/{len(corr_df)}")

    merged.to_csv(OUTPUT_DIR / "gpx4_immune_profile_merged.csv")


if __name__ == "__main__":
    main()
