"""Phase 3: 9p21 co-deletion landscape and deletion-extent classification.

Quantifies co-deletion rates for 9p21 locus genes (CDKN2A, CDKN2B, DMRTA1,
IFNA/IFNB) among MTAP-deleted lines across cancer types. Classifies deletion
extent and tests whether broader deletions create additional vulnerabilities.

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.04_9p21_codeletion
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "pancancer-mtap-prmt5-atlas"

MTAP_CN_THRESHOLD = 0.5

# 9p21 locus genes to analyze
LOCUS_GENES = ["CDKN2A", "CDKN2B", "MTAP", "DMRTA1", "IFNA1", "IFNA2", "IFNB1"]
# Genes used for deletion-extent classification (excluding MTAP itself)
IFN_GENES = ["IFNA1", "IFNA2", "IFNB1"]


def classify_deletion_extent(row: pd.Series) -> str:
    """Classify 9p21 deletion extent for a single cell line.

    Categories:
    - intact: MTAP not deleted
    - CDKN2A+MTAP: focal deletion (MTAP+CDKN2A deleted, DMRTA1/IFN intact)
    - full_9p21: broad deletion (MTAP+CDKN2A+DMRTA1+IFN all deleted)
    """
    if not row.get("MTAP_del", False):
        if row.get("CDKN2A_del", False):
            return "CDKN2A_only"
        return "intact"

    # MTAP is deleted — check extent
    cdkn2a_del = row.get("CDKN2A_del", False)
    dmrta1_del = row.get("DMRTA1_del", False)
    ifn_del = any(row.get(f"{g}_del", False) for g in IFN_GENES)

    if cdkn2a_del and dmrta1_del and ifn_del:
        return "full_9p21"
    return "CDKN2A+MTAP"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "all_cell_lines_classified.csv", index_col=0)

    print("Loading copy number data for 9p21 locus genes...")
    cn = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")

    # Check which genes are available
    available = [g for g in LOCUS_GENES if g in cn.columns]
    missing = [g for g in LOCUS_GENES if g not in cn.columns]
    if missing:
        print(f"  Warning: missing genes in CN data: {missing}")
    print(f"  Available: {available}")

    # Extract CN for locus genes and classify deletion
    for gene in available:
        classified[f"{gene}_CN"] = cn[gene].reindex(classified.index)
        classified[f"{gene}_del"] = classified[f"{gene}_CN"] < MTAP_CN_THRESHOLD

    # Deletion-extent classification
    classified["deletion_extent"] = classified.apply(classify_deletion_extent, axis=1)

    # === Co-deletion rates among MTAP-deleted lines per cancer type ===
    summary = pd.read_csv(OUTPUT_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()

    has_cn = classified.dropna(subset=["MTAP_CN"])
    mtap_deleted = has_cn[has_cn["MTAP_deleted"] == True]

    codeletion_rows = []
    for cancer_type in qualifying:
        ct_deleted = mtap_deleted[mtap_deleted["OncotreeLineage"] == cancer_type]
        n = len(ct_deleted)
        if n == 0:
            continue

        row = {"cancer_type": cancer_type, "N_mtap_deleted": n}
        for gene in available:
            if gene == "MTAP":
                continue
            col = f"{gene}_del"
            if col in ct_deleted.columns:
                rate = ct_deleted[col].sum() / n
                row[f"{gene}_codeletion_rate"] = round(float(rate), 3)
        codeletion_rows.append(row)

    codeletion_df = pd.DataFrame(codeletion_rows)
    codeletion_df.to_csv(OUTPUT_DIR / "9p21_codeletion_rates.csv", index=False)
    print(f"\nSaved co-deletion rates for {len(codeletion_df)} cancer types")

    # === Deletion-extent classification per cell line ===
    extent_cols = ["OncotreeLineage", "deletion_extent"]
    for gene in available:
        extent_cols.extend([f"{gene}_CN", f"{gene}_del"])
    extent_df = classified[extent_cols].copy()
    extent_df.to_csv(OUTPUT_DIR / "deletion_extent_classification.csv")
    print(f"Saved deletion-extent classification for {len(extent_df)} cell lines")

    # === Deletion extent by cancer type ===
    extent_by_ct = []
    for cancer_type in qualifying:
        ct = has_cn[has_cn["OncotreeLineage"] == cancer_type]
        counts = ct["deletion_extent"].value_counts()
        row = {"cancer_type": cancer_type, "N_total": len(ct)}
        for cat in ["intact", "CDKN2A_only", "CDKN2A+MTAP", "full_9p21"]:
            row[cat] = int(counts.get(cat, 0))
        extent_by_ct.append(row)

    extent_by_ct_df = pd.DataFrame(extent_by_ct)
    extent_by_ct_df.to_csv(OUTPUT_DIR / "deletion_extent_by_cancer_type.csv", index=False)
    print(f"Saved deletion-extent counts by cancer type")

    # === Compare PRMT5/MAT2A dependency across deletion-extent groups ===
    print("\nComparing dependency across deletion-extent groups...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    dep_results = {}
    for gene in ["PRMT5", "MAT2A"]:
        if gene not in crispr.columns:
            print(f"  Warning: {gene} not in CRISPR data, skipping")
            continue

        dep = crispr[gene]
        merged = has_cn.join(dep.rename(f"{gene}_dep"), how="inner")
        merged = merged.dropna(subset=[f"{gene}_dep"])

        # Kruskal-Wallis across deletion-extent groups (excluding intact)
        groups_data = {}
        for cat in ["CDKN2A_only", "CDKN2A+MTAP", "full_9p21"]:
            vals = merged[merged["deletion_extent"] == cat][f"{gene}_dep"].values
            if len(vals) >= 3:
                groups_data[cat] = vals

        if len(groups_data) >= 2:
            kw_stat, kw_p = stats.kruskal(*groups_data.values())
            dep_results[gene] = {
                "kruskal_wallis_H": float(kw_stat),
                "kruskal_wallis_p": float(kw_p),
                "group_medians": {k: float(np.median(v)) for k, v in groups_data.items()},
                "group_sizes": {k: len(v) for k, v in groups_data.items()},
            }
            print(f"  {gene}: KW H={kw_stat:.2f}, p={kw_p:.3f}")
            for cat, vals in groups_data.items():
                print(f"    {cat}: median={np.median(vals):.4f}, n={len(vals)}")
        else:
            dep_results[gene] = {"error": "insufficient groups with n>=3"}
            print(f"  {gene}: insufficient groups for comparison")

    # Also compare intact vs all MTAP-deleted for reference
    for gene in ["PRMT5", "MAT2A"]:
        if gene not in crispr.columns:
            continue
        dep = crispr[gene]
        merged = has_cn.join(dep.rename(f"{gene}_dep"), how="inner")
        merged = merged.dropna(subset=[f"{gene}_dep"])

        intact_vals = merged[merged["deletion_extent"] == "intact"][f"{gene}_dep"].values
        focal_vals = merged[merged["deletion_extent"] == "CDKN2A+MTAP"][f"{gene}_dep"].values
        full_vals = merged[merged["deletion_extent"] == "full_9p21"][f"{gene}_dep"].values

        # Test focal vs full 9p21
        if len(focal_vals) >= 3 and len(full_vals) >= 3:
            stat, p = stats.mannwhitneyu(full_vals, focal_vals, alternative="two-sided")
            dep_results[f"{gene}_focal_vs_full"] = {
                "mannwhitney_U": float(stat),
                "mannwhitney_p": float(p),
                "median_focal": float(np.median(focal_vals)),
                "median_full": float(np.median(full_vals)),
                "n_focal": len(focal_vals),
                "n_full": len(full_vals),
            }
            print(f"  {gene} focal vs full: U={stat:.0f}, p={p:.3f}, "
                  f"median focal={np.median(focal_vals):.4f} vs full={np.median(full_vals):.4f}")

    with open(OUTPUT_DIR / "deletion_extent_dependency.json", "w") as f:
        json.dump(dep_results, f, indent=2)
    print("Saved deletion_extent_dependency.json")

    # === Heatmap: co-deletion rates across cancer types ===
    print("\nGenerating visualizations...")
    rate_cols = [c for c in codeletion_df.columns if c.endswith("_codeletion_rate")]
    if rate_cols:
        heatmap_data = codeletion_df.set_index("cancer_type")[rate_cols]
        heatmap_data.columns = [c.replace("_codeletion_rate", "") for c in rate_cols]

        fig, ax = plt.subplots(figsize=(8, max(5, len(heatmap_data) * 0.35)))
        sns.heatmap(heatmap_data, annot=True, fmt=".2f", cmap="YlOrRd",
                    vmin=0, vmax=1, ax=ax, linewidths=0.5)
        ax.set_title("9p21 Co-Deletion Rates Among MTAP-Deleted Lines")
        ax.set_ylabel("")
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / "9p21_codeletion_heatmap.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("  Saved: 9p21_codeletion_heatmap.png")

    # === Boxplot: PRMT5/MAT2A dependency by deletion-extent ===
    for gene in ["PRMT5", "MAT2A"]:
        if gene not in crispr.columns:
            continue
        dep = crispr[gene]
        merged = has_cn.join(dep.rename("dep"), how="inner").dropna(subset=["dep"])

        plot_data = merged[merged["deletion_extent"].isin(
            ["intact", "CDKN2A_only", "CDKN2A+MTAP", "full_9p21"]
        )]

        fig, ax = plt.subplots(figsize=(7, 5))
        order = ["intact", "CDKN2A_only", "CDKN2A+MTAP", "full_9p21"]
        present_cats = [c for c in order if c in plot_data["deletion_extent"].values]
        sns.boxplot(data=plot_data, x="deletion_extent", y="dep",
                    order=present_cats, ax=ax, palette="Set2")
        sns.stripplot(data=plot_data, x="deletion_extent", y="dep",
                      order=present_cats, ax=ax, color="gray", alpha=0.3, size=3)
        ax.set_xlabel("9p21 Deletion Extent")
        ax.set_ylabel(f"{gene} CRISPR Dependency")
        ax.set_title(f"{gene} Dependency by 9p21 Deletion Extent (Pan-Cancer)")
        ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3)
        fig.tight_layout()
        fig.savefig(OUTPUT_DIR / f"deletion_extent_{gene.lower()}_boxplot.png",
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved: deletion_extent_{gene.lower()}_boxplot.png")


if __name__ == "__main__":
    main()
