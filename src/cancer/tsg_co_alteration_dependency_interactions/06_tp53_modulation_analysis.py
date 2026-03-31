"""Phase 4: TP53 modulation analysis across published atlases.

For each published cancer division atlas (PTEN, RB1, KEAP1, NF1), stratifies
per-cancer-type SL effect sizes by TP53 mutation frequency. Tests hypothesis:
cancer types with high TP53 mutation rates show systematically weaker SL effects.

Method:
  1. Compute TP53 mutation frequency per cancer type from TCGA Phase 1a data
  2. For each atlas, collect top SL dependencies and their per-cancer-type Cohen's d
  3. Compute Spearman correlation between TP53 mutation rate and mean |Cohen's d|
  4. Report correlations and generate summary table

Output:
  - phase4_tp53_modulation.csv: per-atlas correlation results
  - phase4_tp53_modulation_detail.csv: per-atlas per-cancer-type data points
  - phase4_tp53_modulation_summary.json: meta-summary

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.06_tp53_modulation_analysis
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"
ATLAS_BASE = REPO_ROOT / "output" / "cancer"

# Minimum cancer types needed for meaningful correlation
MIN_CANCER_TYPES = 5
# Top N SL genes per atlas to use for mean effect size
TOP_SL_N = 50

# TCGA abbreviation -> DepMap OncotreeLineage mapping
TCGA_TO_DEPMAP = {
    "BRCA": "Breast",
    "GBM": "CNS/Brain",
    "LGG": "CNS/Brain",
    "OV": "Ovary/Fallopian Tube",
    "LUAD": "Lung",
    "LUSC": "Lung",
    "UCEC": "Uterus",
    "UCS": "Uterus",
    "KIRC": "Kidney",
    "KIRP": "Kidney",
    "KICH": "Kidney",
    "HNSC": "Head and Neck",
    "THCA": "Thyroid",
    "PRAD": "Prostate",
    "SKCM": "Skin",
    "COAD": "Bowel",
    "READ": "Bowel",
    "STAD": "Esophagus/Stomach",
    "ESCA": "Esophagus/Stomach",
    "BLCA": "Bladder/Urinary Tract",
    "LIHC": "Liver",
    "CESC": "Cervix",
    "SARC": "Soft Tissue",
    "LAML": "Myeloid",
    "PAAD": "Pancreas",
    "ACC": "Adrenal Gland",
    "MESO": "Pleura",
    "CHOL": "Biliary Tract",
    "TGCT": "Testis",
    "THYM": "Thymus",
    "PCPG": "Peripheral Nervous System",
    "DLBC": "Lymphoid",
    "UVM": "Eye",
}

# Atlas definitions: name -> (results_file, cancer_type_col, gene_col, effect_col)
ATLAS_SOURCES = {
    "PTEN-loss": {
        "file": "pten-loss-pancancer-dependency-atlas/phase3/genomewide_all_results.csv",
        "cancer_type_col": "cancer_type",
        "gene_col": "gene",
        "effect_col": "cohens_d",
        "fdr_col": "fdr",
    },
    "RB1-loss": {
        "file": "rb1-loss-pancancer-dependency-atlas/phase3/genomewide_all_results.csv",
        "cancer_type_col": "cancer_type",
        "gene_col": "gene",
        "effect_col": "cohens_d",
        "fdr_col": "fdr",
    },
    "KEAP1-loss": {
        "file": "keap1-nrf2-pancancer-dependency-atlas/phase2/genomewide_all_results.csv",
        "cancer_type_col": "cancer_type",
        "gene_col": "gene",
        "effect_col": "cohens_d",
        "fdr_col": "fdr",
    },
    "NF1-loss": {
        "file": "nf1-loss-pancancer-dependency-atlas/phase3/genomewide_all_results.csv",
        "cancer_type_col": "cancer_type",
        "gene_col": "gene",
        "effect_col": "cohens_d",
        "fdr_col": "fdr",
    },
}


def compute_tp53_freq_per_cancer_type() -> pd.DataFrame:
    """Compute TP53 mutation frequency per cancer type from Phase 1a TCGA data."""
    matrix = pd.read_csv(OUTPUT_DIR / "phase1_alteration_matrix.csv")

    # Compute per cancer type
    grouped = matrix.groupby("cancer_type").agg(
        n_patients=("TP53", "count"),
        n_tp53_altered=("TP53", "sum"),
    ).reset_index()
    grouped["tp53_freq"] = grouped["n_tp53_altered"] / grouped["n_patients"]

    # Map TCGA abbreviations to DepMap lineage names
    grouped["depmap_lineage"] = grouped["cancer_type"].map(TCGA_TO_DEPMAP)

    # For cancer types that map to the same lineage (e.g., COAD+READ -> Bowel),
    # aggregate by taking weighted average of TP53 frequency
    lineage_groups = grouped.dropna(subset=["depmap_lineage"]).groupby("depmap_lineage").apply(
        lambda g: pd.Series({
            "tp53_freq": (g["n_tp53_altered"].sum() / g["n_patients"].sum()),
            "n_patients": g["n_patients"].sum(),
            "n_tp53_altered": int(g["n_tp53_altered"].sum()),
        })
    ).reset_index()
    lineage_groups.columns = ["depmap_lineage", "tp53_freq", "n_patients", "n_tp53_altered"]

    return lineage_groups


def get_top_sl_genes(atlas_df: pd.DataFrame, cfg: dict) -> list[str]:
    """Identify top SL genes from atlas by strongest mean effect across cancer types."""
    gene_col = cfg["gene_col"]
    effect_col = cfg["effect_col"]

    # For each gene, compute mean |Cohen's d| across cancer types (negative d = selective dependency)
    gene_mean = atlas_df.groupby(gene_col)[effect_col].apply(
        lambda x: x.dropna().mean()
    ).reset_index()
    gene_mean.columns = ["gene", "mean_d"]

    # SL genes have negative Cohen's d (more dependency in altered lines)
    gene_mean = gene_mean.sort_values("mean_d")
    return gene_mean.head(TOP_SL_N)["gene"].tolist()


def compute_atlas_cancer_type_effect(
    atlas_df: pd.DataFrame,
    cfg: dict,
    top_genes: list[str],
) -> pd.DataFrame:
    """Compute mean SL effect size per cancer type for top SL genes."""
    gene_col = cfg["gene_col"]
    effect_col = cfg["effect_col"]
    ct_col = cfg["cancer_type_col"]

    # Filter to top SL genes
    subset = atlas_df[atlas_df[gene_col].isin(top_genes)].copy()

    # Mean effect per cancer type
    ct_effect = subset.groupby(ct_col).agg(
        mean_d=(effect_col, "mean"),
        median_d=(effect_col, "median"),
        n_genes=(gene_col, "nunique"),
    ).reset_index()
    ct_effect.columns = ["cancer_type", "mean_sl_effect", "median_sl_effect", "n_sl_genes"]

    # Filter out pooled/pan-cancer entries
    ct_effect = ct_effect[~ct_effect["cancer_type"].str.contains("Pan-cancer|pooled", case=False, na=False)]

    return ct_effect


def main() -> None:
    out_dir = OUTPUT_DIR / "phase4_tp53_modulation"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 4: TP53 Modulation Analysis ===\n")

    # Step 1: Compute TP53 frequency per cancer type
    print("Computing TP53 mutation frequency per cancer type (TCGA)...")
    tp53_freq = compute_tp53_freq_per_cancer_type()
    print(f"  {len(tp53_freq)} cancer lineages with TP53 data")
    tp53_freq.to_csv(out_dir / "tp53_frequency_by_lineage.csv", index=False)

    # Print top/bottom TP53 freq
    tp53_sorted = tp53_freq.sort_values("tp53_freq", ascending=False)
    print("\n  Top 5 TP53 mutation frequency:")
    for _, row in tp53_sorted.head(5).iterrows():
        print(f"    {row['depmap_lineage']:30s}  {row['tp53_freq']:.1%}  (N={int(row['n_patients'])})")
    print("  Bottom 5:")
    for _, row in tp53_sorted.tail(5).iterrows():
        print(f"    {row['depmap_lineage']:30s}  {row['tp53_freq']:.1%}  (N={int(row['n_patients'])})")

    # Step 2: For each atlas, compute correlation
    print("\n" + "=" * 60)
    print("Computing TP53 modulation per atlas...\n")

    correlation_results = []
    detail_rows = []

    for atlas_name, cfg in ATLAS_SOURCES.items():
        atlas_path = ATLAS_BASE / cfg["file"]
        if not atlas_path.exists():
            print(f"  {atlas_name}: SKIPPED (file not found: {atlas_path.name})")
            continue

        print(f"  {atlas_name}:")
        atlas_df = pd.read_csv(atlas_path)
        print(f"    {len(atlas_df)} rows, {atlas_df[cfg['gene_col']].nunique()} genes, "
              f"{atlas_df[cfg['cancer_type_col']].nunique()} cancer types")

        # Get top SL genes
        top_genes = get_top_sl_genes(atlas_df, cfg)
        print(f"    Top {len(top_genes)} SL genes selected")

        # Compute per-cancer-type mean effect
        ct_effect = compute_atlas_cancer_type_effect(atlas_df, cfg, top_genes)

        # Merge with TP53 frequency
        merged = ct_effect.merge(
            tp53_freq[["depmap_lineage", "tp53_freq", "n_patients"]],
            left_on="cancer_type",
            right_on="depmap_lineage",
            how="inner",
        )

        n_types = len(merged)
        print(f"    {n_types} cancer types with both atlas and TP53 data")

        if n_types < MIN_CANCER_TYPES:
            print(f"    SKIPPED: too few cancer types (need >= {MIN_CANCER_TYPES})")
            correlation_results.append({
                "atlas": atlas_name,
                "n_cancer_types": n_types,
                "spearman_r": np.nan,
                "spearman_p": np.nan,
                "sufficient_data": False,
            })
            continue

        # Spearman correlation: TP53 freq vs mean SL effect
        # Hypothesis: high TP53 freq -> weaker SL (less negative d, i.e. positive correlation
        # if d is negative for SL, or negative correlation with |d|)
        r_mean, p_mean = stats.spearmanr(merged["tp53_freq"], merged["mean_sl_effect"])
        r_median, p_median = stats.spearmanr(merged["tp53_freq"], merged["median_sl_effect"])

        # Also correlate with absolute effect size
        r_abs, p_abs = stats.spearmanr(merged["tp53_freq"], merged["mean_sl_effect"].abs())

        print(f"    Spearman r (TP53 freq vs mean SL d):    {r_mean:+.3f}  (p={p_mean:.3e})")
        print(f"    Spearman r (TP53 freq vs median SL d):  {r_median:+.3f}  (p={p_median:.3e})")
        print(f"    Spearman r (TP53 freq vs |mean SL d|):  {r_abs:+.3f}  (p={p_abs:.3e})")

        # Interpretation: if SL effect is negative (dependency in altered lines),
        # positive r means high TP53 freq -> less negative d -> weaker SL effect
        # The hypothesis predicts r > 0.3 (for mean_d) or r < -0.3 (for |d|)
        weakening = r_mean > 0  # positive r with negative d means weakening

        correlation_results.append({
            "atlas": atlas_name,
            "n_cancer_types": n_types,
            "spearman_r_mean_d": round(r_mean, 4),
            "spearman_p_mean_d": p_mean,
            "spearman_r_median_d": round(r_median, 4),
            "spearman_p_median_d": p_median,
            "spearman_r_abs_d": round(r_abs, 4),
            "spearman_p_abs_d": p_abs,
            "tp53_weakens_sl": weakening,
            "sufficient_data": True,
        })

        # Save detail rows
        for _, row in merged.iterrows():
            detail_rows.append({
                "atlas": atlas_name,
                "cancer_type": row["cancer_type"],
                "tp53_freq": round(row["tp53_freq"], 4),
                "n_patients": int(row["n_patients"]),
                "mean_sl_effect": round(row["mean_sl_effect"], 4),
                "median_sl_effect": round(row["median_sl_effect"], 4),
                "n_sl_genes": int(row["n_sl_genes"]),
            })

    # Save results
    print("\n" + "=" * 60)

    corr_df = pd.DataFrame(correlation_results)
    corr_df.to_csv(out_dir / "phase4_tp53_modulation.csv", index=False)
    print(f"Saved: {out_dir / 'phase4_tp53_modulation.csv'}")

    if detail_rows:
        detail_df = pd.DataFrame(detail_rows)
        detail_df.to_csv(out_dir / "phase4_tp53_modulation_detail.csv", index=False)
        print(f"Saved: {out_dir / 'phase4_tp53_modulation_detail.csv'}")

    # Summary
    valid = corr_df[corr_df["sufficient_data"] == True]
    n_weakening = int(valid["tp53_weakens_sl"].sum()) if len(valid) > 0 else 0
    # Check for r < -0.3 on |d| (TP53 weakens absolute SL magnitude)
    n_strong = int((valid["spearman_r_abs_d"] < -0.3).sum()) if "spearman_r_abs_d" in valid.columns else 0

    summary = {
        "atlases_tested": len(correlation_results),
        "atlases_with_sufficient_data": len(valid),
        "atlases_tp53_weakens_sl": n_weakening,
        "atlases_strong_negative_r": n_strong,
        "correlations": correlation_results,
    }
    with open(out_dir / "phase4_tp53_modulation_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Saved: {out_dir / 'phase4_tp53_modulation_summary.json'}")

    print(f"\nSummary:")
    print(f"  Atlases tested: {len(correlation_results)}")
    print(f"  With sufficient data: {len(valid)}")
    print(f"  TP53 weakens SL (positive r with d): {n_weakening}")
    print(f"  Strong |d| correlation (r < -0.3): {n_strong}")

    # Validation
    print(f"\nValidation:")
    print(f"  At least 2 atlases with r < -0.3: {'PASS' if n_strong >= 2 else 'CHECK'} ({n_strong})")

    print("\nDone.")


if __name__ == "__main__":
    main()
