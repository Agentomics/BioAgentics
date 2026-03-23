"""Phase 5: Cross-atlas comparison — NF1-loss vs KRAS-mutant dependency profiles.

Compares NF1-loss dependencies with published KRAS-mutant data to identify:
  (a) shared RAS-pathway dependencies
  (b) NF1-specific dependencies
  (c) KRAS-specific dependencies

Generates MPNST-specific target recommendations and per-tumor-type summaries.

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.05_cross_atlas_comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT

PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)
PHASE3_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)
PHASE4_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase4"
)
KRAS_DIR = REPO_ROOT / "output" / "cancer" / "crc-kras-dependencies"
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase5"
)

FDR_THRESHOLD = 0.1
EFFECT_SIZE_THRESHOLD = 0.3


def load_nf1_results() -> pd.DataFrame:
    """Load NF1-loss pan-cancer genome-wide results."""
    return pd.read_csv(PHASE3_DIR / "genomewide_all_results.csv")


def load_kras_results() -> dict:
    """Load KRAS-mutant CRC dependency results."""
    with open(KRAS_DIR / "allele_dependency_results.json") as f:
        return json.load(f)


def build_kras_gene_table(kras_data: dict) -> pd.DataFrame:
    """Extract KRAS all-mutant dependency scores into a DataFrame."""
    all_mut = kras_data["allele_comparisons"].get("all_KRAS_mut", {})
    top_deps = all_mut.get("top_dependencies", [])
    if not top_deps:
        return pd.DataFrame()

    df = pd.DataFrame(top_deps)
    df = df.rename(columns={
        "gene": "gene",
        "cohens_d": "kras_cohens_d",
        "pvalue": "kras_pvalue",
        "mean_allele": "kras_mean_dep_mut",
        "mean_wt": "kras_mean_dep_wt",
    })
    # Keep relevant columns
    cols = [c for c in ["gene", "kras_cohens_d", "kras_pvalue", "kras_mean_dep_mut", "kras_mean_dep_wt"] if c in df.columns]
    return df[cols]


def build_comparison_matrix(
    nf1_results: pd.DataFrame,
    kras_table: pd.DataFrame,
) -> pd.DataFrame:
    """Build gene-level comparison: NF1-loss vs KRAS-mutant effect sizes."""
    # Use NF1 pan-cancer (RAS-excluded) results
    nf1_pan = nf1_results[nf1_results["cancer_type"] == "Pan-cancer (RAS-excluded)"].copy()
    if len(nf1_pan) == 0:
        nf1_pan = nf1_results[nf1_results["cancer_type"] == "Pan-cancer (pooled)"].copy()

    nf1_pan = nf1_pan[["gene", "cohens_d", "fdr"]].rename(columns={
        "cohens_d": "nf1_cohens_d",
        "fdr": "nf1_fdr",
    })

    # Merge on gene
    merged = pd.merge(nf1_pan, kras_table, on="gene", how="outer")

    # Classify
    merged["category"] = "neither"

    nf1_hit = (merged["nf1_cohens_d"] < -EFFECT_SIZE_THRESHOLD)
    kras_hit = (merged["kras_cohens_d"] < -EFFECT_SIZE_THRESHOLD) if "kras_cohens_d" in merged.columns else pd.Series(False, index=merged.index)

    merged.loc[nf1_hit & kras_hit, "category"] = "shared_RAS_dependency"
    merged.loc[nf1_hit & ~kras_hit, "category"] = "NF1_specific"
    merged.loc[~nf1_hit & kras_hit, "category"] = "KRAS_specific"

    return merged


def generate_mpnst_recommendations(
    nf1_results: pd.DataFrame,
    drug_map: pd.DataFrame,
    mpnst_lines: pd.DataFrame,
) -> list[dict]:
    """Generate MPNST-specific therapeutic target recommendations."""
    # Get PNS (peripheral nervous system) results
    pns_results = nf1_results[nf1_results["cancer_type"] == "Peripheral Nervous System"]

    recommendations = []

    # Tier 1: Dependencies with clinical-stage drugs in PNS lineage
    if len(pns_results) > 0:
        pns_hits = pns_results[pns_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD].sort_values("cohens_d")
        for _, row in pns_hits.head(15).iterrows():
            gene = row["gene"]
            drug_info = drug_map[drug_map["gene"] == gene] if len(drug_map) > 0 else pd.DataFrame()
            has_drug = len(drug_info) > 0

            recommendations.append({
                "gene": gene,
                "tier": 1 if has_drug else 2,
                "cohens_d_pns": round(row["cohens_d"], 4),
                "fdr_pns": row.get("fdr", None),
                "has_drug": has_drug,
                "compounds": drug_info.iloc[0]["compounds"] if has_drug else "None",
                "stage": drug_info.iloc[0]["clinical_stage"] if has_drug else "N/A",
                "n_mpnst_lines": len(mpnst_lines) if len(mpnst_lines) > 0 else 0,
            })

    # Sort by tier then effect size
    recommendations.sort(key=lambda x: (x["tier"], x["cohens_d_pns"]))
    return recommendations


def generate_per_tumor_recommendations(
    nf1_results: pd.DataFrame,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Generate per-tumor-type top dependency recommendations."""
    rows = []
    for cancer_type in qualifying_types:
        ct_results = nf1_results[nf1_results["cancer_type"] == cancer_type]
        ct_hits = ct_results[ct_results["cohens_d"] < -EFFECT_SIZE_THRESHOLD].sort_values("cohens_d")

        for _, row in ct_hits.head(5).iterrows():
            rows.append({
                "cancer_type": cancer_type,
                "gene": row["gene"],
                "cohens_d": round(row["cohens_d"], 4),
                "fdr": row.get("fdr", None),
            })

    return pd.DataFrame(rows)


def plot_scatter_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot: NF1-loss d vs KRAS-mutant d."""
    both = comparison.dropna(subset=["nf1_cohens_d", "kras_cohens_d"])
    if len(both) < 10:
        return

    fig, ax = plt.subplots(figsize=(8, 8))

    colors = {
        "shared_RAS_dependency": "#E53935",
        "NF1_specific": "#1E88E5",
        "KRAS_specific": "#FF9800",
        "neither": "#CCCCCC",
    }

    for cat, color in colors.items():
        mask = both["category"] == cat
        if mask.sum() == 0:
            continue
        subset = both[mask]
        ax.scatter(
            subset["kras_cohens_d"],
            subset["nf1_cohens_d"],
            c=color,
            s=8 if cat == "neither" else 20,
            alpha=0.3 if cat == "neither" else 0.7,
            label=f"{cat} ({mask.sum()})",
        )

    # Label notable genes
    notable = both[both["category"].isin(["shared_RAS_dependency", "NF1_specific", "KRAS_specific"])]
    for cat in ["shared_RAS_dependency", "NF1_specific", "KRAS_specific"]:
        cat_genes = notable[notable["category"] == cat]
        if cat == "shared_RAS_dependency":
            top = cat_genes.nsmallest(5, "nf1_cohens_d")
        elif cat == "NF1_specific":
            top = cat_genes.nsmallest(5, "nf1_cohens_d")
        else:
            top = cat_genes.nsmallest(5, "kras_cohens_d")

        for _, row in top.iterrows():
            ax.annotate(
                row["gene"],
                (row["kras_cohens_d"], row["nf1_cohens_d"]),
                fontsize=6,
                ha="left",
            )

    ax.axhline(-EFFECT_SIZE_THRESHOLD, color="blue", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="orange", linestyle="--", alpha=0.3, linewidth=0.5)
    ax.plot([-3, 1], [-3, 1], "k--", alpha=0.2, linewidth=0.5)

    ax.set_xlabel("KRAS-mutant Cohen's d (CRC)")
    ax.set_ylabel("NF1-loss Cohen's d (Pan-cancer, RAS-excluded)")
    ax.set_title("NF1-Loss vs KRAS-Mutant Dependency Comparison")
    ax.legend(fontsize=7, loc="upper left")

    plt.tight_layout()
    fig.savefig(output_dir / "nf1_vs_kras_scatter.png", dpi=150)
    plt.close(fig)


def plot_category_bar(comparison: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart of category counts."""
    counts = comparison["category"].value_counts()
    if len(counts) == 0:
        return

    fig, ax = plt.subplots(figsize=(6, 4))
    colors = {
        "shared_RAS_dependency": "#E53935",
        "NF1_specific": "#1E88E5",
        "KRAS_specific": "#FF9800",
        "neither": "#CCCCCC",
    }
    bar_colors = [colors.get(c, "#999") for c in counts.index]
    ax.bar(range(len(counts)), counts.values, color=bar_colors, edgecolor="black", linewidth=0.3)
    ax.set_xticks(range(len(counts)))
    ax.set_xticklabels(counts.index, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Number of genes")
    ax.set_title("NF1 vs KRAS Dependency Classification")

    plt.tight_layout()
    fig.savefig(output_dir / "category_counts.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: Cross-Atlas Comparison — NF1 vs KRAS ===\n")

    # Load data
    print("Loading NF1-loss genome-wide results...")
    nf1_results = load_nf1_results()
    print(f"  {len(nf1_results)} results")

    print("Loading KRAS-mutant CRC dependency data...")
    kras_data = load_kras_results()
    kras_table = build_kras_gene_table(kras_data)
    print(f"  {len(kras_table)} KRAS dependency genes")
    kras_summary = kras_data.get("summary", {})
    print(f"  CRC: {kras_summary.get('total_kras_mutant', '?')} KRAS-mut, {kras_summary.get('total_kras_wt', '?')} WT")

    # Build comparison matrix
    print("\nBuilding NF1 vs KRAS comparison matrix...")
    comparison = build_comparison_matrix(nf1_results, kras_table)

    cat_counts = comparison["category"].value_counts()
    for cat, count in cat_counts.items():
        print(f"  {cat}: {count} genes")

    # Shared RAS dependencies
    shared = comparison[comparison["category"] == "shared_RAS_dependency"].sort_values("nf1_cohens_d")
    print(f"\nShared RAS-pathway dependencies ({len(shared)} genes):")
    for _, row in shared.head(15).iterrows():
        print(f"  {row['gene']}: NF1 d={row['nf1_cohens_d']:.3f}, KRAS d={row['kras_cohens_d']:.3f}")

    # NF1-specific
    nf1_specific = comparison[comparison["category"] == "NF1_specific"].sort_values("nf1_cohens_d")
    print(f"\nNF1-specific dependencies ({len(nf1_specific)} genes):")
    for _, row in nf1_specific.head(15).iterrows():
        kras_d = f"KRAS d={row['kras_cohens_d']:.3f}" if pd.notna(row.get("kras_cohens_d")) else "KRAS d=N/A"
        print(f"  {row['gene']}: NF1 d={row['nf1_cohens_d']:.3f}, {kras_d}")

    # KRAS-specific
    kras_specific = comparison[comparison["category"] == "KRAS_specific"]
    if "kras_cohens_d" in kras_specific.columns:
        kras_specific = kras_specific.sort_values("kras_cohens_d")
    print(f"\nKRAS-specific dependencies ({len(kras_specific)} genes):")
    for _, row in kras_specific.head(15).iterrows():
        nf1_d = f"NF1 d={row['nf1_cohens_d']:.3f}" if pd.notna(row.get("nf1_cohens_d")) else "NF1 d=N/A"
        print(f"  {row['gene']}: {nf1_d}, KRAS d={row['kras_cohens_d']:.3f}")

    # Load Phase 1 data for MPNST
    print("\nGenerating MPNST recommendations...")
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()

    mpnst_path = PHASE1_DIR / "mpnst_lines.csv"
    mpnst_lines = pd.read_csv(mpnst_path, index_col=0) if mpnst_path.exists() else pd.DataFrame()

    drug_map_path = PHASE4_DIR / "druggable_dependencies.csv"
    drug_map = pd.read_csv(drug_map_path) if drug_map_path.exists() else pd.DataFrame()

    mpnst_recs = generate_mpnst_recommendations(nf1_results, drug_map, mpnst_lines)
    print(f"  {len(mpnst_recs)} MPNST target recommendations")

    if mpnst_recs:
        print("\nMPNST therapeutic targets:")
        for rec in mpnst_recs[:10]:
            drug_str = f"[{rec['stage']}] {rec['compounds']}" if rec["has_drug"] else "[No drug]"
            print(f"  Tier {rec['tier']}: {rec['gene']} d={rec['cohens_d_pns']:.3f} — {drug_str}")

    # Per-tumor-type recommendations
    print("\nPer-tumor-type top dependencies:")
    per_tumor = generate_per_tumor_recommendations(nf1_results, qualifying_types)
    for ct in qualifying_types:
        ct_recs = per_tumor[per_tumor["cancer_type"] == ct]
        if len(ct_recs) > 0:
            top_genes = ", ".join(
                f"{r['gene']}(d={r['cohens_d']:.2f})"
                for _, r in ct_recs.head(3).iterrows()
            )
            print(f"  {ct}: {top_genes}")

    # Save outputs
    print("\nSaving outputs...")
    comparison.to_csv(OUTPUT_DIR / "nf1_vs_kras_comparison.csv", index=False)
    print(f"  nf1_vs_kras_comparison.csv - {len(comparison)} genes")

    shared.to_csv(OUTPUT_DIR / "shared_ras_dependencies.csv", index=False)
    nf1_specific.to_csv(OUTPUT_DIR / "nf1_specific_dependencies.csv", index=False)
    kras_specific.to_csv(OUTPUT_DIR / "kras_specific_dependencies.csv", index=False)

    if mpnst_recs:
        pd.DataFrame(mpnst_recs).to_csv(OUTPUT_DIR / "mpnst_recommendations.csv", index=False)

    per_tumor.to_csv(OUTPUT_DIR / "per_tumor_recommendations.csv", index=False)

    # Plots
    print("Generating plots...")
    plot_scatter_comparison(comparison, OUTPUT_DIR)
    plot_category_bar(comparison, OUTPUT_DIR)

    # Summary text
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 5: Cross-Atlas Comparison",
        "=" * 70,
        "",
        "NF1 vs KRAS DEPENDENCY CLASSIFICATION",
        f"  Shared RAS dependencies: {len(shared)} genes",
        f"  NF1-specific: {len(nf1_specific)} genes",
        f"  KRAS-specific: {len(kras_specific)} genes",
        "",
        "TOP SHARED RAS DEPENDENCIES",
        "-" * 60,
    ]
    for _, row in shared.head(10).iterrows():
        summary_lines.append(
            f"  {row['gene']}: NF1 d={row['nf1_cohens_d']:.3f}, KRAS d={row['kras_cohens_d']:.3f}"
        )

    summary_lines += [
        "",
        "TOP NF1-SPECIFIC DEPENDENCIES (not shared with KRAS)",
        "-" * 60,
    ]
    for _, row in nf1_specific.head(10).iterrows():
        summary_lines.append(f"  {row['gene']}: NF1 d={row['nf1_cohens_d']:.3f}")

    summary_lines += [
        "",
        "MPNST THERAPEUTIC TARGETS",
        "-" * 60,
    ]
    if mpnst_recs:
        for rec in mpnst_recs[:10]:
            drug_str = f"{rec['stage']}: {rec['compounds']}" if rec["has_drug"] else "No drug available"
            summary_lines.append(
                f"  Tier {rec['tier']}: {rec['gene']} (PNS d={rec['cohens_d_pns']:.3f}) — {drug_str}"
            )
    else:
        summary_lines.append("  No PNS lineage hits found")

    summary_lines += [
        "",
        "PER-TUMOR-TYPE TOP DEPENDENCIES",
        "-" * 60,
    ]
    for ct in qualifying_types:
        ct_recs = per_tumor[per_tumor["cancer_type"] == ct]
        if len(ct_recs) > 0:
            genes = "; ".join(f"{r['gene']}(d={r['cohens_d']:.2f})" for _, r in ct_recs.head(3).iterrows())
            summary_lines.append(f"  {ct}: {genes}")

    summary_lines.append("")

    with open(OUTPUT_DIR / "cross_atlas_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  cross_atlas_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
