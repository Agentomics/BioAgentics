"""Phase 3b: TP53-stratified differential dependency analysis.

71.5% of NF1-lost lines are TP53-mutant vs ~50-55% background rate. This
confounds attribution of DDR (ATR, CHEK1, WEE1), cell cycle (CDK4/6), and
epigenetic (EZH2) dependencies to NF1 loss. This module re-runs the
genome-wide screen within TP53-mutant and TP53-WT strata to separate
NF1-specific effects from TP53-driven confounding.

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.03b_tp53_stratified_screen
"""

from __future__ import annotations

from pathlib import Path

import importlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

# Reuse Phase 3 statistical functions (filename starts with digit, needs importlib)
_phase3 = importlib.import_module(
    "cancer.nf1_loss_pancancer_dependency_atlas.03_genomewide_screen"
)
cohens_d = _phase3.cohens_d
fdr_correction = _phase3.fdr_correction
screen_one_context = _phase3.screen_one_context
GENE_SETS = _phase3.GENE_SETS
FDR_THRESHOLD = _phase3.FDR_THRESHOLD
EFFECT_SIZE_THRESHOLD = _phase3.EFFECT_SIZE_THRESHOLD
MIN_SAMPLES = _phase3.MIN_SAMPLES

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)
PHASE3_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)
OUTPUT_DIR = PHASE3_DIR  # Save alongside existing phase3 outputs

# Genes of particular interest for TP53 confounding
CONFOUND_GENES = {
    "DDR": ["ATR", "CHEK1", "WEE1", "PARP1"],
    "Cell cycle": ["CDK4", "CDK6", "CDK2", "CCND1", "RB1"],
    "Epigenetic/PRC2": ["EZH2", "EED", "SUZ12", "BRD4", "DOT1L"],
}


def plot_stratified_comparison(
    unstratified: pd.DataFrame,
    tp53_mut: pd.DataFrame,
    tp53_wt: pd.DataFrame | None,
    focus_genes: list[str],
    title: str,
    out_path: Path,
) -> None:
    """Bar chart comparing Cohen's d across unstratified vs TP53 strata."""
    genes_present = [g for g in focus_genes if g in unstratified["gene"].values]
    if not genes_present:
        return

    n_groups = 2 if tp53_wt is None or len(tp53_wt) == 0 else 3
    fig, ax = plt.subplots(figsize=(max(8, len(genes_present) * 0.8), 5))

    x = np.arange(len(genes_present))
    width = 0.25 if n_groups == 3 else 0.35

    def get_d(df: pd.DataFrame, gene: str) -> float:
        row = df[df["gene"] == gene]
        return float(row["cohens_d"].iloc[0]) if len(row) > 0 else 0.0

    d_unstrat = [get_d(unstratified, g) for g in genes_present]
    d_tp53m = [get_d(tp53_mut, g) for g in genes_present]

    if n_groups == 3:
        offset = width
        ax.bar(x - offset, d_unstrat, width, label="Unstratified", color="#4DBEEE", alpha=0.8)
        ax.bar(x, d_tp53m, width, label="TP53-mut stratum", color="#D95319", alpha=0.8)
        d_tp53wt = [get_d(tp53_wt, g) for g in genes_present]
        ax.bar(x + offset, d_tp53wt, width, label="TP53-WT stratum", color="#77AC30", alpha=0.8)
    else:
        offset = width / 2
        ax.bar(x - offset, d_unstrat, width, label="Unstratified", color="#4DBEEE", alpha=0.8)
        ax.bar(x + offset, d_tp53m, width, label="TP53-mut stratum", color="#D95319", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(genes_present, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Cohen's d (NF1-lost vs intact)")
    ax.set_title(title)
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def classify_confounding(
    unstratified: pd.DataFrame,
    tp53_mut: pd.DataFrame,
    tp53_wt: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare unstratified vs TP53-stratified results gene-by-gene.

    Returns a merged table with confounding classification:
    - 'NF1-specific': significant in TP53-mut stratum (effect persists after controlling for TP53)
    - 'TP53-confounded': significant unstratified but NOT in TP53-mut stratum
    - 'Both': significant in both strata — genuine NF1 effect, also TP53-associated
    - 'Neither': not significant in any analysis
    """
    merged = unstratified[["gene", "cohens_d", "fdr", "composite_score"]].rename(
        columns={
            "cohens_d": "d_unstratified",
            "fdr": "fdr_unstratified",
            "composite_score": "cs_unstratified",
        }
    )

    tp53m_cols = tp53_mut[["gene", "cohens_d", "fdr", "n_lost", "n_intact"]].rename(
        columns={
            "cohens_d": "d_tp53mut",
            "fdr": "fdr_tp53mut",
            "n_lost": "n_lost_tp53mut",
            "n_intact": "n_intact_tp53mut",
        }
    )
    merged = merged.merge(tp53m_cols, on="gene", how="outer")

    if tp53_wt is not None and len(tp53_wt) > 0:
        tp53wt_cols = tp53_wt[["gene", "cohens_d", "fdr", "n_lost", "n_intact"]].rename(
            columns={
                "cohens_d": "d_tp53wt",
                "fdr": "fdr_tp53wt",
                "n_lost": "n_lost_tp53wt",
                "n_intact": "n_intact_tp53wt",
            }
        )
        merged = merged.merge(tp53wt_cols, on="gene", how="outer")

    def _classify(row: pd.Series) -> str:
        sig_unstrat = (
            pd.notna(row.get("fdr_unstratified"))
            and row["fdr_unstratified"] < FDR_THRESHOLD
            and pd.notna(row.get("d_unstratified"))
            and abs(row["d_unstratified"]) > EFFECT_SIZE_THRESHOLD
        )
        sig_tp53m = (
            pd.notna(row.get("fdr_tp53mut"))
            and row["fdr_tp53mut"] < FDR_THRESHOLD
            and pd.notna(row.get("d_tp53mut"))
            and abs(row["d_tp53mut"]) > EFFECT_SIZE_THRESHOLD
        )
        if sig_unstrat and sig_tp53m:
            return "NF1-specific (confirmed)"
        elif sig_unstrat and not sig_tp53m:
            return "TP53-confounded"
        elif not sig_unstrat and sig_tp53m:
            return "NF1-specific (TP53-masked)"
        else:
            return "not significant"

    merged["confounding_class"] = merged.apply(_classify, axis=1)

    return merged.sort_values("d_unstratified", na_position="last")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3b: TP53-Stratified Dependency Screen ===\n")

    # Load Phase 1 classification
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "nf1_loss_classification.csv", index_col=0)
    print(f"  {len(classified)} lines")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes, {len(crispr)} cell lines")

    merged = classified.join(crispr, how="inner")
    print(f"  {len(merged)} lines with both classification and CRISPR data")

    # RAS-excluded cohort (primary analysis context)
    merged_no_ras = merged[~merged["has_RAS_mutation"]]
    print(f"  {len(merged_no_ras)} lines after excluding RAS-mutant")

    # TP53 stratification
    tp53_mut = merged_no_ras[merged_no_ras["TP53_status"] == "mutant"]
    tp53_wt = merged_no_ras[merged_no_ras["TP53_status"] == "WT"]

    print(f"\nTP53 stratification (RAS-excluded):")
    tp53m_lost = tp53_mut[tp53_mut["NF1_loss"] == True]  # noqa: E712
    tp53m_intact = tp53_mut[tp53_mut["NF1_status"] == "intact"]
    tp53wt_lost = tp53_wt[tp53_wt["NF1_loss"] == True]  # noqa: E712
    tp53wt_intact = tp53_wt[tp53_wt["NF1_status"] == "intact"]

    print(f"  TP53-mutant: {len(tp53m_lost)} NF1-lost, {len(tp53m_intact)} NF1-intact")
    print(f"  TP53-WT:     {len(tp53wt_lost)} NF1-lost, {len(tp53wt_intact)} NF1-intact")

    # --- TP53-mutant stratum screen ---
    print("\nScreening TP53-mutant stratum (NF1-lost vs NF1-intact)...")
    tp53m_rows = screen_one_context(
        tp53m_lost, tp53m_intact, crispr_cols, "Pan-cancer (RAS-excl, TP53-mut)"
    )
    tp53m_results = pd.DataFrame(tp53m_rows)
    print(f"  {len(tp53m_results)} genes tested")

    # --- TP53-WT stratum screen (if sample sizes permit) ---
    tp53wt_results = pd.DataFrame()
    if len(tp53wt_lost) >= MIN_SAMPLES and len(tp53wt_intact) >= MIN_SAMPLES:
        print("Screening TP53-WT stratum (NF1-lost vs NF1-intact)...")
        tp53wt_rows = screen_one_context(
            tp53wt_lost, tp53wt_intact, crispr_cols, "Pan-cancer (RAS-excl, TP53-WT)"
        )
        tp53wt_results = pd.DataFrame(tp53wt_rows)
        print(f"  {len(tp53wt_results)} genes tested")
    else:
        print(f"  Skipping TP53-WT stratum (insufficient samples: "
              f"{len(tp53wt_lost)} lost, {len(tp53wt_intact)} intact)")

    # --- Load unstratified results for comparison ---
    print("\nLoading unstratified Phase 3 results for comparison...")
    unstrat_all = pd.read_csv(PHASE3_DIR / "genomewide_all_results.csv")
    unstrat_pancancer = unstrat_all[
        unstrat_all["cancer_type"] == "Pan-cancer (RAS-excluded)"
    ].copy()
    print(f"  {len(unstrat_pancancer)} genes in unstratified pan-cancer")

    # --- Confounding classification ---
    print("\nClassifying TP53 confounding...")
    confounding = classify_confounding(unstrat_pancancer, tp53m_results, tp53wt_results)

    class_counts = confounding["confounding_class"].value_counts()
    for cls, count in class_counts.items():
        print(f"  {cls}: {count}")

    # --- Focus on confound-prone gene sets ---
    print("\nConfound-prone gene set analysis:")
    all_confound_genes = []
    for genes in CONFOUND_GENES.values():
        all_confound_genes.extend(genes)

    confound_focus = confounding[confounding["gene"].isin(all_confound_genes)].copy()
    for gs_name, gs_genes in CONFOUND_GENES.items():
        gs_data = confound_focus[confound_focus["gene"].isin(gs_genes)]
        print(f"\n  {gs_name}:")
        for _, row in gs_data.iterrows():
            d_u = f"d={row['d_unstratified']:.3f}" if pd.notna(row.get("d_unstratified")) else "d=N/A"
            d_m = f"d={row['d_tp53mut']:.3f}" if pd.notna(row.get("d_tp53mut")) else "d=N/A"
            fdr_u = f"FDR={row['fdr_unstratified']:.3e}" if pd.notna(row.get("fdr_unstratified")) else "FDR=N/A"
            fdr_m = f"FDR={row['fdr_tp53mut']:.3e}" if pd.notna(row.get("fdr_tp53mut")) else "FDR=N/A"
            d_w = ""
            if "d_tp53wt" in row.index and pd.notna(row.get("d_tp53wt")):
                d_w = f" | TP53-WT: d={row['d_tp53wt']:.3f}"
            print(
                f"    {row['gene']}: unstrat({d_u}, {fdr_u}) | "
                f"TP53-mut({d_m}, {fdr_m}){d_w} => {row['confounding_class']}"
            )

    # --- Save results ---
    tp53m_results.to_csv(OUTPUT_DIR / "tp53_stratified_tp53mut_results.csv", index=False)
    if len(tp53wt_results) > 0:
        tp53wt_results.to_csv(OUTPUT_DIR / "tp53_stratified_tp53wt_results.csv", index=False)
    confounding.to_csv(OUTPUT_DIR / "tp53_confounding_classification.csv", index=False)
    confound_focus.to_csv(OUTPUT_DIR / "tp53_confound_focus_genes.csv", index=False)

    print(f"\nSaved to {OUTPUT_DIR}:")
    print("  tp53_stratified_tp53mut_results.csv")
    if len(tp53wt_results) > 0:
        print("  tp53_stratified_tp53wt_results.csv")
    print("  tp53_confounding_classification.csv")
    print("  tp53_confound_focus_genes.csv")

    # --- Comparison plots ---
    print("\nGenerating comparison plots...")
    for gs_name, gs_genes in CONFOUND_GENES.items():
        safe_name = gs_name.replace("/", "_").replace(" ", "_")
        plot_stratified_comparison(
            unstrat_pancancer, tp53m_results,
            tp53wt_results if len(tp53wt_results) > 0 else None,
            gs_genes,
            f"TP53 Confounding Control: {gs_name}",
            OUTPUT_DIR / f"tp53_stratified_{safe_name}.png",
        )
    print(f"  {len(CONFOUND_GENES)} comparison plots")

    # --- Summary ---
    confirmed = confounding[confounding["confounding_class"] == "NF1-specific (confirmed)"]
    confounded = confounding[confounding["confounding_class"] == "TP53-confounded"]
    masked = confounding[confounding["confounding_class"] == "NF1-specific (TP53-masked)"]

    summary_lines = [
        "=" * 70,
        "NF1-Loss Dependency Atlas - Phase 3b: TP53-Stratified Analysis",
        "=" * 70,
        "",
        f"TP53-mutant stratum: {len(tp53m_lost)} NF1-lost vs {len(tp53m_intact)} NF1-intact",
        f"TP53-WT stratum: {len(tp53wt_lost)} NF1-lost vs {len(tp53wt_intact)} NF1-intact",
        "",
        f"NF1-specific (confirmed): {len(confirmed)} genes",
        f"TP53-confounded: {len(confounded)} genes",
        f"NF1-specific (TP53-masked): {len(masked)} genes",
        "",
        "CONFOUND-PRONE GENE SETS",
        "-" * 60,
    ]

    for gs_name, gs_genes in CONFOUND_GENES.items():
        summary_lines.append(f"\n  {gs_name}:")
        gs_data = confound_focus[confound_focus["gene"].isin(gs_genes)]
        for _, row in gs_data.iterrows():
            d_u = f"{row['d_unstratified']:.3f}" if pd.notna(row.get("d_unstratified")) else "N/A"
            d_m = f"{row['d_tp53mut']:.3f}" if pd.notna(row.get("d_tp53mut")) else "N/A"
            summary_lines.append(
                f"    {row['gene']}: d_unstrat={d_u}, d_tp53mut={d_m} => {row['confounding_class']}"
            )

    if len(confirmed) > 0:
        summary_lines += [
            "",
            "TOP NF1-SPECIFIC (CONFIRMED) HITS",
            "-" * 60,
        ]
        for _, row in confirmed.head(20).iterrows():
            d_u = f"{row['d_unstratified']:.3f}" if pd.notna(row.get("d_unstratified")) else "N/A"
            d_m = f"{row['d_tp53mut']:.3f}" if pd.notna(row.get("d_tp53mut")) else "N/A"
            summary_lines.append(f"  {row['gene']}: d_unstrat={d_u}, d_tp53mut={d_m}")

    if len(confounded) > 0:
        summary_lines += [
            "",
            "TP53-CONFOUNDED HITS (unstratified sig, stratified not sig)",
            "-" * 60,
        ]
        for _, row in confounded.head(20).iterrows():
            d_u = f"{row['d_unstratified']:.3f}" if pd.notna(row.get("d_unstratified")) else "N/A"
            d_m = f"{row['d_tp53mut']:.3f}" if pd.notna(row.get("d_tp53mut")) else "N/A"
            summary_lines.append(f"  {row['gene']}: d_unstrat={d_u}, d_tp53mut={d_m}")

    summary_lines.append("")
    with open(OUTPUT_DIR / "tp53_stratified_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  tp53_stratified_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
