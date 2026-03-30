"""Phase 3c: Lineage-corrected genome-wide differential dependency screen.

Addresses lineage bias artifacts (KRTAP21-1, SPRR1A, etc.) in the original
Phase 3 screen. NF1 mutations are enriched in certain lineages (melanoma, PNS),
so tissue-of-origin genes appear as false positives.

Approach: residualize CRISPR scores by subtracting per-lineage means, then
run the same MannWhitney/Cohen's d analysis on residuals. This removes lineage
effects without consuming degrees of freedom (unlike full OLS with dummies).

Usage:
    PYTHONPATH=src/cancer:src uv run python -m nf1_loss_pancancer_dependency_atlas.03c_lineage_corrected_screen
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase1"
)
OUTPUT_DIR = (
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase3"
)

FDR_THRESHOLD = 0.1
STRICT_FDR = 0.05
EFFECT_SIZE_THRESHOLD = 0.5
HIGH_CONFIDENCE_ES = 0.8
MIN_SAMPLES = 3


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def residualize_by_lineage(
    data: pd.DataFrame, crispr_cols: list[str], lineage_col: str = "OncotreeLineage"
) -> pd.DataFrame:
    """Subtract per-lineage mean from each CRISPR gene score.

    For each gene, replaces the raw score with (score - lineage_mean).
    This removes tissue-of-origin effects while preserving within-lineage variance.
    """
    residualized = data[crispr_cols].copy()
    lineages = data[lineage_col]

    # Compute lineage means efficiently
    lineage_means = residualized.groupby(lineages).transform("mean")
    residualized = residualized - lineage_means

    return residualized


def screen_residualized(
    data: pd.DataFrame, residuals: pd.DataFrame, crispr_cols: list[str]
) -> list[dict]:
    """Run MannWhitney + Cohen's d on lineage-residualized CRISPR scores."""
    lost_mask = data["NF1_loss"].values.astype(bool)
    intact_mask = data["NF1_status"].values == "intact"

    rows = []
    pvals = []

    for gene in crispr_cols:
        vals = residuals[gene].values
        lost_vals = vals[lost_mask & ~np.isnan(vals)]
        intact_vals = vals[intact_mask & ~np.isnan(vals)]

        if len(lost_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
        d = cohens_d(lost_vals, intact_vals)

        rows.append({
            "gene": gene,
            "cohens_d": round(d, 4),
            "p_value": float(pval),
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "median_resid_lost": round(float(np.median(lost_vals)), 4),
            "median_resid_intact": round(float(np.median(intact_vals)), 4),
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])
            fdr_val = max(fdrs[i], 1e-300)
            row["composite_score"] = round(abs(row["cohens_d"]) * -np.log10(fdr_val), 4)

    return rows


def compare_with_uncorrected(corrected: pd.DataFrame, uncorrected_path: Path) -> pd.DataFrame:
    """Merge corrected results with uncorrected and flag changes."""
    uncorr = pd.read_csv(uncorrected_path)
    uncorr_pc = uncorr[uncorr["cancer_type"] == "Pan-cancer (RAS-excluded)"].copy()
    uncorr_pc = uncorr_pc[["gene", "cohens_d", "p_value", "fdr", "composite_score"]].rename(
        columns={
            "cohens_d": "uncorr_cohens_d",
            "p_value": "uncorr_pvalue",
            "fdr": "uncorr_fdr",
            "composite_score": "uncorr_composite",
        }
    )

    merged = corrected.merge(uncorr_pc, on="gene", how="left")

    merged["was_significant"] = (
        (merged["uncorr_fdr"] < FDR_THRESHOLD)
        & (merged["uncorr_cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    )
    merged["is_significant"] = (
        (merged["fdr"] < FDR_THRESHOLD)
        & (merged["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    )
    merged["lineage_artifact"] = merged["was_significant"] & ~merged["is_significant"]
    merged["fdr_change"] = merged["fdr"] - merged["uncorr_fdr"]
    merged["rank_change"] = None  # will be computed below

    # Compute rank change
    if "uncorr_cohens_d" in merged.columns:
        uncorr_rank = merged["uncorr_cohens_d"].rank()
        corr_rank = merged["cohens_d"].rank()
        merged["rank_change"] = (corr_rank - uncorr_rank).astype(int)

    return merged


def plot_correction_comparison(comparison: pd.DataFrame, output_dir: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    mask = comparison["uncorr_cohens_d"].notna()
    x = comparison.loc[mask, "uncorr_cohens_d"]
    y = comparison.loc[mask, "cohens_d"]
    artifacts = comparison.loc[mask, "lineage_artifact"]

    ax.scatter(x[~artifacts], y[~artifacts], s=3, alpha=0.3, c="#CCCCCC", label="Other")
    ax.scatter(x[artifacts], y[artifacts], s=20, alpha=0.8, c="#D95319", label="Lineage artifact")

    top_art = comparison[comparison["lineage_artifact"]].nsmallest(5, "uncorr_fdr")
    for _, row in top_art.iterrows():
        ax.annotate(row["gene"], (row["uncorr_cohens_d"], row["cohens_d"]),
                     fontsize=7, ha="left")

    lims = [min(x.min(), y.min()) - 0.1, max(x.max(), y.max()) + 0.1]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("Uncorrected Cohen's d")
    ax.set_ylabel("Lineage-corrected Cohen's d")
    ax.set_title("Effect Size: Uncorrected vs Corrected")
    ax.legend(fontsize=8)

    ax = axes[1]
    x_fdr = -np.log10(comparison.loc[mask, "uncorr_fdr"].clip(lower=1e-50))
    y_fdr = -np.log10(comparison.loc[mask, "fdr"].clip(lower=1e-50))

    ax.scatter(x_fdr[~artifacts], y_fdr[~artifacts], s=3, alpha=0.3, c="#CCCCCC")
    ax.scatter(x_fdr[artifacts], y_fdr[artifacts], s=20, alpha=0.8, c="#D95319")

    for _, row in top_art.iterrows():
        ax.annotate(
            row["gene"],
            (-np.log10(max(row["uncorr_fdr"], 1e-50)),
             -np.log10(max(row["fdr"], 1e-50))),
            fontsize=7, ha="left",
        )

    lims = [0, max(x_fdr.max(), y_fdr.max()) + 0.5]
    ax.plot(lims, lims, "k--", alpha=0.3, linewidth=0.5)
    ax.set_xlabel("-log10(Uncorrected FDR)")
    ax.set_ylabel("-log10(Corrected FDR)")
    ax.set_title("Significance: Uncorrected vs Corrected")

    fig.tight_layout()
    fig.savefig(output_dir / "lineage_correction_comparison.png", dpi=150)
    plt.close(fig)


def plot_corrected_volcano(results: pd.DataFrame, output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    x = results["cohens_d"].values
    y = -np.log10(results["fdr"].values.clip(min=1e-50))

    sig = (results["fdr"] < FDR_THRESHOLD) & (results["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ax.scatter(x[~sig], y[~sig], c="#CCCCCC", s=5, alpha=0.5)

    gained = sig & (results["cohens_d"] < 0)
    lost_dep = sig & (results["cohens_d"] > 0)
    ax.scatter(x[gained], y[gained], c="#D95319", s=15, alpha=0.8, label="Gained dep.")
    ax.scatter(x[lost_dep], y[lost_dep], c="#4DBEEE", s=15, alpha=0.8, label="Lost dep.")

    top = results[gained].nsmallest(10, "cohens_d")
    for _, row in top.iterrows():
        ax.annotate(row["gene"], (row["cohens_d"], -np.log10(max(row["fdr"], 1e-50))),
                     fontsize=7, ha="right")

    ax.axhline(-np.log10(FDR_THRESHOLD), color="grey", linestyle="--", alpha=0.5)
    ax.axvline(-EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)
    ax.axvline(EFFECT_SIZE_THRESHOLD, color="grey", linestyle="--", alpha=0.5)

    ax.set_xlabel("Cohen's d (lineage-residualized)")
    ax.set_ylabel("-log10(FDR)")
    ax.set_title("NF1-Loss Dependency Screen: Lineage-Corrected (RAS-excluded)")
    ax.legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(output_dir / "volcano_lineage_corrected.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 3c: Lineage-Corrected Genome-Wide Screen ===\n")

    # Load Phase 1 classifications
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "nf1_loss_classification.csv", index_col=0)

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    crispr_cols = list(crispr.columns)
    print(f"  {len(crispr_cols)} genes, {len(crispr)} cell lines")

    merged = classified.join(crispr, how="inner")
    print(f"  {len(merged)} lines with both classification and CRISPR data")

    # Exclude RAS-mutant lines (same as Phase 3 "RAS-excluded" context)
    data = merged[~merged["has_RAS_mutation"]].copy()
    print(f"  {len(data)} lines after excluding RAS-mutant")

    # Lineage distribution
    lineage_counts = data.groupby("OncotreeLineage")["NF1_loss"].agg(["sum", "count"])
    lineage_counts.columns = ["n_nf1_lost", "n_total"]
    print(f"\n  Lineages: {len(lineage_counts)}")
    nf1_lineages = lineage_counts[lineage_counts["n_nf1_lost"] > 0]
    print(f"  Lineages with NF1-lost lines: {len(nf1_lineages)}")
    for lin, row in nf1_lineages.iterrows():
        print(f"    {lin}: {int(row['n_nf1_lost'])} lost / {int(row['n_total'])} total")

    # Residualize CRISPR scores by lineage
    print(f"\n  Residualizing {len(crispr_cols)} genes by lineage...")
    residuals = residualize_by_lineage(data, crispr_cols)
    print("  Done.")

    # Run screen on residualized scores
    print("  Running MannWhitney + Cohen's d on residualized scores...")
    rows = screen_residualized(data, residuals, crispr_cols)
    results_df = pd.DataFrame(rows)
    results_df = results_df.sort_values("cohens_d").reset_index(drop=True)

    print(f"  {len(results_df)} genes tested")

    # Significant hits
    sig = results_df[
        (results_df["fdr"] < FDR_THRESHOLD)
        & (results_df["cohens_d"].abs() > EFFECT_SIZE_THRESHOLD)
    ]
    gained = sig[sig["cohens_d"] < 0]
    lost = sig[sig["cohens_d"] > 0]
    print(f"  Significant gained dependencies: {len(gained)}")
    print(f"  Significant lost dependencies: {len(lost)}")

    # Save corrected results
    results_df.to_csv(OUTPUT_DIR / "lineage_corrected_results.csv", index=False)

    # Compare with uncorrected
    uncorrected_path = OUTPUT_DIR / "genomewide_all_results.csv"
    comparison = None
    artifacts = pd.DataFrame()
    if uncorrected_path.exists():
        print("\n  Comparing with uncorrected results...")
        comparison = compare_with_uncorrected(results_df, uncorrected_path)

        artifacts = comparison[comparison["lineage_artifact"]]
        print(f"  Lineage artifacts identified: {len(artifacts)}")
        if len(artifacts) > 0:
            print("  Flagged lineage artifacts (previously significant, now not):")
            for _, row in artifacts.sort_values("uncorr_fdr").iterrows():
                print(
                    f"    {row['gene']}: uncorr_d={row['uncorr_cohens_d']:.3f} "
                    f"(FDR={row['uncorr_fdr']:.3e}) -> corr_d={row['cohens_d']:.3f} "
                    f"(FDR={row['fdr']:.3e})"
                )

        comparison.to_csv(OUTPUT_DIR / "lineage_correction_comparison.csv", index=False)
        plot_correction_comparison(comparison, OUTPUT_DIR)

    # Top hits
    print("\nTop lineage-corrected gained dependencies:")
    for _, row in results_df[results_df["cohens_d"] < 0].head(20).iterrows():
        sig_flag = " ***" if row["fdr"] < FDR_THRESHOLD and abs(row["cohens_d"]) > EFFECT_SIZE_THRESHOLD else ""
        print(
            f"  {row['gene']}: d={row['cohens_d']:.3f}, "
            f"FDR={row['fdr']:.3e}, composite={row['composite_score']:.1f}{sig_flag}"
        )

    # Plots
    print("\nGenerating plots...")
    plot_corrected_volcano(results_df, OUTPUT_DIR)

    # Summary
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 3c: Lineage Correction",
        "=" * 70,
        "",
        "Method: Residualize CRISPR scores by subtracting per-lineage means,",
        "  then run MannWhitney U + Cohen's d on residuals.",
        f"Context: Pan-cancer RAS-excluded ({len(data)} cell lines)",
        f"Lineages: {len(lineage_counts)} ({len(nf1_lineages)} with NF1-lost lines)",
        f"Genes tested: {len(results_df)}",
        "",
        f"Significant gained dependencies (FDR<{FDR_THRESHOLD}, |d|>{EFFECT_SIZE_THRESHOLD}): {len(gained)}",
        f"Significant lost dependencies: {len(lost)}",
        "",
    ]

    if len(artifacts) > 0:
        summary_lines += [
            "LINEAGE ARTIFACTS REMOVED",
            "-" * 60,
            "Genes that were significant before correction but not after:",
        ]
        for _, row in artifacts.sort_values("uncorr_fdr").iterrows():
            summary_lines.append(
                f"  {row['gene']}: uncorr_d={row['uncorr_cohens_d']:.3f} "
                f"(FDR={row['uncorr_fdr']:.3e}) -> corr_d={row['cohens_d']:.3f} "
                f"(FDR={row['fdr']:.3e})"
            )
        summary_lines.append("")

    summary_lines += [
        "TOP LINEAGE-CORRECTED HITS (gained dependencies)",
        "-" * 60,
    ]
    for _, row in results_df[results_df["cohens_d"] < 0].head(30).iterrows():
        sig_flag = " ***" if row["fdr"] < FDR_THRESHOLD and abs(row["cohens_d"]) > EFFECT_SIZE_THRESHOLD else ""
        summary_lines.append(
            f"  {row['gene']}: d={row['cohens_d']:.3f}, "
            f"FDR={row['fdr']:.3e}, composite={row['composite_score']:.1f}{sig_flag}"
        )

    summary_lines.append("")

    with open(OUTPUT_DIR / "lineage_corrected_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))

    print("\nDone.")


if __name__ == "__main__":
    main()
