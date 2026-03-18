"""Phase 3a: Extended SL network — methionine salvage genes + genome-wide screen.

Tests targeted methionine pathway genes and runs a genome-wide differential
dependency screen to identify all genes with enhanced dependency in
MTAP-deleted vs intact NSCLC lines.

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.04_extended_sl_network
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
OUTPUT_DIR = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"
FIG_DIR = OUTPUT_DIR / "figures"

# Targeted methionine salvage / PRMT pathway genes
TARGETED_GENES = ["PRMT5", "PRMT1", "PRMT7", "MAT2A", "MAT2B", "SRM", "SMS"]

# Highlight genes on volcano plot
HIGHLIGHT_GENES = set(TARGETED_GENES) | {"CDKN2A", "CDKN2B", "MTAP"}

# Minimum non-NaN samples per group
MIN_N = 5


def cohens_d(g1: np.ndarray, g2: np.ndarray) -> float:
    n1, n2 = len(g1), len(g2)
    var1, var2 = g1.var(ddof=1), g2.var(ddof=1)
    pooled = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0
    return float((g1.mean() - g2.mean()) / pooled)


def genome_wide_screen(
    crispr: pd.DataFrame, deleted_ids: set, intact_ids: set
) -> pd.DataFrame:
    """Run Wilcoxon test for every gene comparing MTAP-deleted vs intact."""
    del_idx = [i for i in crispr.index if i in deleted_ids]
    int_idx = [i for i in crispr.index if i in intact_ids]

    results = []
    n_genes = len(crispr.columns)

    for i, gene in enumerate(crispr.columns):
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1}/{n_genes} genes processed...", file=sys.stderr)

        del_vals = crispr.loc[del_idx, gene].dropna()
        int_vals = crispr.loc[int_idx, gene].dropna()

        if len(del_vals) < MIN_N or len(int_vals) < MIN_N:
            continue

        del_arr = del_vals.values
        int_arr = int_vals.values
        try:
            stat, pval = stats.mannwhitneyu(del_arr, int_arr, alternative="two-sided")
        except ValueError:
            continue

        d = cohens_d(del_arr, int_arr)
        results.append({
            "gene": gene,
            "n_deleted": len(del_arr),
            "n_intact": len(int_arr),
            "median_deleted": float(np.median(del_arr)),
            "median_intact": float(np.median(int_arr)),
            "cohens_d": d,
            "mannwhitney_p": pval,
        })

    df = pd.DataFrame(results)

    # FDR correction
    reject, fdr, _, _ = multipletests(df["mannwhitney_p"], method="fdr_bh")
    df["fdr"] = fdr
    df["significant"] = reject

    return df.sort_values("cohens_d")


def plot_volcano(screen: pd.DataFrame, out_path: Path) -> None:
    """Volcano plot: Cohen's d vs -log10(FDR)."""
    fig, ax = plt.subplots(figsize=(9, 7))

    neglog_fdr = -np.log10(screen["fdr"].clip(lower=1e-50))
    d = screen["cohens_d"]

    # Color by significance
    sig = screen["fdr"] < 0.05
    strong = sig & (d.abs() > 0.3)

    ax.scatter(d[~sig], neglog_fdr[~sig], alpha=0.15, s=8, color="gray", label="NS")
    ax.scatter(d[sig & ~strong], neglog_fdr[sig & ~strong], alpha=0.4, s=12,
               color="#4DBEEE", label="FDR<0.05")
    ax.scatter(d[strong], neglog_fdr[strong], alpha=0.6, s=20,
               color="#D95319", label="FDR<0.05 & |d|>0.3")

    # Highlight pathway genes
    for _, row in screen.iterrows():
        if row["gene"] in HIGHLIGHT_GENES:
            x = row["cohens_d"]
            y = -np.log10(max(row["fdr"], 1e-50))
            ax.annotate(row["gene"], (x, y), fontsize=7, fontweight="bold",
                        xytext=(5, 5), textcoords="offset points")
            ax.scatter([x], [y], s=50, edgecolors="black", facecolors="none",
                       linewidths=1.5, zorder=5)

    ax.axhline(-np.log10(0.05), color="gray", linestyle="--", alpha=0.5)
    ax.axvline(-0.3, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(0.3, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Cohen's d (deleted − intact)")
    ax.set_ylabel("-log₁₀(FDR)")
    ax.set_title("Genome-wide Differential Dependency: MTAP-deleted vs Intact NSCLC")
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_top_genes(screen: pd.DataFrame, out_path: Path, n: int = 20) -> None:
    """Bar chart of top N differential dependencies (most negative Cohen's d)."""
    top = screen.head(n).copy()
    top = top.iloc[::-1]  # Reverse for horizontal bar

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#D95319" if fdr < 0.05 else "#4DBEEE" for fdr in top["fdr"]]
    bars = ax.barh(range(len(top)), top["cohens_d"], color=colors, alpha=0.8)

    gene_labels = [
        f"{g} *" if g in HIGHLIGHT_GENES else g for g in top["gene"]
    ]
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(gene_labels, fontsize=8)
    ax.set_xlabel("Cohen's d (negative = stronger dep. in MTAP-deleted)")
    ax.set_title(f"Top {n} Differential Dependencies (MTAP-deleted NSCLC)")
    ax.axvline(0, color="black", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load data
    classified = pd.read_csv(OUTPUT_DIR / "nsclc_cell_lines_classified.csv", index_col=0)
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    deleted_ids = set(classified[classified["MTAP_deleted"]].index)
    intact_ids = set(classified[~classified["MTAP_deleted"]].index)

    # Restrict to NSCLC lines with CRISPR data
    deleted_ids &= set(crispr.index)
    intact_ids &= set(crispr.index)
    print(f"NSCLC lines with CRISPR data: {len(deleted_ids)} deleted, {len(intact_ids)} intact")

    # Targeted genes
    print("\nTargeted methionine pathway genes:")
    for gene in TARGETED_GENES:
        if gene not in crispr.columns:
            print(f"  {gene}: not found in CRISPR data")
            continue
        del_vals = crispr.loc[list(deleted_ids), gene].dropna().values
        int_vals = crispr.loc[list(intact_ids), gene].dropna().values
        if len(del_vals) < MIN_N or len(int_vals) < MIN_N:
            print(f"  {gene}: insufficient samples")
            continue
        stat, pval = stats.mannwhitneyu(del_vals, int_vals, alternative="two-sided")
        d = cohens_d(del_vals, int_vals)
        print(f"  {gene}: d={d:.3f}, p={pval:.4f}, "
              f"median del={np.median(del_vals):.4f}, intact={np.median(int_vals):.4f}")

    # Genome-wide screen
    print(f"\nRunning genome-wide screen ({len(crispr.columns)} genes)...")
    screen = genome_wide_screen(crispr, deleted_ids, intact_ids)

    n_sig = (screen["fdr"] < 0.05).sum()
    n_strong = ((screen["fdr"] < 0.05) & (screen["cohens_d"].abs() > 0.3)).sum()
    print(f"\nResults: {n_sig} genes FDR<0.05, {n_strong} with |d|>0.3")

    # Save full results
    out_path = OUTPUT_DIR / "extended_sl_genes.csv"
    screen.to_csv(out_path, index=False)
    print(f"Saved: {out_path}")

    # Print top hits
    sig_strong = screen[(screen["fdr"] < 0.05) & (screen["cohens_d"].abs() > 0.3)]
    if len(sig_strong) > 0:
        print(f"\nTop significant genes (FDR<0.05, |d|>0.3):")
        for _, row in sig_strong.head(20).iterrows():
            marker = " ***" if row["gene"] in HIGHLIGHT_GENES else ""
            print(f"  {row['gene']}: d={row['cohens_d']:.3f}, "
                  f"FDR={row['fdr']:.2e}{marker}")

    # Plots
    print("\nGenerating plots...")
    plot_volcano(screen, FIG_DIR / "volcano_genome_wide.png")
    plot_top_genes(screen, FIG_DIR / "top20_diff_dep.png")


if __name__ == "__main__":
    main()
