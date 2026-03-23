"""Phase 2: RAS/MAPK pathway dependency baseline — NF1-lost vs NF1-intact.

Compares RAS/MAPK pathway gene dependencies between NF1-lost and intact lines
across qualifying cancer types. Establishes the baseline for identifying
dependencies BEYOND the expected MAPK axis.

Statistics: Cohen's d, Mann-Whitney U, bootstrap 95% CI (1000 iter, seed=42), BH FDR.

Usage:
    uv run python -m nf1_loss_pancancer_dependency_atlas.02_ras_mapk_baseline
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
    REPO_ROOT / "output" / "cancer" / "nf1-loss-pancancer-dependency-atlas" / "phase2"
)

# RAS/MAPK pathway genes per plan
RAS_MAPK_GENES = [
    "BRAF", "RAF1", "MAP2K1", "MAP2K2", "MAPK1", "MAPK3",
    "SOS1", "GRB2", "PTPN11",
]

# Additional RAS pathway genes for context
RAS_EXTENDED = ["KRAS", "NRAS", "HRAS", "NF1", "ARAF", "SHOC2", "MRAS"]

BOOTSTRAP_N = 1000
BOOTSTRAP_SEED = 42
FDR_THRESHOLD = 0.1
MIN_SAMPLES = 3


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def bootstrap_ci(
    group1: np.ndarray, group2: np.ndarray, n_boot: int = BOOTSTRAP_N, seed: int = BOOTSTRAP_SEED
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.RandomState(seed)
    boot_ds = []
    for _ in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        boot_ds.append(cohens_d(b1, b2))
    return float(np.percentile(boot_ds, 2.5)), float(np.percentile(boot_ds, 97.5))


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
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


def screen_ras_mapk(
    lost_data: pd.DataFrame,
    intact_data: pd.DataFrame,
    genes: list[str],
    context_name: str,
) -> list[dict]:
    """Compare RAS/MAPK gene dependencies between NF1-lost and intact."""
    rows = []
    pvals = []

    for gene in genes:
        if gene not in lost_data.columns or gene not in intact_data.columns:
            continue

        lost_vals = lost_data[gene].dropna().values
        intact_vals = intact_data[gene].dropna().values

        if len(lost_vals) < MIN_SAMPLES or len(intact_vals) < MIN_SAMPLES:
            continue

        d = cohens_d(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
        ci_lo, ci_hi = bootstrap_ci(lost_vals, intact_vals)

        rows.append({
            "cancer_type": context_name,
            "gene": gene,
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "p_value": float(pval),
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "median_dep_lost": round(float(np.median(lost_vals)), 4),
            "median_dep_intact": round(float(np.median(intact_vals)), 4),
            "is_core_mapk": gene in RAS_MAPK_GENES,
        })
        pvals.append(pval)

    if pvals:
        fdrs = fdr_correction(np.array(pvals))
        for i, row in enumerate(rows):
            row["fdr"] = float(fdrs[i])

    return rows


def plot_mapk_heatmap(results: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of RAS/MAPK gene effect sizes across cancer types."""
    core = results[results["is_core_mapk"]]
    if len(core) == 0:
        return

    pivot = core.pivot_table(
        index="gene", columns="cancer_type", values="cohens_d", aggfunc="first"
    )
    if pivot.empty:
        return

    # Order genes by plan order
    gene_order = [g for g in RAS_MAPK_GENES if g in pivot.index]
    pivot = pivot.loc[gene_order]

    fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.0), max(4, len(gene_order) * 0.5)))
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-1.5, vmax=1.5)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=10)

    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7)

    plt.colorbar(im, ax=ax, label="Cohen's d (NF1-lost vs intact)")
    ax.set_title("RAS/MAPK Pathway Dependency: NF1-Lost vs Intact")
    plt.tight_layout()
    fig.savefig(output_dir / "ras_mapk_heatmap.png", dpi=150)
    plt.close(fig)


def plot_forest(results: pd.DataFrame, output_dir: Path) -> None:
    """Forest plot of pan-cancer RAS/MAPK effect sizes with CIs."""
    pancancer = results[
        (results["cancer_type"] == "Pan-cancer (pooled)") & results["is_core_mapk"]
    ].copy()
    if len(pancancer) == 0:
        return

    pancancer = pancancer.sort_values("cohens_d")

    fig, ax = plt.subplots(figsize=(8, max(3, len(pancancer) * 0.4)))

    y_pos = range(len(pancancer))
    ax.errorbar(
        pancancer["cohens_d"],
        y_pos,
        xerr=[
            pancancer["cohens_d"] - pancancer["ci_lower"],
            pancancer["ci_upper"] - pancancer["cohens_d"],
        ],
        fmt="o",
        color="#1E88E5",
        capsize=3,
        markersize=6,
    )

    ax.axvline(0, color="grey", linestyle="--", alpha=0.5)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(pancancer["gene"].values, fontsize=9)
    ax.set_xlabel("Cohen's d (NF1-lost vs intact)\n← more essential in NF1-lost")
    ax.set_title("Pan-Cancer RAS/MAPK Pathway Dependencies")

    for i, (_, row) in enumerate(pancancer.iterrows()):
        sig = "*" if row.get("fdr", 1) < FDR_THRESHOLD else ""
        ax.text(
            row["ci_upper"] + 0.05,
            i,
            f"d={row['cohens_d']:.2f}{sig}",
            va="center",
            fontsize=7,
        )

    plt.tight_layout()
    fig.savefig(output_dir / "ras_mapk_forest.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: RAS/MAPK Pathway Dependency Baseline ===\n")

    # Load Phase 1
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "nf1_loss_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(classified)} lines, {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {len(crispr.columns)} genes, {len(crispr)} cell lines")

    merged = classified.join(crispr, how="inner")
    print(f"  {len(merged)} lines with both classification and CRISPR data")

    # Check which RAS/MAPK genes are available
    all_genes = RAS_MAPK_GENES + RAS_EXTENDED
    available = [g for g in all_genes if g in crispr.columns]
    missing = [g for g in all_genes if g not in crispr.columns]
    print(f"  RAS/MAPK genes available: {len(available)}/{len(all_genes)}")
    if missing:
        print(f"  Missing: {', '.join(missing)}")

    # Screen each qualifying cancer type + pan-cancer
    # Exclude lines with concurrent RAS mutations to isolate NF1-specific effects
    all_rows = []
    contexts = qualifying_types + ["Pan-cancer (pooled)", "Pan-cancer (RAS-excluded)"]

    for context in contexts:
        if context == "Pan-cancer (pooled)":
            ct_data = merged
        elif context == "Pan-cancer (RAS-excluded)":
            ct_data = merged[~merged["has_RAS_mutation"]]
        else:
            ct_data = merged[merged["OncotreeLineage"] == context]

        lost_lines = ct_data[ct_data["NF1_loss"] == True]  # noqa: E712
        intact_lines = ct_data[ct_data["NF1_status"] == "intact"]
        print(f"  Screening {context} ({len(lost_lines)} lost, {len(intact_lines)} intact)...")

        rows = screen_ras_mapk(lost_lines, intact_lines, available, context)
        all_rows.extend(rows)

    all_results = pd.DataFrame(all_rows)
    print(f"\n  Total comparisons: {len(all_results)}")

    # Save full results
    all_results.to_csv(OUTPUT_DIR / "ras_mapk_results.csv", index=False)

    # Significant MAPK pathway dependencies
    sig = all_results[
        (all_results["fdr"] < FDR_THRESHOLD) & (all_results["cohens_d"] < 0)
    ]
    print(f"  Significant MAPK gains (FDR<{FDR_THRESHOLD}, d<0): {len(sig)}")

    # Pan-cancer core MAPK results
    print(f"\nPan-cancer (pooled) RAS/MAPK results:")
    pancancer = all_results[
        (all_results["cancer_type"] == "Pan-cancer (pooled)") & all_results["is_core_mapk"]
    ].sort_values("cohens_d")
    for _, row in pancancer.iterrows():
        sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < 0 else ""
        print(
            f"  {row['gene']}: d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
            f"FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}"
        )

    # RAS-excluded comparison
    print(f"\nPan-cancer (RAS-excluded) core MAPK results:")
    ras_excl = all_results[
        (all_results["cancer_type"] == "Pan-cancer (RAS-excluded)") & all_results["is_core_mapk"]
    ].sort_values("cohens_d")
    for _, row in ras_excl.iterrows():
        sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < 0 else ""
        print(
            f"  {row['gene']}: d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
            f"FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}"
        )

    # Plots
    print("\nGenerating plots...")
    plot_mapk_heatmap(all_results, OUTPUT_DIR)
    plot_forest(all_results, OUTPUT_DIR)

    # Summary text
    summary_lines = [
        "=" * 70,
        "NF1-Loss Pan-Cancer Dependency Atlas - Phase 2: RAS/MAPK Baseline",
        "=" * 70,
        "",
        f"Contexts screened: {', '.join(contexts)}",
        f"Total comparisons: {len(all_results)}",
        f"Significant MAPK gains (FDR<{FDR_THRESHOLD}, d<0): {len(sig)}",
        "",
        "PAN-CANCER (POOLED) CORE MAPK RESULTS",
        "-" * 60,
    ]
    for _, row in pancancer.iterrows():
        sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < 0 else ""
        summary_lines.append(
            f"  {row['gene']}: d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
            f"FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}"
        )

    summary_lines += [
        "",
        "PAN-CANCER (RAS-EXCLUDED) CORE MAPK RESULTS",
        "-" * 60,
    ]
    for _, row in ras_excl.iterrows():
        sig_flag = " ***" if row.get("fdr", 1) < FDR_THRESHOLD and row["cohens_d"] < 0 else ""
        summary_lines.append(
            f"  {row['gene']}: d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
            f"FDR={row.get('fdr', 'N/A'):.3e}{sig_flag}"
        )

    summary_lines.append("")

    with open(OUTPUT_DIR / "ras_mapk_baseline_summary.txt", "w") as f:
        f.write("\n".join(summary_lines))
    print("  ras_mapk_baseline_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
