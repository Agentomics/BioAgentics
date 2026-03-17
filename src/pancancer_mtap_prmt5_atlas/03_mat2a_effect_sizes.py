"""Phase 2b: Compute MAT2A SL effect size per cancer type and compare with PRMT5.

Parallels Phase 2a for MAT2A, then compares PRMT5 vs MAT2A rankings to
identify dual-target opportunities (relevant to IDEAYA IDE892+IDE397 combo trial).

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.03_mat2a_effect_sizes
"""

from __future__ import annotations

import json
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
OUTPUT_DIR = REPO_ROOT / "output" / "pancancer-mtap-prmt5-atlas"

N_BOOTSTRAP = 1000
SEED = 42

# Threshold for "strong" SL: d < -0.5 (medium effect)
STRONG_SL_THRESHOLD = -0.5


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def cohens_d_bootstrap_ci(
    group1: np.ndarray, group2: np.ndarray,
    n_boot: int = N_BOOTSTRAP, alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(SEED)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        ds[i] = cohens_d(b1, b2)
    lo = float(np.percentile(ds, 100 * alpha / 2))
    hi = float(np.percentile(ds, 100 * (1 - alpha / 2)))
    return lo, hi


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
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


def compute_mat2a_effect_sizes(classified: pd.DataFrame, mat2a_dep: pd.Series) -> pd.DataFrame:
    """Compute MAT2A SL effect size per qualifying cancer type."""
    summary = pd.read_csv(OUTPUT_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()

    merged = classified.join(mat2a_dep.rename("MAT2A_dep"), how="inner")
    merged = merged.dropna(subset=["MTAP_deleted", "MAT2A_dep"])

    rows = []
    for cancer_type in qualifying:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        deleted = ct_data[ct_data["MTAP_deleted"]]["MAT2A_dep"].values
        intact = ct_data[~ct_data["MTAP_deleted"]]["MAT2A_dep"].values

        if len(deleted) < 3 or len(intact) < 3:
            continue

        stat, pval = stats.mannwhitneyu(deleted, intact, alternative="two-sided")
        d = cohens_d(deleted, intact)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(deleted, intact)

        rows.append({
            "cancer_type": cancer_type,
            "N_deleted": len(deleted),
            "N_intact": len(intact),
            "median_dep_deleted": float(np.median(deleted)),
            "median_dep_intact": float(np.median(intact)),
            "cohens_d": d,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "p_value": float(pval),
        })

        print(f"  {cancer_type}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
              f"p={pval:.2e}, N={len(deleted)}+{len(intact)}")

    result = pd.DataFrame(rows)
    result["fdr"] = fdr_correction(result["p_value"].values)
    result["significant"] = result["fdr"] < 0.05
    result = result.sort_values("cohens_d").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)
    return result


def plot_mat2a_forest(result: pd.DataFrame, out_path: Path) -> None:
    """Forest plot for MAT2A effect sizes."""
    fig, ax = plt.subplots(figsize=(8, max(6, len(result) * 0.4)))

    y_pos = np.arange(len(result))
    colors = ["#D95319" if (row["significant"] and row["cohens_d"] < 0)
              else "#4DBEEE" if row["significant"]
              else "#999999"
              for _, row in result.iterrows()]

    ax.barh(y_pos, result["cohens_d"], xerr=[
        result["cohens_d"] - result["ci_lower"],
        result["ci_upper"] - result["cohens_d"],
    ], color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    labels = [f"{row['cancer_type']} (n={row['N_deleted']}+{row['N_intact']})"
              for _, row in result.iterrows()]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (MTAP-deleted vs intact MAT2A dependency)")
    ax.set_title("MAT2A Synthetic Lethality by Cancer Type\n(negative = stronger SL)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def compare_prmt5_mat2a(mat2a: pd.DataFrame, out_dir: Path) -> dict:
    """Compare PRMT5 vs MAT2A rankings across cancer types."""
    prmt5 = pd.read_csv(out_dir / "prmt5_effect_sizes.csv")

    # Merge on cancer_type
    merged = prmt5[["cancer_type", "cohens_d", "significant"]].merge(
        mat2a[["cancer_type", "cohens_d", "significant"]],
        on="cancer_type", suffixes=("_prmt5", "_mat2a"),
    )

    # Spearman correlation of effect sizes
    r, p = stats.spearmanr(merged["cohens_d_prmt5"], merged["cohens_d_mat2a"])

    # Classify each cancer type
    classifications = []
    for _, row in merged.iterrows():
        prmt5_strong = row["cohens_d_prmt5"] < STRONG_SL_THRESHOLD
        mat2a_strong = row["cohens_d_mat2a"] < STRONG_SL_THRESHOLD
        if prmt5_strong and mat2a_strong:
            cat = "dual_target"
        elif prmt5_strong:
            cat = "PRMT5_only"
        elif mat2a_strong:
            cat = "MAT2A_only"
        else:
            cat = "neither"
        classifications.append({
            "cancer_type": row["cancer_type"],
            "prmt5_d": float(row["cohens_d_prmt5"]),
            "mat2a_d": float(row["cohens_d_mat2a"]),
            "classification": cat,
        })

    # Scatter plot
    fig, ax = plt.subplots(figsize=(7, 7))
    for entry in classifications:
        color = {"dual_target": "#D95319", "PRMT5_only": "#0072BD",
                 "MAT2A_only": "#77AC30", "neither": "#999999"}[entry["classification"]]
        ax.scatter(entry["prmt5_d"], entry["mat2a_d"], c=color, s=60, zorder=3)
        ax.annotate(entry["cancer_type"], (entry["prmt5_d"], entry["mat2a_d"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")

    ax.axhline(y=STRONG_SL_THRESHOLD, color="gray", linestyle="--", alpha=0.4)
    ax.axvline(x=STRONG_SL_THRESHOLD, color="gray", linestyle="--", alpha=0.4)
    ax.set_xlabel("PRMT5 Cohen's d")
    ax.set_ylabel("MAT2A Cohen's d")
    ax.set_title(f"PRMT5 vs MAT2A SL by Cancer Type\n(Spearman r={r:.2f}, p={p:.2e})")

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D95319', markersize=8, label='Dual target'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#0072BD', markersize=8, label='PRMT5 only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#77AC30', markersize=8, label='MAT2A only'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#999999', markersize=8, label='Neither'),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_dir / "prmt5_vs_mat2a_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: prmt5_vs_mat2a_scatter.png")

    comparison = {
        "spearman_r": float(r),
        "spearman_p": float(p),
        "n_cancer_types": len(merged),
        "classifications": classifications,
        "counts": {cat: sum(1 for c in classifications if c["classification"] == cat)
                   for cat in ["dual_target", "PRMT5_only", "MAT2A_only", "neither"]},
    }
    return comparison


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "all_cell_lines_classified.csv", index_col=0)

    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    if "MAT2A" not in crispr.columns:
        raise ValueError("MAT2A not found in CRISPRGeneEffect")
    mat2a_dep = crispr["MAT2A"]

    print("\nComputing MAT2A effect sizes per cancer type:")
    result = compute_mat2a_effect_sizes(classified, mat2a_dep)

    out_csv = OUTPUT_DIR / "mat2a_effect_sizes.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {len(result)} cancer types to {out_csv.name}")

    print("\nGenerating MAT2A forest plot...")
    plot_mat2a_forest(result, OUTPUT_DIR / "mat2a_forest_plot.png")

    print("\nComparing PRMT5 vs MAT2A rankings...")
    comparison = compare_prmt5_mat2a(result, OUTPUT_DIR)

    out_json = OUTPUT_DIR / "dual_target_comparison.json"
    with open(out_json, "w") as f:
        json.dump(comparison, f, indent=2)
    print(f"Saved {out_json.name}")

    # Summary
    counts = comparison["counts"]
    print(f"\nDual-target classification (d < {STRONG_SL_THRESHOLD}):")
    for cat, n in counts.items():
        print(f"  {cat}: {n}")
    print(f"Spearman correlation: r={comparison['spearman_r']:.2f}, p={comparison['spearman_p']:.2e}")


if __name__ == "__main__":
    main()
