"""Phase 1b: Compare PRMT5 dependency between MTAP-deleted and intact NSCLC lines.

Tests the synthetic lethal relationship between MTAP deletion and PRMT5
dependency using DepMap CRISPR data. Includes pan-cancer positive control.

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.02_prmt5_dependency
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
OUTPUT_DIR = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl"
FIG_DIR = OUTPUT_DIR / "figures"

# MTAP CN threshold (same as classifier)
MTAP_CN_THRESHOLD = 0.5
N_BOOTSTRAP = 10000


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return (group1.mean() - group2.mean()) / pooled_std


def cohens_d_bootstrap_ci(
    group1: np.ndarray, group2: np.ndarray, n_boot: int = N_BOOTSTRAP, alpha: float = 0.05
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(42)
    ds = []
    for _ in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        ds.append(cohens_d(b1, b2))
    ds = np.array(ds)
    lo = np.percentile(ds, 100 * alpha / 2)
    hi = np.percentile(ds, 100 * (1 - alpha / 2))
    return float(lo), float(hi)


def compare_dependency(
    deleted: np.ndarray, intact: np.ndarray, label: str
) -> dict:
    """Compare PRMT5 dependency between MTAP-deleted and intact groups."""
    stat, pval = stats.mannwhitneyu(deleted, intact, alternative="two-sided")
    d = cohens_d(deleted, intact)
    ci_lo, ci_hi = cohens_d_bootstrap_ci(deleted, intact)

    result = {
        "label": label,
        "n_deleted": len(deleted),
        "n_intact": len(intact),
        "median_deleted": float(np.median(deleted)),
        "median_intact": float(np.median(intact)),
        "mean_deleted": float(np.mean(deleted)),
        "mean_intact": float(np.mean(intact)),
        "mannwhitney_U": float(stat),
        "mannwhitney_p": float(pval),
        "cohens_d": float(d),
        "cohens_d_ci_95": [float(ci_lo), float(ci_hi)],
    }

    print(f"\n{label}:")
    print(f"  N: deleted={len(deleted)}, intact={len(intact)}")
    print(f"  Median PRMT5 dep: deleted={np.median(deleted):.4f}, intact={np.median(intact):.4f}")
    print(f"  Wilcoxon p={pval:.2e}")
    print(f"  Cohen's d={d:.3f} (95% CI: [{ci_lo:.3f}, {ci_hi:.3f}])")

    return result


def plot_dependency_boxviolin(
    deleted: np.ndarray, intact: np.ndarray, title: str, out_path: Path
) -> None:
    """Box+strip plot of PRMT5 dependency by MTAP status."""
    fig, ax = plt.subplots(figsize=(5, 6))

    data = [intact, deleted]
    labels = [f"MTAP intact\n(n={len(intact)})", f"MTAP deleted\n(n={len(deleted)})"]
    bp = ax.boxplot(data, tick_labels=labels, widths=0.5, patch_artist=True,
                    medianprops=dict(color="black", linewidth=2))
    bp["boxes"][0].set_facecolor("#4DBEEE")
    bp["boxes"][1].set_facecolor("#D95319")

    # Overlay strip points
    for i, d in enumerate(data):
        jitter = np.random.default_rng(0).normal(0, 0.04, size=len(d))
        ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.4, s=20,
                   color="gray", zorder=3)

    ax.set_ylabel("PRMT5 CRISPR dependency (GeneEffect)")
    ax.set_title(title)
    ax.axhline(y=-1, color="red", linestyle="--", alpha=0.3, label="Strong dep. threshold")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cn_vs_dependency(
    cn_values: np.ndarray, dep_values: np.ndarray, out_path: Path
) -> None:
    """Scatter: MTAP CN vs PRMT5 dependency in NSCLC."""
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(cn_values, dep_values, alpha=0.5, s=30, edgecolors="none")
    ax.axvline(x=MTAP_CN_THRESHOLD, color="red", linestyle="--", alpha=0.5,
               label=f"MTAP del threshold ({MTAP_CN_THRESHOLD})")
    ax.set_xlabel("MTAP CN (DepMap ratio)")
    ax.set_ylabel("PRMT5 CRISPR dependency (GeneEffect)")
    ax.set_title("MTAP Copy Number vs PRMT5 Dependency (NSCLC)")

    # Spearman correlation
    r, p = stats.spearmanr(cn_values, dep_values)
    ax.annotate(f"Spearman r={r:.3f}, p={p:.2e}", xy=(0.05, 0.95),
                xycoords="axes fraction", fontsize=9, va="top")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Load classified NSCLC lines
    classified = pd.read_csv(OUTPUT_DIR / "nsclc_cell_lines_classified.csv", index_col=0)

    # Load CRISPR dependency
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    if "PRMT5" not in crispr.columns:
        raise ValueError("PRMT5 not found in CRISPRGeneEffect")

    prmt5_dep = crispr["PRMT5"]

    # === NSCLC analysis ===
    nsclc_dep = classified.join(prmt5_dep.rename("PRMT5_dep"), how="inner")
    nsclc_dep = nsclc_dep.dropna(subset=["MTAP_deleted", "PRMT5_dep"])

    deleted_dep = nsclc_dep[nsclc_dep["MTAP_deleted"]]["PRMT5_dep"].values
    intact_dep = nsclc_dep[~nsclc_dep["MTAP_deleted"]]["PRMT5_dep"].values

    nsclc_result = compare_dependency(deleted_dep, intact_dep, "NSCLC: PRMT5 dependency by MTAP status")

    # === Pan-cancer positive control ===
    cn_all = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
    mtap_cn_all = cn_all["MTAP"]
    pan = pd.DataFrame({
        "MTAP_CN": mtap_cn_all,
        "PRMT5_dep": prmt5_dep,
    }).dropna()
    pan["MTAP_deleted"] = pan["MTAP_CN"] < MTAP_CN_THRESHOLD

    pan_deleted = pan[pan["MTAP_deleted"]]["PRMT5_dep"].values
    pan_intact = pan[~pan["MTAP_deleted"]]["PRMT5_dep"].values

    pan_result = compare_dependency(pan_deleted, pan_intact, "Pan-cancer: PRMT5 dependency by MTAP status")

    # === Visualizations ===
    print("\nGenerating plots...")
    plot_dependency_boxviolin(
        deleted_dep, intact_dep,
        "PRMT5 Dependency by MTAP Status (NSCLC)",
        FIG_DIR / "prmt5_dep_nsclc_boxplot.png",
    )
    plot_dependency_boxviolin(
        pan_deleted, pan_intact,
        "PRMT5 Dependency by MTAP Status (Pan-cancer)",
        FIG_DIR / "prmt5_dep_pancancer_boxplot.png",
    )

    # CN vs dependency scatter (NSCLC lines with both CN and dep data)
    nsclc_cn_dep = nsclc_dep.dropna(subset=["MTAP_CN_log2"])
    plot_cn_vs_dependency(
        nsclc_cn_dep["MTAP_CN_log2"].values,
        nsclc_cn_dep["PRMT5_dep"].values,
        FIG_DIR / "mtap_cn_vs_prmt5_dep_nsclc.png",
    )

    # === Save results ===
    results = {
        "nsclc": nsclc_result,
        "pan_cancer": pan_result,
    }
    out_path = OUTPUT_DIR / "prmt5_dependency_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
