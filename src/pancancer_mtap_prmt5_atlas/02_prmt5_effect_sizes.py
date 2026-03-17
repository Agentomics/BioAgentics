"""Phase 2a: Compute PRMT5 SL effect size per cancer type.

For each qualifying cancer type, compares PRMT5 CRISPR dependency between
MTAP-deleted and MTAP-intact lines using Mann-Whitney U, Cohen's d with
bootstrap 95% CI, and FDR correction. Produces a forest plot ranking all
cancer types by SL strength.

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.02_prmt5_effect_sizes
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

# Bootstrap parameters
N_BOOTSTRAP = 1000
SEED = 42

# NSCLC reference from prior analysis
NSCLC_REFERENCE_D = -1.19


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
    # Enforce monotonicity (from largest rank down)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def compute_effect_sizes(classified: pd.DataFrame, prmt5_dep: pd.Series) -> pd.DataFrame:
    """Compute PRMT5 SL effect size per qualifying cancer type."""
    # Load cancer type summary to get qualifying types
    summary = pd.read_csv(OUTPUT_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()

    # Join PRMT5 dependency
    merged = classified.join(prmt5_dep.rename("PRMT5_dep"), how="inner")
    merged = merged.dropna(subset=["MTAP_deleted", "PRMT5_dep"])

    rows = []
    for cancer_type in qualifying:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        deleted = ct_data[ct_data["MTAP_deleted"]]["PRMT5_dep"].values
        intact = ct_data[~ct_data["MTAP_deleted"]]["PRMT5_dep"].values

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
            "stronger_than_nsclc": d < NSCLC_REFERENCE_D,
        })

        print(f"  {cancer_type}: d={d:.3f} [{ci_lo:.3f}, {ci_hi:.3f}], "
              f"p={pval:.2e}, N={len(deleted)}+{len(intact)}")

    result = pd.DataFrame(rows)

    # FDR correction
    result["fdr"] = fdr_correction(result["p_value"].values)
    result["significant"] = result["fdr"] < 0.05

    # Rank by Cohen's d (most negative first = strongest SL)
    result = result.sort_values("cohens_d").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


def plot_forest(result: pd.DataFrame, out_path: Path) -> None:
    """Forest plot: cancer types ranked by Cohen's d ± 95% CI."""
    fig, ax = plt.subplots(figsize=(8, max(6, len(result) * 0.4)))

    y_pos = np.arange(len(result))
    colors = []
    for _, row in result.iterrows():
        if row["significant"] and row["cohens_d"] < 0:
            colors.append("#D95319")  # Significant negative (SL)
        elif row["significant"]:
            colors.append("#4DBEEE")  # Significant positive
        else:
            colors.append("#999999")  # Not significant

    ax.barh(y_pos, result["cohens_d"], xerr=[
        result["cohens_d"] - result["ci_lower"],
        result["ci_upper"] - result["cohens_d"],
    ], color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    # Labels with sample sizes
    labels = [
        f"{row['cancer_type']} (n={row['N_deleted']}+{row['N_intact']})"
        for _, row in result.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    # Reference lines
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.axvline(x=NSCLC_REFERENCE_D, color="blue", linestyle="--", alpha=0.5,
               label=f"NSCLC reference (d={NSCLC_REFERENCE_D})")

    ax.set_xlabel("Cohen's d (MTAP-deleted vs intact PRMT5 dependency)")
    ax.set_title("PRMT5 Synthetic Lethality by Cancer Type\n(negative = stronger SL)")
    ax.legend(loc="lower right", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading classified cell lines...")
    classified = pd.read_csv(OUTPUT_DIR / "all_cell_lines_classified.csv", index_col=0)

    print("Loading CRISPR dependency data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    if "PRMT5" not in crispr.columns:
        raise ValueError("PRMT5 not found in CRISPRGeneEffect")
    prmt5_dep = crispr["PRMT5"]

    print("\nComputing PRMT5 effect sizes per cancer type:")
    result = compute_effect_sizes(classified, prmt5_dep)

    # Save CSV
    out_csv = OUTPUT_DIR / "prmt5_effect_sizes.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {len(result)} cancer types to {out_csv.name}")

    # Save JSON for downstream scripts
    out_json = OUTPUT_DIR / "prmt5_effect_sizes.json"
    with open(out_json, "w") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2)
    print(f"Saved {out_json.name}")

    # Forest plot
    print("\nGenerating forest plot...")
    plot_forest(result, OUTPUT_DIR / "prmt5_forest_plot.png")

    # Summary
    n_sig = result["significant"].sum()
    n_stronger = result["stronger_than_nsclc"].sum()
    print(f"\nSummary: {n_sig} cancer types with FDR < 0.05")
    print(f"  {n_stronger} cancer types with stronger SL than NSCLC reference (d < {NSCLC_REFERENCE_D})")

    # Validate Lung
    lung_row = result[result["cancer_type"] == "Lung"]
    if not lung_row.empty:
        d = lung_row.iloc[0]["cohens_d"]
        print(f"\nValidation: Lung d = {d:.3f} (NSCLC reference = {NSCLC_REFERENCE_D})")


if __name__ == "__main__":
    main()
