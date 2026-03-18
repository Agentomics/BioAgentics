"""Phase 7: OncotreeSubtype-level PRMT5 SL analysis.

Breaks down qualifying cancer types by OncotreeSubtype to reveal
subtype-specific PRMT5 synthetic lethality signals hidden by lineage-level
aggregation. Primarily motivated by CNS/Brain where GBM (OncotreeSubtype=
"Glioblastoma") shows stronger SL (d=-0.74) than the aggregated lineage
(d=-0.68).

Usage:
    uv run python -m pancancer_mtap_prmt5_atlas.07_subtype_prmt5_analysis
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

# Minimum samples per group for subtype-level analysis
MIN_DELETED = 5
MIN_INTACT = 5


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


def compute_subtype_effect_sizes(
    classified: pd.DataFrame, prmt5_dep: pd.Series,
) -> pd.DataFrame:
    """Compute PRMT5 SL effect size per OncotreeSubtype within qualifying lineages."""
    # Load lineage-level summary to get qualifying lineages
    summary = pd.read_csv(OUTPUT_DIR / "cancer_type_summary.csv")
    qualifying_lineages = summary[summary["qualifies"]]["cancer_type"].tolist()

    merged = classified.join(prmt5_dep.rename("PRMT5_dep"), how="inner")
    merged = merged.dropna(subset=["MTAP_deleted", "PRMT5_dep"])

    rows = []
    for lineage in qualifying_lineages:
        lin_data = merged[merged["OncotreeLineage"] == lineage]

        # Get subtypes within this lineage
        subtypes = lin_data["OncotreeSubtype"].dropna().unique()
        if len(subtypes) <= 1:
            continue

        for subtype in subtypes:
            sub_data = lin_data[lin_data["OncotreeSubtype"] == subtype]
            deleted = sub_data[sub_data["MTAP_deleted"]]["PRMT5_dep"].values
            intact = sub_data[~sub_data["MTAP_deleted"]]["PRMT5_dep"].values

            if len(deleted) < MIN_DELETED or len(intact) < MIN_INTACT:
                continue

            stat, pval = stats.mannwhitneyu(deleted, intact, alternative="two-sided")
            d = cohens_d(deleted, intact)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(deleted, intact)

            rows.append({
                "lineage": lineage,
                "subtype": subtype,
                "N_deleted": len(deleted),
                "N_intact": len(intact),
                "median_dep_deleted": float(np.median(deleted)),
                "median_dep_intact": float(np.median(intact)),
                "cohens_d": d,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(pval),
            })

    result = pd.DataFrame(rows)
    if result.empty:
        return result

    result["fdr"] = fdr_correction(result["p_value"].values)
    result["significant"] = result["fdr"] < 0.05
    result = result.sort_values("cohens_d").reset_index(drop=True)
    result["rank"] = range(1, len(result) + 1)

    return result


def compute_lineage_comparison(
    classified: pd.DataFrame, prmt5_dep: pd.Series,
    subtype_results: pd.DataFrame,
) -> pd.DataFrame:
    """For each subtype result, add the parent lineage effect size for comparison."""
    lineage_es = pd.read_csv(OUTPUT_DIR / "prmt5_effect_sizes.csv")
    lineage_map = lineage_es.set_index("cancer_type")["cohens_d"].to_dict()

    subtype_results = subtype_results.copy()
    subtype_results["lineage_cohens_d"] = subtype_results["lineage"].map(lineage_map)
    subtype_results["stronger_than_lineage"] = (
        subtype_results["cohens_d"] < subtype_results["lineage_cohens_d"]
    )
    return subtype_results


def plot_subtype_forest(result: pd.DataFrame, lineage: str, out_path: Path) -> None:
    """Forest plot for subtypes within a specific lineage."""
    sub = result[result["lineage"] == lineage].copy()
    if sub.empty:
        return

    # Also show the lineage-level effect for reference
    lineage_es = pd.read_csv(OUTPUT_DIR / "prmt5_effect_sizes.csv")
    lineage_row = lineage_es[lineage_es["cancer_type"] == lineage]

    fig, ax = plt.subplots(figsize=(8, max(4, (len(sub) + 1) * 0.5)))

    y_pos = np.arange(len(sub))
    colors = []
    for _, row in sub.iterrows():
        if row["significant"] and row["cohens_d"] < 0:
            colors.append("#D95319")
        elif row["significant"]:
            colors.append("#4DBEEE")
        else:
            colors.append("#999999")

    ax.barh(y_pos, sub["cohens_d"], xerr=[
        sub["cohens_d"] - sub["ci_lower"],
        sub["ci_upper"] - sub["cohens_d"],
    ], color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    labels = [
        f"{row['subtype']} (n={row['N_deleted']}+{row['N_intact']})"
        for _, row in sub.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linewidth=0.8)

    # Add lineage-level reference line
    if not lineage_row.empty:
        lin_d = lineage_row.iloc[0]["cohens_d"]
        ax.axvline(x=lin_d, color="blue", linestyle="--", alpha=0.5,
                   label=f"{lineage} aggregate (d={lin_d:.2f})")
        ax.legend(loc="lower right", fontsize=8)

    ax.set_xlabel("Cohen's d (MTAP-deleted vs intact PRMT5 dependency)")
    ax.set_title(f"PRMT5 SL by OncotreeSubtype within {lineage}\n(negative = stronger SL)")

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

    print("\nComputing PRMT5 effect sizes by OncotreeSubtype:")
    result = compute_subtype_effect_sizes(classified, prmt5_dep)

    if result.empty:
        print("  No subtypes qualified (≥5 deleted + ≥5 intact)")
        return

    # Add lineage comparison
    result = compute_lineage_comparison(classified, prmt5_dep, result)

    for _, row in result.iterrows():
        flag = "*" if row["significant"] else " "
        stronger = "↑" if row.get("stronger_than_lineage", False) else " "
        print(f" {flag}{stronger} {row['lineage']}/{row['subtype']}: "
              f"d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}], "
              f"p={row['p_value']:.2e}, fdr={row['fdr']:.3f}, "
              f"N={row['N_deleted']}+{row['N_intact']}")

    # Save CSV
    out_csv = OUTPUT_DIR / "subtype_prmt5_effect_sizes.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {len(result)} subtypes to {out_csv.name}")

    # Save JSON
    out_json = OUTPUT_DIR / "subtype_prmt5_effect_sizes.json"
    with open(out_json, "w") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2)
    print(f"Saved {out_json.name}")

    # Forest plots per lineage that has qualifying subtypes
    print("\nGenerating subtype forest plots...")
    for lineage in result["lineage"].unique():
        safe_name = lineage.lower().replace("/", "_").replace(" ", "_")
        plot_subtype_forest(
            result, lineage,
            OUTPUT_DIR / f"subtype_forest_{safe_name}.png",
        )

    # Summary
    n_sig = result["significant"].sum()
    n_stronger = result["stronger_than_lineage"].sum()
    print(f"\nSummary: {n_sig} subtypes with FDR < 0.05")
    print(f"  {n_stronger} subtypes with stronger SL than parent lineage")

    # GBM highlight
    gbm = result[result["subtype"].str.contains("Glioblastoma", case=False, na=False)]
    if not gbm.empty:
        g = gbm.iloc[0]
        print(f"\nGBM highlight: d={g['cohens_d']:.3f}, p={g['p_value']:.2e}, "
              f"fdr={g['fdr']:.3f}, N={g['N_deleted']}+{g['N_intact']}")
        print(f"  vs CNS/Brain aggregate: d={g['lineage_cohens_d']:.3f}")
        print(f"  Stronger than aggregate: {g['stronger_than_lineage']}")


if __name__ == "__main__":
    main()
