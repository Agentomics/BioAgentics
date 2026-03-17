"""Phase 2: Compute known SL target effect sizes per cancer type.

For each qualifying cancer type, compares CRISPR dependency between
ARID1A-mutant and WT lines for known SL target genes using Mann-Whitney U,
Cohen's d with bootstrap 95% CI, and BH-FDR correction.

Usage:
    uv run python -m arid1a_pancancer_sl_atlas.02_known_sl_effect_sizes
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
PHASE1_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "arid1a-pancancer-sl-atlas" / "phase2"

# Bootstrap parameters
N_BOOTSTRAP = 1000
SEED = 42

# Known SL target genes
SL_TARGETS = [
    "EZH2",       # PRC2 - well-established SL (positive control)
    "ARID1B",     # Paralog SL - most direct
    "USP8",       # Deubiquitinase - FGFR2/STAT3 axis (Saito et al. 2025)
    "BRD2",       # Bromodomain reader
    "BRD4",       # Bromodomain reader - BRD4-driven transcription
    "HDAC1",      # Histone deacetylase
    "HDAC2",      # Histone deacetylase
    "HDAC3",      # Histone deacetylase
    "HDAC6",      # Chromatin compensation
    "ATR",        # Replication stress SL
    "ATRIP",      # Replication stress SL
    "PARP1",      # DNA damage repair SL
    "PARP2",      # DNA damage repair SL
    "HSP90AA1",   # Proteostatic stress SL
    "PIK3CA",     # Co-activated pathway
    "AKT1",       # Co-activated pathway
    "MTOR",       # Co-activated pathway
]

# Reference cancer types with clinical validation (tulmimetostat)
# Ovary = OCCC, Uterus = EC
REFERENCE_LINEAGES = {"Ovary/Fallopian Tube": "OCCC", "Uterus": "EC"}


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


def compute_all_effect_sizes(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Compute SL effect sizes for all target genes across all qualifying cancer types."""
    # Find which SL targets are in CRISPR data
    available_targets = [g for g in SL_TARGETS if g in crispr.columns]
    missing = [g for g in SL_TARGETS if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")
    print(f"  Testing {len(available_targets)} target genes across {len(qualifying_types)} cancer types")

    # Merge classification with CRISPR data
    merged = classified.join(crispr[available_targets], how="inner")

    rows = []
    for cancer_type in qualifying_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        mutant = ct_data[ct_data["ARID1A_status"] == "mutant"]
        wt = ct_data[ct_data["ARID1A_status"] == "WT"]

        for gene in available_targets:
            mut_vals = mutant[gene].dropna().values
            wt_vals = wt[gene].dropna().values

            if len(mut_vals) < 3 or len(wt_vals) < 3:
                continue

            _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")
            d = cohens_d(mut_vals, wt_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(mut_vals, wt_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "cohens_d": d,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(pval),
                "n_mut": len(mut_vals),
                "n_wt": len(wt_vals),
                "median_dep_mut": float(np.median(mut_vals)),
                "median_dep_wt": float(np.median(wt_vals)),
            })

    result = pd.DataFrame(rows)

    # BH-FDR correction across all tests
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    else:
        result["fdr"] = []

    return result


def build_sl_hierarchy(result: pd.DataFrame) -> pd.DataFrame:
    """Rank SL targets per cancer type by effect size."""
    hierarchy_rows = []
    for cancer_type, group in result.groupby("cancer_type"):
        ranked = group.sort_values("cohens_d")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            hierarchy_rows.append({
                "cancer_type": cancer_type,
                "rank": rank,
                "gene": row["gene"],
                "cohens_d": row["cohens_d"],
                "fdr": row["fdr"],
                "n_mut": row["n_mut"],
                "n_wt": row["n_wt"],
            })
    return pd.DataFrame(hierarchy_rows)


def flag_clinical_insights(result: pd.DataFrame) -> None:
    """Print clinical calibration insights using OCCC/EC as anchors."""
    print("\n--- Clinical Calibration (OCCC/EC anchors) ---")

    for lineage, label in REFERENCE_LINEAGES.items():
        ref = result[(result["cancer_type"] == lineage) & (result["gene"] == "EZH2")]
        if ref.empty:
            print(f"  {label} ({lineage}): EZH2 not available")
            continue
        ref_d = ref.iloc[0]["cohens_d"]
        print(f"  {label} ({lineage}): EZH2 d={ref_d:.3f}")

        # Find cancer types that exceed this reference
        ezh2_all = result[result["gene"] == "EZH2"]
        stronger = ezh2_all[ezh2_all["cohens_d"] < ref_d]
        if len(stronger) > 0:
            print(f"    Cancer types with STRONGER EZH2 SL than {label}:")
            for _, row in stronger.iterrows():
                print(f"      {row['cancer_type']}: d={row['cohens_d']:.3f}")

    # Check for non-EZH2 dominant SL per cancer type
    print("\n  Cancer types where non-EZH2 target is strongest SL:")
    for cancer_type, group in result.groupby("cancer_type"):
        strongest = group.loc[group["cohens_d"].idxmin()]
        if strongest["gene"] != "EZH2" and strongest["cohens_d"] < -0.3:
            ezh2_row = group[group["gene"] == "EZH2"]
            ezh2_d = ezh2_row.iloc[0]["cohens_d"] if len(ezh2_row) > 0 else float("nan")
            print(f"    {cancer_type}: {strongest['gene']} d={strongest['cohens_d']:.3f} "
                  f"(vs EZH2 d={ezh2_d:.3f})")


def plot_ezh2_forest(result: pd.DataFrame, out_path: Path) -> None:
    """Forest plot: cancer types ranked by EZH2 SL strength."""
    ezh2 = result[result["gene"] == "EZH2"].sort_values("cohens_d").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(6, len(ezh2) * 0.45)))
    y_pos = np.arange(len(ezh2))

    colors = []
    for _, row in ezh2.iterrows():
        if row["fdr"] < 0.05 and row["cohens_d"] < 0:
            colors.append("#D95319")
        elif row["fdr"] < 0.05:
            colors.append("#4DBEEE")
        else:
            colors.append("#999999")

    ax.barh(y_pos, ezh2["cohens_d"], xerr=[
        ezh2["cohens_d"] - ezh2["ci_lower"],
        ezh2["ci_upper"] - ezh2["cohens_d"],
    ], color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    labels = [
        f"{row['cancer_type']} (n={row['n_mut']}+{row['n_wt']})"
        for _, row in ezh2.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (ARID1A-mutant vs WT EZH2 dependency)")
    ax.set_title("EZH2 Synthetic Lethality by Cancer Type\n(negative = stronger SL in ARID1A-mutant)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(result: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of effect sizes: cancer types x SL target genes."""
    pivot = result.pivot_table(
        index="cancer_type", columns="gene", values="cohens_d", aggfunc="first"
    )
    # Order rows by EZH2 effect size (if available)
    if "EZH2" in pivot.columns:
        pivot = pivot.sort_values("EZH2", ascending=True)

    # Order columns by gene list order
    ordered_cols = [g for g in SL_TARGETS if g in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered_cols) * 0.8), max(6, len(pivot) * 0.45)))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells with values
    for i in range(len(pivot)):
        for j in range(len(ordered_cols)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (negative = SL)")
    ax.set_title("ARID1A SL Effect Sizes: Cancer Type x Target Gene")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Known SL Target Effect Sizes ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "all_cell_lines_classified.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    # Compute effect sizes
    print("\nComputing effect sizes for known SL targets...")
    result = compute_all_effect_sizes(classified, crispr, qualifying)
    print(f"  {len(result)} total tests computed")

    # Save CSV
    out_csv = OUTPUT_DIR / "known_sl_effect_sizes.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved to {out_csv.name}")

    # Save JSON
    out_json = OUTPUT_DIR / "known_sl_effect_sizes.json"
    with open(out_json, "w") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2)

    # Build SL hierarchy
    hierarchy = build_sl_hierarchy(result)
    hierarchy.to_csv(OUTPUT_DIR / "sl_hierarchy_by_cancer_type.csv", index=False)

    # Print top SL hits
    sig_hits = result[result["fdr"] < 0.05].sort_values("cohens_d")
    print(f"\nSignificant SL hits (FDR < 0.05): {len(sig_hits)}")
    for _, row in sig_hits.head(20).iterrows():
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] FDR={row['fdr']:.3e}")

    # Clinical calibration
    flag_clinical_insights(result)

    # Plots
    print("\nGenerating plots...")
    plot_ezh2_forest(result, OUTPUT_DIR / "ezh2_forest_plot.png")
    print("  Saved ezh2_forest_plot.png")
    plot_heatmap(result, OUTPUT_DIR / "all_targets_heatmap.png")
    print("  Saved all_targets_heatmap.png")


if __name__ == "__main__":
    main()
