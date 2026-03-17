"""Phase 2: Compute known SL target effect sizes per cancer type.

For each qualifying cancer type, compares CRISPR dependency between
SMARCA4-deficient and intact lines for known SL target genes.

Primary target: SMARCA2 (BRM) - canonical paralog SL.
Also runs 3-way mutation class analysis:
  (a) all deficient vs intact
  (b) Class 1 only vs intact
  (c) Class 2 only vs intact

Statistical framework: Mann-Whitney U, Cohen's d with bootstrap 95% CI,
BH-FDR correction.

Usage:
    uv run python -m smarca4_pancancer_sl_atlas.02_known_sl_effect_sizes
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
PHASE1_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "smarca4-pancancer-sl-atlas" / "phase2"

# Bootstrap parameters
N_BOOTSTRAP = 1000
SEED = 42

# Known SL target genes for SMARCA4-deficient context
SL_TARGETS = [
    "SMARCA2",    # BRM - canonical paralog SL (primary target)
    "ARID1A",     # SWI/SNF subunit
    "ARID1B",     # SWI/SNF subunit - paralog
    "PBRM1",      # SWI/SNF (PBAF complex)
    "EZH2",       # PRC2 - known SWI/SNF interactor
    "BRD2",       # Bromodomain reader
    "BRD4",       # Bromodomain reader
    "HDAC1",      # Histone deacetylase
    "HDAC2",      # Histone deacetylase
    "ATR",        # Replication stress SL
    "PARP1",      # DNA damage repair
    "HSP90AA1",   # Proteostatic stress
    "CDK4",       # Cell cycle (clinical relevance in NSCLC)
    "CDK6",       # Cell cycle
]


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


def compute_effect_sizes(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    qualifying_types: list[str],
    deficient_mask: pd.Series | None = None,
    label: str = "all",
) -> pd.DataFrame:
    """Compute SL effect sizes for target genes across qualifying cancer types.

    If deficient_mask is provided, uses it to further filter deficient lines
    (e.g., Class 1 only or Class 2 only).
    """
    available_targets = [g for g in SL_TARGETS if g in crispr.columns]
    merged = classified.join(crispr[available_targets], how="inner")

    rows = []
    for cancer_type in qualifying_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]

        if deficient_mask is not None:
            deficient = ct_data[
                (ct_data["smarca4_status"] == "deficient") &
                ct_data.index.isin(deficient_mask.index[deficient_mask])
            ]
        else:
            deficient = ct_data[ct_data["smarca4_status"] == "deficient"]

        intact = ct_data[ct_data["smarca4_status"] == "intact"]

        for gene in available_targets:
            def_vals = deficient[gene].dropna().values
            int_vals = intact[gene].dropna().values

            if len(def_vals) < 3 or len(int_vals) < 3:
                continue

            _, pval = stats.mannwhitneyu(def_vals, int_vals, alternative="two-sided")
            d = cohens_d(def_vals, int_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(def_vals, int_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "cohort": label,
                "cohens_d": d,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(pval),
                "n_def": len(def_vals),
                "n_intact": len(int_vals),
                "median_dep_def": float(np.median(def_vals)),
                "median_dep_intact": float(np.median(int_vals)),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
    else:
        result["fdr"] = []

    return result


def build_sl_hierarchy(result: pd.DataFrame) -> pd.DataFrame:
    """Rank SL targets per cancer type by effect size."""
    hierarchy_rows = []
    all_df = result[result["cohort"] == "all"]
    for cancer_type, group in all_df.groupby("cancer_type"):
        ranked = group.sort_values("cohens_d")
        for rank, (_, row) in enumerate(ranked.iterrows(), 1):
            hierarchy_rows.append({
                "cancer_type": cancer_type,
                "rank": rank,
                "gene": row["gene"],
                "cohens_d": row["cohens_d"],
                "fdr": row["fdr"],
                "n_def": row["n_def"],
                "n_intact": row["n_intact"],
            })
    return pd.DataFrame(hierarchy_rows)


def plot_smarca2_forest(result: pd.DataFrame, out_path: Path) -> None:
    """Forest plot: cancer types ranked by SMARCA2 SL strength."""
    all_df = result[result["cohort"] == "all"]
    smarca2 = all_df[all_df["gene"] == "SMARCA2"].sort_values("cohens_d").reset_index(drop=True)

    if len(smarca2) == 0:
        print("  WARNING: No SMARCA2 data for forest plot")
        return

    fig, ax = plt.subplots(figsize=(8, max(6, len(smarca2) * 0.6)))
    y_pos = np.arange(len(smarca2))

    colors = []
    for _, row in smarca2.iterrows():
        if row["fdr"] < 0.05 and row["cohens_d"] < 0:
            colors.append("#D95319")
        elif row["fdr"] < 0.05:
            colors.append("#4DBEEE")
        else:
            colors.append("#999999")

    ax.barh(y_pos, smarca2["cohens_d"], xerr=[
        smarca2["cohens_d"] - smarca2["ci_lower"],
        smarca2["ci_upper"] - smarca2["cohens_d"],
    ], color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    labels = [
        f"{row['cancer_type']} (n={row['n_def']}+{row['n_intact']})"
        for _, row in smarca2.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (SMARCA4-deficient vs intact SMARCA2 dependency)")
    ax.set_title("SMARCA2 Synthetic Lethality by Cancer Type\n(negative = stronger SL in SMARCA4-deficient)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_mutation_class_comparison(result: pd.DataFrame, out_path: Path) -> None:
    """Compare SMARCA2 effect sizes between Class 1 and Class 2 mutations."""
    smarca2 = result[result["gene"] == "SMARCA2"].copy()

    # Get cancer types that have both class 1 and class 2 data, or at least class 1
    class1 = smarca2[smarca2["cohort"] == "class_1"]
    class2 = smarca2[smarca2["cohort"] == "class_2"]
    all_cohort = smarca2[smarca2["cohort"] == "all"]

    if len(class1) == 0 and len(all_cohort) == 0:
        print("  WARNING: No data for mutation class comparison plot")
        return

    # Use all cancer types that have at least "all" data
    cancer_types = all_cohort["cancer_type"].unique()

    fig, ax = plt.subplots(figsize=(10, max(5, len(cancer_types) * 0.8)))
    y_pos = np.arange(len(cancer_types))
    bar_height = 0.25

    for i, ct in enumerate(cancer_types):
        for cohort, offset, color, label_prefix in [
            ("all", -bar_height, "#1f77b4", "All"),
            ("class_1", 0, "#E53935", "Class 1"),
            ("class_2", bar_height, "#FB8C00", "Class 2"),
        ]:
            row = smarca2[(smarca2["cancer_type"] == ct) & (smarca2["cohort"] == cohort)]
            if len(row) > 0:
                row = row.iloc[0]
                ax.barh(
                    i + offset, row["cohens_d"],
                    xerr=[[row["cohens_d"] - row["ci_lower"]], [row["ci_upper"] - row["cohens_d"]]],
                    height=bar_height, color=color, alpha=0.7, capsize=2, ecolor="gray",
                    label=label_prefix if i == 0 else "",
                )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cancer_types, fontsize=9)
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Cohen's d (SMARCA2 dependency)")
    ax.set_title("SMARCA2 SL: Mutation Class Comparison\n(Class 1=truncating, Class 2=missense)")
    ax.legend(loc="lower left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(result: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of effect sizes: cancer types x SL target genes (all cohort)."""
    all_df = result[result["cohort"] == "all"]
    pivot = all_df.pivot_table(
        index="cancer_type", columns="gene", values="cohens_d", aggfunc="first"
    )
    if "SMARCA2" in pivot.columns:
        pivot = pivot.sort_values("SMARCA2", ascending=True)

    ordered_cols = [g for g in SL_TARGETS if g in pivot.columns]
    pivot = pivot[ordered_cols]

    fig, ax = plt.subplots(figsize=(max(10, len(ordered_cols) * 0.8), max(6, len(pivot) * 0.5)))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    for i in range(len(pivot)):
        for j in range(len(ordered_cols)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7,
                        color="white" if abs(val) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (negative = SL)")
    ax.set_title("SMARCA4 SL Effect Sizes: Cancer Type x Target Gene")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Known SL Target Effect Sizes (SMARCA2 Primary) ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "smarca4_classified_lines.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types: {qualifying}")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    available_targets = [g for g in SL_TARGETS if g in crispr.columns]
    missing = [g for g in SL_TARGETS if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")
    print(f"  Testing {len(available_targets)} target genes")

    # (a) All deficient vs intact
    print("\n--- (a) All deficient vs intact ---")
    result_all = compute_effect_sizes(classified, crispr, qualifying, label="all")
    print(f"  {len(result_all)} tests computed")

    # (b) Class 1 only vs intact
    print("\n--- (b) Class 1 (truncating) vs intact ---")
    class1_mask = classified["mutation_class"] == "Class_1"
    result_c1 = compute_effect_sizes(classified, crispr, qualifying,
                                     deficient_mask=class1_mask, label="class_1")
    print(f"  {len(result_c1)} tests computed")

    # (c) Class 2 only vs intact
    print("\n--- (c) Class 2 (missense) vs intact ---")
    class2_mask = classified["mutation_class"] == "Class_2"
    result_c2 = compute_effect_sizes(classified, crispr, qualifying,
                                     deficient_mask=class2_mask, label="class_2")
    print(f"  {len(result_c2)} tests computed")

    # Combine all results
    result = pd.concat([result_all, result_c1, result_c2], ignore_index=True)

    # Save CSVs
    out_csv = OUTPUT_DIR / "known_sl_effect_sizes.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved {len(result)} results to {out_csv.name}")

    out_json = OUTPUT_DIR / "known_sl_effect_sizes.json"
    with open(out_json, "w") as f:
        json.dump(result.to_dict(orient="records"), f, indent=2)

    # Mutation class comparison (SMARCA2 focus)
    mutation_class = result[result["gene"] == "SMARCA2"].copy()
    mutation_class.to_csv(OUTPUT_DIR / "mutation_class_comparison.csv", index=False)

    # Build SL hierarchy
    hierarchy = build_sl_hierarchy(result)
    hierarchy.to_csv(OUTPUT_DIR / "sl_hierarchy_by_cancer_type.csv", index=False)

    # Print significant hits (all cohort)
    all_sig = result_all[result_all["fdr"] < 0.05].sort_values("cohens_d")
    print(f"\nSignificant SL hits (FDR < 0.05, all deficient): {len(all_sig)}")
    for _, row in all_sig.head(20).iterrows():
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] FDR={row['fdr']:.3e}")

    # Print SMARCA2 mutation class comparison
    print("\n--- SMARCA2 Mutation Class Comparison ---")
    for ct in qualifying:
        parts = []
        for cohort_label, cohort_df in [("All", result_all), ("C1", result_c1), ("C2", result_c2)]:
            if len(cohort_df) > 0 and "cancer_type" in cohort_df.columns:
                row = cohort_df[(cohort_df["cancer_type"] == ct) & (cohort_df["gene"] == "SMARCA2")]
                if len(row) > 0:
                    r = row.iloc[0]
                    parts.append(f"{cohort_label}: d={r['cohens_d']:.3f} (n={r['n_def']})")
                else:
                    parts.append(f"{cohort_label}: N/A")
            else:
                parts.append(f"{cohort_label}: N/A (too few)")
        print(f"  {ct}: {' | '.join(parts)}")

    # Plots
    print("\nGenerating plots...")
    plot_smarca2_forest(result, OUTPUT_DIR / "smarca2_sl_ranking.png")
    print("  Saved smarca2_sl_ranking.png")
    plot_mutation_class_comparison(result, OUTPUT_DIR / "mutation_class_effect.png")
    print("  Saved mutation_class_effect.png")
    plot_heatmap(result, OUTPUT_DIR / "all_targets_heatmap.png")
    print("  Saved all_targets_heatmap.png")

    print("\n=== Phase 2 Complete ===")


if __name__ == "__main__":
    main()
