"""Phase 2: HR pathway dependency validation — positive and negative controls.

For each qualifying cancer type, computes differential dependency between
BRCA-deficient vs proficient lines for HR pathway positive controls (PARP1,
RAD51 paralogs, POLQ, RPA1, POLB) and validates the 53BP1/SHLD negative
control (loss should reduce PARP1 dependency).

Runs analyses SEPARATELY for BRCA1-deficient, BRCA2-deficient, and
any-BRCA-deficient groups.

Usage:
    uv run python -m brca_pancancer_sl_atlas.02_hr_dependency_validation
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
PHASE1_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "brca-pancancer-sl-atlas" / "phase2"

N_BOOTSTRAP = 1000
SEED = 42

# Positive control genes — must validate with FDR < 0.05
POSITIVE_CONTROLS = [
    "PARP1",      # primary SL partner
    "RAD51C",     # RAD51 paralog
    "RAD51D",     # RAD51 paralog
    "XRCC2",      # RAD51 paralog
    "XRCC3",      # RAD51 paralog
    "POLQ",       # POLθ, validated by ART6043 Phase 2
    "RPA1",       # replication protein A
    "POLB",       # DNA Pol Beta, BER (PMID 40326293)
]

# BRCA group labels for stratified analysis
BRCA_GROUPS = ["any_brca", "brca1_only", "brca2_only"]

# Minimum sample sizes
MIN_DEF = 3
MIN_PROF = 3


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d (pooled SD) for two groups."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return 0.0
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_sd)


def cohens_d_bootstrap_ci(
    group1: np.ndarray, group2: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = SEED
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.RandomState(seed)
    ds = []
    for _ in range(n_boot):
        g1 = rng.choice(group1, size=len(group1), replace=True)
        g2 = rng.choice(group2, size=len(group2), replace=True)
        ds.append(cohens_d(g1, g2))
    ds = np.array(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def fdr_correction(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return pvals.copy()
    order = np.argsort(pvals)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    fdr = pvals * n / ranks
    # Monotonicity enforcement
    sorted_idx = np.argsort(pvals)[::-1]
    sorted_fdr = fdr[sorted_idx]
    for i in range(1, len(sorted_fdr)):
        sorted_fdr[i] = min(sorted_fdr[i], sorted_fdr[i - 1])
    fdr[sorted_idx] = sorted_fdr
    return np.minimum(fdr, 1.0)


def compute_control_effects(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    qualifying_types: list[str],
    genes: list[str],
    brca_group: str,
) -> pd.DataFrame:
    """Compute effect sizes for control genes across qualifying cancer types.

    Args:
        classified: Phase 1 classified lines (indexed by ModelID)
        crispr: CRISPR gene effect matrix (indexed by ModelID)
        qualifying_types: cancer types to analyze
        genes: control genes to test
        brca_group: 'any_brca', 'brca1_only', or 'brca2_only'
    """
    # Merge classified lines with CRISPR data
    common = classified.index.intersection(crispr.index)
    merged = classified.loc[common].copy()

    # Define deficient mask based on BRCA group
    if brca_group == "any_brca":
        def_mask = merged["brca_combined_status"] == "deficient"
    elif brca_group == "brca1_only":
        def_mask = merged["brca1_status"] == "deficient"
    elif brca_group == "brca2_only":
        def_mask = merged["brca2_status"] == "deficient"
    else:
        raise ValueError(f"Unknown brca_group: {brca_group}")

    prof_mask = merged["brca_combined_status"] == "proficient"

    rows = []
    for cancer_type in qualifying_types:
        ct_mask = merged["OncotreeLineage"] == cancer_type
        ct_def = merged[ct_mask & def_mask].index
        ct_prof = merged[ct_mask & prof_mask].index

        if len(ct_def) < MIN_DEF or len(ct_prof) < MIN_PROF:
            continue

        for gene in genes:
            if gene not in crispr.columns:
                continue

            def_vals = crispr.loc[ct_def, gene].dropna().values
            prof_vals = crispr.loc[ct_prof, gene].dropna().values

            if len(def_vals) < MIN_DEF or len(prof_vals) < MIN_PROF:
                continue

            _, pval = stats.mannwhitneyu(def_vals, prof_vals, alternative="two-sided")
            d = cohens_d(def_vals, prof_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(def_vals, prof_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "brca_group": brca_group,
                "cohens_d": round(d, 4),
                "ci_low": round(ci_lo, 4),
                "ci_high": round(ci_hi, 4),
                "pvalue": pval,
                "n_mut": len(def_vals),
                "n_wt": len(prof_vals),
                "mean_def": round(float(def_vals.mean()), 4),
                "mean_prof": round(float(prof_vals.mean()), 4),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    else:
        result["fdr"] = []
    return result


def compute_53bp1_negative_control(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
) -> pd.DataFrame:
    """Validate negative control: 53BP1/SHLD loss should reduce PARP1 dependency.

    Compares PARP1 dependency between BRCA-mut/53BP1-intact vs BRCA-mut/53BP1-lost.
    """
    if "PARP1" not in crispr.columns:
        return pd.DataFrame()

    common = classified.index.intersection(crispr.index)
    merged = classified.loc[common].copy()

    brca_def = merged[merged["brca_combined_status"] == "deficient"]

    rows = []
    for cancer_type in brca_def["OncotreeLineage"].unique():
        ct = brca_def[brca_def["OncotreeLineage"] == cancer_type]
        intact = ct[ct["shld_complex_status"] == "SHLD-intact"].index
        lost = ct[ct["shld_complex_status"] == "SHLD-lost"].index

        # Also do pan-cancer analysis
        if cancer_type != "__pan_cancer__":
            if len(intact) < 2 or len(lost) < 2:
                continue

        intact_vals = crispr.loc[intact.intersection(crispr.index), "PARP1"].dropna().values
        lost_vals = crispr.loc[lost.intersection(crispr.index), "PARP1"].dropna().values

        if len(intact_vals) < 2 or len(lost_vals) < 2:
            continue

        _, pval = stats.mannwhitneyu(intact_vals, lost_vals, alternative="two-sided")
        d = cohens_d(lost_vals, intact_vals)  # lost - intact
        ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)

        rows.append({
            "cancer_type": cancer_type,
            "comparison": "SHLD-lost vs SHLD-intact",
            "gene": "PARP1",
            "cohens_d": round(d, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "pvalue": pval,
            "n_shld_intact": len(intact_vals),
            "n_shld_lost": len(lost_vals),
            "mean_intact": round(float(intact_vals.mean()), 4),
            "mean_lost": round(float(lost_vals.mean()), 4),
        })

    # Pan-cancer analysis
    all_intact = brca_def[brca_def["shld_complex_status"] == "SHLD-intact"].index
    all_lost = brca_def[brca_def["shld_complex_status"] == "SHLD-lost"].index
    intact_vals = crispr.loc[all_intact.intersection(crispr.index), "PARP1"].dropna().values
    lost_vals = crispr.loc[all_lost.intersection(crispr.index), "PARP1"].dropna().values

    if len(intact_vals) >= 2 and len(lost_vals) >= 2:
        _, pval = stats.mannwhitneyu(intact_vals, lost_vals, alternative="two-sided")
        d = cohens_d(lost_vals, intact_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)
        rows.append({
            "cancer_type": "Pan-Cancer",
            "comparison": "SHLD-lost vs SHLD-intact",
            "gene": "PARP1",
            "cohens_d": round(d, 4),
            "ci_low": round(ci_lo, 4),
            "ci_high": round(ci_hi, 4),
            "pvalue": pval,
            "n_shld_intact": len(intact_vals),
            "n_shld_lost": len(lost_vals),
            "mean_intact": round(float(intact_vals.mean()), 4),
            "mean_lost": round(float(lost_vals.mean()), 4),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
    return result


def plot_positive_control_forest(results: pd.DataFrame, output_dir: Path) -> None:
    """Forest plot of all positive controls across cancer types and BRCA groups."""
    if len(results) == 0:
        return

    # Filter to significant or near-significant results for readability
    plot_data = results.copy()

    # Create label combining gene, cancer type, and BRCA group
    plot_data["label"] = (
        plot_data["gene"] + " | " + plot_data["cancer_type"] + " | " + plot_data["brca_group"]
    )
    plot_data = plot_data.sort_values("cohens_d")

    n_rows = len(plot_data)
    fig_height = max(6, n_rows * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    y_pos = range(n_rows)
    colors = []
    for _, row in plot_data.iterrows():
        if row["fdr"] < 0.05 and row["cohens_d"] < 0:
            colors.append("#E53935")  # significant SL (more dependent in deficient)
        elif row["fdr"] < 0.05:
            colors.append("#1E88E5")  # significant but wrong direction
        else:
            colors.append("#9E9E9E")  # not significant

    ax.barh(
        y_pos,
        plot_data["cohens_d"],
        xerr=[
            plot_data["cohens_d"] - plot_data["ci_low"],
            plot_data["ci_high"] - plot_data["cohens_d"],
        ],
        color=colors,
        alpha=0.8,
        capsize=3,
        height=0.7,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_data["label"], fontsize=7)
    ax.axvline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("Cohen's d (negative = more essential in BRCA-deficient)")
    ax.set_title("Positive Control Validation: HR Pathway Dependencies")

    # Add significance markers
    for i, (_, row) in enumerate(plot_data.iterrows()):
        marker = ""
        if row["fdr"] < 0.001:
            marker = "***"
        elif row["fdr"] < 0.01:
            marker = "**"
        elif row["fdr"] < 0.05:
            marker = "*"
        if marker:
            x = row["ci_high"] if row["cohens_d"] > 0 else row["ci_low"]
            ax.text(x + 0.02, i, marker, va="center", fontsize=8, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "positive_control_forest_plot.png", dpi=150)
    plt.close(fig)


def plot_53bp1_comparison(neg_control: pd.DataFrame, output_dir: Path) -> None:
    """Bar plot comparing PARP1 dependency: SHLD-intact vs SHLD-lost."""
    if len(neg_control) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    x = range(len(neg_control))
    colors = ["#E53935" if row["cohens_d"] > 0 else "#1E88E5"
              for _, row in neg_control.iterrows()]

    bars = ax.bar(
        x,
        neg_control["cohens_d"],
        yerr=[
            neg_control["cohens_d"] - neg_control["ci_low"],
            neg_control["ci_high"] - neg_control["cohens_d"],
        ],
        color=colors,
        alpha=0.8,
        capsize=5,
    )

    labels = [
        f"{row['cancer_type']}\n(n={row['n_shld_intact']}v{row['n_shld_lost']})"
        for _, row in neg_control.iterrows()
    ]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Cohen's d (SHLD-lost minus SHLD-intact)")
    ax.set_title("Negative Control: PARP1 Dependency in 53BP1/SHLD-Lost vs Intact\n"
                 "(Positive d = less dependent = PARPi resistance)")

    # Add FDR labels
    for i, (_, row) in enumerate(neg_control.iterrows()):
        marker = ""
        if row["fdr"] < 0.001:
            marker = "***"
        elif row["fdr"] < 0.01:
            marker = "**"
        elif row["fdr"] < 0.05:
            marker = "*"
        if marker:
            y = row["ci_high"] if row["cohens_d"] > 0 else row["ci_low"]
            ax.text(i, y + 0.05, marker, ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(output_dir / "53bp1_parp1_comparison.png", dpi=150)
    plt.close(fig)


def plot_brca1_vs_brca2(results: pd.DataFrame, output_dir: Path) -> None:
    """Grouped bar plot comparing BRCA1 vs BRCA2 effect sizes for each control gene."""
    if len(results) == 0:
        return

    # Pivot to compare brca1 vs brca2
    genes_present = [g for g in POSITIVE_CONTROLS if g in results["gene"].values]
    if not genes_present:
        return

    fig, ax = plt.subplots(figsize=(12, 6))

    # Get unique cancer types with both BRCA1 and BRCA2 data
    b1_data = results[results["brca_group"] == "brca1_only"]
    b2_data = results[results["brca_group"] == "brca2_only"]

    # Create grouped bars by gene, colored by BRCA group, pan-cancer average
    gene_b1 = b1_data.groupby("gene")["cohens_d"].mean()
    gene_b2 = b2_data.groupby("gene")["cohens_d"].mean()

    genes = sorted(set(gene_b1.index) | set(gene_b2.index),
                   key=lambda g: POSITIVE_CONTROLS.index(g) if g in POSITIVE_CONTROLS else 99)
    x = np.arange(len(genes))
    width = 0.35

    b1_vals = [gene_b1.get(g, 0) for g in genes]
    b2_vals = [gene_b2.get(g, 0) for g in genes]

    ax.bar(x - width / 2, b1_vals, width, label="BRCA1-deficient", color="#E53935", alpha=0.8)
    ax.bar(x + width / 2, b2_vals, width, label="BRCA2-deficient", color="#1E88E5", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(genes, fontsize=9, rotation=45, ha="right")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_ylabel("Mean Cohen's d (across cancer types)")
    ax.set_title("BRCA1 vs BRCA2: Differential Dependency on HR Controls\n"
                 "(Negative = more essential in BRCA-deficient)")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / "brca1_vs_brca2_comparison.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: HR Pathway Dependency Validation ===\n")

    # --- Step 1: Load Phase 1 outputs ---
    print("Loading Phase 1 classifier output...")
    classified = pd.read_csv(PHASE1_DIR / "brca_classified_lines.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "cancer_type_summary.csv")
    print(f"  {len(classified)} classified lines")

    # Get qualifying cancer types (primary + exploratory)
    qualifying = summary[
        summary["qualifies_primary"] | summary["qualifies_exploratory"]
    ]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types: {qualifying}")

    # --- Step 2: Load CRISPR dependency data ---
    print("\nLoading CRISPR gene effect data...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {crispr.shape[0]} cell lines, {crispr.shape[1]} genes")

    # Check which control genes are in CRISPR data
    available = [g for g in POSITIVE_CONTROLS if g in crispr.columns]
    missing = [g for g in POSITIVE_CONTROLS if g not in crispr.columns]
    print(f"  Available controls: {available}")
    if missing:
        print(f"  Missing from CRISPR data: {missing}")

    # --- Step 3: Positive control validation ---
    print("\n--- Positive Control Validation ---")
    all_results = []

    for brca_group in BRCA_GROUPS:
        print(f"\nRunning for {brca_group}...")
        result = compute_control_effects(
            classified, crispr, qualifying, available, brca_group
        )
        if len(result) > 0:
            all_results.append(result)
            sig = result[result["fdr"] < 0.05]
            print(f"  {len(result)} tests, {len(sig)} significant (FDR < 0.05)")
            for _, row in sig.iterrows():
                print(f"    {row['gene']} | {row['cancer_type']}: "
                      f"d={row['cohens_d']:.3f} [{row['ci_low']:.3f}, {row['ci_high']:.3f}] "
                      f"FDR={row['fdr']:.4f} (n={row['n_mut']}v{row['n_wt']})")
        else:
            print(f"  No tests could be run (insufficient samples)")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
    else:
        combined = pd.DataFrame()

    # --- Step 4: Negative control — 53BP1/SHLD ---
    print("\n--- Negative Control: 53BP1/SHLD →  PARP1 Dependency ---")
    neg_control = compute_53bp1_negative_control(classified, crispr)
    if len(neg_control) > 0:
        print(f"  {len(neg_control)} comparisons")
        for _, row in neg_control.iterrows():
            direction = "LESS dependent (PARPi resistant)" if row["cohens_d"] > 0 else "MORE dependent"
            sig = f"FDR={row['fdr']:.4f}" if row["fdr"] < 0.05 else "n.s."
            print(f"    {row['cancer_type']}: SHLD-lost {direction}, "
                  f"d={row['cohens_d']:.3f} ({sig}) "
                  f"n={row['n_shld_intact']}v{row['n_shld_lost']}")
    else:
        print("  Insufficient SHLD-lost lines for comparison")

    # --- Step 5: Save outputs ---
    print("\nSaving outputs...")

    if len(combined) > 0:
        combined.to_csv(OUTPUT_DIR / "hr_validation_results.csv", index=False)
        print(f"  hr_validation_results.csv ({len(combined)} rows)")
    else:
        print("  WARNING: No positive control results to save")

    if len(neg_control) > 0:
        neg_control.to_csv(OUTPUT_DIR / "negative_control_53bp1.csv", index=False)
        print(f"  negative_control_53bp1.csv ({len(neg_control)} rows)")

    # --- Step 6: Plots ---
    print("\nGenerating plots...")

    plot_positive_control_forest(combined, OUTPUT_DIR)
    print("  positive_control_forest_plot.png")

    plot_53bp1_comparison(neg_control, OUTPUT_DIR)
    print("  53bp1_parp1_comparison.png")

    plot_brca1_vs_brca2(combined, OUTPUT_DIR)
    print("  brca1_vs_brca2_comparison.png")

    # --- Step 7: Validation summary JSON ---
    summary_json = {
        "total_tests": len(combined) if len(combined) > 0 else 0,
        "significant_tests": int((combined["fdr"] < 0.05).sum()) if len(combined) > 0 else 0,
        "controls_tested": available,
        "controls_missing": missing,
        "qualifying_cancer_types": qualifying,
        "positive_controls": {},
        "negative_control_53bp1": {},
    }

    if len(combined) > 0:
        for gene in available:
            gene_data = combined[combined["gene"] == gene]
            sig_data = gene_data[gene_data["fdr"] < 0.05]
            summary_json["positive_controls"][gene] = {
                "n_tests": len(gene_data),
                "n_significant": len(sig_data),
                "validated": len(sig_data) > 0,
                "cancer_types_validated": sig_data["cancer_type"].unique().tolist() if len(sig_data) > 0 else [],
                "best_effect_size": float(gene_data["cohens_d"].min()) if len(gene_data) > 0 else None,
            }

    if len(neg_control) > 0:
        pan = neg_control[neg_control["cancer_type"] == "Pan-Cancer"]
        summary_json["negative_control_53bp1"] = {
            "pan_cancer_d": float(pan["cohens_d"].iloc[0]) if len(pan) > 0 else None,
            "pan_cancer_fdr": float(pan["fdr"].iloc[0]) if len(pan) > 0 else None,
            "direction_correct": bool(pan["cohens_d"].iloc[0] > 0) if len(pan) > 0 else None,
            "n_cancer_types_tested": len(neg_control[neg_control["cancer_type"] != "Pan-Cancer"]),
        }

    with open(OUTPUT_DIR / "validation_summary.json", "w") as f:
        json.dump(summary_json, f, indent=2)
    print("  validation_summary.json")

    # --- Summary ---
    n_validated = sum(
        1 for v in summary_json["positive_controls"].values() if v.get("validated")
    )
    print(f"\n=== Phase 2 Complete ===")
    print(f"  {n_validated}/{len(available)} positive controls validated (FDR < 0.05)")
    if summary_json["negative_control_53bp1"].get("direction_correct"):
        print("  53BP1/SHLD negative control: CORRECT direction (SHLD-lost = less PARP1 dependent)")
    elif summary_json["negative_control_53bp1"].get("direction_correct") is False:
        print("  53BP1/SHLD negative control: UNEXPECTED direction")
    else:
        print("  53BP1/SHLD negative control: insufficient data")


if __name__ == "__main__":
    main()
