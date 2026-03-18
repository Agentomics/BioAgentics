"""Phase 2: PI3K/AKT/mTOR pathway dependency effect sizes by PTEN status.

For each qualifying cancer type, compares CRISPR dependency between
PTEN-lost and PTEN-intact lines for PI3K/AKT/mTOR pathway genes.
PIK3CB is the key positive control (PTEN-null cells depend on p110beta).

Usage:
    uv run python -m pten_loss_pancancer_dependency_atlas.02_pi3k_akt_effect_sizes
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
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "pten-loss-pancancer-dependency-atlas" / "phase2"

# Bootstrap parameters
N_BOOTSTRAP = 1000
SEED = 42

# PI3K/AKT/mTOR pathway target genes
PATHWAY_TARGETS = [
    "PIK3CB",  # KEY POSITIVE CONTROL: p110beta, PTEN-null dependency
    "PIK3CA",  # NEGATIVE CONTROL: p110alpha, should NOT show PTEN selectivity
    "AKT1",    # Direct PTEN substrate
    "AKT2",    # Direct PTEN substrate
    "MTOR",    # Downstream effector
    "RICTOR",  # mTORC2 component
    "RPTOR",   # mTORC1 component
]

# Classification thresholds
ROBUST_FDR = 0.05
ROBUST_D = 0.5
MARGINAL_FDR = 0.10


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2).

    Negative d = group1 more dependent (more negative CRISPR scores).
    """
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


def classify_hit(fdr: float, abs_d: float) -> str:
    """Classify as ROBUST, MARGINAL, or NOT_SIGNIFICANT."""
    if fdr < ROBUST_FDR and abs_d > ROBUST_D:
        return "ROBUST"
    elif fdr < ROBUST_FDR or fdr < MARGINAL_FDR:
        return "MARGINAL"
    return "NOT_SIGNIFICANT"


def compute_pathway_effect_sizes(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Compute effect sizes for PI3K/AKT/mTOR pathway genes."""
    available_targets = [g for g in PATHWAY_TARGETS if g in crispr.columns]
    missing = [g for g in PATHWAY_TARGETS if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")
    print(f"  Testing {len(available_targets)} target genes across {len(qualifying_types)} cancer types")

    merged = classified.join(crispr[available_targets], how="inner")

    rows = []
    for cancer_type in qualifying_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        lost = ct_data[ct_data["PTEN_status"] == "lost"]
        intact = ct_data[ct_data["PTEN_status"] == "intact"]

        for gene in available_targets:
            lost_vals = lost[gene].dropna().values
            intact_vals = intact[gene].dropna().values

            if len(lost_vals) < 3 or len(intact_vals) < 3:
                continue

            _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
            d = cohens_d(lost_vals, intact_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "cohens_d": d,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "p_value": float(pval),
                "n_lost": len(lost_vals),
                "n_intact": len(intact_vals),
                "median_dep_lost": float(np.median(lost_vals)),
                "median_dep_intact": float(np.median(intact_vals)),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["p_value"].values)
        result["classification"] = result.apply(
            lambda row: classify_hit(row["fdr"], abs(row["cohens_d"])), axis=1
        )
    else:
        result["fdr"] = []
        result["classification"] = []

    return result


def leave_one_out_robustness(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    robust_hits: pd.DataFrame,
) -> pd.DataFrame:
    """Leave-one-out robustness test for ROBUST hits."""
    merged = classified.join(
        crispr[[g for g in PATHWAY_TARGETS if g in crispr.columns]], how="inner"
    )

    loo_rows = []
    for _, hit in robust_hits.iterrows():
        cancer_type = hit["cancer_type"]
        gene = hit["gene"]

        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        lost = ct_data[ct_data["PTEN_status"] == "lost"]
        intact = ct_data[ct_data["PTEN_status"] == "intact"]

        lost_vals = lost[gene].dropna()
        intact_vals = intact[gene].dropna().values
        original_d = hit["cohens_d"]

        # Leave-one-out from lost group
        d_values = []
        for idx in lost_vals.index:
            loo_vals = lost_vals.drop(idx).values
            if len(loo_vals) < 2:
                continue
            d_loo = cohens_d(loo_vals, intact_vals)
            d_values.append(d_loo)

        if d_values:
            loo_rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "original_d": original_d,
                "loo_mean_d": float(np.mean(d_values)),
                "loo_min_d": float(np.min(d_values)),
                "loo_max_d": float(np.max(d_values)),
                "loo_std_d": float(np.std(d_values)),
                "n_loo_tests": len(d_values),
                "stable": bool(np.all(np.array(d_values) < -0.3) if original_d < 0
                               else np.all(np.array(d_values) > 0.3)),
            })

    return pd.DataFrame(loo_rows) if loo_rows else pd.DataFrame()


def pik3ca_comutation_stratification(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    qualifying_types: list[str],
) -> pd.DataFrame:
    """Stratify PTEN-lost by PIK3CA co-mutation for double-hit analysis."""
    available_targets = [g for g in PATHWAY_TARGETS if g in crispr.columns]
    merged = classified.join(crispr[available_targets], how="inner")

    rows = []
    for cancer_type in qualifying_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        lost = ct_data[ct_data["PTEN_status"] == "lost"]
        intact = ct_data[ct_data["PTEN_status"] == "intact"]

        lost_pik3ca_wt = lost[~lost["PIK3CA_hotspot"]]
        lost_pik3ca_mut = lost[lost["PIK3CA_hotspot"]]

        for gene in available_targets:
            intact_vals = intact[gene].dropna().values

            # PTEN-lost / PIK3CA-WT vs intact
            pik3ca_wt_vals = lost_pik3ca_wt[gene].dropna().values
            if len(pik3ca_wt_vals) >= 3 and len(intact_vals) >= 3:
                _, p_wt = stats.mannwhitneyu(pik3ca_wt_vals, intact_vals, alternative="two-sided")
                d_wt = cohens_d(pik3ca_wt_vals, intact_vals)
            else:
                p_wt, d_wt = float("nan"), float("nan")

            # PTEN-lost / PIK3CA-mutant vs intact
            pik3ca_mut_vals = lost_pik3ca_mut[gene].dropna().values
            if len(pik3ca_mut_vals) >= 3 and len(intact_vals) >= 3:
                _, p_mut = stats.mannwhitneyu(pik3ca_mut_vals, intact_vals, alternative="two-sided")
                d_mut = cohens_d(pik3ca_mut_vals, intact_vals)
            else:
                p_mut, d_mut = float("nan"), float("nan")

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "n_lost_pik3ca_wt": len(pik3ca_wt_vals),
                "n_lost_pik3ca_mut": len(pik3ca_mut_vals),
                "n_intact": len(intact_vals),
                "d_pten_lost_pik3ca_wt": d_wt,
                "p_pten_lost_pik3ca_wt": p_wt,
                "d_pten_lost_pik3ca_mut": d_mut,
                "p_pten_lost_pik3ca_mut": p_mut,
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def plot_effect_size_heatmap(result: pd.DataFrame, out_path: Path) -> None:
    """Heatmap of Cohen's d across targets x cancer types."""
    pivot = result.pivot_table(
        index="cancer_type", columns="gene", values="cohens_d", aggfunc="first"
    )
    # Order columns by pathway target list order
    ordered_cols = [g for g in PATHWAY_TARGETS if g in pivot.columns]
    pivot = pivot[ordered_cols]

    # Order rows by PIK3CB effect size if available
    if "PIK3CB" in pivot.columns:
        pivot = pivot.sort_values("PIK3CB", ascending=True)

    fig, ax = plt.subplots(figsize=(max(8, len(ordered_cols) * 1.2), max(6, len(pivot) * 0.5)))

    vmax = max(abs(pivot.min().min()), abs(pivot.max().max()), 1.0)
    im = ax.imshow(pivot.values, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(ordered_cols)))
    ax.set_xticklabels(ordered_cols, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate cells
    for i in range(len(pivot)):
        for j in range(len(ordered_cols)):
            val = pivot.iloc[i, j]
            if pd.notna(val):
                # Look up classification
                mask = (result["cancer_type"] == pivot.index[i]) & (result["gene"] == ordered_cols[j])
                cls = result.loc[mask, "classification"].values
                marker = ""
                if len(cls) > 0:
                    if cls[0] == "ROBUST":
                        marker = " **"
                    elif cls[0] == "MARGINAL":
                        marker = " *"
                ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center",
                        fontsize=7, color="white" if abs(val) > vmax * 0.6 else "black")

    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Cohen's d (negative = PTEN-lost more dependent)")
    ax.set_title("PI3K/AKT/mTOR Pathway Dependencies by PTEN Status\n** ROBUST  * MARGINAL")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_text(
    output_dir: Path,
    result: pd.DataFrame,
    loo_df: pd.DataFrame,
    pik3ca_strat: pd.DataFrame,
) -> None:
    """Write human-readable summary text."""
    lines = []
    lines.append("=" * 70)
    lines.append("PTEN Loss Pan-Cancer Dependency Atlas - Phase 2")
    lines.append("PI3K/AKT/mTOR Pathway Dependency Effect Sizes")
    lines.append("=" * 70)
    lines.append("")

    # Overall counts
    n_tests = len(result)
    n_robust = (result["classification"] == "ROBUST").sum()
    n_marginal = (result["classification"] == "MARGINAL").sum()
    lines.append(f"Total tests: {n_tests}")
    lines.append(f"ROBUST hits (FDR<{ROBUST_FDR}, |d|>{ROBUST_D}): {n_robust}")
    lines.append(f"MARGINAL hits: {n_marginal}")
    lines.append("")

    # PIK3CB positive control
    lines.append("--- PIK3CB POSITIVE CONTROL ---")
    pik3cb = result[result["gene"] == "PIK3CB"].sort_values("cohens_d")
    pik3cb_robust = pik3cb[pik3cb["classification"] == "ROBUST"]
    if len(pik3cb_robust) >= 2:
        lines.append(f"PASS: PIK3CB is ROBUST in {len(pik3cb_robust)} cancer types")
    else:
        lines.append(f"** CONCERN: PIK3CB is ROBUST in only {len(pik3cb_robust)} cancer types (need >=2)")
    for _, row in pik3cb.iterrows():
        lines.append(f"  {row['cancer_type']:30s} d={row['cohens_d']:.3f} "
                      f"FDR={row['fdr']:.3e} [{row['classification']}]")
    lines.append("")

    # PIK3CA negative control
    lines.append("--- PIK3CA NEGATIVE CONTROL ---")
    pik3ca = result[result["gene"] == "PIK3CA"].sort_values("cohens_d")
    pik3ca_sig = pik3ca[pik3ca["fdr"] < 0.05]
    if len(pik3ca_sig) == 0:
        lines.append("PASS: PIK3CA shows NO PTEN-selective dependency (as expected)")
    else:
        lines.append(f"** INVESTIGATE: PIK3CA shows significant PTEN-selectivity in {len(pik3ca_sig)} types")
    for _, row in pik3ca.iterrows():
        lines.append(f"  {row['cancer_type']:30s} d={row['cohens_d']:.3f} "
                      f"FDR={row['fdr']:.3e} [{row['classification']}]")
    lines.append("")

    # All ROBUST hits
    robust = result[result["classification"] == "ROBUST"].sort_values("cohens_d")
    lines.append("--- ALL ROBUST HITS ---")
    for _, row in robust.iterrows():
        lines.append(f"  {row['cancer_type']:30s} {row['gene']:10s} d={row['cohens_d']:.3f} "
                      f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] FDR={row['fdr']:.3e}")
    lines.append("")

    # LOO robustness
    if len(loo_df) > 0:
        lines.append("--- LEAVE-ONE-OUT ROBUSTNESS ---")
        for _, row in loo_df.iterrows():
            stability = "STABLE" if row["stable"] else "UNSTABLE"
            lines.append(f"  {row['cancer_type']:30s} {row['gene']:10s} "
                          f"orig_d={row['original_d']:.3f} loo_range=[{row['loo_min_d']:.3f}, "
                          f"{row['loo_max_d']:.3f}] {stability}")
        lines.append("")

    # PIK3CA co-mutation stratification highlights
    if len(pik3ca_strat) > 0:
        lines.append("--- PIK3CA CO-MUTATION STRATIFICATION ---")
        for _, row in pik3ca_strat.iterrows():
            if pd.notna(row["d_pten_lost_pik3ca_mut"]) and row["n_lost_pik3ca_mut"] >= 3:
                lines.append(
                    f"  {row['cancer_type']:30s} {row['gene']:10s} "
                    f"PIK3CA-WT: d={row['d_pten_lost_pik3ca_wt']:.3f} (n={row['n_lost_pik3ca_wt']}) "
                    f"PIK3CA-mut: d={row['d_pten_lost_pik3ca_mut']:.3f} (n={row['n_lost_pik3ca_mut']})"
                )
        lines.append("")

    text = "\n".join(lines)
    (output_dir / "pi3k_akt_summary.txt").write_text(text)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: PI3K/AKT/mTOR Pathway Dependency Effect Sizes ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "pten_classification.csv", index_col=0)
    summary = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying = summary[summary["qualifies"]]["cancer_type"].tolist()
    print(f"  {len(qualifying)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")

    # Compute effect sizes
    print("\nComputing PI3K/AKT/mTOR pathway effect sizes...")
    result = compute_pathway_effect_sizes(classified, crispr, qualifying)
    print(f"  {len(result)} total tests computed")

    # Save main results
    result.to_csv(OUTPUT_DIR / "pi3k_akt_effect_sizes.csv", index=False)

    # PIK3CB positive control report
    pik3cb = result[result["gene"] == "PIK3CB"].copy()
    pik3cb.to_csv(OUTPUT_DIR / "pikb_positive_control.csv", index=False)

    pik3cb_robust = pik3cb[pik3cb["classification"] == "ROBUST"]
    print(f"\n--- PIK3CB Positive Control ---")
    if len(pik3cb_robust) >= 2:
        print(f"  PASS: PIK3CB ROBUST in {len(pik3cb_robust)} cancer types")
    else:
        print(f"  ** CONCERN: PIK3CB ROBUST in only {len(pik3cb_robust)} cancer types (need >=2)")
    for _, row in pik3cb.sort_values("cohens_d").iterrows():
        print(f"  {row['cancer_type']:30s} d={row['cohens_d']:.3f} "
              f"FDR={row['fdr']:.3e} [{row['classification']}]")

    # PIK3CA negative control
    pik3ca = result[result["gene"] == "PIK3CA"]
    pik3ca_sig = pik3ca[pik3ca["fdr"] < 0.05]
    print(f"\n--- PIK3CA Negative Control ---")
    if len(pik3ca_sig) == 0:
        print(f"  PASS: No PTEN-selective PIK3CA dependency")
    else:
        print(f"  ** INVESTIGATE: PIK3CA significant in {len(pik3ca_sig)} types")

    # All significant hits
    sig = result[result["fdr"] < 0.05].sort_values("cohens_d")
    print(f"\nSignificant hits (FDR < 0.05): {len(sig)}")
    for _, row in sig.iterrows():
        print(f"  {row['cancer_type']:30s} {row['gene']:10s} d={row['cohens_d']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] FDR={row['fdr']:.3e} "
              f"[{row['classification']}]")

    # Leave-one-out robustness
    robust_hits = result[result["classification"] == "ROBUST"]
    if len(robust_hits) > 0:
        print("\nRunning leave-one-out robustness tests...")
        loo_df = leave_one_out_robustness(classified, crispr, robust_hits)
        if len(loo_df) > 0:
            loo_df.to_csv(OUTPUT_DIR / "loo_robustness.csv", index=False)
            for _, row in loo_df.iterrows():
                stability = "STABLE" if row["stable"] else "UNSTABLE"
                print(f"  {row['cancer_type']:30s} {row['gene']:10s} "
                      f"loo_range=[{row['loo_min_d']:.3f}, {row['loo_max_d']:.3f}] {stability}")
    else:
        loo_df = pd.DataFrame()
        print("\nNo ROBUST hits — skipping LOO test")

    # PIK3CA co-mutation stratification
    print("\nRunning PIK3CA co-mutation stratification...")
    pik3ca_strat = pik3ca_comutation_stratification(classified, crispr, qualifying)
    if len(pik3ca_strat) > 0:
        pik3ca_strat.to_csv(OUTPUT_DIR / "pik3ca_comutation_stratification.csv", index=False)

    # Heatmap
    print("\nGenerating effect size heatmap...")
    plot_effect_size_heatmap(result, OUTPUT_DIR / "effect_size_heatmap.png")
    print("  Saved effect_size_heatmap.png")

    # Summary text
    write_summary_text(OUTPUT_DIR, result, loo_df, pik3ca_strat)
    print("  Saved pi3k_akt_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
