"""Phase 2: Candidate SL dependency analysis for RB1-loss cancers.

Compares CRISPR dependency in RB1-loss vs RB1-intact per qualifying cancer type
for established SL candidates: CDK2, AURKA, AURKB, CHEK1, WEE1, CSNK2A1 (CK2),
TTK, FOXM1, MYBL2.

Negative controls: CDK4, CDK6 (should NOT show SL — no RB1 to phosphorylate).
CCNE1-amplified subgroup analysis for enhanced CDK2 dependency.

Statistics: Cohen's d, bootstrap 95% CI, Mann-Whitney U, BH FDR, leave-one-out
robustness, permutation testing.

Usage:
    uv run python -m rb1_loss_pancancer_dependency_atlas.02_candidate_dependencies
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
PHASE1_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "rb1-loss-pancancer-dependency-atlas" / "phase2"

# SL candidate genes
SL_CANDIDATES = ["CDK2", "AURKA", "AURKB", "CHEK1", "WEE1", "CSNK2A1", "TTK", "FOXM1", "MYBL2"]

# Negative controls: CDK4/CDK6 should NOT show SL with RB1-loss
NEGATIVE_CONTROLS = ["CDK4", "CDK6"]

# Bootstrap/permutation parameters
N_BOOTSTRAP = 1000
N_PERMUTATIONS = 10000
SEED = 42


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
    n_boot: int = N_BOOTSTRAP, seed: int = SEED,
) -> tuple[float, float]:
    """Bootstrap 95% CI for Cohen's d."""
    rng = np.random.default_rng(seed)
    ds = np.empty(n_boot)
    for i in range(n_boot):
        b1 = rng.choice(group1, size=len(group1), replace=True)
        b2 = rng.choice(group2, size=len(group2), replace=True)
        ds[i] = cohens_d(b1, b2)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5))


def permutation_test(
    group1: np.ndarray, group2: np.ndarray,
    n_perm: int = N_PERMUTATIONS, seed: int = SEED,
) -> float:
    """Permutation test: fraction of permuted |d| >= observed |d|."""
    rng = np.random.default_rng(seed)
    observed_d = abs(cohens_d(group1, group2))
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    count = 0
    for _ in range(n_perm):
        rng.shuffle(combined)
        perm_d = abs(cohens_d(combined[:n1], combined[n1:]))
        if perm_d >= observed_d:
            count += 1
    return (count + 1) / (n_perm + 1)


def leave_one_out_robust(
    group1: np.ndarray, group2: np.ndarray,
) -> tuple[bool, float]:
    """Leave-one-out: remove each group1 line, check if significance flips."""
    if len(group1) < 3:
        return False, 0.0
    base_d = cohens_d(group1, group2)
    base_sign = np.sign(base_d)
    min_abs_d = abs(base_d)
    for i in range(len(group1)):
        reduced = np.delete(group1, i)
        if len(reduced) < 2:
            continue
        d_i = cohens_d(reduced, group2)
        abs_d_i = abs(d_i)
        if abs_d_i < min_abs_d:
            min_abs_d = abs_d_i
        if np.sign(d_i) != base_sign and abs_d_i > 0.1:
            return False, min_abs_d
    return True, min_abs_d


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


def classify_result(fdr: float, ci_lo: float, ci_hi: float, perm_p: float) -> str:
    """Classify as ROBUST, MARGINAL, or NOT_SIGNIFICANT."""
    sig_fdr = fdr < 0.05
    ci_excludes_zero = (ci_lo > 0 and ci_hi > 0) or (ci_lo < 0 and ci_hi < 0)
    sig_perm = perm_p < 0.05
    if sig_fdr and ci_excludes_zero and sig_perm:
        return "ROBUST"
    elif sig_fdr or (ci_excludes_zero and sig_perm):
        return "MARGINAL"
    else:
        return "NOT_SIGNIFICANT"


def compute_effect_sizes(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    cancer_types: list[str],
    genes: list[str],
) -> pd.DataFrame:
    """Compute effect sizes for RB1-loss vs RB1-intact across cancer types."""
    available = [g for g in genes if g in crispr.columns]
    missing = [g for g in genes if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")

    merged = classified.join(crispr[available], how="inner")

    rows = []
    for cancer_type in cancer_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        lost_lines = ct_data[ct_data["RB1_status"] == "lost"]
        intact_lines = ct_data[ct_data["RB1_status"] == "intact"]

        for gene in available:
            lost_vals = lost_lines[gene].dropna().values
            intact_vals = intact_lines[gene].dropna().values

            if len(lost_vals) < 3 or len(intact_vals) < 3:
                continue

            d = cohens_d(lost_vals, intact_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)
            _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
            perm_p = permutation_test(lost_vals, intact_vals)
            loo_robust, loo_min_d = leave_one_out_robust(lost_vals, intact_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "n_lost": len(lost_vals),
                "n_intact": len(intact_vals),
                "cohens_d": round(d, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "pvalue": float(pval),
                "permutation_p": round(perm_p, 4),
                "loo_robust": loo_robust,
                "loo_min_abs_d": round(loo_min_d, 4),
                "median_dep_lost": round(float(np.median(lost_vals)), 4),
                "median_dep_intact": round(float(np.median(intact_vals)), 4),
            })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
        result["classification"] = result.apply(
            lambda r: classify_result(r["fdr"], r["ci_lower"], r["ci_upper"], r["permutation_p"]),
            axis=1,
        )
    return result


def compute_pancancer_pooled(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    genes: list[str],
) -> pd.DataFrame:
    """Compute pan-cancer pooled effect sizes."""
    available = [g for g in genes if g in crispr.columns]
    merged = classified[classified["has_crispr"]].join(crispr[available], how="inner")

    lost_lines = merged[merged["RB1_status"] == "lost"]
    intact_lines = merged[merged["RB1_status"] == "intact"]

    rows = []
    for gene in available:
        lost_vals = lost_lines[gene].dropna().values
        intact_vals = intact_lines[gene].dropna().values

        if len(lost_vals) < 3 or len(intact_vals) < 3:
            continue

        d = cohens_d(lost_vals, intact_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")
        perm_p = permutation_test(lost_vals, intact_vals)
        loo_robust, loo_min_d = leave_one_out_robust(lost_vals, intact_vals)

        rows.append({
            "cancer_type": "Pan-cancer (pooled)",
            "gene": gene,
            "n_lost": len(lost_vals),
            "n_intact": len(intact_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "permutation_p": round(perm_p, 4),
            "loo_robust": loo_robust,
            "loo_min_abs_d": round(loo_min_d, 4),
            "median_dep_lost": round(float(np.median(lost_vals)), 4),
            "median_dep_intact": round(float(np.median(intact_vals)), 4),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
        result["classification"] = result.apply(
            lambda r: classify_result(r["fdr"], r["ci_lower"], r["ci_upper"], r["permutation_p"]),
            axis=1,
        )
    return result


def compute_ccne1_subgroup(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
) -> pd.DataFrame:
    """Test CDK2 dependency in CCNE1-amplified RB1-loss subgroup.

    CCNE1 amplification drives cyclin E/CDK2 overactivity. Combined with RB1
    loss, this should intensify CDK2 dependency.
    """
    if "CDK2" not in crispr.columns:
        print("  WARNING: CDK2 not found in CRISPRGeneEffect")
        return pd.DataFrame()

    merged = classified[classified["has_crispr"]].join(crispr[["CDK2"]], how="inner")

    rows = []
    # Test 1: CCNE1-amp + RB1-loss vs RB1-intact (all CCNE1 status)
    ccne1_rb1loss = merged[(merged["CCNE1_amplified"] == True) & (merged["RB1_status"] == "lost")]
    rb1_intact = merged[merged["RB1_status"] == "intact"]

    if len(ccne1_rb1loss) >= 3 and len(rb1_intact) >= 3:
        lost_vals = ccne1_rb1loss["CDK2"].dropna().values
        intact_vals = rb1_intact["CDK2"].dropna().values
        d = cohens_d(lost_vals, intact_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(lost_vals, intact_vals)
        _, pval = stats.mannwhitneyu(lost_vals, intact_vals, alternative="two-sided")

        rows.append({
            "comparison": "CCNE1-amp+RB1-loss vs RB1-intact",
            "n_group1": len(lost_vals),
            "n_group2": len(intact_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "median_dep_group1": round(float(np.median(lost_vals)), 4),
            "median_dep_group2": round(float(np.median(intact_vals)), 4),
        })

    # Test 2: RB1-loss/CCNE1-amp vs RB1-loss/CCNE1-normal (intensification)
    rb1_loss = merged[merged["RB1_status"] == "lost"]
    ccne1_amp = rb1_loss[rb1_loss["CCNE1_amplified"] == True]
    ccne1_normal = rb1_loss[rb1_loss["CCNE1_amplified"] == False]

    if len(ccne1_amp) >= 3 and len(ccne1_normal) >= 3:
        amp_vals = ccne1_amp["CDK2"].dropna().values
        norm_vals = ccne1_normal["CDK2"].dropna().values
        d = cohens_d(amp_vals, norm_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(amp_vals, norm_vals)
        _, pval = stats.mannwhitneyu(amp_vals, norm_vals, alternative="two-sided")

        rows.append({
            "comparison": "RB1-loss/CCNE1-amp vs RB1-loss/CCNE1-normal",
            "n_group1": len(amp_vals),
            "n_group2": len(norm_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "median_dep_group1": round(float(np.median(amp_vals)), 4),
            "median_dep_group2": round(float(np.median(norm_vals)), 4),
        })

    return pd.DataFrame(rows)


def plot_forest(result: pd.DataFrame, gene: str, out_path: Path) -> None:
    """Forest plot of effect sizes across cancer types for a single gene."""
    gene_data = result[result["gene"] == gene].sort_values("cohens_d").reset_index(drop=True)
    if len(gene_data) == 0:
        return

    fig, ax = plt.subplots(figsize=(8, max(4, len(gene_data) * 0.6)))
    y_pos = np.arange(len(gene_data))

    colors = []
    for _, row in gene_data.iterrows():
        cls = row.get("classification", "NOT_SIGNIFICANT")
        if cls == "ROBUST":
            colors.append("#D95319")
        elif cls == "MARGINAL":
            colors.append("#EDB120")
        else:
            colors.append("#999999")

    xerr_lo = gene_data["cohens_d"].values - gene_data["ci_lower"].values
    xerr_hi = gene_data["ci_upper"].values - gene_data["cohens_d"].values

    ax.barh(y_pos, gene_data["cohens_d"], xerr=[xerr_lo, xerr_hi],
            color=colors, alpha=0.7, height=0.6, capsize=3, ecolor="gray")

    labels = [
        f"{row['cancer_type']} (n={row['n_lost']}+{row['n_intact']}) [{row.get('classification', '')}]"
        for _, row in gene_data.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Cohen's d ({gene} dependency: RB1-loss vs intact)")
    ax.set_title(f"{gene} Dependency by Cancer Type\n"
                 "(negative = more essential in RB1-loss)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap(result: pd.DataFrame, genes: list[str], output_dir: Path) -> None:
    """Heatmap of effect sizes: genes x cancer types."""
    if len(result) == 0:
        return

    pivot = result.pivot_table(index="cancer_type", columns="gene", values="cohens_d")
    present_genes = [g for g in genes if g in pivot.columns]
    if not present_genes:
        return
    pivot = pivot[present_genes]

    fig, ax = plt.subplots(figsize=(max(8, len(present_genes) * 1.2), max(6, len(pivot) * 0.5)))

    data = pivot.values
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.5)
    im = ax.imshow(data, cmap="RdBu_r", aspect="auto", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(present_genes)))
    ax.set_xticklabels(present_genes, rotation=45, ha="right", fontsize=10)
    ax.set_yticks(range(len(pivot)))
    ax.set_yticklabels(pivot.index, fontsize=9)

    # Annotate with d values and classification
    cls_pivot = result.pivot_table(index="cancer_type", columns="gene", values="classification", aggfunc="first")
    for i in range(len(pivot)):
        for j in range(len(present_genes)):
            val = data[i, j]
            if np.isnan(val):
                continue
            gene = present_genes[j]
            ct = pivot.index[i]
            cls = cls_pivot.loc[ct, gene] if ct in cls_pivot.index and gene in cls_pivot.columns else ""
            marker = ""
            if cls == "ROBUST":
                marker = " **"
            elif cls == "MARGINAL":
                marker = " *"
            color = "white" if abs(val) > vmax * 0.6 else "black"
            ax.text(j, i, f"{val:.2f}{marker}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Cohen's d (negative = more essential in RB1-loss)")
    ax.set_title("RB1-Loss SL Candidate Dependencies\n(** ROBUST, * MARGINAL)")
    fig.tight_layout()
    fig.savefig(output_dir / "sl_candidate_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_txt(
    sl_result: pd.DataFrame,
    control_result: pd.DataFrame,
    ccne1_result: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary text file."""
    lines = [
        "=" * 70,
        "RB1-Loss Pan-Cancer Dependency Atlas - Phase 2: Candidate SL Dependencies",
        "=" * 70,
        "",
    ]

    # SL candidates by gene
    for gene in SL_CANDIDATES:
        gene_data = sl_result[sl_result["gene"] == gene]
        if len(gene_data) == 0:
            lines.append(f"{gene}: No data (missing from CRISPR dataset)")
            lines.append("")
            continue

        n_robust = (gene_data["classification"] == "ROBUST").sum()
        n_marginal = (gene_data["classification"] == "MARGINAL").sum()
        n_ns = (gene_data["classification"] == "NOT_SIGNIFICANT").sum()

        lines.append(f"{gene} DEPENDENCY (RB1-loss vs intact)")
        lines.append(f"  {n_robust} ROBUST, {n_marginal} MARGINAL, {n_ns} NOT_SIGNIFICANT")
        lines.append("-" * 60)

        for _, row in gene_data.sort_values("cohens_d").iterrows():
            lines.append(
                f"  {row['cancer_type']}: d={row['cohens_d']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                f"FDR={row['fdr']:.3e} perm_p={row['permutation_p']:.4f} "
                f"LOO={'robust' if row['loo_robust'] else 'fragile'} "
                f"-> {row['classification']}"
            )
        lines.append("")

    # Universality analysis
    lines.append("UNIVERSALITY ANALYSIS")
    lines.append("-" * 60)
    for gene in SL_CANDIDATES:
        gene_data = sl_result[sl_result["gene"] == gene]
        if len(gene_data) == 0:
            continue
        robust_types = gene_data[gene_data["classification"] == "ROBUST"]["cancer_type"].tolist()
        if robust_types:
            lines.append(f"  {gene}: ROBUST in {len(robust_types)} types: {', '.join(robust_types)}")
        else:
            lines.append(f"  {gene}: No ROBUST results")
    lines.append("")

    # Negative controls
    lines.append("NEGATIVE CONTROLS: CDK4, CDK6 (should NOT show SL with RB1-loss)")
    lines.append("-" * 60)
    for _, row in control_result.iterrows():
        cls = row.get("classification", "N/A")
        expected = "PASS" if cls == "NOT_SIGNIFICANT" else "UNEXPECTED"
        lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"FDR={row.get('fdr', float('nan')):.3e} -> {cls} ({expected})"
        )
    lines.append("")

    # CCNE1 subgroup
    lines.append("CCNE1 AMPLIFICATION SUBGROUP (CDK2 dependency)")
    lines.append("-" * 60)
    if len(ccne1_result) == 0:
        lines.append("  No results (insufficient CCNE1-amplified lines)")
    else:
        for _, row in ccne1_result.iterrows():
            lines.append(
                f"  {row['comparison']}: d={row['cohens_d']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                f"p={row['pvalue']:.3e}"
            )
    lines.append("")

    with open(output_dir / "candidate_dependencies_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: Candidate SL Dependency Analysis ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "rb1_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {len(crispr)} lines, {len(crispr.columns)} genes")

    # --- SL candidate effect sizes ---
    print(f"\nComputing effect sizes for SL candidates: {SL_CANDIDATES}")
    per_type = compute_effect_sizes(classified, crispr, qualifying_types, SL_CANDIDATES)
    pooled = compute_pancancer_pooled(classified, crispr, SL_CANDIDATES)
    sl_result = pd.concat([per_type, pooled], ignore_index=True)

    for gene in SL_CANDIDATES:
        gene_data = sl_result[sl_result["gene"] == gene]
        n_robust = (gene_data["classification"] == "ROBUST").sum() if len(gene_data) > 0 else 0
        n_total = len(gene_data)
        print(f"  {gene}: {n_total} tests, {n_robust} ROBUST")

    # --- Negative controls ---
    print(f"\nComputing negative controls: {NEGATIVE_CONTROLS}")
    ctrl_per_type = compute_effect_sizes(classified, crispr, qualifying_types, NEGATIVE_CONTROLS)
    ctrl_pooled = compute_pancancer_pooled(classified, crispr, NEGATIVE_CONTROLS)
    control_result = pd.concat([ctrl_per_type, ctrl_pooled], ignore_index=True)

    for gene in NEGATIVE_CONTROLS:
        gene_data = control_result[control_result["gene"] == gene]
        n_unexpected = (gene_data["classification"] != "NOT_SIGNIFICANT").sum() if len(gene_data) > 0 else 0
        print(f"  {gene}: {n_unexpected} unexpected significant results")

    # --- CCNE1 subgroup analysis ---
    print("\nCCNE1 amplification subgroup (CDK2 dependency)...")
    ccne1_result = compute_ccne1_subgroup(classified, crispr)
    for _, row in ccne1_result.iterrows():
        print(f"  {row['comparison']}: d={row['cohens_d']:.3f}, p={row['pvalue']:.3e}")

    # --- Key results summary ---
    print("\nKEY RESULTS (pan-cancer pooled):")
    pooled_sl = sl_result[sl_result["cancer_type"] == "Pan-cancer (pooled)"]
    for _, row in pooled_sl.sort_values("cohens_d").iterrows():
        print(f"  {row['gene']}: d={row['cohens_d']:.3f} [{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
              f"FDR={row['fdr']:.3e} -> {row['classification']}")

    # --- Save outputs ---
    print("\nSaving outputs...")
    sl_result.to_csv(OUTPUT_DIR / "sl_candidate_effects.csv", index=False)
    control_result.to_csv(OUTPUT_DIR / "negative_control_effects.csv", index=False)
    ccne1_result.to_csv(OUTPUT_DIR / "ccne1_subgroup_cdk2.csv", index=False)

    # Forest plots for each SL candidate
    for gene in SL_CANDIDATES:
        if gene in sl_result["gene"].values:
            plot_forest(sl_result, gene, OUTPUT_DIR / f"{gene.lower()}_forest_plot.png")

    # Negative control forest plots
    for gene in NEGATIVE_CONTROLS:
        if gene in control_result["gene"].values:
            plot_forest(control_result, gene, OUTPUT_DIR / f"{gene.lower()}_control_forest_plot.png")

    # Heatmap
    plot_heatmap(sl_result, SL_CANDIDATES, OUTPUT_DIR)

    # Summary
    write_summary_txt(sl_result, control_result, ccne1_result, OUTPUT_DIR)

    print("  Saved all CSVs, forest plots, heatmap, and summary.")
    print("\nDone.")


if __name__ == "__main__":
    main()
