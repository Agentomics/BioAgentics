"""Phase 2: CDK4/6 dependency effect sizes by cancer type in CDKN2A-deleted lines.

Compares CDK4 and CDK6 CRISPR dependency (separately) in CDKN2A-deleted vs
intact lines. Stratifies by RB1 status. Tests MDM2 dependency in
CDKN2A-del/TP53-WT context. CDK2/CDK1 as negative controls.

Statistics: Cohen's d with bootstrap 95% CI, Mann-Whitney U, BH-FDR,
leave-one-out robustness, and permutation testing.

Usage:
    uv run python -m cdkn2a_pancancer_dependency_atlas.02_cdk46_effect_sizes
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
PHASE1_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "cdkn2a-pancancer-dependency-atlas" / "phase2"

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
    del_vals: np.ndarray, intact_vals: np.ndarray,
) -> tuple[bool, float]:
    """Leave-one-out: remove each deleted line, check if significance flips."""
    if len(del_vals) < 3:
        return False, 0.0
    base_d = cohens_d(del_vals, intact_vals)
    base_sign = np.sign(base_d)
    min_abs_d = abs(base_d)
    for i in range(len(del_vals)):
        reduced = np.delete(del_vals, i)
        if len(reduced) < 2:
            continue
        d_i = cohens_d(reduced, intact_vals)
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
    status_col: str = "CDKN2A_status",
    group_del: str = "deleted",
    group_intact: str = "intact",
    rb1_stratum: str = "all",
) -> pd.DataFrame:
    """Compute effect sizes for given genes across cancer types.

    rb1_stratum: 'all' (no filter), 'rb1_intact', or 'rb1_lost'.
    """
    available = [g for g in genes if g in crispr.columns]
    missing = [g for g in genes if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")

    merged = classified.join(crispr[available], how="inner")

    # Apply RB1 stratum filter to deleted lines
    if rb1_stratum == "rb1_intact":
        merged = merged[
            (merged[status_col] == group_intact) |
            ((merged[status_col] == group_del) & (merged["RB1_status"] == "intact"))
        ]
    elif rb1_stratum == "rb1_lost":
        merged = merged[
            (merged[status_col] == group_intact) |
            ((merged[status_col] == group_del) & (merged["RB1_status"] == "lost"))
        ]

    rows = []
    for cancer_type in cancer_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        del_lines = ct_data[ct_data[status_col] == group_del]
        intact_lines = ct_data[ct_data[status_col] == group_intact]

        for gene in available:
            del_vals = del_lines[gene].dropna().values
            intact_vals = intact_lines[gene].dropna().values

            if len(del_vals) < 3 or len(intact_vals) < 3:
                continue

            d = cohens_d(del_vals, intact_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(del_vals, intact_vals)
            _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")
            perm_p = permutation_test(del_vals, intact_vals)
            loo_robust, loo_min_d = leave_one_out_robust(del_vals, intact_vals)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "rb1_stratum": rb1_stratum,
                "n_del": len(del_vals),
                "n_intact": len(intact_vals),
                "cohens_d": round(d, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "pvalue": float(pval),
                "permutation_p": round(perm_p, 4),
                "loo_robust": loo_robust,
                "loo_min_abs_d": round(loo_min_d, 4),
                "median_dep_del": round(float(np.median(del_vals)), 4),
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


def add_pancancer_pooled(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    genes: list[str],
    rb1_stratum: str = "all",
) -> pd.DataFrame:
    """Compute pan-cancer pooled effect sizes."""
    available = [g for g in genes if g in crispr.columns]
    merged = classified[classified["has_crispr"]].join(crispr[available], how="inner")

    if rb1_stratum == "rb1_intact":
        merged = merged[
            (merged["CDKN2A_status"] == "intact") |
            ((merged["CDKN2A_status"] == "deleted") & (merged["RB1_status"] == "intact"))
        ]
    elif rb1_stratum == "rb1_lost":
        merged = merged[
            (merged["CDKN2A_status"] == "intact") |
            ((merged["CDKN2A_status"] == "deleted") & (merged["RB1_status"] == "lost"))
        ]

    del_lines = merged[merged["CDKN2A_status"] == "deleted"]
    intact_lines = merged[merged["CDKN2A_status"] == "intact"]

    rows = []
    for gene in available:
        del_vals = del_lines[gene].dropna().values
        intact_vals = intact_lines[gene].dropna().values

        if len(del_vals) < 3 or len(intact_vals) < 3:
            continue

        d = cohens_d(del_vals, intact_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(del_vals, intact_vals)
        _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")
        perm_p = permutation_test(del_vals, intact_vals)
        loo_robust, loo_min_d = leave_one_out_robust(del_vals, intact_vals)

        rows.append({
            "cancer_type": "Pan-cancer (pooled)",
            "gene": gene,
            "rb1_stratum": rb1_stratum,
            "n_del": len(del_vals),
            "n_intact": len(intact_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "permutation_p": round(perm_p, 4),
            "loo_robust": loo_robust,
            "loo_min_abs_d": round(loo_min_d, 4),
            "median_dep_del": round(float(np.median(del_vals)), 4),
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


def compute_mdm2_effect(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
    cancer_types: list[str],
) -> pd.DataFrame:
    """Test MDM2 dependency in CDKN2A-del/TP53-WT vs CDKN2A-intact/TP53-WT.

    ARF loss (via CDKN2A deletion) liberates MDM2, potentially creating
    MDM2 dependency in TP53-WT tumors.
    """
    if "MDM2" not in crispr.columns:
        print("  WARNING: MDM2 not found in CRISPRGeneEffect")
        return pd.DataFrame()

    merged = classified.join(crispr[["MDM2"]], how="inner")
    tp53_wt = merged[merged["TP53_status"] == "WT"]

    rows = []
    for cancer_type in cancer_types + ["Pan-cancer (pooled)"]:
        if cancer_type == "Pan-cancer (pooled)":
            ct_data = tp53_wt
        else:
            ct_data = tp53_wt[tp53_wt["OncotreeLineage"] == cancer_type]

        del_vals = ct_data.loc[ct_data["CDKN2A_status"] == "deleted", "MDM2"].dropna().values
        intact_vals = ct_data.loc[ct_data["CDKN2A_status"] == "intact", "MDM2"].dropna().values

        if len(del_vals) < 3 or len(intact_vals) < 3:
            continue

        d = cohens_d(del_vals, intact_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(del_vals, intact_vals)
        _, pval = stats.mannwhitneyu(del_vals, intact_vals, alternative="two-sided")
        perm_p = permutation_test(del_vals, intact_vals)
        loo_robust, loo_min_d = leave_one_out_robust(del_vals, intact_vals)

        rows.append({
            "cancer_type": cancer_type,
            "gene": "MDM2",
            "context": "CDKN2A-del/TP53-WT vs CDKN2A-intact/TP53-WT",
            "n_del": len(del_vals),
            "n_intact": len(intact_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "permutation_p": round(perm_p, 4),
            "loo_robust": loo_robust,
            "loo_min_abs_d": round(loo_min_d, 4),
            "median_dep_del": round(float(np.median(del_vals)), 4),
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


def compute_rb1_coloss_impact(
    classified: pd.DataFrame,
    crispr: pd.DataFrame,
) -> pd.DataFrame:
    """Quantify how RB1 co-loss abolishes CDK4/6 dependency.

    Pan-cancer: compare CDK4/CDK6 dependency in CDKN2A-del/RB1-intact vs
    CDKN2A-del/RB1-lost.
    """
    genes = [g for g in ["CDK4", "CDK6"] if g in crispr.columns]
    merged = classified[
        (classified["CDKN2A_status"] == "deleted") & classified["has_crispr"]
    ].join(crispr[genes], how="inner")

    rb1_intact = merged[merged["RB1_status"] == "intact"]
    rb1_lost = merged[merged["RB1_status"] == "lost"]

    rows = []
    for gene in genes:
        intact_vals = rb1_intact[gene].dropna().values
        lost_vals = rb1_lost[gene].dropna().values

        if len(intact_vals) < 3 or len(lost_vals) < 2:
            rows.append({
                "gene": gene,
                "n_rb1_intact": len(intact_vals),
                "n_rb1_lost": len(lost_vals),
                "cohens_d": float("nan"),
                "pvalue": float("nan"),
                "median_dep_rb1_intact": round(float(np.median(intact_vals)), 4) if len(intact_vals) > 0 else float("nan"),
                "median_dep_rb1_lost": round(float(np.median(lost_vals)), 4) if len(lost_vals) > 0 else float("nan"),
                "note": "insufficient RB1-lost samples for statistical test",
            })
            continue

        d = cohens_d(intact_vals, lost_vals)
        _, pval = stats.mannwhitneyu(intact_vals, lost_vals, alternative="two-sided")

        rows.append({
            "gene": gene,
            "n_rb1_intact": len(intact_vals),
            "n_rb1_lost": len(lost_vals),
            "cohens_d": round(d, 4),
            "pvalue": float(pval),
            "median_dep_rb1_intact": round(float(np.median(intact_vals)), 4),
            "median_dep_rb1_lost": round(float(np.median(lost_vals)), 4),
            "note": "",
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

    stratum = gene_data["rb1_stratum"].iloc[0] if "rb1_stratum" in gene_data.columns else "all"
    labels = [
        f"{row['cancer_type']} (n={row['n_del']}+{row['n_intact']}) [{row.get('classification', '')}]"
        for _, row in gene_data.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Cohen's d ({gene} dependency: CDKN2A-del vs intact)")
    ax.set_title(f"{gene} Dependency by Cancer Type (RB1 stratum: {stratum})\n"
                 "(negative = more essential in CDKN2A-deleted)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_txt(
    cdk4_result: pd.DataFrame,
    cdk6_result: pd.DataFrame,
    control_result: pd.DataFrame,
    mdm2_result: pd.DataFrame,
    rb1_impact: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 60,
        "CDKN2A Pan-Cancer Dependency Atlas - Phase 2: CDK4/6 Effect Sizes",
        "=" * 60,
        "",
    ]

    for label, result, gene in [
        ("CDK4", cdk4_result, "CDK4"),
        ("CDK6", cdk6_result, "CDK6"),
    ]:
        # Show primary analysis (RB1-intact stratum)
        primary = result[result["rb1_stratum"] == "rb1_intact"]
        lines.append(f"{label} DEPENDENCY (CDKN2A-del/RB1-intact vs intact)")
        lines.append("-" * 50)
        if len(primary) == 0:
            lines.append("  No results (insufficient samples in RB1-intact stratum)")
        else:
            for _, row in primary.iterrows():
                lines.append(
                    f"  {row['cancer_type']}: d={row['cohens_d']:.3f} "
                    f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                    f"FDR={row['fdr']:.3e} perm_p={row['permutation_p']:.4f} "
                    f"LOO={'robust' if row['loo_robust'] else 'fragile'} "
                    f"-> {row['classification']}"
                )
        lines.append("")

    # RB1 co-loss impact
    lines.append("RB1 CO-LOSS IMPACT (within CDKN2A-deleted)")
    lines.append("-" * 50)
    for _, row in rb1_impact.iterrows():
        if row.get("note"):
            lines.append(f"  {row['gene']}: {row['note']}")
        else:
            lines.append(
                f"  {row['gene']}: RB1-intact median={row['median_dep_rb1_intact']:.3f} "
                f"vs RB1-lost median={row['median_dep_rb1_lost']:.3f} "
                f"d={row['cohens_d']:.3f} p={row['pvalue']:.3e}"
            )
    lines.append("")

    # MDM2
    lines.append("MDM2 DEPENDENCY (CDKN2A-del/TP53-WT vs intact/TP53-WT)")
    lines.append("-" * 50)
    if len(mdm2_result) == 0:
        lines.append("  No results")
    else:
        for _, row in mdm2_result.iterrows():
            lines.append(
                f"  {row['cancer_type']}: d={row['cohens_d']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                f"FDR={row['fdr']:.3e} -> {row['classification']}"
            )
    lines.append("")

    # Controls
    lines.append("CONTROL GENES CDK2/CDK1 (expect NOT_SIGNIFICANT)")
    lines.append("-" * 50)
    for _, row in control_result.iterrows():
        cls = row.get("classification", "N/A")
        lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"FDR={row.get('fdr', float('nan')):.3e} -> {cls}"
        )
    lines.append("")

    with open(output_dir / "cdk46_effect_sizes_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: CDK4/6 Effect Sizes by Cancer Type ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "cdkn2a_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {len(crispr)} lines, {len(crispr.columns)} genes")

    # --- CDK4 effect sizes across 3 RB1 strata ---
    cdk4_frames = []
    cdk6_frames = []

    for stratum in ["all", "rb1_intact", "rb1_lost"]:
        print(f"\nComputing CDK4/CDK6 effect sizes (RB1 stratum: {stratum})...")

        per_type = compute_effect_sizes(
            classified, crispr, qualifying_types, ["CDK4", "CDK6"],
            rb1_stratum=stratum,
        )

        pooled = add_pancancer_pooled(classified, crispr, ["CDK4", "CDK6"], rb1_stratum=stratum)
        combined = pd.concat([per_type, pooled], ignore_index=True)

        cdk4 = combined[combined["gene"] == "CDK4"]
        cdk6 = combined[combined["gene"] == "CDK6"]

        n_robust_4 = (cdk4["classification"] == "ROBUST").sum() if len(cdk4) > 0 else 0
        n_robust_6 = (cdk6["classification"] == "ROBUST").sum() if len(cdk6) > 0 else 0
        print(f"  CDK4: {len(cdk4)} tests, {n_robust_4} ROBUST")
        print(f"  CDK6: {len(cdk6)} tests, {n_robust_6} ROBUST")

        cdk4_frames.append(cdk4)
        cdk6_frames.append(cdk6)

    cdk4_result = pd.concat(cdk4_frames, ignore_index=True)
    cdk6_result = pd.concat(cdk6_frames, ignore_index=True)

    # --- Control genes CDK2, CDK1 (should NOT show CDKN2A-selective dependency) ---
    print("\nComputing control gene effect sizes (CDK2, CDK1)...")
    control_per_type = compute_effect_sizes(
        classified, crispr, qualifying_types, ["CDK2", "CDK1"],
        rb1_stratum="all",
    )
    control_pooled = add_pancancer_pooled(classified, crispr, ["CDK2", "CDK1"])
    control_result = pd.concat([control_per_type, control_pooled], ignore_index=True)
    print(f"  {len(control_result)} control tests")

    # --- MDM2 dependency in CDKN2A-del/TP53-WT context ---
    print("\nComputing MDM2 dependency (CDKN2A-del/TP53-WT context)...")
    mdm2_result = compute_mdm2_effect(classified, crispr, qualifying_types)
    n_mdm2_robust = (mdm2_result["classification"] == "ROBUST").sum() if len(mdm2_result) > 0 else 0
    print(f"  {len(mdm2_result)} MDM2 tests, {n_mdm2_robust} ROBUST")

    # --- RB1 co-loss impact ---
    print("\nQuantifying RB1 co-loss impact on CDK4/6 dependency...")
    rb1_impact = compute_rb1_coloss_impact(classified, crispr)
    for _, row in rb1_impact.iterrows():
        if not row.get("note"):
            print(f"  {row['gene']}: RB1-intact={row['median_dep_rb1_intact']:.3f} "
                  f"vs RB1-lost={row['median_dep_rb1_lost']:.3f} d={row['cohens_d']:.3f}")
        else:
            print(f"  {row['gene']}: {row['note']}")

    # Print key results
    print("\nKEY RESULTS (RB1-intact stratum):")
    for gene, result in [("CDK4", cdk4_result), ("CDK6", cdk6_result)]:
        primary = result[result["rb1_stratum"] == "rb1_intact"]
        for _, row in primary.iterrows():
            print(f"  {row['cancer_type']} / {gene}: d={row['cohens_d']:.3f} "
                  f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                  f"FDR={row['fdr']:.3e} -> {row['classification']}")

    # Save outputs
    print("\nSaving outputs...")
    cdk4_result.to_csv(OUTPUT_DIR / "cdk4_effect_sizes.csv", index=False)
    cdk6_result.to_csv(OUTPUT_DIR / "cdk6_effect_sizes.csv", index=False)
    control_result.to_csv(OUTPUT_DIR / "control_gene_effects.csv", index=False)
    mdm2_result.to_csv(OUTPUT_DIR / "mdm2_effect_sizes.csv", index=False)
    rb1_impact.to_csv(OUTPUT_DIR / "rb1_coloss_impact.csv", index=False)
    print("  cdk4_effect_sizes.csv, cdk6_effect_sizes.csv")
    print("  control_gene_effects.csv, mdm2_effect_sizes.csv, rb1_coloss_impact.csv")

    # Forest plots (primary analysis = rb1_intact stratum)
    primary_cdk4 = cdk4_result[cdk4_result["rb1_stratum"] == "rb1_intact"]
    primary_cdk6 = cdk6_result[cdk6_result["rb1_stratum"] == "rb1_intact"]
    plot_forest(primary_cdk4, "CDK4", OUTPUT_DIR / "cdk4_forest_plot.png")
    plot_forest(primary_cdk6, "CDK6", OUTPUT_DIR / "cdk6_forest_plot.png")
    if len(mdm2_result) > 0:
        plot_forest(mdm2_result, "MDM2", OUTPUT_DIR / "mdm2_forest_plot.png")
    print("  cdk4_forest_plot.png, cdk6_forest_plot.png, mdm2_forest_plot.png")

    # Summary text
    write_summary_txt(cdk4_result, cdk6_result, control_result, mdm2_result, rb1_impact, OUTPUT_DIR)
    print("  cdk46_effect_sizes_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
