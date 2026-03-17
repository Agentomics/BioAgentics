"""Phase 2: WRN synthetic lethality effect sizes by cancer type.

For each qualifying cancer type (and pan-cancer pooled), compares WRN CRISPR
dependency in MSI-H vs MSS lines. Includes WRNIP1 as secondary target and
BLM/RECQL as negative control helicases.

Statistics: Cohen's d with bootstrap 95% CI, Mann-Whitney U, BH-FDR,
leave-one-out robustness, and permutation testing.

Usage:
    uv run python -m wrn_msi_pancancer_atlas.02_wrn_effect_sizes
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
PHASE1_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase1"
OUTPUT_DIR = REPO_ROOT / "output" / "wrn-msi-pancancer-atlas" / "phase2"

# Target genes
PRIMARY_TARGETS = ["WRN", "WRNIP1"]
CONTROL_HELICASES = ["BLM", "RECQL"]

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
    return (count + 1) / (n_perm + 1)  # +1 for observed


def leave_one_out_robust(
    msi_vals: np.ndarray, mss_vals: np.ndarray,
) -> tuple[bool, float]:
    """Leave-one-out: remove each MSI-H line, check if significance flips.

    Returns (is_robust, min_abs_d_after_removal).
    """
    if len(msi_vals) < 3:
        return False, 0.0

    base_d = cohens_d(msi_vals, mss_vals)
    base_sign = np.sign(base_d)
    min_abs_d = abs(base_d)

    for i in range(len(msi_vals)):
        reduced = np.delete(msi_vals, i)
        if len(reduced) < 2:
            continue
        d_i = cohens_d(reduced, mss_vals)
        abs_d_i = abs(d_i)
        if abs_d_i < min_abs_d:
            min_abs_d = abs_d_i
        # Check if sign flipped
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
    do_permutation: bool = True,
    do_loo: bool = True,
) -> pd.DataFrame:
    """Compute effect sizes for given genes across cancer types."""
    available = [g for g in genes if g in crispr.columns]
    missing = [g for g in genes if g not in crispr.columns]
    if missing:
        print(f"  WARNING: Missing from CRISPRGeneEffect: {missing}")

    merged = classified.join(crispr[available], how="inner")

    rows = []
    for cancer_type in cancer_types:
        ct_data = merged[merged["OncotreeLineage"] == cancer_type]
        msi_h = ct_data[ct_data["msi_status"] == "MSI-H"]
        mss = ct_data[ct_data["msi_status"] == "MSS"]

        for gene in available:
            msi_vals = msi_h[gene].dropna().values
            mss_vals = mss[gene].dropna().values

            if len(msi_vals) < 3 or len(mss_vals) < 3:
                continue

            d = cohens_d(msi_vals, mss_vals)
            ci_lo, ci_hi = cohens_d_bootstrap_ci(msi_vals, mss_vals)
            _, pval = stats.mannwhitneyu(msi_vals, mss_vals, alternative="two-sided")

            perm_p = permutation_test(msi_vals, mss_vals) if do_permutation else float("nan")

            if do_loo:
                loo_robust, loo_min_d = leave_one_out_robust(msi_vals, mss_vals)
            else:
                loo_robust, loo_min_d = True, abs(d)

            rows.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "n_msi": len(msi_vals),
                "n_mss": len(mss_vals),
                "cohens_d": round(d, 4),
                "ci_lower": round(ci_lo, 4),
                "ci_upper": round(ci_hi, 4),
                "pvalue": float(pval),
                "permutation_p": round(perm_p, 4) if not np.isnan(perm_p) else float("nan"),
                "loo_robust": loo_robust,
                "loo_min_abs_d": round(loo_min_d, 4),
                "median_dep_msi": round(float(np.median(msi_vals)), 4),
                "median_dep_mss": round(float(np.median(mss_vals)), 4),
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
) -> pd.DataFrame:
    """Compute pan-cancer pooled effect sizes (all qualifying lines combined)."""
    available = [g for g in genes if g in crispr.columns]
    merged = classified[classified["has_crispr"]].join(crispr[available], how="inner")

    msi_h = merged[merged["msi_status"] == "MSI-H"]
    mss = merged[merged["msi_status"] == "MSS"]

    rows = []
    for gene in available:
        msi_vals = msi_h[gene].dropna().values
        mss_vals = mss[gene].dropna().values

        if len(msi_vals) < 3 or len(mss_vals) < 3:
            continue

        d = cohens_d(msi_vals, mss_vals)
        ci_lo, ci_hi = cohens_d_bootstrap_ci(msi_vals, mss_vals)
        _, pval = stats.mannwhitneyu(msi_vals, mss_vals, alternative="two-sided")
        perm_p = permutation_test(msi_vals, mss_vals)
        loo_robust, loo_min_d = leave_one_out_robust(msi_vals, mss_vals)

        rows.append({
            "cancer_type": "Pan-cancer (pooled)",
            "gene": gene,
            "n_msi": len(msi_vals),
            "n_mss": len(mss_vals),
            "cohens_d": round(d, 4),
            "ci_lower": round(ci_lo, 4),
            "ci_upper": round(ci_hi, 4),
            "pvalue": float(pval),
            "permutation_p": round(perm_p, 4),
            "loo_robust": loo_robust,
            "loo_min_abs_d": round(loo_min_d, 4),
            "median_dep_msi": round(float(np.median(msi_vals)), 4),
            "median_dep_mss": round(float(np.median(mss_vals)), 4),
        })

    result = pd.DataFrame(rows)
    if len(result) > 0:
        result["fdr"] = fdr_correction(result["pvalue"].values)
        result["classification"] = result.apply(
            lambda r: classify_result(r["fdr"], r["ci_lower"], r["ci_upper"], r["permutation_p"]),
            axis=1,
        )
    return result


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
        f"{row['cancer_type']} (n={row['n_msi']}+{row['n_mss']}) [{row.get('classification', '')}]"
        for _, row in gene_data.iterrows()
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel(f"Cohen's d ({gene} dependency: MSI-H vs MSS)")
    ax.set_title(f"{gene} Synthetic Lethality by Cancer Type\n"
                 "(negative = more essential in MSI-H)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def write_summary_txt(
    wrn_result: pd.DataFrame,
    control_result: pd.DataFrame,
    output_dir: Path,
) -> None:
    """Write human-readable summary."""
    lines = [
        "=" * 60,
        "WRN-MSI Pan-Cancer Atlas — Phase 2: WRN Effect Sizes",
        "=" * 60,
        "",
    ]

    for gene in PRIMARY_TARGETS:
        gene_data = wrn_result[wrn_result["gene"] == gene]
        if len(gene_data) == 0:
            continue
        lines.append(f"{gene} DEPENDENCY (MSI-H vs MSS)")
        lines.append("-" * 50)
        for _, row in gene_data.iterrows():
            lines.append(
                f"  {row['cancer_type']}: d={row['cohens_d']:.3f} "
                f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
                f"FDR={row['fdr']:.3e} perm_p={row['permutation_p']:.4f} "
                f"LOO={'robust' if row['loo_robust'] else 'fragile'} "
                f"→ {row['classification']}"
            )
        lines.append("")

    lines.append("CONTROL HELICASES (expect NOT_SIGNIFICANT)")
    lines.append("-" * 50)
    for _, row in control_result.iterrows():
        lines.append(
            f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
            f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
            f"FDR={row.get('fdr', float('nan')):.3e} → {row.get('classification', 'N/A')}"
        )
    lines.append("")

    # Interpretation
    wrn_data = wrn_result[wrn_result["gene"] == "WRN"]
    robust = wrn_data[wrn_data["classification"] == "ROBUST"]
    lines.append("INTERPRETATION")
    lines.append("-" * 50)
    if len(robust) > 0:
        lines.append(f"  ROBUST WRN-MSI SL in {len(robust)} context(s):")
        for _, row in robust.iterrows():
            lines.append(f"    {row['cancer_type']}: d={row['cohens_d']:.3f}")
    else:
        lines.append("  No ROBUST WRN-MSI SL detected in per-type analysis.")
        pooled = wrn_data[wrn_data["cancer_type"] == "Pan-cancer (pooled)"]
        if len(pooled) > 0:
            lines.append(f"  Pan-cancer pooled: d={pooled.iloc[0]['cohens_d']:.3f} "
                         f"→ {pooled.iloc[0]['classification']}")
    lines.append("")

    with open(output_dir / "wrn_effect_sizes_summary.txt", "w") as f:
        f.write("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: WRN Effect Sizes by Cancer Type ===\n")

    # Load Phase 1 outputs
    print("Loading Phase 1 classified cell lines...")
    classified = pd.read_csv(PHASE1_DIR / "msi_classification.csv", index_col=0)
    qualifying = pd.read_csv(PHASE1_DIR / "qualifying_cancer_types.csv")
    qualifying_types = qualifying["cancer_type"].tolist()
    print(f"  {len(qualifying_types)} qualifying cancer types: {qualifying_types}")

    # Load CRISPR data
    print("Loading CRISPRGeneEffect...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {len(crispr)} lines, {len(crispr.columns)} genes")

    # Primary targets: WRN + WRNIP1
    print(f"\nComputing WRN/WRNIP1 effect sizes (permutation n={N_PERMUTATIONS})...")
    wrn_per_type = compute_effect_sizes(
        classified, crispr, qualifying_types, PRIMARY_TARGETS,
        do_permutation=True, do_loo=True,
    )

    # Pan-cancer pooled
    print("Computing pan-cancer pooled effect sizes...")
    wrn_pooled = add_pancancer_pooled(classified, crispr, PRIMARY_TARGETS)

    wrn_result = pd.concat([wrn_per_type, wrn_pooled], ignore_index=True)
    print(f"  {len(wrn_result)} total WRN/WRNIP1 tests")

    # Control helicases: BLM, RECQL
    print("\nComputing control helicase effect sizes...")
    control_per_type = compute_effect_sizes(
        classified, crispr, qualifying_types, CONTROL_HELICASES,
        do_permutation=True, do_loo=False,
    )
    control_pooled = add_pancancer_pooled(classified, crispr, CONTROL_HELICASES)
    control_result = pd.concat([control_per_type, control_pooled], ignore_index=True)
    print(f"  {len(control_result)} control helicase tests")

    # Print results
    print("\nWRN RESULTS:")
    for _, row in wrn_result[wrn_result["gene"] == "WRN"].iterrows():
        print(f"  {row['cancer_type']}: d={row['cohens_d']:.3f} "
              f"[{row['ci_lower']:.3f}, {row['ci_upper']:.3f}] "
              f"FDR={row['fdr']:.3e} → {row['classification']}")

    print("\nCONTROL HELICASES:")
    for _, row in control_result.iterrows():
        print(f"  {row['cancer_type']} / {row['gene']}: d={row['cohens_d']:.3f} "
              f"→ {row.get('classification', 'N/A')}")

    # Save outputs
    print("\nSaving outputs...")
    wrn_result.to_csv(OUTPUT_DIR / "wrn_effect_sizes.csv", index=False)
    print("  wrn_effect_sizes.csv")

    control_result.to_csv(OUTPUT_DIR / "control_helicase_effects.csv", index=False)
    print("  control_helicase_effects.csv")

    # Forest plot
    plot_forest(wrn_result, "WRN", OUTPUT_DIR / "wrn_forest_plot.png")
    print("  wrn_forest_plot.png")

    # Summary text
    write_summary_txt(wrn_result, control_result, OUTPUT_DIR)
    print("  wrn_effect_sizes_summary.txt")

    print("\nDone.")


if __name__ == "__main__":
    main()
