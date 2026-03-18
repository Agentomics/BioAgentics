"""Phase 2: Co-mutation modulation of PRMT5 dependency in MTAP-deleted NSCLC.

Tests whether KRAS allele, STK11, KEAP1, TP53, or NFE2L2 mutations modulate
PRMT5 dependency within MTAP-deleted NSCLC lines. Builds a multivariate
linear model with interaction terms.

Usage:
    uv run python -m mtap_prmt5_nsclc_sl.03_comutation_modulation
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

BINARY_GENES = ["STK11_mut", "KEAP1_mut", "TP53_mut", "NFE2L2_mut"]


def load_merged_data() -> pd.DataFrame:
    """Load classified NSCLC lines merged with PRMT5 dependency."""
    classified = pd.read_csv(OUTPUT_DIR / "nsclc_cell_lines_classified.csv", index_col=0)
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    prmt5_dep = crispr["PRMT5"].rename("PRMT5_dep")
    merged = classified.join(prmt5_dep, how="inner")
    return merged.dropna(subset=["MTAP_deleted", "PRMT5_dep"])


def stratified_tests(df: pd.DataFrame) -> list[dict]:
    """Run stratified comparisons within MTAP-deleted NSCLC lines."""
    deleted = df[df["MTAP_deleted"]].copy()
    results = []

    # Binary gene mutations
    for gene_col in BINARY_GENES:
        gene_name = gene_col.replace("_mut", "")
        mut = deleted[deleted[gene_col]]["PRMT5_dep"]
        wt = deleted[~deleted[gene_col]]["PRMT5_dep"]

        result = {
            "gene": gene_name,
            "test": "mann_whitney",
            "n_mutant": len(mut),
            "n_wt": len(wt),
            "median_mutant": float(mut.median()) if len(mut) > 0 else None,
            "median_wt": float(wt.median()) if len(wt) > 0 else None,
        }

        if len(mut) >= 3 and len(wt) >= 3:
            stat, pval = stats.mannwhitneyu(mut, wt, alternative="two-sided")
            result["p_value"] = float(pval)
            result["U_stat"] = float(stat)
        else:
            result["p_value"] = None
            result["note"] = "Too few samples for test"

        results.append(result)
        print(f"  {gene_name}: mut={len(mut)}, wt={len(wt)}, "
              f"p={result['p_value']:.4f}" if result["p_value"] is not None
              else f"  {gene_name}: mut={len(mut)}, wt={len(wt)}, insufficient N")

    # KRAS allele (Kruskal-Wallis across allele subgroups)
    kras_groups = deleted.groupby("KRAS_allele")["PRMT5_dep"]
    allele_results = {}
    group_arrays = []
    for allele, group in kras_groups:
        allele_results[allele] = {
            "n": len(group),
            "median": float(group.median()),
        }
        if len(group) >= 2:
            group_arrays.append(group.values)

    kras_result = {
        "gene": "KRAS_allele",
        "test": "kruskal_wallis",
        "subgroups": allele_results,
    }

    if len(group_arrays) >= 2:
        h_stat, pval = stats.kruskal(*group_arrays)
        kras_result["H_stat"] = float(h_stat)
        kras_result["p_value"] = float(pval)
        print(f"  KRAS allele: H={h_stat:.3f}, p={pval:.4f}")
    else:
        kras_result["p_value"] = None
        kras_result["note"] = "Too few groups with >= 2 samples"
        print("  KRAS allele: insufficient subgroups for test")

    results.append(kras_result)
    return results


def build_multivariate_model(df: pd.DataFrame) -> dict:
    """Build multivariate OLS model with interaction terms.

    PRMT5_dep ~ MTAP_deleted + STK11 + KEAP1 + TP53 + NFE2L2 + KRAS_mut
                + MTAP:STK11 + MTAP:KEAP1 + MTAP:TP53 + MTAP:NFE2L2 + MTAP:KRAS
    """
    import statsmodels.api as sm

    model_df = df.copy()
    model_df["MTAP_del"] = model_df["MTAP_deleted"].astype(int)
    model_df["STK11"] = model_df["STK11_mut"].astype(int)
    model_df["KEAP1"] = model_df["KEAP1_mut"].astype(int)
    model_df["TP53"] = model_df["TP53_mut"].astype(int)
    model_df["NFE2L2"] = model_df["NFE2L2_mut"].astype(int)
    model_df["KRAS_mut"] = model_df["KRAS_status"].astype(int)

    # Interaction terms
    for gene in ["STK11", "KEAP1", "TP53", "NFE2L2", "KRAS_mut"]:
        model_df[f"MTAP_x_{gene}"] = model_df["MTAP_del"] * model_df[gene]

    feature_cols = [
        "MTAP_del", "STK11", "KEAP1", "TP53", "NFE2L2", "KRAS_mut",
        "MTAP_x_STK11", "MTAP_x_KEAP1", "MTAP_x_TP53", "MTAP_x_NFE2L2", "MTAP_x_KRAS_mut",
    ]

    X = sm.add_constant(model_df[feature_cols].astype(float))
    y = model_df["PRMT5_dep"]

    model = sm.OLS(y, X).fit()

    result = {
        "r_squared": float(model.rsquared),
        "adj_r_squared": float(model.rsquared_adj),
        "f_statistic": float(model.fvalue),
        "f_pvalue": float(model.f_pvalue),
        "n_obs": int(model.nobs),
        "coefficients": {},
    }

    for name in model.params.index:
        result["coefficients"][name] = {
            "coef": float(model.params[name]),
            "std_err": float(model.bse[name]),
            "t_stat": float(model.tvalues[name]),
            "p_value": float(model.pvalues[name]),
        }

    print(f"\n  OLS Model: R²={model.rsquared:.3f}, adj R²={model.rsquared_adj:.3f}, "
          f"F={model.fvalue:.2f} (p={model.f_pvalue:.2e})")
    print(f"  N={int(model.nobs)}")
    print("\n  Significant terms (p < 0.1):")
    for name in model.params.index:
        if model.pvalues[name] < 0.1:
            print(f"    {name}: coef={model.params[name]:.4f}, p={model.pvalues[name]:.4f}")

    return result


def plot_comutation_boxplots(df: pd.DataFrame, out_path: Path) -> None:
    """Grouped box plots: PRMT5 dependency by co-mutation within MTAP-deleted."""
    deleted = df[df["MTAP_deleted"]].copy()
    genes = ["STK11_mut", "KEAP1_mut", "TP53_mut", "NFE2L2_mut"]
    labels = ["STK11", "KEAP1", "TP53", "NFE2L2"]

    fig, axes = plt.subplots(1, 4, figsize=(14, 5), sharey=True)
    for ax, gene_col, label in zip(axes, genes, labels):
        mut = deleted[deleted[gene_col]]["PRMT5_dep"]
        wt = deleted[~deleted[gene_col]]["PRMT5_dep"]

        bp = ax.boxplot(
            [wt, mut],
            tick_labels=[f"WT\n(n={len(wt)})", f"Mut\n(n={len(mut)})"],
            widths=0.5, patch_artist=True,
            medianprops=dict(color="black", linewidth=2),
        )
        bp["boxes"][0].set_facecolor("#4DBEEE")
        bp["boxes"][1].set_facecolor("#D95319")

        for i, d in enumerate([wt, mut]):
            if len(d) > 0:
                jitter = np.random.default_rng(0).normal(0, 0.05, size=len(d))
                ax.scatter(np.full(len(d), i + 1) + jitter, d, alpha=0.4, s=15,
                           color="gray", zorder=3)

        ax.set_title(label)
        if ax == axes[0]:
            ax.set_ylabel("PRMT5 dependency")

    fig.suptitle("PRMT5 Dependency by Co-mutation (MTAP-deleted NSCLC)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_forest(model_result: dict, out_path: Path) -> None:
    """Forest plot of interaction term effect sizes from OLS model."""
    interaction_terms = {
        k: v for k, v in model_result["coefficients"].items()
        if k.startswith("MTAP_x_")
    }
    if not interaction_terms:
        return

    names = list(interaction_terms.keys())
    coefs = [interaction_terms[n]["coef"] for n in names]
    errors = [1.96 * interaction_terms[n]["std_err"] for n in names]
    pvals = [interaction_terms[n]["p_value"] for n in names]

    # Clean names for display
    display_names = [n.replace("MTAP_x_", "MTAP × ") for n in names]

    fig, ax = plt.subplots(figsize=(7, 4))
    y_pos = range(len(names))
    colors = ["#D95319" if p < 0.05 else "#4DBEEE" if p < 0.1 else "gray" for p in pvals]

    ax.barh(y_pos, coefs, xerr=errors, color=colors, alpha=0.7, height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{n} (p={p:.3f})" for n, p in zip(display_names, pvals)])
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Interaction coefficient (effect on PRMT5 dep.)")
    ax.set_title("MTAP × Co-mutation Interaction Effects (OLS)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading merged data...")
    merged = load_merged_data()
    n_deleted = merged["MTAP_deleted"].sum()
    n_total = len(merged)
    print(f"  {n_total} NSCLC lines with dependency data, {n_deleted} MTAP-deleted")

    print("\nStratified tests (within MTAP-deleted):")
    stratified = stratified_tests(merged)

    print("\nBuilding multivariate model (all NSCLC lines)...")
    model_result = build_multivariate_model(merged)

    print("\nGenerating plots...")
    plot_comutation_boxplots(merged, FIG_DIR / "comutation_boxplots.png")
    plot_forest(model_result, FIG_DIR / "interaction_forest.png")

    results = {
        "stratified_tests": stratified,
        "multivariate_model": model_result,
    }
    out_path = OUTPUT_DIR / "comutation_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")


if __name__ == "__main__":
    main()
