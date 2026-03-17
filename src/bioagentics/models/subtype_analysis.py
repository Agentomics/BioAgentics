"""Subtype-specific dependency analysis.

Statistical testing for differential dependencies between NSCLC molecular
subtypes using Kruskal-Wallis and pairwise Wilcoxon rank-sum tests with
Benjamini-Hochberg FDR correction.

Usage:
    from bioagentics.models.subtype_analysis import analyze_subtype_dependencies
    results = analyze_subtype_dependencies(dep_matrix, subtypes)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu

logger = logging.getLogger(__name__)


@dataclass
class SubtypeResults:
    """Results of subtype-specific dependency analysis."""

    kruskal_wallis: pd.DataFrame      # per-gene KW test (statistic, p, fdr)
    pairwise: pd.DataFrame            # pairwise Wilcoxon results for KW-significant genes
    significant_genes: list[str] = field(default_factory=list)
    n_tested: int = 0
    n_significant: int = 0


def _benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Apply Benjamini-Hochberg FDR correction."""
    n = len(pvals)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvals)
    sorted_p = pvals[sorted_idx]
    fdr = np.empty(n)
    fdr[sorted_idx[-1]] = sorted_p[-1]
    for i in range(n - 2, -1, -1):
        fdr[sorted_idx[i]] = min(fdr[sorted_idx[i + 1]], sorted_p[i] * n / (i + 1))
    return np.clip(fdr, 0, 1)


def _rank_biserial(x: np.ndarray, y: np.ndarray) -> float:
    """Compute rank-biserial correlation as effect size for Wilcoxon test."""
    n_x, n_y = len(x), len(y)
    if n_x == 0 or n_y == 0:
        return 0.0
    stat, _ = mannwhitneyu(x, y, alternative="two-sided")
    return 2 * stat / (n_x * n_y) - 1


def analyze_subtype_dependencies(
    dep_matrix: pd.DataFrame,
    subtypes: pd.Series,
    kras_alleles: pd.Series | None = None,
    fdr_threshold: float = 0.05,
) -> SubtypeResults:
    """Test for differential dependencies between NSCLC molecular subtypes.

    Parameters
    ----------
    dep_matrix : DataFrame (patients x genes)
        Predicted dependency scores.
    subtypes : Series
        Molecular subtype label per patient (index must match dep_matrix).
    kras_alleles : Series, optional
        KRAS allele per patient for allele-stratified analysis.
    fdr_threshold : float
        FDR threshold for significance.

    Returns
    -------
    SubtypeResults with KW test results, pairwise comparisons, and gene lists.
    """
    # Align indices
    common = dep_matrix.index.intersection(subtypes.index)
    dep = dep_matrix.loc[common]
    sub = subtypes.loc[common]
    unique_subtypes = sorted(sub.unique())

    logger.info(
        "Testing %d genes across %d patients (%s)",
        dep.shape[1], len(common), ", ".join(f"{s}: {(sub == s).sum()}" for s in unique_subtypes),
    )

    # Kruskal-Wallis test per gene
    kw_rows = []
    for gene in dep.columns:
        groups = [dep.loc[sub == st, gene].dropna().values for st in unique_subtypes]
        groups = [g for g in groups if len(g) >= 2]
        if len(groups) < 2:
            kw_rows.append({"gene": gene, "kw_stat": np.nan, "kw_p": 1.0})
            continue
        stat, p = kruskal(*groups)
        kw_rows.append({"gene": gene, "kw_stat": stat, "kw_p": p})

    kw_df = pd.DataFrame(kw_rows).set_index("gene")
    kw_df["kw_fdr"] = _benjamini_hochberg(kw_df["kw_p"].values)

    # Pairwise Wilcoxon for significant genes
    sig_genes = kw_df[kw_df["kw_fdr"] < fdr_threshold].index.tolist()
    pairs = list(combinations(unique_subtypes, 2))

    pw_rows = []
    for gene in sig_genes:
        for st_a, st_b in pairs:
            x = dep.loc[sub == st_a, gene].dropna().values
            y = dep.loc[sub == st_b, gene].dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            stat, p = mannwhitneyu(x, y, alternative="two-sided")
            effect = _rank_biserial(x, y)
            pw_rows.append({
                "gene": gene,
                "group_a": st_a,
                "group_b": st_b,
                "u_stat": stat,
                "p_value": p,
                "effect_size": effect,
                "mean_a": np.mean(x),
                "mean_b": np.mean(y),
                "direction": "a_higher" if np.mean(x) > np.mean(y) else "b_higher",
            })

    pw_df = pd.DataFrame(pw_rows)
    if len(pw_df) > 0:
        pw_df["fdr"] = _benjamini_hochberg(pw_df["p_value"].values)
    else:
        pw_df["fdr"] = []

    # KRAS allele analysis (within KRAS-mutant patients)
    if kras_alleles is not None:
        allele_common = dep.index.intersection(kras_alleles.index)
        kras_mut_mask = kras_alleles.loc[allele_common] != "WT"
        kras_patients = allele_common[kras_mut_mask]

        if len(kras_patients) >= 10:
            allele_labels = kras_alleles.loc[kras_patients]
            allele_subtypes = sorted(allele_labels.unique())

            if len(allele_subtypes) >= 2:
                dep_kras = dep.loc[kras_patients]
                allele_rows = []
                for gene in sig_genes:
                    groups = [dep_kras.loc[allele_labels == a, gene].dropna().values
                              for a in allele_subtypes]
                    groups = [g for g in groups if len(g) >= 2]
                    if len(groups) < 2:
                        continue
                    stat, p = kruskal(*groups)
                    allele_rows.append({
                        "gene": gene, "kw_stat": stat, "kw_p": p,
                        "alleles_tested": ";".join(allele_subtypes),
                    })

                if allele_rows:
                    allele_df = pd.DataFrame(allele_rows).set_index("gene")
                    allele_df["kw_fdr"] = _benjamini_hochberg(allele_df["kw_p"].values)
                    # Append allele results to existing pairwise DataFrame
                    # (not pw_rows, which lacks the BH-corrected fdr column)
                    allele_pw = pd.DataFrame([
                        {
                            "gene": row.name,
                            "group_a": "KRAS_allele_axis",
                            "group_b": row["alleles_tested"],
                            "u_stat": row["kw_stat"],
                            "p_value": row["kw_p"],
                            "effect_size": np.nan,
                            "mean_a": np.nan, "mean_b": np.nan,
                            "direction": "allele_kw",
                            "fdr": row["kw_fdr"],
                        }
                        for _, row in allele_df.iterrows()
                    ])
                    pw_df = pd.concat([pw_df, allele_pw], ignore_index=True)

    logger.info("%d / %d genes significant (FDR < %.2f)", len(sig_genes), len(kw_df), fdr_threshold)

    return SubtypeResults(
        kruskal_wallis=kw_df,
        pairwise=pw_df,
        significant_genes=sig_genes,
        n_tested=len(kw_df),
        n_significant=len(sig_genes),
    )


def save_subtype_results(results: SubtypeResults, results_dir: str | Path) -> None:
    """Save subtype analysis results to disk."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results.kruskal_wallis.to_csv(results_dir / "kruskal_wallis_results.csv")
    results.pairwise.to_csv(results_dir / "pairwise_comparisons.csv", index=False)

    with open(results_dir / "significant_genes.txt", "w") as f:
        for gene in results.significant_genes:
            f.write(gene + "\n")

    logger.info("Saved subtype results to %s", results_dir)
