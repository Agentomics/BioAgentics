"""Enrichment testing for gene sets in striosomal vs. matrix compartments.

Tests whether a set of genes (e.g., TS risk genes) is enriched in
striosome-associated vs. matrix-associated zones using:
1. Fisher exact test (2x2 contingency)
2. Permutation-based enrichment (10,000 permutations)
3. FDR correction (Benjamini-Hochberg)

Input: gene set + zone compartment scores (from Phase 1 compartment_scoring).
Output: enrichment p-values, odds ratios, FDR-corrected q-values.

Usage:
    from bioagentics.tourettes.striosomal_matrix.enrichment_test import (
        fisher_enrichment,
        permutation_enrichment,
        run_enrichment_battery,
    )
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass
class EnrichmentResult:
    """Result of a single enrichment test."""

    test_name: str
    gene_set_name: str
    compartment: str  # "striosome" or "matrix"
    n_genes_tested: int
    n_genes_in_compartment: int
    odds_ratio: float
    p_value: float
    q_value: float  # FDR-corrected, filled in after multiple testing
    direction: str  # "enriched" or "depleted"
    method: str  # "fisher" or "permutation"


def fisher_enrichment(
    gene_scores: dict[str, float],
    test_genes: set[str],
    threshold: float = 0.0,
    gene_set_name: str = "test_set",
) -> tuple[EnrichmentResult, EnrichmentResult]:
    """Fisher exact test for gene set enrichment in striosome vs. matrix.

    Args:
        gene_scores: Dict mapping gene symbol -> compartment score.
            Positive = striosome-biased, negative = matrix-biased.
        test_genes: Set of gene symbols to test for enrichment.
        threshold: Score threshold for classifying genes. Genes with
            score > threshold are "striosome", score < -threshold are "matrix".
        gene_set_name: Label for the gene set being tested.

    Returns:
        Tuple of (striosome_result, matrix_result).
    """
    # Intersect test genes with scored genes
    scored_test = test_genes & set(gene_scores.keys())
    if len(scored_test) == 0:
        return _empty_results(gene_set_name, len(test_genes))

    all_genes = set(gene_scores.keys())
    background = all_genes - scored_test

    # Classify all genes
    def _classify(genes: set[str]) -> tuple[int, int, int]:
        strio = sum(1 for g in genes if gene_scores.get(g, 0) > threshold)
        matrix = sum(1 for g in genes if gene_scores.get(g, 0) < -threshold)
        neutral = len(genes) - strio - matrix
        return strio, matrix, neutral

    test_s, test_m, _test_n = _classify(scored_test)
    bg_s, bg_m, _bg_n = _classify(background)

    # Fisher exact for striosome enrichment: test genes in striosome vs not
    table_s = np.array([[test_s, len(scored_test) - test_s],
                        [bg_s, len(background) - bg_s]])
    res_s = stats.fisher_exact(table_s, alternative="two-sided")
    or_s, p_s = float(res_s[0]), float(res_s[1])

    # Fisher exact for matrix enrichment
    table_m = np.array([[test_m, len(scored_test) - test_m],
                        [bg_m, len(background) - bg_m]])
    res_m = stats.fisher_exact(table_m, alternative="two-sided")
    or_m, p_m = float(res_m[0]), float(res_m[1])

    strio_result = EnrichmentResult(
        test_name=f"fisher_{gene_set_name}_striosome",
        gene_set_name=gene_set_name,
        compartment="striosome",
        n_genes_tested=len(scored_test),
        n_genes_in_compartment=test_s,
        odds_ratio=or_s,
        p_value=p_s,
        q_value=np.nan,
        direction="enriched" if or_s > 1 else "depleted",
        method="fisher",
    )

    matrix_result = EnrichmentResult(
        test_name=f"fisher_{gene_set_name}_matrix",
        gene_set_name=gene_set_name,
        compartment="matrix",
        n_genes_tested=len(scored_test),
        n_genes_in_compartment=test_m,
        odds_ratio=or_m,
        p_value=p_m,
        q_value=np.nan,
        direction="enriched" if or_m > 1 else "depleted",
        method="fisher",
    )

    return strio_result, matrix_result


def permutation_enrichment(
    gene_scores: dict[str, float],
    test_genes: set[str],
    n_permutations: int = 10_000,
    gene_set_name: str = "test_set",
    rng_seed: int = 42,
) -> tuple[EnrichmentResult, EnrichmentResult]:
    """Permutation-based enrichment test for compartment bias.

    Computes the mean compartment score for test genes and compares against
    the null distribution from random gene sets of the same size.

    Args:
        gene_scores: Dict mapping gene symbol -> compartment score.
        test_genes: Set of gene symbols to test.
        n_permutations: Number of permutations (default 10,000).
        gene_set_name: Label for the gene set.
        rng_seed: Random seed for reproducibility.

    Returns:
        Tuple of (striosome_result, matrix_result).
    """
    scored_test = sorted(test_genes & set(gene_scores.keys()))
    if len(scored_test) == 0:
        return _empty_results(gene_set_name, len(test_genes))

    all_genes = sorted(gene_scores.keys())
    all_scores = np.array([gene_scores[g] for g in all_genes])
    test_scores = np.array([gene_scores[g] for g in scored_test])
    observed_mean = float(np.mean(test_scores))

    # Permutation null
    rng = np.random.default_rng(rng_seed)
    n_test = len(scored_test)
    null_means = np.empty(n_permutations)
    for i in range(n_permutations):
        idx = rng.choice(len(all_genes), size=n_test, replace=False)
        null_means[i] = np.mean(all_scores[idx])

    # Two-sided p-value for striosome (positive direction)
    p_strio = float(np.mean(null_means >= observed_mean))
    # Two-sided p-value for matrix (negative direction)
    p_matrix = float(np.mean(null_means <= observed_mean))

    # Compute effect size as z-score
    null_std = float(np.std(null_means))
    null_mean = float(np.mean(null_means))
    z = (observed_mean - null_mean) / null_std if null_std > 0 else 0.0

    strio_result = EnrichmentResult(
        test_name=f"perm_{gene_set_name}_striosome",
        gene_set_name=gene_set_name,
        compartment="striosome",
        n_genes_tested=n_test,
        n_genes_in_compartment=int(np.sum(test_scores > 0)),
        odds_ratio=z,  # z-score as effect size for permutation
        p_value=p_strio,
        q_value=np.nan,
        direction="enriched" if observed_mean > 0 else "depleted",
        method="permutation",
    )

    matrix_result = EnrichmentResult(
        test_name=f"perm_{gene_set_name}_matrix",
        gene_set_name=gene_set_name,
        compartment="matrix",
        n_genes_tested=n_test,
        n_genes_in_compartment=int(np.sum(test_scores < 0)),
        odds_ratio=-z,
        p_value=p_matrix,
        q_value=np.nan,
        direction="enriched" if observed_mean < 0 else "depleted",
        method="permutation",
    )

    return strio_result, matrix_result


def fdr_correct(results: list[EnrichmentResult]) -> list[EnrichmentResult]:
    """Apply Benjamini-Hochberg FDR correction to a list of results."""
    if not results:
        return results

    p_values = np.array([r.p_value for r in results])
    n = len(p_values)

    # BH procedure
    sorted_idx = np.argsort(p_values)
    sorted_p = p_values[sorted_idx]
    ranks = np.arange(1, n + 1)
    q_values = np.minimum(1.0, sorted_p * n / ranks)

    # Enforce monotonicity (cumulative minimum from the end)
    for i in range(n - 2, -1, -1):
        q_values[i] = min(q_values[i], q_values[i + 1])

    # Map back to original order
    q_out = np.empty(n)
    q_out[sorted_idx] = q_values

    for i, r in enumerate(results):
        r.q_value = float(q_out[i])

    return results


def run_enrichment_battery(
    gene_scores: dict[str, float],
    gene_sets: dict[str, set[str]],
    threshold: float = 0.0,
    n_permutations: int = 10_000,
    rng_seed: int = 42,
) -> list[EnrichmentResult]:
    """Run Fisher + permutation enrichment tests for multiple gene sets.

    Args:
        gene_scores: Dict mapping gene symbol -> compartment score.
        gene_sets: Dict mapping gene set name -> set of gene symbols.
        threshold: Score threshold for Fisher test classification.
        n_permutations: Number of permutations.
        rng_seed: Random seed.

    Returns:
        FDR-corrected list of EnrichmentResult.
    """
    all_results: list[EnrichmentResult] = []

    for name, genes in sorted(gene_sets.items()):
        # Fisher exact
        s_fisher, m_fisher = fisher_enrichment(
            gene_scores, genes, threshold=threshold, gene_set_name=name,
        )
        all_results.extend([s_fisher, m_fisher])

        # Permutation
        s_perm, m_perm = permutation_enrichment(
            gene_scores, genes,
            n_permutations=n_permutations,
            gene_set_name=name,
            rng_seed=rng_seed,
        )
        all_results.extend([s_perm, m_perm])

    return fdr_correct(all_results)


def results_to_rows(results: list[EnrichmentResult]) -> list[dict]:
    """Convert EnrichmentResult list to list of dicts for CSV export."""
    return [
        {
            "test_name": r.test_name,
            "gene_set_name": r.gene_set_name,
            "compartment": r.compartment,
            "n_genes_tested": r.n_genes_tested,
            "n_genes_in_compartment": r.n_genes_in_compartment,
            "odds_ratio": f"{r.odds_ratio:.4f}",
            "p_value": f"{r.p_value:.6f}",
            "q_value": f"{r.q_value:.6f}",
            "direction": r.direction,
            "method": r.method,
        }
        for r in results
    ]


def _empty_results(
    gene_set_name: str, n_genes: int,
) -> tuple[EnrichmentResult, EnrichmentResult]:
    """Return empty results when no genes overlap with scored genes."""
    return (
        EnrichmentResult(
            test_name=f"empty_{gene_set_name}_striosome",
            gene_set_name=gene_set_name,
            compartment="striosome",
            n_genes_tested=0,
            n_genes_in_compartment=0,
            odds_ratio=np.nan,
            p_value=1.0,
            q_value=1.0,
            direction="none",
            method="none",
        ),
        EnrichmentResult(
            test_name=f"empty_{gene_set_name}_matrix",
            gene_set_name=gene_set_name,
            compartment="matrix",
            n_genes_tested=0,
            n_genes_in_compartment=0,
            odds_ratio=np.nan,
            p_value=1.0,
            q_value=1.0,
            direction="none",
            method="none",
        ),
    )
