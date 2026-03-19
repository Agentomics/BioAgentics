"""Tests for striosomal-matrix enrichment testing module."""

import numpy as np
import pytest

from bioagentics.tourettes.striosomal_matrix.enrichment_test import (
    EnrichmentResult,
    fdr_correct,
    fisher_enrichment,
    permutation_enrichment,
    results_to_rows,
    run_enrichment_battery,
)


def _make_gene_scores(n_strio=50, n_matrix=50, n_neutral=100, seed=42):
    """Create synthetic gene scores: positive=striosome, negative=matrix."""
    rng = np.random.default_rng(seed)
    scores = {}
    for i in range(n_strio):
        scores[f"STRIO_{i}"] = float(rng.uniform(0.5, 2.0))
    for i in range(n_matrix):
        scores[f"MATRIX_{i}"] = float(rng.uniform(-2.0, -0.5))
    for i in range(n_neutral):
        scores[f"NEUTRAL_{i}"] = float(rng.uniform(-0.3, 0.3))
    return scores


class TestFisherEnrichment:
    def test_striosome_enriched_set(self):
        scores = _make_gene_scores()
        strio_genes = {f"STRIO_{i}" for i in range(20)}
        s_res, m_res = fisher_enrichment(scores, strio_genes, gene_set_name="strio_set")
        assert s_res.compartment == "striosome"
        assert s_res.direction == "enriched"
        assert s_res.odds_ratio > 1
        assert s_res.p_value < 0.05

    def test_matrix_enriched_set(self):
        scores = _make_gene_scores()
        matrix_genes = {f"MATRIX_{i}" for i in range(20)}
        s_res, m_res = fisher_enrichment(scores, matrix_genes, gene_set_name="matrix_set")
        assert m_res.compartment == "matrix"
        assert m_res.direction == "enriched"
        assert m_res.odds_ratio > 1
        assert m_res.p_value < 0.05

    def test_neutral_set_not_enriched(self):
        scores = _make_gene_scores()
        neutral_genes = {f"NEUTRAL_{i}" for i in range(20)}
        s_res, m_res = fisher_enrichment(scores, neutral_genes, gene_set_name="neutral_set")
        # Neither compartment should be strongly enriched
        assert s_res.p_value > 0.01 or s_res.odds_ratio < 2
        assert m_res.p_value > 0.01 or m_res.odds_ratio < 2

    def test_empty_overlap(self):
        scores = _make_gene_scores()
        no_overlap = {"NONEXISTENT_1", "NONEXISTENT_2"}
        s_res, m_res = fisher_enrichment(scores, no_overlap)
        assert s_res.n_genes_tested == 0
        assert s_res.p_value == 1.0

    def test_returns_correct_gene_counts(self):
        scores = _make_gene_scores()
        genes = {f"STRIO_{i}" for i in range(10)} | {f"MATRIX_{i}" for i in range(5)}
        s_res, m_res = fisher_enrichment(scores, genes)
        assert s_res.n_genes_tested == 15
        assert s_res.n_genes_in_compartment == 10
        assert m_res.n_genes_in_compartment == 5


class TestPermutationEnrichment:
    def test_striosome_enrichment(self):
        scores = _make_gene_scores()
        strio_genes = {f"STRIO_{i}" for i in range(20)}
        s_res, m_res = permutation_enrichment(
            scores, strio_genes, n_permutations=5000, gene_set_name="strio_set",
        )
        assert s_res.compartment == "striosome"
        assert s_res.direction == "enriched"
        assert s_res.p_value < 0.05

    def test_reproducibility(self):
        scores = _make_gene_scores()
        genes = {f"STRIO_{i}" for i in range(10)}
        r1 = permutation_enrichment(scores, genes, rng_seed=123)
        r2 = permutation_enrichment(scores, genes, rng_seed=123)
        assert r1[0].p_value == r2[0].p_value

    def test_empty_overlap(self):
        scores = _make_gene_scores()
        s_res, m_res = permutation_enrichment(scores, {"FAKE_GENE"})
        assert s_res.n_genes_tested == 0


class TestFDRCorrection:
    def test_monotonic_q_values(self):
        results = [
            EnrichmentResult("t1", "gs1", "s", 10, 5, 2.0, 0.01, np.nan, "e", "f"),
            EnrichmentResult("t2", "gs1", "m", 10, 5, 0.5, 0.5, np.nan, "d", "f"),
            EnrichmentResult("t3", "gs2", "s", 10, 3, 1.5, 0.03, np.nan, "e", "f"),
            EnrichmentResult("t4", "gs2", "m", 10, 7, 3.0, 0.001, np.nan, "e", "f"),
        ]
        corrected = fdr_correct(results)
        q_values = [r.q_value for r in corrected]
        assert all(0 <= q <= 1 for q in q_values)
        # Most significant should still have smallest q
        assert corrected[3].q_value <= corrected[1].q_value

    def test_single_result(self):
        results = [
            EnrichmentResult("t1", "gs", "s", 10, 5, 2.0, 0.04, np.nan, "e", "f"),
        ]
        corrected = fdr_correct(results)
        assert corrected[0].q_value == pytest.approx(0.04)

    def test_empty_list(self):
        assert fdr_correct([]) == []


class TestRunEnrichmentBattery:
    def test_battery_produces_results(self):
        scores = _make_gene_scores()
        gene_sets = {
            "strio_set": {f"STRIO_{i}" for i in range(15)},
            "matrix_set": {f"MATRIX_{i}" for i in range(15)},
        }
        results = run_enrichment_battery(scores, gene_sets, n_permutations=1000)
        # 2 gene sets x 2 methods x 2 compartments = 8 results
        assert len(results) == 8
        # All should have q_values filled
        assert all(not np.isnan(r.q_value) for r in results)

    def test_results_to_rows(self):
        scores = _make_gene_scores()
        gene_sets = {"test": {f"STRIO_{i}" for i in range(5)}}
        results = run_enrichment_battery(scores, gene_sets, n_permutations=100)
        rows = results_to_rows(results)
        assert len(rows) == 4
        assert all("gene_set_name" in r for r in rows)
        assert all("p_value" in r for r in rows)
