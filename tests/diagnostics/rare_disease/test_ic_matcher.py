"""Tests for the IC-based phenotype matcher."""

from __future__ import annotations

import math

import networkx as nx
import pytest

from bioagentics.diagnostics.rare_disease.ic_matcher import (
    ICScorer,
    rank_diseases,
)

# Build a small HPO DAG for testing:
#
#   HP:0000001 (All)
#       ↑
#   HP:0000118 (Phenotypic abnormality)
#       ↑                ↑
#   HP:0000707 (Neuro)   HP:0000152 (Head/neck)
#       ↑                    ↑
#   HP:0002011 (CNS)     HP:0000234 (Head)
#       ↑
#   HP:0012443 (Brain)
#


def _build_test_dag() -> nx.DiGraph:
    g = nx.DiGraph()
    terms = {
        "HP:0000001": "All",
        "HP:0000118": "Phenotypic abnormality",
        "HP:0000707": "Abnormality of the nervous system",
        "HP:0000152": "Abnormality of head or neck",
        "HP:0000234": "Abnormality of the head",
        "HP:0002011": "Morphological CNS abnormality",
        "HP:0012443": "Abnormality of brain morphology",
    }
    for tid, name in terms.items():
        g.add_node(tid, name=name, node_type="phenotype")

    # Child → Parent edges
    g.add_edge("HP:0000118", "HP:0000001", relation="is_a")
    g.add_edge("HP:0000707", "HP:0000118", relation="is_a")
    g.add_edge("HP:0000152", "HP:0000118", relation="is_a")
    g.add_edge("HP:0000234", "HP:0000152", relation="is_a")
    g.add_edge("HP:0002011", "HP:0000707", relation="is_a")
    g.add_edge("HP:0012443", "HP:0002011", relation="is_a")
    return g


# Disease annotations for testing
DISEASE_ANNOTATIONS = {
    "OMIM:100001": ["HP:0000707", "HP:0000234"],  # Neuro + Head
    "OMIM:100002": ["HP:0012443", "HP:0002011"],  # Brain + CNS (deep neuro)
    "OMIM:100003": ["HP:0000152"],  # Head/neck only
}


@pytest.fixture
def scorer() -> ICScorer:
    dag = _build_test_dag()
    s = ICScorer(hpo_dag=dag)
    s.compute_ic(DISEASE_ANNOTATIONS)
    return s


class TestICComputation:
    def test_root_has_zero_ic(self, scorer: ICScorer):
        # Root (HP:0000001) is ancestor of all diseases, so p=1.0, IC=0
        assert scorer.ic.get("HP:0000001", 0.0) == 0.0

    def test_ic_monotonically_increasing_with_depth(self, scorer: ICScorer):
        # More specific terms should have higher IC
        ic_root = scorer.ic.get("HP:0000001", 0.0)
        ic_pa = scorer.ic.get("HP:0000118", 0.0)
        ic_neuro = scorer.ic.get("HP:0000707", 0.0)
        ic_cns = scorer.ic.get("HP:0002011", 0.0)
        ic_brain = scorer.ic.get("HP:0012443", 0.0)

        assert ic_root <= ic_pa
        assert ic_pa <= ic_neuro
        assert ic_neuro <= ic_cns
        assert ic_cns <= ic_brain

    def test_leaf_has_highest_ic(self, scorer: ICScorer):
        # HP:0012443 (brain) is annotated to only 1 disease
        ic_brain = scorer.ic.get("HP:0012443", 0.0)
        ic_root = scorer.ic.get("HP:0000001", 0.0)
        assert ic_brain > ic_root

    def test_ic_values_positive(self, scorer: ICScorer):
        for term, ic_val in scorer.ic.items():
            assert ic_val >= 0.0, f"IC for {term} is negative: {ic_val}"

    def test_ic_formula_correct(self, scorer: ICScorer):
        # HP:0000001 (root) — all 3 diseases have it as ancestor
        # p = 3/3 = 1.0, IC = -log2(1.0) = 0.0
        assert scorer.ic["HP:0000001"] == pytest.approx(0.0)

        # HP:0012443 (brain) — only OMIM:100002 has it
        # p = 1/3, IC = -log2(1/3) ≈ 1.585
        assert scorer.ic["HP:0012443"] == pytest.approx(-math.log2(1 / 3))


class TestMICA:
    def test_same_term_mica_is_self(self, scorer: ICScorer):
        mica = scorer.mica("HP:0000707", "HP:0000707")
        assert mica == "HP:0000707"

    def test_parent_child_mica_is_parent(self, scorer: ICScorer):
        # MICA of HP:0002011 (CNS) and HP:0000707 (Neuro) should be HP:0000707
        # Since CNS's ancestors include Neuro, their MICA is Neuro (which has higher
        # IC than the shared ancestors above it)
        mica = scorer.mica("HP:0002011", "HP:0000707")
        assert mica == "HP:0000707"

    def test_disjoint_branches_mica_is_common_ancestor(self, scorer: ICScorer):
        # HP:0000707 (Neuro) and HP:0000234 (Head) share HP:0000118 as MICA
        mica = scorer.mica("HP:0000707", "HP:0000234")
        assert mica == "HP:0000118"

    def test_nonexistent_term_returns_none(self, scorer: ICScorer):
        mica = scorer.mica("HP:FAKE", "HP:0000707")
        assert mica is None


class TestSimilarity:
    def test_resnik_same_term(self, scorer: ICScorer):
        sim = scorer.resnik_similarity("HP:0000707", "HP:0000707")
        assert sim == scorer.ic["HP:0000707"]

    def test_resnik_related_terms_positive(self, scorer: ICScorer):
        sim = scorer.resnik_similarity("HP:0002011", "HP:0000707")
        assert sim > 0

    def test_resnik_unrelated_terms_lower(self, scorer: ICScorer):
        sim_related = scorer.resnik_similarity("HP:0002011", "HP:0000707")
        sim_unrelated = scorer.resnik_similarity("HP:0000707", "HP:0000234")
        assert sim_related > sim_unrelated

    def test_lin_same_term_is_one(self, scorer: ICScorer):
        sim = scorer.lin_similarity("HP:0000707", "HP:0000707")
        assert sim == pytest.approx(1.0)

    def test_lin_bounded_zero_one(self, scorer: ICScorer):
        for t1 in scorer.ic:
            for t2 in scorer.ic:
                sim = scorer.lin_similarity(t1, t2)
                assert 0.0 <= sim <= 1.0 + 1e-9, f"Lin({t1},{t2})={sim}"

    def test_lin_symmetric(self, scorer: ICScorer):
        sim_ab = scorer.lin_similarity("HP:0002011", "HP:0000234")
        sim_ba = scorer.lin_similarity("HP:0000234", "HP:0002011")
        assert sim_ab == pytest.approx(sim_ba)


class TestRanking:
    def test_correct_disease_ranks_high(self, scorer: ICScorer):
        # Query with neuro terms — OMIM:100002 (Brain+CNS) should rank high
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0012443", "HP:0002011"],
            disease_annotations=DISEASE_ANNOTATIONS,
            method="resnik",
        )
        top_disease = results[0].disease_id
        assert top_disease == "OMIM:100002"

    def test_ranking_returns_all_diseases(self, scorer: ICScorer):
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        assert len(results) == 3

    def test_ranks_are_sequential(self, scorer: ICScorer):
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        ranks = [r.rank for r in results]
        assert ranks == [1, 2, 3]

    def test_scores_descending(self, scorer: ICScorer):
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_lin_method_works(self, scorer: ICScorer):
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0012443"],
            disease_annotations=DISEASE_ANNOTATIONS,
            method="lin",
        )
        assert len(results) == 3
        assert results[0].score > 0

    def test_invalid_method_raises(self, scorer: ICScorer):
        with pytest.raises(ValueError, match="Unknown"):
            rank_diseases(
                scorer,
                query_hpo_terms=["HP:0000707"],
                disease_annotations=DISEASE_ANNOTATIONS,
                method="invalid",
            )

    def test_head_query_matches_head_disease(self, scorer: ICScorer):
        # Query with head term — OMIM:100003 (Head/neck) should rank high
        results = rank_diseases(
            scorer,
            query_hpo_terms=["HP:0000234"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        # OMIM:100001 has HP:0000234 directly, OMIM:100003 has HP:0000152 (parent)
        assert results[0].disease_id == "OMIM:100001"
