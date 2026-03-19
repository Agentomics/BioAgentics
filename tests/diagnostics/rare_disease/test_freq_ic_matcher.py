"""Tests for the frequency-weighted IC phenotype matcher."""

from __future__ import annotations

import networkx as nx
import pytest

from bioagentics.diagnostics.rare_disease.freq_ic_matcher import (
    FreqICMatcher,
    rank_diseases,
)
from bioagentics.diagnostics.rare_disease.ic_matcher import ICScorer


# Same test DAG as test_ic_matcher.py:
#   HP:0000001 (All)
#       |
#   HP:0000118 (Phenotypic abnormality)
#       |                |
#   HP:0000707 (Neuro)   HP:0000152 (Head/neck)
#       |                    |
#   HP:0002011 (CNS)     HP:0000234 (Head)
#       |
#   HP:0012443 (Brain)


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
    g.add_edge("HP:0000118", "HP:0000001", relation="is_a")
    g.add_edge("HP:0000707", "HP:0000118", relation="is_a")
    g.add_edge("HP:0000152", "HP:0000118", relation="is_a")
    g.add_edge("HP:0000234", "HP:0000152", relation="is_a")
    g.add_edge("HP:0002011", "HP:0000707", relation="is_a")
    g.add_edge("HP:0012443", "HP:0002011", relation="is_a")
    return g


DISEASE_ANNOTATIONS = {
    "OMIM:100001": ["HP:0000707", "HP:0000234"],  # Neuro + Head
    "OMIM:100002": ["HP:0012443", "HP:0002011"],  # Brain + CNS
    "OMIM:100003": ["HP:0000152"],  # Head/neck only
}


@pytest.fixture
def scorer() -> ICScorer:
    dag = _build_test_dag()
    s = ICScorer(hpo_dag=dag)
    s.compute_ic(DISEASE_ANNOTATIONS)
    return s


@pytest.fixture
def matcher(scorer: ICScorer) -> FreqICMatcher:
    m = FreqICMatcher(scorer=scorer)
    # OMIM:100001: Neuro is obligate, Head is occasional
    m.add_disease_frequencies("OMIM:100001", [
        {"hpo_id": "HP:0000707", "frequency_category": "obligate", "frequency_value": 1.0},
        {"hpo_id": "HP:0000234", "frequency_category": "occasional", "frequency_value": 0.12},
    ])
    # OMIM:100002: Brain is very_frequent, CNS is obligate
    m.add_disease_frequencies("OMIM:100002", [
        {"hpo_id": "HP:0012443", "frequency_category": "very_frequent", "frequency_value": 0.90},
        {"hpo_id": "HP:0002011", "frequency_category": "obligate", "frequency_value": 1.0},
    ])
    # OMIM:100003: Head/neck is frequent
    m.add_disease_frequencies("OMIM:100003", [
        {"hpo_id": "HP:0000152", "frequency_category": "frequent", "frequency_value": 0.50},
    ])
    return m


class TestFreqAnnotationLoading:
    def test_add_disease_frequencies(self, matcher: FreqICMatcher):
        assert "OMIM:100001" in matcher.disease_freq
        assert len(matcher.disease_freq["OMIM:100001"]) == 2

    def test_excluded_terms_tracked(self, scorer: ICScorer):
        m = FreqICMatcher(scorer=scorer)
        m.add_disease_frequencies("OMIM:999", [
            {"hpo_id": "HP:0000707", "frequency_category": "obligate", "frequency_value": 1.0},
            {"hpo_id": "HP:0000234", "frequency_category": "excluded", "frequency_value": 0.0},
        ])
        assert "HP:0000234" in m.excluded_terms["OMIM:999"]
        assert "HP:0000234" not in m.disease_freq["OMIM:999"]

    def test_hpoa_style_annotations(self, scorer: ICScorer):
        m = FreqICMatcher(scorer=scorer)
        m.add_disease_frequencies("OMIM:999", [
            {"hpo_id": "HP:0000707", "frequency_label": "obligate", "frequency": 1.0},
        ])
        ann = m.disease_freq["OMIM:999"]["HP:0000707"]
        assert ann.frequency == 1.0
        assert ann.category == "obligate"


class TestObligateScoresHigher:
    def test_obligate_beats_occasional(self, matcher: FreqICMatcher):
        """A disease with obligate frequency for the matching term should score
        higher than the same disease with only occasional frequency."""
        # Query with neuro term — OMIM:100001 has it as obligate (1.0)
        score_obligate = matcher.score_disease(
            ["HP:0000707"], "OMIM:100001", ["HP:0000707", "HP:0000234"]
        )
        # OMIM:100003 has head/neck as frequent (0.50)
        score_frequent = matcher.score_disease(
            ["HP:0000152"], "OMIM:100003", ["HP:0000152"]
        )
        # The obligate-weighted match should score higher (same self-similarity
        # but multiplied by 1.0 vs 0.50)
        assert score_obligate > score_frequent

    def test_frequency_weight_scales_score(self, matcher: FreqICMatcher, scorer: ICScorer):
        """Frequency weight should directly scale the similarity contribution."""
        sim_neuro = scorer.resnik_similarity("HP:0000707", "HP:0000707")
        # OMIM:100001 has HP:0000707 as obligate (weight=1.0)
        score = matcher.score_disease(
            ["HP:0000707"], "OMIM:100001", ["HP:0000707"]
        )
        assert score == pytest.approx(1.0 * sim_neuro)

    def test_occasional_weight_reduces_score(self, matcher: FreqICMatcher, scorer: ICScorer):
        """Occasional frequency (0.12) should substantially reduce score."""
        sim_head = scorer.resnik_similarity("HP:0000234", "HP:0000234")
        # OMIM:100001 has HP:0000234 as occasional (weight=0.12)
        score = matcher.score_disease(
            ["HP:0000234"], "OMIM:100001", ["HP:0000234"]
        )
        # The weighted score should be 0.12 * sim
        assert score == pytest.approx(0.12 * sim_head)


class TestExcludedTermPenalty:
    def test_excluded_term_reduces_score(self, scorer: ICScorer):
        """A query term matching an excluded phenotype should penalize."""
        m = FreqICMatcher(scorer=scorer)
        # Disease with neuro as obligate, head as excluded
        m.add_disease_frequencies("OMIM:EX1", [
            {"hpo_id": "HP:0000707", "frequency_category": "obligate", "frequency_value": 1.0},
            {"hpo_id": "HP:0000234", "frequency_category": "excluded", "frequency_value": 0.0},
        ])
        # Score with no excluded match
        score_no_excluded = m.score_disease(
            ["HP:0000707"], "OMIM:EX1", ["HP:0000707"]
        )
        # Score with a query that includes the excluded term
        score_with_excluded = m.score_disease(
            ["HP:0000707", "HP:0000234"], "OMIM:EX1", ["HP:0000707"]
        )
        assert score_with_excluded < score_no_excluded

    def test_excluded_penalty_does_not_go_negative(self, scorer: ICScorer):
        """Score should be clamped at zero."""
        m = FreqICMatcher(scorer=scorer)
        m.add_disease_frequencies("OMIM:EX2", [
            {"hpo_id": "HP:0000707", "frequency_category": "excluded", "frequency_value": 0.0},
            {"hpo_id": "HP:0000234", "frequency_category": "excluded", "frequency_value": 0.0},
            {"hpo_id": "HP:0002011", "frequency_category": "excluded", "frequency_value": 0.0},
            {"hpo_id": "HP:0012443", "frequency_category": "excluded", "frequency_value": 0.0},
        ])
        score = m.score_disease(
            ["HP:0000707", "HP:0000234", "HP:0002011", "HP:0012443"],
            "OMIM:EX2",
            [],  # no non-excluded terms
        )
        assert score >= 0.0


class TestRanking:
    def test_ranking_returns_all_diseases(self, matcher: FreqICMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        assert len(results) == 3

    def test_ranks_sequential(self, matcher: FreqICMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_descending(self, matcher: FreqICMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:0000707"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_freq_weighting_changes_ranking(self, scorer: ICScorer):
        """Frequency weighting should change ranking vs unweighted IC."""
        from bioagentics.diagnostics.rare_disease.ic_matcher import (
            rank_diseases as ic_rank_diseases,
        )

        # Create matcher where OMIM:100003 has obligate head and
        # OMIM:100001 has only occasional head
        m = FreqICMatcher(scorer=scorer)
        m.add_disease_frequencies("OMIM:100001", [
            {"hpo_id": "HP:0000707", "frequency_category": "occasional", "frequency_value": 0.12},
            {"hpo_id": "HP:0000234", "frequency_category": "occasional", "frequency_value": 0.12},
        ])
        m.add_disease_frequencies("OMIM:100003", [
            {"hpo_id": "HP:0000152", "frequency_category": "obligate", "frequency_value": 1.0},
        ])

        query = ["HP:0000234"]  # Head term

        # Unweighted: OMIM:100001 has exact HP:0000234 match, should rank #1
        ic_results = ic_rank_diseases(scorer, query, DISEASE_ANNOTATIONS)
        assert ic_results[0].disease_id == "OMIM:100001"

        # Freq-weighted: OMIM:100001's head is only occasional (0.12),
        # while OMIM:100003's head/neck is obligate (1.0).
        # OMIM:100003 should now score higher despite parent-level match
        freq_results = rank_diseases(m, query, DISEASE_ANNOTATIONS)
        assert freq_results[0].disease_id == "OMIM:100003"

    def test_lin_method(self, matcher: FreqICMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:0012443"],
            disease_annotations=DISEASE_ANNOTATIONS,
            method="lin",
        )
        assert len(results) == 3
        assert results[0].score > 0

    def test_invalid_method_raises(self, matcher: FreqICMatcher):
        with pytest.raises(ValueError, match="Unknown"):
            rank_diseases(
                matcher,
                query_hpo_terms=["HP:0000707"],
                disease_annotations=DISEASE_ANNOTATIONS,
                method="invalid",
            )

    def test_empty_query(self, matcher: FreqICMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=[],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        assert all(r.score == 0.0 for r in results)

    def test_no_freq_data_uses_default(self, scorer: ICScorer):
        """Without frequency data, all terms get default weight 0.5."""
        m = FreqICMatcher(scorer=scorer)
        results = rank_diseases(
            m,
            query_hpo_terms=["HP:0012443"],
            disease_annotations=DISEASE_ANNOTATIONS,
        )
        # Should still work, just with uniform weighting
        assert len(results) == 3
        assert results[0].score > 0
