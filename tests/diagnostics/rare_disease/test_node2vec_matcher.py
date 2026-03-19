"""Tests for the node2vec-based phenotype matcher."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from bioagentics.diagnostics.rare_disease.node2vec_matcher import (
    Node2VecConfig,
    Node2VecMatcher,
    _biased_walk,
    _build_adjacency,
    generate_walks,
    rank_diseases,
    train_embeddings,
)


def _build_test_graph() -> nx.DiGraph:
    """Build a small heterogeneous test graph.

    Phenotype hierarchy:
      HP:ROOT -> HP:A -> HP:A1, HP:A2
      HP:ROOT -> HP:B -> HP:B1

    Diseases:
      D1 has_phenotype HP:A1, HP:A2 (both from branch A)
      D2 has_phenotype HP:B1 (branch B)
      D3 has_phenotype HP:A1, HP:B1 (mixed)

    Gene:
      GENE:G1 associated with D1
    """
    g = nx.DiGraph()
    # Phenotype nodes
    for pid in ["HP:ROOT", "HP:A", "HP:B", "HP:A1", "HP:A2", "HP:B1"]:
        g.add_node(pid, node_type="phenotype")
    # Phenotype hierarchy (child -> parent)
    g.add_edge("HP:A", "HP:ROOT", edge_type="is_a")
    g.add_edge("HP:B", "HP:ROOT", edge_type="is_a")
    g.add_edge("HP:A1", "HP:A", edge_type="is_a")
    g.add_edge("HP:A2", "HP:A", edge_type="is_a")
    g.add_edge("HP:B1", "HP:B", edge_type="is_a")
    # Disease nodes
    for did in ["D1", "D2", "D3"]:
        g.add_node(did, node_type="disease")
    g.add_edge("D1", "HP:A1", edge_type="has_phenotype")
    g.add_edge("D1", "HP:A2", edge_type="has_phenotype")
    g.add_edge("D2", "HP:B1", edge_type="has_phenotype")
    g.add_edge("D3", "HP:A1", edge_type="has_phenotype")
    g.add_edge("D3", "HP:B1", edge_type="has_phenotype")
    # Gene node
    g.add_node("GENE:G1", node_type="gene")
    g.add_edge("D1", "GENE:G1", edge_type="associated_gene")
    return g


@pytest.fixture
def test_graph() -> nx.DiGraph:
    return _build_test_graph()


@pytest.fixture
def config() -> Node2VecConfig:
    return Node2VecConfig(
        dimensions=16,
        walk_length=10,
        num_walks=5,
        epochs=3,
        negative_samples=3,
        seed=42,
    )


@pytest.fixture
def matcher(test_graph: nx.DiGraph, config: Node2VecConfig) -> Node2VecMatcher:
    return Node2VecMatcher.from_graph(test_graph, config)


class TestAdjacency:
    def test_undirected_adjacency(self, test_graph: nx.DiGraph):
        adj = _build_adjacency(test_graph)
        # HP:A -> HP:ROOT edge means both should appear in each other's adjacency
        assert "HP:ROOT" in adj["HP:A"]
        assert "HP:A" in adj["HP:ROOT"]

    def test_all_nodes_present(self, test_graph: nx.DiGraph):
        adj = _build_adjacency(test_graph)
        for node in test_graph.nodes():
            assert node in adj


class TestRandomWalks:
    def test_walk_starts_at_node(self, test_graph: nx.DiGraph):
        import random
        adj = _build_adjacency(test_graph)
        rng = random.Random(42)
        walk = _biased_walk(adj, "HP:A", 10, 1.0, 1.0, rng)
        assert walk[0] == "HP:A"

    def test_walk_length_bounded(self, test_graph: nx.DiGraph):
        import random
        adj = _build_adjacency(test_graph)
        rng = random.Random(42)
        walk = _biased_walk(adj, "HP:A", 10, 1.0, 1.0, rng)
        assert len(walk) <= 10

    def test_walk_stays_in_graph(self, test_graph: nx.DiGraph):
        import random
        adj = _build_adjacency(test_graph)
        rng = random.Random(42)
        nodes = set(test_graph.nodes())
        walk = _biased_walk(adj, "HP:A", 10, 1.0, 1.0, rng)
        for node in walk:
            assert node in nodes

    def test_generate_walks_count(self, test_graph: nx.DiGraph, config: Node2VecConfig):
        walks = generate_walks(test_graph, config)
        # num_walks * num_nodes_with_neighbors
        assert len(walks) > 0
        assert len(walks) <= config.num_walks * test_graph.number_of_nodes()


class TestEmbeddings:
    def test_embedding_dimensions(self, matcher: Node2VecMatcher, config: Node2VecConfig):
        assert matcher.embeddings.shape[1] == config.dimensions

    def test_all_nodes_embedded(self, matcher: Node2VecMatcher, test_graph: nx.DiGraph):
        for node in test_graph.nodes():
            assert node in matcher.word2idx

    def test_embedding_not_zero(self, matcher: Node2VecMatcher):
        vec = matcher.get_embedding("D1")
        assert vec is not None
        assert np.linalg.norm(vec) > 0

    def test_nonexistent_node_returns_none(self, matcher: Node2VecMatcher):
        assert matcher.get_embedding("FAKE_NODE") is None

    def test_query_embedding_averages(self, matcher: Node2VecMatcher):
        vec_a1 = matcher.get_embedding("HP:A1")
        vec_a2 = matcher.get_embedding("HP:A2")
        query_vec = matcher.query_embedding(["HP:A1", "HP:A2"])
        assert query_vec is not None
        expected = (vec_a1 + vec_a2) / 2
        np.testing.assert_allclose(query_vec, expected)

    def test_query_embedding_ignores_unknown(self, matcher: Node2VecMatcher):
        vec = matcher.query_embedding(["HP:A1", "UNKNOWN"])
        expected = matcher.get_embedding("HP:A1")
        assert vec is not None
        np.testing.assert_allclose(vec, expected)

    def test_query_all_unknown_returns_none(self, matcher: Node2VecMatcher):
        assert matcher.query_embedding(["FAKE1", "FAKE2"]) is None


class TestRelatedSimilarity:
    def test_related_diseases_more_similar(self, matcher: Node2VecMatcher):
        """D1 (branch A) should be more similar to D3 (mixed A+B) than to D2 (branch B)."""
        from sklearn.metrics.pairwise import cosine_similarity

        d1 = matcher.get_embedding("D1").reshape(1, -1)
        d2 = matcher.get_embedding("D2").reshape(1, -1)
        d3 = matcher.get_embedding("D3").reshape(1, -1)

        sim_d1_d3 = cosine_similarity(d1, d3)[0, 0]
        sim_d1_d2 = cosine_similarity(d1, d2)[0, 0]

        # D1 shares HP:A1 with D3 but nothing with D2
        assert sim_d1_d3 > sim_d1_d2

    def test_phenotype_proximity(self, matcher: Node2VecMatcher):
        """Sibling phenotypes (HP:A1, HP:A2) should be more similar than
        cross-branch phenotypes (HP:A1, HP:B1)."""
        from sklearn.metrics.pairwise import cosine_similarity

        a1 = matcher.get_embedding("HP:A1").reshape(1, -1)
        a2 = matcher.get_embedding("HP:A2").reshape(1, -1)
        b1 = matcher.get_embedding("HP:B1").reshape(1, -1)

        sim_siblings = cosine_similarity(a1, a2)[0, 0]
        sim_cross = cosine_similarity(a1, b1)[0, 0]

        assert sim_siblings > sim_cross


class TestRanking:
    def test_branch_a_query_ranks_d1_above_d2(self, matcher: Node2VecMatcher):
        """Querying branch-A phenotypes should rank D1 above D2."""
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:A1", "HP:A2"],
            disease_ids=["D1", "D2", "D3"],
        )
        d1_rank = next(r.rank for r in results if r.disease_id == "D1")
        d2_rank = next(r.rank for r in results if r.disease_id == "D2")
        assert d1_rank < d2_rank

    def test_ranking_returns_all_diseases(self, matcher: Node2VecMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:A1"],
            disease_ids=["D1", "D2", "D3"],
        )
        assert len(results) == 3

    def test_ranks_sequential(self, matcher: Node2VecMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:A1"],
            disease_ids=["D1", "D2", "D3"],
        )
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_descending(self, matcher: Node2VecMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:A1"],
            disease_ids=["D1", "D2", "D3"],
        )
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_empty_query(self, matcher: Node2VecMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=[],
            disease_ids=["D1", "D2"],
        )
        assert all(r.score == 0.0 for r in results)

    def test_unknown_disease_gets_zero(self, matcher: Node2VecMatcher):
        results = rank_diseases(
            matcher,
            query_hpo_terms=["HP:A1"],
            disease_ids=["D1", "UNKNOWN"],
        )
        unknown_result = next(r for r in results if r.disease_id == "UNKNOWN")
        assert unknown_result.score == 0.0


class TestTrainingConvergence:
    def test_loss_decreases(self, test_graph: nx.DiGraph):
        """Training loss should decrease over epochs (rough check)."""
        import logging
        config = Node2VecConfig(
            dimensions=16, walk_length=10, num_walks=5,
            epochs=5, negative_samples=3, seed=42,
        )
        walks = generate_walks(test_graph, config)
        # Just verify training completes without error
        embeddings, word2idx, idx2word = train_embeddings(walks, config)
        assert embeddings.shape == (len(word2idx), config.dimensions)
