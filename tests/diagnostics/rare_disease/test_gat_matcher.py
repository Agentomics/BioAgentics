"""Tests for the GAT-based phenotype matcher."""

from __future__ import annotations

import networkx as nx
import pytest
import torch

from bioagentics.diagnostics.rare_disease.gat_matcher import (
    GATConfig,
    GATLayer,
    GATLinkPredictor,
    GATMatcher,
    _prepare_graph_data,
    rank_diseases,
    train_gat,
)


def _build_test_graph() -> nx.DiGraph:
    """Build a small heterogeneous test graph for GAT testing.

    Phenotype hierarchy:
      HP:ROOT -> HP:A -> HP:A1, HP:A2
      HP:ROOT -> HP:B -> HP:B1

    Diseases with has_phenotype edges:
      D1 -> HP:A1, HP:A2
      D2 -> HP:B1
      D3 -> HP:A1, HP:B1
    """
    g = nx.DiGraph()
    for pid in ["HP:ROOT", "HP:A", "HP:B", "HP:A1", "HP:A2", "HP:B1"]:
        g.add_node(pid, node_type="phenotype")
    g.add_edge("HP:A", "HP:ROOT", edge_type="is_a")
    g.add_edge("HP:B", "HP:ROOT", edge_type="is_a")
    g.add_edge("HP:A1", "HP:A", edge_type="is_a")
    g.add_edge("HP:A2", "HP:A", edge_type="is_a")
    g.add_edge("HP:B1", "HP:B", edge_type="is_a")

    for did in ["D1", "D2", "D3"]:
        g.add_node(did, node_type="disease")
    g.add_edge("D1", "HP:A1", edge_type="has_phenotype")
    g.add_edge("D1", "HP:A2", edge_type="has_phenotype")
    g.add_edge("D2", "HP:B1", edge_type="has_phenotype")
    g.add_edge("D3", "HP:A1", edge_type="has_phenotype")
    g.add_edge("D3", "HP:B1", edge_type="has_phenotype")

    g.add_node("GENE:G1", node_type="gene")
    g.add_edge("D1", "GENE:G1", edge_type="associated_gene")
    return g


@pytest.fixture
def test_graph() -> nx.DiGraph:
    return _build_test_graph()


@pytest.fixture
def config() -> GATConfig:
    return GATConfig(
        hidden_dim=16,
        num_heads=2,
        num_layers=2,
        epochs=100,
        negative_ratio=2,
        seed=42,
    )


@pytest.fixture
def matcher(test_graph: nx.DiGraph, config: GATConfig) -> GATMatcher:
    return GATMatcher.from_graph(test_graph, config)


class TestGraphPreparation:
    def test_node_mapping(self, test_graph: nx.DiGraph):
        node2idx, _, _ = _prepare_graph_data(test_graph)
        assert len(node2idx) == test_graph.number_of_nodes()
        for node in test_graph.nodes():
            assert node in node2idx

    def test_edge_index_shape(self, test_graph: nx.DiGraph):
        _, edge_index, _ = _prepare_graph_data(test_graph)
        assert edge_index.shape[0] == 2
        assert edge_index.shape[1] > 0

    def test_has_phenotype_edges_found(self, test_graph: nx.DiGraph):
        _, _, pos_edges = _prepare_graph_data(test_graph)
        # D1->HP:A1, D1->HP:A2, D2->HP:B1, D3->HP:A1, D3->HP:B1
        assert len(pos_edges) == 5


class TestGATLayer:
    def test_output_shape(self):
        layer = GATLayer(in_features=16, out_features=16, num_heads=4)
        x = torch.randn(10, 16)
        edges = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
        out = layer(x, edges)
        assert out.shape == (10, 16)

    def test_no_edges(self):
        layer = GATLayer(in_features=16, out_features=16, num_heads=4)
        x = torch.randn(5, 16)
        edges = torch.empty(2, 0, dtype=torch.long)
        out = layer(x, edges)
        assert out.shape == (5, 16)


class TestModelTraining:
    def test_model_trains_without_error(self, test_graph: nx.DiGraph, config: GATConfig):
        model, node2idx, loss_history = train_gat(test_graph, config)
        assert len(loss_history) == config.epochs
        assert len(node2idx) == test_graph.number_of_nodes()

    def test_loss_decreases(self, test_graph: nx.DiGraph, config: GATConfig):
        """Loss should generally decrease (compare first few vs last few)."""
        _, _, loss_history = train_gat(test_graph, config)
        first_avg = sum(loss_history[:3]) / 3
        last_avg = sum(loss_history[-3:]) / 3
        assert last_avg < first_avg

    def test_model_parameters_updated(self, test_graph: nx.DiGraph, config: GATConfig):
        model, _, _ = train_gat(test_graph, config)
        # Embeddings should not be zero after training
        norm = model.node_embed.weight.norm().item()
        assert norm > 0


class TestGATMatcher:
    def test_score_existing_link(self, matcher: GATMatcher):
        """Known disease-phenotype links should score > 0."""
        score = matcher.score_disease_phenotype("D1", "HP:A1")
        assert score > 0

    def test_score_unknown_node(self, matcher: GATMatcher):
        assert matcher.score_disease_phenotype("UNKNOWN", "HP:A1") == 0.0

    def test_score_disease_with_query(self, matcher: GATMatcher):
        score = matcher.score_disease(["HP:A1", "HP:A2"], "D1")
        assert score > 0

    def test_score_bounded(self, matcher: GATMatcher):
        """Scores should be probabilities in [0, 1]."""
        score = matcher.score_disease(["HP:A1"], "D1")
        assert 0 <= score <= 1

    def test_known_link_scores_higher(self, matcher: GATMatcher):
        """D1 (linked to HP:A1, HP:A2) should score higher for branch-A
        query than D2 (linked to HP:B1 only)."""
        score_d1 = matcher.score_disease(["HP:A1", "HP:A2"], "D1")
        score_d2 = matcher.score_disease(["HP:A1", "HP:A2"], "D2")
        assert score_d1 > score_d2


class TestRanking:
    def test_ranking_returns_all(self, matcher: GATMatcher):
        results = rank_diseases(matcher, ["HP:A1"], ["D1", "D2", "D3"])
        assert len(results) == 3

    def test_ranks_sequential(self, matcher: GATMatcher):
        results = rank_diseases(matcher, ["HP:A1"], ["D1", "D2", "D3"])
        assert [r.rank for r in results] == [1, 2, 3]

    def test_scores_descending(self, matcher: GATMatcher):
        results = rank_diseases(matcher, ["HP:A1"], ["D1", "D2", "D3"])
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top10_includes_known_disease(self, matcher: GATMatcher):
        """D1 should appear in top results when queried with its phenotypes."""
        results = rank_diseases(
            matcher,
            ["HP:A1", "HP:A2"],
            ["D1", "D2", "D3"],
        )
        top_ids = [r.disease_id for r in results[:10]]
        assert "D1" in top_ids

    def test_empty_query(self, matcher: GATMatcher):
        results = rank_diseases(matcher, [], ["D1", "D2"])
        assert all(r.score == 0.0 for r in results)

    def test_unknown_disease(self, matcher: GATMatcher):
        results = rank_diseases(matcher, ["HP:A1"], ["D1", "UNKNOWN"])
        unknown = next(r for r in results if r.disease_id == "UNKNOWN")
        assert unknown.score == 0.0
