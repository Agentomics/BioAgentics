"""Tests for heterogeneous knowledge graph construction."""

import json
from pathlib import Path

import networkx as nx
import pytest

from bioagentics.diagnostics.rare_disease.knowledge_graph import (
    build_knowledge_graph,
    get_graph_stats,
    validate_knowledge_graph,
    save_knowledge_graph,
    load_knowledge_graph,
)


# --- Fixtures: small test data ---


@pytest.fixture
def hpo_dag() -> nx.DiGraph:
    """Minimal HPO DAG with a few phenotype terms."""
    g = nx.DiGraph()
    g.add_node("HP:0000001", node_type="phenotype", name="All")
    g.add_node("HP:0000118", node_type="phenotype", name="Phenotypic abnormality")
    g.add_node("HP:0000924", node_type="phenotype", name="Abnormality of the skeletal system")
    g.add_node("HP:0001166", node_type="phenotype", name="Arachnodactyly")
    g.add_node("HP:0001382", node_type="phenotype", name="Joint hypermobility")
    g.add_node("HP:0000256", node_type="phenotype", name="Macrocephaly")
    g.add_node("HP:0000707", node_type="phenotype", name="Abnormality of the nervous system")
    g.add_node("HP:0002011", node_type="phenotype", name="Brain morphology abnormality")

    # is_a edges (child → parent)
    g.add_edge("HP:0000118", "HP:0000001", relation="is_a")
    g.add_edge("HP:0000924", "HP:0000118", relation="is_a")
    g.add_edge("HP:0001166", "HP:0000924", relation="is_a")
    g.add_edge("HP:0001382", "HP:0000924", relation="is_a")
    g.add_edge("HP:0000707", "HP:0000118", relation="is_a")
    g.add_edge("HP:0000256", "HP:0002011", relation="is_a")
    g.add_edge("HP:0002011", "HP:0000707", relation="is_a")
    return g


@pytest.fixture
def omim_disease_hpo() -> dict[str, list[dict]]:
    """OMIM disease-HPO annotations."""
    return {
        "OMIM:154700": [
            {"hpo_id": "HP:0001166", "frequency": 0.90, "evidence": "PCS", "onset": ""},
            {"hpo_id": "HP:0000256", "frequency": 0.30, "evidence": "TAS", "onset": ""},
        ],
        "OMIM:130000": [
            {"hpo_id": "HP:0001382", "frequency": 0.80, "evidence": "PCS", "onset": ""},
        ],
    }


@pytest.fixture
def orphanet_disease_hpo() -> dict[str, list[dict]]:
    """Orphanet disease-HPO annotations with frequency detail."""
    return {
        "ORPHA:558": [
            {
                "hpo_id": "HP:0001166",
                "hpo_term_name": "Arachnodactyly",
                "frequency_category": "obligate",
                "frequency_value": 1.00,
            },
            {
                "hpo_id": "HP:0000256",
                "hpo_term_name": "Macrocephaly",
                "frequency_category": "occasional",
                "frequency_value": 0.12,
            },
        ],
    }


@pytest.fixture
def disease_gene_map() -> dict[str, list[dict]]:
    """Disease→gene associations from genemap2."""
    return {
        "OMIM:154700": [
            {
                "gene_symbol": "FBN1",
                "gene_mim": "134797",
                "mapping_key": 3,
                "inheritance_patterns": ["Autosomal dominant"],
            },
        ],
        "OMIM:130000": [
            {
                "gene_symbol": "COL5A1",
                "gene_mim": "120215",
                "mapping_key": 3,
                "inheritance_patterns": ["Autosomal dominant"],
            },
        ],
    }


@pytest.fixture
def orphanet_crossref() -> dict[str, list[str]]:
    return {"ORPHA:558": ["OMIM:154700"]}


@pytest.fixture
def knowledge_graph(
    hpo_dag, omim_disease_hpo, orphanet_disease_hpo, disease_gene_map, orphanet_crossref
) -> nx.DiGraph:
    return build_knowledge_graph(
        hpo_dag, omim_disease_hpo, orphanet_disease_hpo,
        disease_gene_map, orphanet_crossref,
    )


# --- Tests: build_knowledge_graph ---


class TestBuildKnowledgeGraph:
    def test_has_phenotype_nodes(self, knowledge_graph):
        phenotype_nodes = [
            n for n, d in knowledge_graph.nodes(data=True)
            if d.get("node_type") == "phenotype"
        ]
        assert len(phenotype_nodes) == 8  # All HPO terms from fixture

    def test_has_disease_nodes(self, knowledge_graph):
        disease_nodes = [
            n for n, d in knowledge_graph.nodes(data=True)
            if d.get("node_type") == "disease"
        ]
        # OMIM:154700, OMIM:130000, ORPHA:558
        assert len(disease_nodes) == 3

    def test_has_gene_nodes(self, knowledge_graph):
        gene_nodes = [
            n for n, d in knowledge_graph.nodes(data=True)
            if d.get("node_type") == "gene"
        ]
        assert len(gene_nodes) == 2  # FBN1, COL5A1

    def test_is_a_edges(self, knowledge_graph):
        is_a_edges = [
            (u, v) for u, v, d in knowledge_graph.edges(data=True)
            if d.get("edge_type") == "is_a"
        ]
        assert len(is_a_edges) == 7  # Same as HPO DAG

    def test_has_phenotype_edges(self, knowledge_graph):
        hp_edges = [
            (u, v) for u, v, d in knowledge_graph.edges(data=True)
            if d.get("edge_type") == "has_phenotype"
        ]
        # OMIM:154700→HP:0001166, HP:0000256; OMIM:130000→HP:0001382; ORPHA:558→HP:0001166, HP:0000256
        assert len(hp_edges) == 5

    def test_associated_gene_edges(self, knowledge_graph):
        gene_edges = [
            (u, v) for u, v, d in knowledge_graph.edges(data=True)
            if d.get("edge_type") == "associated_gene"
        ]
        assert len(gene_edges) == 2  # FBN1, COL5A1

    def test_same_as_edges(self, knowledge_graph):
        same_as_edges = [
            (u, v) for u, v, d in knowledge_graph.edges(data=True)
            if d.get("edge_type") == "same_as"
        ]
        # Bidirectional: ORPHA:558↔OMIM:154700
        assert len(same_as_edges) == 2

    def test_frequency_on_omim_edges(self, knowledge_graph):
        edge_data = knowledge_graph.edges["OMIM:154700", "HP:0001166"]
        assert edge_data["frequency"] == 0.90

    def test_frequency_on_orphanet_edges(self, knowledge_graph):
        edge_data = knowledge_graph.edges["ORPHA:558", "HP:0001166"]
        assert edge_data["frequency"] == 1.00
        assert edge_data["frequency_category"] == "obligate"

    def test_gene_node_attributes(self, knowledge_graph):
        fbn1 = knowledge_graph.nodes["GENE:FBN1"]
        assert fbn1["node_type"] == "gene"
        assert fbn1["name"] == "FBN1"

    def test_skips_unknown_hpo_terms(self, hpo_dag, disease_gene_map, orphanet_crossref):
        """Phenotype edges to unknown HPO terms are silently skipped."""
        omim_map = {
            "OMIM:999999": [
                {"hpo_id": "HP:9999999", "frequency": 0.5, "evidence": "IEA", "onset": ""},
            ],
        }
        g = build_knowledge_graph(
            hpo_dag, omim_map, {}, disease_gene_map, orphanet_crossref,
        )
        # Disease node added but no has_phenotype edge to unknown term
        assert g.has_node("OMIM:999999")
        hp_edges = [
            (u, v) for u, v, d in g.edges(data=True)
            if u == "OMIM:999999" and d.get("edge_type") == "has_phenotype"
        ]
        assert len(hp_edges) == 0

    def test_no_crossref_param(self, hpo_dag, omim_disease_hpo, orphanet_disease_hpo, disease_gene_map):
        """Graph builds fine without cross-references."""
        g = build_knowledge_graph(
            hpo_dag, omim_disease_hpo, orphanet_disease_hpo, disease_gene_map, None,
        )
        same_as = [
            (u, v) for u, v, d in g.edges(data=True)
            if d.get("edge_type") == "same_as"
        ]
        assert len(same_as) == 0


# --- Tests: get_graph_stats ---


class TestGetGraphStats:
    def test_total_nodes(self, knowledge_graph):
        stats = get_graph_stats(knowledge_graph)
        # 8 phenotype + 3 disease + 2 gene = 13
        assert stats["total_nodes"] == 13

    def test_total_edges(self, knowledge_graph):
        stats = get_graph_stats(knowledge_graph)
        # 7 is_a + 5 has_phenotype + 2 associated_gene + 2 same_as = 16
        assert stats["total_edges"] == 16

    def test_node_type_counts(self, knowledge_graph):
        stats = get_graph_stats(knowledge_graph)
        assert stats["node_types"]["phenotype"] == 8
        assert stats["node_types"]["disease"] == 3
        assert stats["node_types"]["gene"] == 2

    def test_edge_type_counts(self, knowledge_graph):
        stats = get_graph_stats(knowledge_graph)
        assert stats["edge_types"]["is_a"] == 7
        assert stats["edge_types"]["has_phenotype"] == 5
        assert stats["edge_types"]["associated_gene"] == 2
        assert stats["edge_types"]["same_as"] == 2

    def test_connectivity(self, knowledge_graph):
        stats = get_graph_stats(knowledge_graph)
        # All nodes should be in one connected component
        assert stats["num_components"] == 1
        assert stats["largest_component_fraction"] == 1.0


# --- Tests: validate_knowledge_graph ---


class TestValidateKnowledgeGraph:
    def test_valid_graph_no_warnings(self, knowledge_graph):
        warnings = validate_knowledge_graph(knowledge_graph)
        assert warnings == []

    def test_missing_node_type(self):
        g = nx.DiGraph()
        g.add_node("HP:001", node_type="phenotype")
        g.add_node("OMIM:001", node_type="disease")
        g.add_edge("OMIM:001", "HP:001", edge_type="has_phenotype")
        warnings = validate_knowledge_graph(g)
        assert any("Missing node types" in w for w in warnings)

    def test_low_connectivity(self):
        g = nx.DiGraph()
        # Two disconnected components
        g.add_node("HP:001", node_type="phenotype")
        g.add_node("OMIM:001", node_type="disease")
        g.add_node("GENE:X", node_type="gene")
        g.add_edge("OMIM:001", "HP:001", edge_type="has_phenotype")
        # GENE:X is disconnected
        warnings = validate_knowledge_graph(g)
        assert any("Largest component" in w for w in warnings)


# --- Tests: save/load ---


class TestSaveLoad:
    def test_roundtrip(self, knowledge_graph, tmp_path):
        save_knowledge_graph(knowledge_graph, tmp_path)
        loaded = load_knowledge_graph(tmp_path / "knowledge_graph.graphml")
        assert loaded.number_of_nodes() == knowledge_graph.number_of_nodes()
        assert loaded.number_of_edges() == knowledge_graph.number_of_edges()

    def test_load_nonexistent(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_knowledge_graph(tmp_path / "nonexistent.graphml")
