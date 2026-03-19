"""Tests for the HPO OBO parser."""

from __future__ import annotations

import textwrap
from pathlib import Path

import networkx as nx
import pytest

from bioagentics.diagnostics.rare_disease.hpo_parser import (
    HPO_ROOT_ID,
    build_hpo_dag,
    get_ancestors,
    get_descendants,
    parse_obo,
    validate_dag,
)

# Minimal OBO content for testing
MINIMAL_OBO = textwrap.dedent("""\
    format-version: 1.2
    ontology: hp

    [Term]
    id: HP:0000001
    name: All

    [Term]
    id: HP:0000118
    name: Phenotypic abnormality
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000707
    name: Abnormality of the nervous system
    def: "Any abnormality of the nervous system." [HPO:probinson]
    is_a: HP:0000118 ! Phenotypic abnormality

    [Term]
    id: HP:0000152
    name: Abnormality of head or neck
    is_a: HP:0000118 ! Phenotypic abnormality

    [Term]
    id: HP:0000234
    name: Abnormality of the head
    is_a: HP:0000152 ! Abnormality of head or neck

    [Term]
    id: HP:0002011
    name: Morphological central nervous system abnormality
    is_a: HP:0000707 ! Abnormality of the nervous system

    [Term]
    id: HP:0012443
    name: Abnormality of brain morphology
    is_a: HP:0002011 ! Morphological central nervous system abnormality
    is_a: HP:0000234 ! Abnormality of the head

    [Term]
    id: HP:9999999
    name: Obsolete term
    is_obsolete: true
    is_a: HP:0000118 ! Phenotypic abnormality
""")

# OBO with part_of relationship
OBO_WITH_PART_OF = textwrap.dedent("""\
    format-version: 1.2
    ontology: hp

    [Term]
    id: HP:0000001
    name: All

    [Term]
    id: HP:0000118
    name: Phenotypic abnormality
    is_a: HP:0000001 ! All

    [Term]
    id: HP:0000707
    name: Abnormality of the nervous system
    is_a: HP:0000118 ! Phenotypic abnormality

    [Term]
    id: HP:0100022
    name: Abnormality of movement
    is_a: HP:0000707 ! Abnormality of the nervous system

    [Term]
    id: HP:0100023
    name: Tremor
    is_a: HP:0100022 ! Abnormality of movement
    relationship: part_of HP:0000707 ! Abnormality of the nervous system
""")

# OBO with alt_id
OBO_WITH_ALT_ID = textwrap.dedent("""\
    format-version: 1.2
    ontology: hp

    [Term]
    id: HP:0000001
    name: All

    [Term]
    id: HP:0000118
    name: Phenotypic abnormality
    alt_id: HP:0000000
    alt_id: HP:9000000
    is_a: HP:0000001 ! All
""")


@pytest.fixture
def obo_file(tmp_path: Path) -> Path:
    p = tmp_path / "hp.obo"
    p.write_text(MINIMAL_OBO)
    return p


@pytest.fixture
def obo_with_part_of(tmp_path: Path) -> Path:
    p = tmp_path / "hp_partof.obo"
    p.write_text(OBO_WITH_PART_OF)
    return p


@pytest.fixture
def obo_with_alt_id(tmp_path: Path) -> Path:
    p = tmp_path / "hp_alt.obo"
    p.write_text(OBO_WITH_ALT_ID)
    return p


class TestParseObo:
    def test_parses_correct_term_count(self, obo_file: Path):
        terms = parse_obo(obo_file)
        # 7 active + 1 obsolete = 8 total
        assert len(terms) == 8

    def test_parses_term_id_and_name(self, obo_file: Path):
        terms = parse_obo(obo_file)
        root = next(t for t in terms if t.id == HPO_ROOT_ID)
        assert root.name == "All"

    def test_parses_definition(self, obo_file: Path):
        terms = parse_obo(obo_file)
        ns = next(t for t in terms if t.id == "HP:0000707")
        assert "nervous system" in ns.definition.lower()

    def test_parses_is_a_relationships(self, obo_file: Path):
        terms = parse_obo(obo_file)
        ns = next(t for t in terms if t.id == "HP:0000707")
        assert "HP:0000118" in ns.is_a

    def test_parses_multiple_is_a(self, obo_file: Path):
        terms = parse_obo(obo_file)
        brain = next(t for t in terms if t.id == "HP:0012443")
        assert len(brain.is_a) == 2
        assert "HP:0002011" in brain.is_a
        assert "HP:0000234" in brain.is_a

    def test_marks_obsolete_terms(self, obo_file: Path):
        terms = parse_obo(obo_file)
        obsolete = next(t for t in terms if t.id == "HP:9999999")
        assert obsolete.is_obsolete is True

    def test_parses_part_of(self, obo_with_part_of: Path):
        terms = parse_obo(obo_with_part_of)
        tremor = next(t for t in terms if t.id == "HP:0100023")
        assert "HP:0000707" in tremor.part_of

    def test_parses_alt_ids(self, obo_with_alt_id: Path):
        terms = parse_obo(obo_with_alt_id)
        pa = next(t for t in terms if t.id == "HP:0000118")
        assert len(pa.alt_ids) == 2
        assert "HP:0000000" in pa.alt_ids


class TestBuildDag:
    def test_correct_node_count(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms, include_obsolete=False)
        # 7 active terms (obsolete excluded)
        assert g.number_of_nodes() == 7

    def test_correct_edge_count(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms, include_obsolete=False)
        # Edges: 118→001, 707→118, 152→118, 234→152, 2011→707, 12443→2011, 12443→234
        assert g.number_of_edges() == 7

    def test_root_node_exists(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        assert HPO_ROOT_ID in g

    def test_root_has_no_parents(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        # Out-edges are child→parent, so root should have 0 out-edges
        assert g.out_degree(HPO_ROOT_ID) == 0

    def test_is_dag(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        assert nx.is_directed_acyclic_graph(g)

    def test_excludes_obsolete_by_default(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms, include_obsolete=False)
        assert "HP:9999999" not in g

    def test_includes_obsolete_when_requested(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms, include_obsolete=True)
        assert "HP:9999999" in g

    def test_node_attributes(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        assert g.nodes[HPO_ROOT_ID]["name"] == "All"
        assert g.nodes[HPO_ROOT_ID]["node_type"] == "phenotype"

    def test_edge_relation_attribute(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        assert g.edges["HP:0000118", HPO_ROOT_ID]["relation"] == "is_a"

    def test_part_of_edges(self, obo_with_part_of: Path):
        terms = parse_obo(obo_with_part_of)
        g = build_hpo_dag(terms)
        assert g.has_edge("HP:0100023", "HP:0000707")
        assert g.edges["HP:0100023", "HP:0000707"]["relation"] == "part_of"


class TestValidateDag:
    def test_valid_dag_passes(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        validate_dag(g)  # Should not raise

    def test_cycle_raises(self):
        g = nx.DiGraph()
        g.add_node(HPO_ROOT_ID, name="All", definition="", alt_ids="[]", node_type="phenotype")
        g.add_node("HP:0000002", name="A", definition="", alt_ids="[]", node_type="phenotype")
        g.add_edge("HP:0000002", HPO_ROOT_ID, relation="is_a")
        g.add_edge(HPO_ROOT_ID, "HP:0000002", relation="is_a")  # creates cycle
        with pytest.raises(ValueError, match="cycles"):
            validate_dag(g)

    def test_missing_root_raises(self):
        g = nx.DiGraph()
        g.add_node("HP:0000002", name="A", definition="", alt_ids="[]", node_type="phenotype")
        with pytest.raises(ValueError, match="Root"):
            validate_dag(g)


class TestTraversal:
    def test_get_ancestors_of_leaf(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        # HP:0012443 → HP:0002011 → HP:0000707 → HP:0000118 → HP:0000001
        # HP:0012443 → HP:0000234 → HP:0000152 → HP:0000118 → HP:0000001
        ancestors = get_ancestors(g, "HP:0012443")
        assert "HP:0012443" in ancestors  # inclusive
        assert HPO_ROOT_ID in ancestors
        assert "HP:0000707" in ancestors
        assert "HP:0000234" in ancestors

    def test_get_ancestors_of_root(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        ancestors = get_ancestors(g, HPO_ROOT_ID)
        assert ancestors == {HPO_ROOT_ID}

    def test_get_descendants_of_root(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        descendants = get_descendants(g, HPO_ROOT_ID)
        assert len(descendants) == g.number_of_nodes()

    def test_nonexistent_term_returns_empty(self, obo_file: Path):
        terms = parse_obo(obo_file)
        g = build_hpo_dag(terms)
        assert get_ancestors(g, "HP:FAKE") == set()
