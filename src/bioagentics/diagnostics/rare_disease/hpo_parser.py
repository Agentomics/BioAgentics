"""Parse the Human Phenotype Ontology (HPO) OBO file into a NetworkX DAG.

Reads hp.obo and produces a directed acyclic graph where edges point from
child to parent (is_a) or part_of relationships. Each node stores:
  - id: HPO term ID (e.g. "HP:0000001")
  - name: human-readable term name
  - definition: term definition text (may be empty)
  - alt_ids: list of alternative IDs
  - is_obsolete: whether the term is obsolete

Output:
    data/diagnostics/rare-disease-phenotype-matcher/hpo_graph.graphml

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.hpo_parser [--obo PATH]
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "rare-disease-phenotype-matcher"
HPO_ROOT_ID = "HP:0000001"


@dataclass
class HPOTerm:
    id: str = ""
    name: str = ""
    definition: str = ""
    alt_ids: list[str] = field(default_factory=list)
    is_obsolete: bool = False
    is_a: list[str] = field(default_factory=list)
    part_of: list[str] = field(default_factory=list)


def parse_obo(path: Path) -> list[HPOTerm]:
    """Parse an OBO file and return a list of HPO terms.

    Only [Term] stanzas are parsed. Typedefs and headers are skipped.
    """
    terms: list[HPOTerm] = []
    current: HPOTerm | None = None

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")

            if line == "[Term]":
                if current is not None and current.id:
                    terms.append(current)
                current = HPOTerm()
                continue

            if line.startswith("[") and line.endswith("]"):
                # Other stanza type (e.g. [Typedef]) — flush current term
                if current is not None and current.id:
                    terms.append(current)
                current = None
                continue

            if current is None:
                continue

            if line.startswith("id: "):
                current.id = line[4:].strip()
            elif line.startswith("name: "):
                current.name = line[6:].strip()
            elif line.startswith("def: "):
                # Definition is in double quotes: def: "..." [source]
                m = re.match(r'^def: "(.*?)"', line)
                current.definition = m.group(1) if m else line[5:].strip()
            elif line.startswith("alt_id: "):
                current.alt_ids.append(line[8:].strip())
            elif line.startswith("is_obsolete: true"):
                current.is_obsolete = True
            elif line.startswith("is_a: "):
                # is_a: HP:0000001 ! All
                parent_id = line[6:].split("!")[0].strip()
                current.is_a.append(parent_id)
            elif line.startswith("relationship: part_of "):
                # relationship: part_of HP:0000001 ! All
                target = line[len("relationship: part_of ") :].split("!")[0].strip()
                current.part_of.append(target)

    # Flush last term
    if current is not None and current.id:
        terms.append(current)

    return terms


def build_hpo_dag(terms: list[HPOTerm], include_obsolete: bool = False) -> nx.DiGraph:
    """Build a directed acyclic graph from parsed HPO terms.

    Edges point from child → parent (is_a) or child → parent (part_of).
    This convention means ancestors are reachable via successors/descendants.

    Args:
        terms: Parsed HPO terms from parse_obo().
        include_obsolete: If False (default), skip obsolete terms.

    Returns:
        NetworkX DiGraph with HPO terms as nodes and is_a/part_of edges.
    """
    g = nx.DiGraph()

    for term in terms:
        if not include_obsolete and term.is_obsolete:
            continue
        g.add_node(
            term.id,
            name=term.name,
            definition=term.definition,
            alt_ids=json.dumps(term.alt_ids),
            node_type="phenotype",
        )

    # Add edges (child → parent)
    for term in terms:
        if not include_obsolete and term.is_obsolete:
            continue
        for parent_id in term.is_a:
            if g.has_node(parent_id):
                g.add_edge(term.id, parent_id, relation="is_a")
        for parent_id in term.part_of:
            if g.has_node(parent_id):
                g.add_edge(term.id, parent_id, relation="part_of")

    return g


def validate_dag(g: nx.DiGraph) -> None:
    """Validate that the graph is a proper DAG with the expected root.

    Raises:
        ValueError: If the graph has cycles, no root, or unexpected structure.
    """
    if not nx.is_directed_acyclic_graph(g):
        raise ValueError("HPO graph contains cycles — not a valid DAG")

    if HPO_ROOT_ID not in g:
        raise ValueError(f"Root node {HPO_ROOT_ID} not found in graph")

    # Root should have no outgoing edges (no parents)
    if g.out_degree(HPO_ROOT_ID) != 0:
        raise ValueError(f"Root {HPO_ROOT_ID} has parents — unexpected structure")


def get_ancestors(g: nx.DiGraph, term_id: str) -> set[str]:
    """Get all ancestor terms (inclusive of term itself) via is_a/part_of edges."""
    if term_id not in g:
        return set()
    return {term_id} | nx.descendants(g, term_id)


def get_descendants(g: nx.DiGraph, term_id: str) -> set[str]:
    """Get all descendant terms (inclusive of term itself)."""
    if term_id not in g:
        return set()
    return {term_id} | nx.ancestors(g, term_id)


def load_hpo_dag(path: Path | None = None) -> nx.DiGraph:
    """Load a previously saved HPO DAG from GraphML.

    Args:
        path: Path to GraphML file. Defaults to standard data directory location.
    """
    if path is None:
        path = DATA_DIR / "hpo_graph.graphml"
    if not path.exists():
        raise FileNotFoundError(f"HPO graph not found at {path}. Run the parser first.")
    return nx.read_graphml(path)


def parse_and_build(obo_path: Path, output_dir: Path | None = None) -> nx.DiGraph:
    """End-to-end: parse OBO file, build DAG, validate, and save.

    Args:
        obo_path: Path to hp.obo file.
        output_dir: Directory to save outputs. Defaults to DATA_DIR.

    Returns:
        The validated HPO DAG.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    logger.info("Parsing HPO OBO file: %s", obo_path)
    terms = parse_obo(obo_path)
    logger.info("Parsed %d terms (including obsolete)", len(terms))

    active = [t for t in terms if not t.is_obsolete]
    obsolete = [t for t in terms if t.is_obsolete]
    logger.info("Active terms: %d, Obsolete: %d", len(active), len(obsolete))

    g = build_hpo_dag(terms, include_obsolete=False)
    logger.info("Graph: %d nodes, %d edges", g.number_of_nodes(), g.number_of_edges())

    validate_dag(g)
    logger.info("DAG validation passed")

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    graphml_path = output_dir / "hpo_graph.graphml"
    nx.write_graphml(g, graphml_path)
    logger.info("Saved HPO DAG to %s", graphml_path)

    return g


def main():
    parser = argparse.ArgumentParser(description="Parse HPO OBO file into a NetworkX DAG")
    parser.add_argument(
        "--obo",
        type=Path,
        default=DATA_DIR / "hp.obo",
        help="Path to hp.obo file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_DIR,
        help="Output directory for graph files",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    g = parse_and_build(args.obo, args.output_dir)

    # Summary stats
    roots = [n for n in g.nodes() if g.out_degree(n) == 0]
    leaves = [n for n in g.nodes() if g.in_degree(n) == 0]
    print(f"\nHPO DAG Summary:")
    print(f"  Nodes: {g.number_of_nodes()}")
    print(f"  Edges: {g.number_of_edges()}")
    print(f"  Root nodes: {len(roots)} ({', '.join(roots[:5])})")
    print(f"  Leaf nodes: {len(leaves)}")
    print(f"  Max depth: {nx.dag_longest_path_length(g)}")


if __name__ == "__main__":
    main()
