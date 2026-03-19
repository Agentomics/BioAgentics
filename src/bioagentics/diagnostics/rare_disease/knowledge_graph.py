"""Construct a heterogeneous knowledge graph combining HPO, OMIM, Orphanet, and gene data.

Combines:
  - HPO DAG (phenotype is_a/part_of edges)
  - OMIM disease-phenotype associations (from phenotype.hpoa)
  - Orphanet disease-phenotype associations (with frequency weights)
  - Gene-disease associations (from genemap2)

Node types: Phenotype, Disease, Gene
Edge types: is_a, part_of, has_phenotype, associated_gene

The graph uses NetworkX DiGraph with node/edge attributes for type info
and weights. For downstream GNN use, it can be converted to PyG HeteroData.

Output:
    data/diagnostics/rare-disease-phenotype-matcher/knowledge_graph.graphml

Usage:
    uv run python -m bioagentics.diagnostics.rare_disease.knowledge_graph
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import networkx as nx

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "rare-disease-phenotype-matcher"


def build_knowledge_graph(
    hpo_dag: nx.DiGraph,
    omim_disease_hpo: dict[str, list[dict]],
    orphanet_disease_hpo: dict[str, list[dict]],
    disease_gene_map: dict[str, list[dict]],
    orphanet_crossref: dict[str, list[str]] | None = None,
) -> nx.DiGraph:
    """Build a heterogeneous knowledge graph from all data sources.

    Args:
        hpo_dag: HPO DAG from hpo_parser (child→parent edges).
        omim_disease_hpo: OMIM disease→HPO map from omim_mapper.
        orphanet_disease_hpo: Orphanet disease→HPO map from orphanet_parser.
        disease_gene_map: Disease→gene map from gene_disease_mapper.
        orphanet_crossref: Optional ORPHA→OMIM cross-references.

    Returns:
        NetworkX DiGraph with Phenotype, Disease, and Gene nodes.
    """
    g = nx.DiGraph()

    # 1. Add HPO phenotype nodes and edges
    for node, attrs in hpo_dag.nodes(data=True):
        g.add_node(
            node,
            node_type="phenotype",
            name=attrs.get("name", ""),
        )

    for u, v, attrs in hpo_dag.edges(data=True):
        relation = attrs.get("relation", "is_a")
        g.add_edge(u, v, edge_type=relation, source="hpo")

    logger.info(
        "Added HPO: %d phenotype nodes, %d edges",
        sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "phenotype"),
        g.number_of_edges(),
    )

    # 2. Add OMIM disease nodes and disease→phenotype edges
    omim_edges = 0
    for disease_id, annotations in omim_disease_hpo.items():
        if not g.has_node(disease_id):
            g.add_node(disease_id, node_type="disease", name="", source="omim")

        for ann in annotations:
            hpo_id = ann["hpo_id"]
            if not g.has_node(hpo_id):
                continue  # Skip unknown HPO terms

            freq = ann.get("frequency", 0.5)
            g.add_edge(
                disease_id,
                hpo_id,
                edge_type="has_phenotype",
                source="omim",
                frequency=freq,
            )
            omim_edges += 1

    omim_diseases = sum(
        1 for _, d in g.nodes(data=True)
        if d.get("node_type") == "disease" and d.get("source") == "omim"
    )
    logger.info("Added OMIM: %d disease nodes, %d has_phenotype edges", omim_diseases, omim_edges)

    # 3. Add Orphanet disease nodes and edges
    orphanet_edges = 0
    for orphanet_id, annotations in orphanet_disease_hpo.items():
        if not g.has_node(orphanet_id):
            g.add_node(orphanet_id, node_type="disease", name="", source="orphanet")

        for ann in annotations:
            hpo_id = ann["hpo_id"]
            if not g.has_node(hpo_id):
                continue

            freq_val = ann.get("frequency_value", 0.5)
            freq_cat = ann.get("frequency_category", "unknown")

            # If edge already exists (from OMIM), keep both as separate provenance
            # by checking source; if same edge from orphanet, update with frequency
            edge_key = (orphanet_id, hpo_id)
            if g.has_edge(*edge_key) and g.edges[edge_key].get("source") == "orphanet":
                continue  # Duplicate

            g.add_edge(
                orphanet_id,
                hpo_id,
                edge_type="has_phenotype",
                source="orphanet",
                frequency=freq_val,
                frequency_category=freq_cat,
            )
            orphanet_edges += 1

    orphanet_diseases = sum(
        1 for _, d in g.nodes(data=True)
        if d.get("node_type") == "disease" and d.get("source") == "orphanet"
    )
    logger.info(
        "Added Orphanet: %d disease nodes, %d has_phenotype edges",
        orphanet_diseases,
        orphanet_edges,
    )

    # 4. Add cross-reference edges between ORPHA and OMIM IDs
    xref_edges = 0
    if orphanet_crossref:
        for orpha_id, omim_ids in orphanet_crossref.items():
            if not g.has_node(orpha_id):
                continue
            for omim_id in omim_ids:
                if not g.has_node(omim_id):
                    continue
                g.add_edge(orpha_id, omim_id, edge_type="same_as", source="crossref")
                g.add_edge(omim_id, orpha_id, edge_type="same_as", source="crossref")
                xref_edges += 1
        logger.info("Added %d cross-reference edges (ORPHA↔OMIM)", xref_edges)

    # 5. Add gene nodes and disease→gene edges
    gene_edges = 0
    for disease_id, genes in disease_gene_map.items():
        if not g.has_node(disease_id):
            # Add disease node if not already present
            g.add_node(disease_id, node_type="disease", name="", source="genemap2")

        for gene_info in genes:
            gene_symbol = gene_info["gene_symbol"]
            gene_node_id = f"GENE:{gene_symbol}"

            if not g.has_node(gene_node_id):
                g.add_node(
                    gene_node_id,
                    node_type="gene",
                    name=gene_symbol,
                    gene_mim=gene_info.get("gene_mim", ""),
                )

            g.add_edge(
                disease_id,
                gene_node_id,
                edge_type="associated_gene",
                source="genemap2",
                mapping_key=gene_info.get("mapping_key", 0),
                inheritance=json.dumps(gene_info.get("inheritance_patterns", [])),
            )
            gene_edges += 1

    gene_count = sum(1 for _, d in g.nodes(data=True) if d.get("node_type") == "gene")
    logger.info("Added genes: %d gene nodes, %d associated_gene edges", gene_count, gene_edges)

    return g


def get_graph_stats(g: nx.DiGraph) -> dict:
    """Compute summary statistics for the knowledge graph.

    Returns:
        Dict with node/edge counts by type and connectivity info.
    """
    node_types: dict[str, int] = {}
    for _, data in g.nodes(data=True):
        nt = data.get("node_type", "unknown")
        node_types[nt] = node_types.get(nt, 0) + 1

    edge_types: dict[str, int] = {}
    for _, _, data in g.edges(data=True):
        et = data.get("edge_type", "unknown")
        edge_types[et] = edge_types.get(et, 0) + 1

    # Connectivity (undirected for component analysis)
    ug = g.to_undirected()
    components = list(nx.connected_components(ug))
    largest_component = max(len(c) for c in components) if components else 0

    return {
        "total_nodes": g.number_of_nodes(),
        "total_edges": g.number_of_edges(),
        "node_types": node_types,
        "edge_types": edge_types,
        "num_components": len(components),
        "largest_component_size": largest_component,
        "largest_component_fraction": largest_component / max(g.number_of_nodes(), 1),
    }


def validate_knowledge_graph(g: nx.DiGraph) -> list[str]:
    """Validate the knowledge graph structure.

    Returns:
        List of warning messages (empty if valid).
    """
    warnings: list[str] = []

    stats = get_graph_stats(g)

    # Check all node types present
    expected_types = {"phenotype", "disease", "gene"}
    present_types = set(stats["node_types"].keys())
    missing = expected_types - present_types
    if missing:
        warnings.append(f"Missing node types: {missing}")

    # Check largest component covers >90% of nodes
    if stats["largest_component_fraction"] < 0.9:
        warnings.append(
            f"Largest component covers only {stats['largest_component_fraction']:.1%} "
            f"of nodes (expected >90%)"
        )

    # Check edge types
    expected_edges = {"is_a", "has_phenotype", "associated_gene"}
    present_edges = set(stats["edge_types"].keys())
    missing_edges = expected_edges - present_edges
    if missing_edges:
        warnings.append(f"Missing edge types: {missing_edges}")

    return warnings


def load_knowledge_graph(path: Path | None = None) -> nx.DiGraph:
    """Load a previously saved knowledge graph from GraphML."""
    if path is None:
        path = DATA_DIR / "knowledge_graph.graphml"
    if not path.exists():
        raise FileNotFoundError(
            f"Knowledge graph not found at {path}. Run the builder first."
        )
    return nx.read_graphml(path)


def save_knowledge_graph(g: nx.DiGraph, output_dir: Path | None = None) -> Path:
    """Save the knowledge graph to GraphML format.

    Returns:
        Path to the saved file.
    """
    if output_dir is None:
        output_dir = DATA_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "knowledge_graph.graphml"
    nx.write_graphml(g, path)
    logger.info("Saved knowledge graph to %s", path)
    return path


def build_from_files(
    hpo_graphml: Path | None = None,
    omim_map_json: Path | None = None,
    orphanet_map_json: Path | None = None,
    disease_gene_json: Path | None = None,
    orphanet_crossref_json: Path | None = None,
    output_dir: Path | None = None,
) -> nx.DiGraph:
    """Build knowledge graph from saved intermediate files.

    All paths default to standard DATA_DIR locations.
    """
    if output_dir is None:
        output_dir = DATA_DIR

    # Load HPO DAG
    hpo_path = hpo_graphml or DATA_DIR / "hpo_graph.graphml"
    logger.info("Loading HPO DAG from %s", hpo_path)
    hpo_dag = nx.read_graphml(hpo_path)

    # Load OMIM disease-HPO map
    omim_path = omim_map_json or DATA_DIR / "disease_hpo_map.json"
    logger.info("Loading OMIM map from %s", omim_path)
    with open(omim_path) as f:
        omim_disease_hpo = json.load(f)

    # Load Orphanet disease-HPO map
    orphanet_path = orphanet_map_json or DATA_DIR / "orphanet_disease_hpo.json"
    logger.info("Loading Orphanet map from %s", orphanet_path)
    with open(orphanet_path) as f:
        orphanet_disease_hpo = json.load(f)

    # Load gene-disease map
    gene_path = disease_gene_json or DATA_DIR / "disease_gene_map.json"
    logger.info("Loading disease-gene map from %s", gene_path)
    with open(gene_path) as f:
        disease_gene_map = json.load(f)

    # Optional cross-references
    orphanet_crossref = None
    xref_path = orphanet_crossref_json or DATA_DIR / "orphanet_omim_crossref.json"
    if xref_path.exists():
        logger.info("Loading cross-references from %s", xref_path)
        with open(xref_path) as f:
            orphanet_crossref = json.load(f)

    # Build
    g = build_knowledge_graph(
        hpo_dag, omim_disease_hpo, orphanet_disease_hpo,
        disease_gene_map, orphanet_crossref,
    )

    # Validate
    warnings = validate_knowledge_graph(g)
    for w in warnings:
        logger.warning("Validation: %s", w)

    # Stats
    stats = get_graph_stats(g)
    logger.info("Knowledge graph stats:")
    logger.info("  Total nodes: %d", stats["total_nodes"])
    logger.info("  Total edges: %d", stats["total_edges"])
    for nt, count in stats["node_types"].items():
        logger.info("  %s nodes: %d", nt, count)
    for et, count in stats["edge_types"].items():
        logger.info("  %s edges: %d", et, count)
    logger.info(
        "  Connectivity: %d components, largest covers %.1f%% of nodes",
        stats["num_components"],
        stats["largest_component_fraction"] * 100,
    )

    # Save
    save_knowledge_graph(g, output_dir)

    return g


def main():
    parser = argparse.ArgumentParser(
        description="Build heterogeneous knowledge graph from HPO, OMIM, Orphanet, and gene data"
    )
    parser.add_argument("--hpo-graphml", type=Path, default=None)
    parser.add_argument("--omim-map", type=Path, default=None)
    parser.add_argument("--orphanet-map", type=Path, default=None)
    parser.add_argument("--disease-gene-map", type=Path, default=None)
    parser.add_argument("--orphanet-crossref", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DATA_DIR)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    g = build_from_files(
        hpo_graphml=args.hpo_graphml,
        omim_map_json=args.omim_map,
        orphanet_map_json=args.orphanet_map,
        disease_gene_json=args.disease_gene_map,
        orphanet_crossref_json=args.orphanet_crossref,
        output_dir=args.output_dir,
    )

    stats = get_graph_stats(g)
    print(f"\nKnowledge Graph Summary:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Total edges: {stats['total_edges']}")
    for nt, count in sorted(stats["node_types"].items()):
        print(f"  {nt} nodes: {count}")
    for et, count in sorted(stats["edge_types"].items()):
        print(f"  {et} edges: {count}")
    print(f"  Components: {stats['num_components']}")
    print(f"  Largest component: {stats['largest_component_fraction']:.1%} of nodes")


if __name__ == "__main__":
    main()
