"""Network convergence and hub analysis for PANDAS autoantibody target network.

Analyzes the extended two-layer network (autoantibody PPI + cytokine amplification)
for convergent hub proteins where multiple disrupted pathways intersect.

Computes:
  - Degree centrality, betweenness centrality, PageRank
  - Pathway convergence (nodes shared by multiple enriched pathways)
  - Cross-layer convergence (nodes where autoantibody and cytokine layers meet)
  - Druggability annotations via DGIdb API
  - Ranked hub list with all annotations

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.convergence_hub_analysis

Output:
    data/pandas_pans/autoantibody_network/hub_centrality_metrics.tsv
    data/pandas_pans/autoantibody_network/convergence_analysis.tsv
    data/pandas_pans/autoantibody_network/druggability_annotations.tsv
    data/pandas_pans/autoantibody_network/convergence_stats.json
"""

import json
import logging
import time
from pathlib import Path

import networkx as nx
import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.cytokine_layer import (
    get_cytokine_gene_symbols,
)
from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
    get_seed_dict,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/pandas_pans/autoantibody_network")
DGIDB_API = "https://dgidb.org/api/graphql"
RATE_LIMIT_DELAY = 0.3


def load_extended_network() -> nx.Graph:
    """Load extended_network.tsv into a networkx graph.

    Edge attributes include combined_score and layer annotation.
    """
    path = DATA_DIR / "extended_network.tsv"
    if not path.exists():
        logger.error("Extended network not found: %s", path)
        return nx.Graph()

    # Stream in chunks to stay memory-safe
    G = nx.Graph()
    chunks = pd.read_csv(
        path,
        sep="\t",
        usecols=["source", "target", "combined_score", "layer"],
        chunksize=5000,
    )
    for chunk in chunks:
        for _, row in chunk.iterrows():
            src = str(row["source"])
            tgt = str(row["target"])
            score = float(row["combined_score"]) if pd.notna(row["combined_score"]) else 0.0
            layer = str(row.get("layer", "unknown"))
            if G.has_edge(src, tgt):
                # Keep higher score, merge layer info
                existing = G[src][tgt]
                if score > existing.get("combined_score", 0):
                    existing["combined_score"] = score
                existing_layers = set(existing.get("layers", "").split(","))
                existing_layers.add(layer)
                existing["layers"] = ",".join(sorted(existing_layers - {""}))
            else:
                G.add_edge(src, tgt, combined_score=score, layers=layer)

    logger.info("Loaded network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    return G


def compute_centrality_metrics(G: nx.Graph) -> pd.DataFrame:
    """Compute degree, betweenness, and PageRank centrality for all nodes."""
    logger.info("Computing centrality metrics for %d nodes...", G.number_of_nodes())

    degree_cent = nx.degree_centrality(G)
    logger.info("  Degree centrality done")

    betweenness_cent = nx.betweenness_centrality(G, k=min(500, G.number_of_nodes()))
    logger.info("  Betweenness centrality done (k=%d)", min(500, G.number_of_nodes()))

    pagerank = nx.pagerank(G, weight="combined_score", max_iter=200)
    logger.info("  PageRank done")

    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    seed_dict = get_seed_dict()

    rows = []
    for node in G.nodes():
        # Determine node type
        if node in seed_set:
            node_type = "seed"
            mechanism = seed_dict.get(node, {}).get("mechanism_category", "")
        elif node in cytokine_set:
            node_type = "cytokine"
            mechanism = ""
        else:
            node_type = "interactor"
            mechanism = ""

        # Determine which layers this node participates in
        node_layers = set()
        for _, _, edata in G.edges(node, data=True):
            for l in edata.get("layers", "").split(","):
                if l:
                    node_layers.add(l)

        rows.append({
            "gene_symbol": node,
            "node_type": node_type,
            "mechanism_category": mechanism,
            "degree": G.degree(node),
            "degree_centrality": round(degree_cent[node], 6),
            "betweenness_centrality": round(betweenness_cent[node], 6),
            "pagerank": round(pagerank[node], 8),
            "layers": ",".join(sorted(node_layers)),
            "is_cross_layer": "cross_layer" in node_layers or (
                bool(node_layers & {"autoantibody_ppi"}) and bool(node_layers & {"cytokine_amplification"})
            ),
        })

    df = pd.DataFrame(rows)
    # Composite hub score: weighted combination of centrality metrics
    # Normalize each to 0-1 range before combining
    for col in ["degree_centrality", "betweenness_centrality", "pagerank"]:
        cmax = df[col].max()
        if cmax > 0:
            df[f"{col}_norm"] = df[col] / cmax
        else:
            df[f"{col}_norm"] = 0.0

    df["hub_score"] = (
        0.3 * df["degree_centrality_norm"]
        + 0.4 * df["betweenness_centrality_norm"]
        + 0.3 * df["pagerank_norm"]
    )
    df = df.drop(columns=["degree_centrality_norm", "betweenness_centrality_norm", "pagerank_norm"])
    df = df.sort_values("hub_score", ascending=False)

    return df


def load_pathway_memberships() -> dict[str, list[str]]:
    """Load KEGG + Reactome pathway memberships per gene.

    Returns gene_symbol -> list of pathway names.
    """
    pathways: dict[str, list[str]] = {}

    # KEGG
    kegg_path = DATA_DIR / "kegg_node_pathway_mapping.tsv"
    if kegg_path.exists():
        chunks = pd.read_csv(kegg_path, sep="\t", usecols=["gene_symbol", "pathway_name"], chunksize=5000)
        for chunk in chunks:
            for _, row in chunk.iterrows():
                sym = str(row["gene_symbol"])
                pw = str(row["pathway_name"])
                pathways.setdefault(sym, []).append(f"KEGG:{pw}")

    # Reactome
    reactome_path = DATA_DIR / "reactome_node_pathway_mapping.tsv"
    if reactome_path.exists():
        chunks = pd.read_csv(reactome_path, sep="\t", usecols=["gene_symbol", "pathway_name"], chunksize=5000)
        for chunk in chunks:
            for _, row in chunk.iterrows():
                sym = str(row["gene_symbol"])
                pw = str(row["pathway_name"])
                pathways.setdefault(sym, []).append(f"Reactome:{pw}")

    # Deduplicate
    for sym in pathways:
        pathways[sym] = sorted(set(pathways[sym]))

    logger.info("Loaded pathway memberships for %d genes", len(pathways))
    return pathways


def load_enriched_pathways() -> tuple[set[str], set[str]]:
    """Load significantly enriched pathway names from KEGG and Reactome.

    Returns (kegg_enriched_names, reactome_enriched_names) for pathways with FDR < 0.05.
    """
    kegg_enriched: set[str] = set()
    reactome_enriched: set[str] = set()

    kegg_path = DATA_DIR / "kegg_pathway_enrichment.tsv"
    if kegg_path.exists():
        df = pd.read_csv(kegg_path, sep="\t")
        sig = df[df["fdr"] < 0.05]
        kegg_enriched = set(sig["pathway_name"])
        logger.info("KEGG: %d enriched pathways (FDR<0.05)", len(kegg_enriched))

    reactome_path = DATA_DIR / "reactome_pathway_enrichment.tsv"
    if reactome_path.exists():
        df = pd.read_csv(reactome_path, sep="\t")
        sig = df[df["fdr"] < 0.05]
        reactome_enriched = set(sig["pathway_name"])
        logger.info("Reactome: %d enriched pathways (FDR<0.05)", len(reactome_enriched))

    return kegg_enriched, reactome_enriched


def compute_pathway_convergence(
    centrality_df: pd.DataFrame,
    pathways: dict[str, list[str]],
    kegg_enriched: set[str],
    reactome_enriched: set[str],
) -> pd.DataFrame:
    """Compute pathway convergence: how many enriched pathways each node belongs to.

    Nodes in many enriched pathways are convergence hubs — disruption of these
    nodes affects multiple signaling cascades simultaneously.
    """
    rows = []
    for _, row in centrality_df.iterrows():
        sym = row["gene_symbol"]
        gene_pathways = pathways.get(sym, [])

        kegg_pws = [p.replace("KEGG:", "") for p in gene_pathways if p.startswith("KEGG:")]
        reactome_pws = [p.replace("Reactome:", "") for p in gene_pathways if p.startswith("Reactome:")]

        kegg_enriched_count = sum(1 for p in kegg_pws if p in kegg_enriched)
        reactome_enriched_count = sum(1 for p in reactome_pws if p in reactome_enriched)
        total_enriched = kegg_enriched_count + reactome_enriched_count

        # Collect enriched pathway names for this gene
        enriched_names = []
        enriched_names.extend(p for p in kegg_pws if p in kegg_enriched)
        enriched_names.extend(p for p in reactome_pws if p in reactome_enriched)

        rows.append({
            "gene_symbol": sym,
            "total_pathways": len(gene_pathways),
            "kegg_pathways": len(kegg_pws),
            "reactome_pathways": len(reactome_pws),
            "kegg_enriched_pathways": kegg_enriched_count,
            "reactome_enriched_pathways": reactome_enriched_count,
            "total_enriched_pathways": total_enriched,
            "enriched_pathway_names": ";".join(sorted(set(enriched_names))[:20]),
        })

    df = pd.DataFrame(rows)
    return df


def load_symptom_domains() -> dict[str, list[str]]:
    """Load symptom domain mappings per gene.

    Returns gene_symbol -> list of symptom domains.
    """
    path = DATA_DIR / "symptom_domain_mapping.tsv"
    if not path.exists():
        return {}

    domains: dict[str, list[str]] = {}
    chunks = pd.read_csv(path, sep="\t", usecols=["gene_symbol", "symptom_domain"], chunksize=5000)
    for chunk in chunks:
        for _, row in chunk.iterrows():
            sym = str(row["gene_symbol"])
            dom = str(row["symptom_domain"])
            domains.setdefault(sym, []).append(dom)

    for sym in domains:
        domains[sym] = sorted(set(domains[sym]))

    logger.info("Loaded symptom domains for %d genes", len(domains))
    return domains


def load_brain_expression() -> dict[str, list[str]]:
    """Load brain region expression data per gene.

    Returns gene_symbol -> list of brain regions with expression.
    The Allen expression file is in long format: gene_symbol, region, mean_expression, ...
    """
    path = DATA_DIR / "allen_expression_by_region.tsv"
    if not path.exists():
        return {}

    regions: dict[str, list[str]] = {}
    chunks = pd.read_csv(
        path, sep="\t", usecols=["gene_symbol", "region", "mean_expression"], chunksize=5000
    )
    for chunk in chunks:
        for _, row in chunk.iterrows():
            sym = str(row["gene_symbol"])
            region = str(row["region"])
            expr = float(row["mean_expression"]) if pd.notna(row["mean_expression"]) else 0.0
            if expr > 0:
                regions.setdefault(sym, []).append(region)

    # Deduplicate
    for sym in regions:
        regions[sym] = sorted(set(regions[sym]))

    logger.info("Loaded brain expression for %d genes", len(regions))
    return regions


def query_dgidb_druggability(gene_symbols: list[str]) -> dict[str, list[dict]]:
    """Query DGIdb for drug-gene interactions.

    Queries in batches to avoid overloading the API.
    Returns gene_symbol -> list of {drug_name, interaction_type, sources} dicts.
    """
    logger.info("Querying DGIdb for %d genes...", len(gene_symbols))

    query = """
    query($genes: [String!]!) {
      genes(names: $genes) {
        nodes {
          name
          interactions {
            interactionScore
            interactionTypes {
              type
              directionality
            }
            drug {
              name
              approved
              conceptId
            }
            interactionClaims {
              source {
                fullName
              }
            }
          }
        }
      }
    }
    """

    results: dict[str, list[dict]] = {}
    batch_size = 50

    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i : i + batch_size]
        try:
            resp = requests.post(
                DGIDB_API,
                json={"query": query, "variables": {"genes": batch}},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                nodes = data.get("data", {}).get("genes", {}).get("nodes", [])
                for node in nodes:
                    gene_name = node.get("name", "")
                    interactions = node.get("interactions", [])
                    if interactions:
                        drug_list = []
                        for ix in interactions:
                            drug = ix.get("drug", {})
                            ix_types = ix.get("interactionTypes", [])
                            type_str = ", ".join(t.get("type", "") for t in ix_types if t.get("type"))
                            sources = ix.get("interactionClaims", [])
                            source_names = [s.get("source", {}).get("fullName", "") for s in sources]
                            drug_list.append({
                                "drug_name": drug.get("name", ""),
                                "approved": drug.get("approved", False),
                                "interaction_type": type_str,
                                "score": ix.get("interactionScore"),
                                "sources": "; ".join(s for s in source_names if s)[:200],
                            })
                        results[gene_name] = drug_list
            else:
                logger.warning("DGIdb returned status %d for batch %d", resp.status_code, i)
        except requests.RequestException as e:
            logger.error("DGIdb query failed for batch %d: %s", i, e)
        time.sleep(RATE_LIMIT_DELAY)

        if (i + batch_size) % 200 == 0:
            logger.info("  DGIdb: queried %d / %d genes", min(i + batch_size, len(gene_symbols)), len(gene_symbols))

    logger.info("DGIdb: %d genes have drug interactions", len(results))
    return results


def build_convergence_table(
    centrality_df: pd.DataFrame,
    convergence_df: pd.DataFrame,
    druggability: dict[str, list[dict]],
    symptom_domains: dict[str, list[str]],
    brain_regions: dict[str, list[str]],
) -> pd.DataFrame:
    """Build the final ranked convergence/hub table with all annotations."""
    merged = centrality_df.merge(convergence_df, on="gene_symbol", how="left")

    # Add druggability info
    merged["is_druggable"] = merged["gene_symbol"].apply(lambda g: g in druggability)
    merged["n_drug_interactions"] = merged["gene_symbol"].apply(
        lambda g: len(druggability.get(g, []))
    )
    merged["approved_drugs"] = merged["gene_symbol"].apply(
        lambda g: "; ".join(
            d["drug_name"] for d in druggability.get(g, [])
            if d.get("approved")
        )[:300]
    )
    merged["drug_interaction_types"] = merged["gene_symbol"].apply(
        lambda g: "; ".join(
            set(d["interaction_type"] for d in druggability.get(g, []) if d.get("interaction_type"))
        )[:200]
    )

    # Add symptom domains
    merged["symptom_domains"] = merged["gene_symbol"].apply(
        lambda g: ",".join(symptom_domains.get(g, []))
    )
    merged["n_symptom_domains"] = merged["gene_symbol"].apply(
        lambda g: len(symptom_domains.get(g, []))
    )

    # Add brain expression
    merged["brain_regions"] = merged["gene_symbol"].apply(
        lambda g: ",".join(brain_regions.get(g, []))[:200]
    )

    # Convergence score: combination of hub_score + pathway convergence + cross-layer bonus
    max_enriched = merged["total_enriched_pathways"].max()
    if max_enriched > 0:
        merged["pathway_convergence_norm"] = merged["total_enriched_pathways"] / max_enriched
    else:
        merged["pathway_convergence_norm"] = 0.0

    merged["convergence_score"] = (
        0.4 * merged["hub_score"]
        + 0.3 * merged["pathway_convergence_norm"]
        + 0.15 * merged["is_cross_layer"].astype(float)
        + 0.15 * merged["is_druggable"].astype(float)
    )

    merged = merged.drop(columns=["pathway_convergence_norm"])
    merged = merged.sort_values("convergence_score", ascending=False)

    return merged


def identify_novel_targets(convergence_table: pd.DataFrame, top_n: int = 50) -> pd.DataFrame:
    """Identify hub nodes NOT previously proposed as PANDAS therapeutic targets.

    Filters out known seed proteins and cytokine targets; returns interactor
    nodes with highest convergence scores.
    """
    novel = convergence_table[convergence_table["node_type"] == "interactor"].head(top_n)
    return novel


def run() -> None:
    """Execute the convergence and hub analysis pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Load network
    logger.info("Step 1: Loading extended network")
    G = load_extended_network()
    if G.number_of_nodes() == 0:
        logger.error("Empty network — cannot proceed")
        return

    # Step 2: Compute centrality metrics
    logger.info("Step 2: Computing centrality metrics")
    centrality_df = compute_centrality_metrics(G)
    centrality_path = DATA_DIR / "hub_centrality_metrics.tsv"
    centrality_df.to_csv(centrality_path, sep="\t", index=False)
    logger.info("Saved centrality metrics to %s", centrality_path)

    # Step 3: Load pathway data and compute convergence
    logger.info("Step 3: Computing pathway convergence")
    pathways = load_pathway_memberships()
    kegg_enriched, reactome_enriched = load_enriched_pathways()
    convergence_df = compute_pathway_convergence(
        centrality_df, pathways, kegg_enriched, reactome_enriched
    )

    # Step 4: Load symptom domains and brain expression
    logger.info("Step 4: Loading symptom domains and brain expression")
    symptom_domains = load_symptom_domains()
    brain_regions = load_brain_expression()

    # Step 5: Query DGIdb for druggability
    logger.info("Step 5: Querying DGIdb for druggability annotations")
    # Query top hub genes (all seeds + cytokines + top interactors by degree)
    top_genes = centrality_df.nlargest(300, "degree")["gene_symbol"].tolist()
    druggability = query_dgidb_druggability(top_genes)

    # Save druggability annotations
    drug_rows = []
    for gene, drugs in druggability.items():
        for d in drugs:
            drug_rows.append({"gene_symbol": gene, **d})
    if drug_rows:
        drug_df = pd.DataFrame(drug_rows)
        drug_path = DATA_DIR / "druggability_annotations.tsv"
        drug_df.to_csv(drug_path, sep="\t", index=False)
        logger.info("Saved druggability annotations to %s (%d entries)", drug_path, len(drug_df))

    # Step 6: Build final convergence table
    logger.info("Step 6: Building final convergence table")
    convergence_table = build_convergence_table(
        centrality_df, convergence_df, druggability, symptom_domains, brain_regions
    )
    conv_path = DATA_DIR / "convergence_analysis.tsv"
    convergence_table.to_csv(conv_path, sep="\t", index=False)
    logger.info("Saved convergence analysis to %s", conv_path)

    # Step 7: Identify novel targets
    novel_targets = identify_novel_targets(convergence_table, top_n=50)
    novel_path = DATA_DIR / "novel_therapeutic_targets.tsv"
    novel_targets.to_csv(novel_path, sep="\t", index=False)
    logger.info("Identified %d novel therapeutic target candidates", len(novel_targets))

    # Step 8: Compute and save stats
    seed_set = set(get_gene_symbols())
    cytokine_set = set(get_cytokine_gene_symbols())
    cross_layer_nodes = convergence_table[convergence_table["is_cross_layer"]]
    druggable_hubs = convergence_table[
        (convergence_table["is_druggable"]) & (convergence_table["hub_score"] > convergence_table["hub_score"].median())
    ]

    # Top 10 hubs overall
    top10 = convergence_table.head(10)
    top10_info = []
    for _, row in top10.iterrows():
        top10_info.append({
            "gene": row["gene_symbol"],
            "type": row["node_type"],
            "hub_score": round(float(row["hub_score"]), 4),
            "convergence_score": round(float(row["convergence_score"]), 4),
            "degree": int(row["degree"]),
            "enriched_pathways": int(row["total_enriched_pathways"]),
            "druggable": bool(row["is_druggable"]),
            "cross_layer": bool(row["is_cross_layer"]),
        })

    # Top 10 novel targets
    top10_novel = novel_targets.head(10)
    top10_novel_info = []
    for _, row in top10_novel.iterrows():
        top10_novel_info.append({
            "gene": row["gene_symbol"],
            "hub_score": round(float(row["hub_score"]), 4),
            "convergence_score": round(float(row["convergence_score"]), 4),
            "degree": int(row["degree"]),
            "enriched_pathways": int(row["total_enriched_pathways"]),
            "druggable": bool(row["is_druggable"]),
            "cross_layer": bool(row["is_cross_layer"]),
            "symptom_domains": row["symptom_domains"],
            "approved_drugs": row["approved_drugs"][:200] if row["approved_drugs"] else "",
        })

    stats = {
        "network_nodes": G.number_of_nodes(),
        "network_edges": G.number_of_edges(),
        "seed_nodes": len(seed_set & set(G.nodes())),
        "cytokine_nodes": len(cytokine_set & set(G.nodes())),
        "cross_layer_nodes": len(cross_layer_nodes),
        "genes_with_pathway_data": sum(1 for _, r in convergence_table.iterrows() if r["total_pathways"] > 0),
        "genes_with_enriched_pathways": sum(1 for _, r in convergence_table.iterrows() if r["total_enriched_pathways"] > 0),
        "druggable_genes": len(druggability),
        "druggable_hub_genes": len(druggable_hubs),
        "novel_targets_identified": len(novel_targets),
        "top10_hubs": top10_info,
        "top10_novel_targets": top10_novel_info,
    }

    stats_path = DATA_DIR / "convergence_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved convergence stats to %s", stats_path)

    # Summary
    logger.info("=== Convergence & Hub Analysis Summary ===")
    logger.info("Network: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())
    logger.info("Cross-layer nodes: %d", len(cross_layer_nodes))
    logger.info("Druggable genes queried: %d, with interactions: %d", len(top_genes), len(druggability))
    logger.info("Novel therapeutic targets: %d", len(novel_targets))
    logger.info("--- Top 10 Overall Hubs ---")
    for h in top10_info:
        marker = " [NOVEL]" if h["type"] == "interactor" else f" [{h['type']}]"
        drug = " [DRUGGABLE]" if h["druggable"] else ""
        cross = " [CROSS-LAYER]" if h["cross_layer"] else ""
        logger.info(
            "  %s: hub=%.4f conv=%.4f deg=%d pw=%d%s%s%s",
            h["gene"], h["hub_score"], h["convergence_score"],
            h["degree"], h["enriched_pathways"], marker, drug, cross,
        )
    logger.info("--- Top 10 Novel Targets ---")
    for h in top10_novel_info:
        drug = " [DRUGGABLE]" if h["druggable"] else ""
        logger.info(
            "  %s: conv=%.4f deg=%d pw=%d domains=%s%s",
            h["gene"], h["convergence_score"], h["degree"],
            h["enriched_pathways"], h["symptom_domains"][:50], drug,
        )


if __name__ == "__main__":
    run()
