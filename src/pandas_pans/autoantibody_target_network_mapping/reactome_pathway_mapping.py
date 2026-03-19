"""Reactome pathway mapping for PANDAS autoantibody target network nodes.

Maps network nodes to Reactome pathways via the Reactome Analysis Service API.
Focus: GPCR signaling, CaM pathway, folate metabolism (for FOLR1 hub),
one-carbon metabolism -> neurotransmitter synthesis.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.reactome_pathway_mapping

Output:
    data/pandas_pans/autoantibody_network/reactome_node_pathway_mapping.tsv
    data/pandas_pans/autoantibody_network/reactome_pathway_enrichment.tsv
    data/pandas_pans/autoantibody_network/reactome_mapping_stats.json
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

REACTOME_ANALYSIS_URL = "https://reactome.org/AnalysisService"
REACTOME_CONTENT_URL = "https://reactome.org/ContentService"
DATA_DIR = Path("data/pandas_pans/autoantibody_network")
RATE_LIMIT_DELAY = 0.3

# PANDAS-relevant Reactome pathway keywords for flagging
PANDAS_FOCUS_KEYWORDS = [
    "dopamine",
    "GPCR",
    "G protein",
    "calcium",
    "calmodulin",
    "CaMK",
    "folate",
    "one carbon",
    "neurotransmitter",
    "synap",
    "glutamat",
    "GABA",
    "serotonin",
    "axon",
    "long-term potentiation",
    "JAK",
    "STAT",
    "cytokine",
    "interleukin",
    "interferon",
    "NF-kB",
    "NFkB",
    "immune",
    "basal ganglia",
]


def get_network_nodes() -> list[str]:
    """Extract all unique gene symbols from the extended network."""
    network_path = DATA_DIR / "extended_network.tsv"
    if not network_path.exists():
        logger.error("Extended network file not found: %s", network_path)
        return []

    chunks = pd.read_csv(
        network_path, sep="\t", usecols=["source", "target"], chunksize=5000
    )
    nodes: set[str] = set()
    for chunk in chunks:
        nodes.update(chunk["source"].dropna())
        nodes.update(chunk["target"].dropna())
    logger.info("Found %d unique nodes in extended network", len(nodes))
    return sorted(nodes)


def run_reactome_analysis(gene_symbols: list[str]) -> dict:
    """Submit gene list to Reactome Analysis Service.

    Uses the overrepresentation analysis endpoint with projection
    (maps identifiers to human orthologs if needed).
    Returns the full analysis result JSON.
    """
    # Filter out entries with spaces/special chars (non-standard gene names)
    clean_symbols = [
        s for s in gene_symbols
        if " " not in s and "\t" not in s and s.replace("-", "").replace("_", "").isalnum()
    ]
    logger.info(
        "Submitting %d genes to Reactome Analysis Service (filtered %d non-standard names)...",
        len(clean_symbols),
        len(gene_symbols) - len(clean_symbols),
    )

    # Submit as newline-separated gene symbols
    payload = "\n".join(clean_symbols)
    url = f"{REACTOME_ANALYSIS_URL}/identifiers/projection"
    params = {
        "interactors": "false",
        "pageSize": 1000,
        "page": 1,
        "sortBy": "ENTITIES_PVALUE",
        "order": "ASC",
        "resource": "TOTAL",
        "pValue": "1",
        "includeDisease": "true",
    }

    try:
        resp = requests.post(
            url,
            headers={"Content-Type": "text/plain"},
            params=params,
            data=payload,
            timeout=120,
        )
        resp.raise_for_status()
        result = resp.json()
        logger.info(
            "Analysis complete: %d pathways found, %d identifiers mapped",
            result.get("pathwaysFound", 0),
            result.get("identifiersNotFound", 0),
        )
        return result
    except requests.RequestException as e:
        logger.error("Reactome analysis failed: %s", e)
        return {}


def fetch_pathway_participants(
    pathway_ids: list[str],
) -> dict[str, list[str]]:
    """Fetch participant gene symbols for specific pathways from Reactome.

    Queries in batches to avoid overloading the API.
    Returns pathway_id -> list of gene symbols found in network.
    """
    pathway_genes: dict[str, list[str]] = {}
    batch_size = 20

    for i in range(0, len(pathway_ids), batch_size):
        batch = pathway_ids[i : i + batch_size]
        for pid in batch:
            url = f"{REACTOME_CONTENT_URL}/data/participants/{pid}"
            try:
                resp = requests.get(
                    url,
                    headers={"Accept": "application/json"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    genes = set()
                    for participant in data:
                        # Extract gene names from participant entities
                        display_name = participant.get("displayName", "")
                        if display_name:
                            # Reactome names often include compartment: "GENE [compartment]"
                            gene = display_name.split(" [")[0].strip()
                            if gene:
                                genes.add(gene)
                        # Also check refEntities for gene symbols
                        for ref in participant.get("refEntities", []):
                            if ref.get("geneName"):
                                genes.update(ref["geneName"])
                    pathway_genes[pid] = sorted(genes)
            except requests.RequestException:
                pass
            time.sleep(RATE_LIMIT_DELAY)

        logger.info(
            "Fetched participants for %d / %d pathways",
            min(i + batch_size, len(pathway_ids)),
            len(pathway_ids),
        )

    return pathway_genes


def is_focus_pathway(name: str) -> bool:
    """Check if pathway name matches PANDAS-relevant keywords."""
    name_lower = name.lower()
    return any(kw.lower() in name_lower for kw in PANDAS_FOCUS_KEYWORDS)


def parse_analysis_results(
    analysis_result: dict,
    network_nodes: set[str],
    seed_symbols: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse Reactome analysis results into mapping and enrichment DataFrames.

    Returns:
        mapping_df: gene_symbol -> pathway mapping
        enrichment_df: pathway enrichment with p-values
    """
    pathways = analysis_result.get("pathways", [])
    if not pathways:
        return pd.DataFrame(), pd.DataFrame()

    # Token needed for detailed results
    token = analysis_result.get("summary", {}).get("token", "")

    enrichment_rows = []
    mapping_rows = []

    for pw in pathways:
        pw_id = pw.get("stId", "")
        pw_name = pw.get("name", "")
        entities = pw.get("entities", {})
        pvalue = entities.get("pValue", 1.0)
        fdr = entities.get("fdr", 1.0)
        found = entities.get("found", 0)
        total = entities.get("total", 0)
        ratio = entities.get("ratio", 0)
        is_focus = is_focus_pathway(pw_name)

        enrichment_rows.append({
            "pathway_id": pw_id,
            "pathway_name": pw_name,
            "network_genes": found,
            "pathway_total_genes": total,
            "ratio": round(ratio, 6) if ratio else 0,
            "pvalue": pvalue,
            "fdr": fdr,
            "is_focus_pathway": is_focus,
        })

    enrichment_df = pd.DataFrame(enrichment_rows)

    # For top pathways, get the actual mapped genes using the token
    if token:
        # Get found identifiers per pathway
        sig_pathways = enrichment_df[enrichment_df["fdr"] < 0.1]["pathway_id"].tolist()
        # Limit to top 200 to avoid too many API calls
        sig_pathways = sig_pathways[:200]

        logger.info(
            "Fetching mapped genes for %d significant pathways...", len(sig_pathways)
        )

        for pw_id in sig_pathways:
            url = (
                f"{REACTOME_ANALYSIS_URL}/token/{token}"
                f"/found/all/{pw_id}"
            )
            try:
                resp = requests.get(
                    url,
                    params={"resource": "TOTAL"},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    pw_name = enrichment_df.loc[
                        enrichment_df["pathway_id"] == pw_id, "pathway_name"
                    ].iloc[0]
                    # Response is a dict with "entities" key
                    entities = data.get("entities", []) if isinstance(data, dict) else data
                    for entity in entities:
                        gene = entity.get("id", "")
                        if gene in network_nodes:
                            mapping_rows.append({
                                "gene_symbol": gene,
                                "pathway_id": pw_id,
                                "pathway_name": pw_name,
                                "is_seed": gene in seed_symbols,
                                "is_focus_pathway": is_focus_pathway(pw_name),
                            })
            except requests.RequestException:
                pass
            time.sleep(RATE_LIMIT_DELAY)

    mapping_df = pd.DataFrame(mapping_rows)
    if not mapping_df.empty:
        mapping_df = mapping_df.drop_duplicates(
            subset=["gene_symbol", "pathway_id"]
        )

    return mapping_df, enrichment_df


def run() -> None:
    """Execute the full Reactome pathway mapping pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    seed_symbols = set(get_gene_symbols())

    # Step 1: Get network nodes
    logger.info("Step 1: Loading network nodes")
    network_nodes_list = get_network_nodes()
    if not network_nodes_list:
        logger.error("No network nodes found — cannot proceed")
        return
    network_nodes = set(network_nodes_list)

    # Step 2: Run Reactome overrepresentation analysis
    logger.info("Step 2: Running Reactome analysis")
    analysis_result = run_reactome_analysis(network_nodes_list)
    if not analysis_result:
        logger.error("Reactome analysis failed")
        return

    # Step 3: Parse results
    logger.info("Step 3: Parsing analysis results and fetching gene mappings")
    mapping_df, enrichment_df = parse_analysis_results(
        analysis_result, network_nodes, seed_symbols
    )

    # Step 4: Save results
    if not enrichment_df.empty:
        enrichment_path = DATA_DIR / "reactome_pathway_enrichment.tsv"
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        logger.info("Saved %d pathway enrichments to %s", len(enrichment_df), enrichment_path)

    if not mapping_df.empty:
        mapping_path = DATA_DIR / "reactome_node_pathway_mapping.tsv"
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        logger.info("Saved %d node-pathway mappings to %s", len(mapping_df), mapping_path)

    # Step 5: Stats
    sig_enriched = enrichment_df[enrichment_df["fdr"] < 0.05] if not enrichment_df.empty else pd.DataFrame()
    focus_enriched = sig_enriched[sig_enriched["is_focus_pathway"]] if not sig_enriched.empty else pd.DataFrame()

    mapped_genes = set(mapping_df["gene_symbol"]) if not mapping_df.empty else set()

    stats = {
        "total_network_nodes": len(network_nodes),
        "nodes_submitted": len(network_nodes_list),
        "total_pathways_found": len(enrichment_df),
        "significant_pathways_fdr05": len(sig_enriched),
        "focus_pathways_significant": len(focus_enriched),
        "genes_mapped_to_pathways": len(mapped_genes),
        "seed_proteins_mapped": len(mapped_genes & seed_symbols),
        "total_node_pathway_mappings": len(mapping_df),
    }

    # Add top focus pathway details
    if not focus_enriched.empty:
        stats["top_focus_pathways"] = []
        for _, row in focus_enriched.head(15).iterrows():
            pw_genes = set(
                mapping_df.loc[
                    mapping_df["pathway_id"] == row["pathway_id"], "gene_symbol"
                ]
            ) if not mapping_df.empty else set()
            seeds_in = pw_genes & seed_symbols
            stats["top_focus_pathways"].append({
                "pathway_id": row["pathway_id"],
                "pathway_name": row["pathway_name"],
                "network_genes": int(row["network_genes"]),
                "fdr": round(float(row["fdr"]), 8),
                "seed_proteins": sorted(seeds_in),
            })

    stats_path = DATA_DIR / "reactome_mapping_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved stats to %s", stats_path)

    # Summary
    logger.info("=== Reactome Pathway Mapping Summary ===")
    logger.info(
        "Total pathways: %d, Significant (FDR<0.05): %d, Focus pathways significant: %d",
        len(enrichment_df),
        len(sig_enriched),
        len(focus_enriched),
    )
    if not sig_enriched.empty:
        for _, row in sig_enriched.head(10).iterrows():
            marker = " ***" if row["is_focus_pathway"] else ""
            logger.info(
                "  %s %s: %d genes, FDR=%.2e%s",
                row["pathway_id"],
                row["pathway_name"],
                row["network_genes"],
                row["fdr"],
                marker,
            )


if __name__ == "__main__":
    run()
