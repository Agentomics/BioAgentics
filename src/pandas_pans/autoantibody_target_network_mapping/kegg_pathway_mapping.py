"""KEGG pathway mapping for PANDAS autoantibody target network nodes.

Maps all network nodes (seed + interactors) to KEGG pathways using the KEGG
REST API. Performs pathway enrichment analysis with focus on PANDAS-relevant
pathways: dopaminergic synapse, calcium signaling, basal ganglia circuits,
GPCR signaling.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.kegg_pathway_mapping

Output:
    data/pandas_pans/autoantibody_network/kegg_node_pathway_mapping.tsv
    data/pandas_pans/autoantibody_network/kegg_pathway_enrichment.tsv
    data/pandas_pans/autoantibody_network/kegg_mapping_stats.json
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from scipy import stats as scipy_stats

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

KEGG_API_BASE = "https://rest.kegg.jp"
RATE_LIMIT_DELAY = 0.5  # seconds between API calls
DATA_DIR = Path("data/pandas_pans/autoantibody_network")

# PANDAS-relevant KEGG pathways to highlight
PANDAS_FOCUS_PATHWAYS = {
    "hsa04728": "Dopaminergic synapse",
    "hsa04020": "Calcium signaling pathway",
    "hsa04080": "Neuroactive ligand-receptor interaction",
    "hsa04024": "cAMP signaling pathway",
    "hsa04062": "Chemokine signaling pathway",
    "hsa04060": "Cytokine-cytokine receptor interaction",
    "hsa04630": "JAK-STAT signaling pathway",
    "hsa04010": "MAPK signaling pathway",
    "hsa04150": "mTOR signaling pathway",
    "hsa04151": "PI3K-Akt signaling pathway",
    "hsa04720": "Long-term potentiation",
    "hsa04723": "Retrograde endocannabinoid signaling",
    "hsa04724": "Glutamatergic synapse",
    "hsa04725": "Cholinergic synapse",
    "hsa04726": "Serotonergic synapse",
    "hsa04727": "GABAergic synapse",
    "hsa04730": "Long-term depression",
    "hsa05012": "Parkinson disease",
    "hsa05030": "Cocaine addiction",
    "hsa05031": "Amphetamine addiction",
    "hsa04360": "Axon guidance",
    "hsa00670": "One carbon pool by folate",
    "hsa00790": "Folate biosynthesis",
}


def _kegg_get(path: str) -> str:
    """Make a KEGG REST API request and return the text."""
    url = f"{KEGG_API_BASE}/{path}"
    try:
        resp = requests.get(url, timeout=60)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.error("KEGG API error for %s: %s", path, e)
        return ""


def download_human_gene_list() -> dict[str, str]:
    """Download human gene list from KEGG. Returns symbol -> kegg_id mapping.

    KEGG list/hsa format (tab-separated):
        hsa:ENTREZ_ID  CDS  CHROM  SYMBOLS; description
    where SYMBOLS is "SYM1, SYM2, SYM3" before the semicolon.
    """
    logger.info("Downloading KEGG human gene list...")
    text = _kegg_get("list/hsa")
    if not text:
        return {}

    symbol_to_kegg: dict[str, str] = {}
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) < 4:
            continue
        kegg_id = parts[0].strip()  # e.g., "hsa:1234"
        # Symbols + description are in the 4th column (index 3)
        desc = parts[3]
        # Gene symbols are before the first semicolon
        if ";" in desc:
            symbols_part = desc.split(";")[0]
        else:
            symbols_part = desc
        symbols = [s.strip() for s in symbols_part.split(",")]
        for sym in symbols:
            if sym:
                symbol_to_kegg[sym] = kegg_id

    logger.info("Parsed %d gene symbol -> KEGG ID mappings", len(symbol_to_kegg))
    return symbol_to_kegg


def download_gene_pathway_links() -> list[tuple[str, str]]:
    """Download all human gene-pathway associations from KEGG.

    Returns list of (kegg_gene_id, pathway_id) tuples.
    """
    logger.info("Downloading KEGG gene-pathway links for human...")
    text = _kegg_get("link/pathway/hsa")
    time.sleep(RATE_LIMIT_DELAY)
    if not text:
        return []

    links = []
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) == 2:
            gene_id = parts[0].strip()  # e.g., "hsa:1234"
            pathway_id = parts[1].strip().replace("path:", "")  # e.g., "hsa04728"
            links.append((gene_id, pathway_id))

    logger.info("Parsed %d gene-pathway links", len(links))
    return links


def download_pathway_names() -> dict[str, str]:
    """Download human pathway names from KEGG. Returns pathway_id -> name."""
    logger.info("Downloading KEGG pathway names...")
    text = _kegg_get("list/pathway/hsa")
    time.sleep(RATE_LIMIT_DELAY)
    if not text:
        return {}

    names: dict[str, str] = {}
    for line in text.strip().split("\n"):
        parts = line.split("\t")
        if len(parts) >= 2:
            pid = parts[0].strip().replace("path:", "")
            name = parts[1].strip()
            # Remove " - Homo sapiens (human)" suffix
            if " - Homo sapiens" in name:
                name = name.split(" - Homo sapiens")[0]
            names[pid] = name

    logger.info("Parsed %d pathway names", len(names))
    return names


def get_network_nodes() -> set[str]:
    """Extract all unique gene symbols from the extended network."""
    network_path = DATA_DIR / "extended_network.tsv"
    if not network_path.exists():
        logger.error("Extended network file not found: %s", network_path)
        return set()

    # Read only source and target columns to save memory
    chunks = pd.read_csv(
        network_path, sep="\t", usecols=["source", "target"], chunksize=5000
    )
    nodes: set[str] = set()
    for chunk in chunks:
        nodes.update(chunk["source"].dropna())
        nodes.update(chunk["target"].dropna())
    logger.info("Found %d unique nodes in extended network", len(nodes))
    return nodes


def map_nodes_to_pathways(
    nodes: set[str],
    symbol_to_kegg: dict[str, str],
    gene_pathway_links: list[tuple[str, str]],
    pathway_names: dict[str, str],
) -> pd.DataFrame:
    """Map network nodes to KEGG pathways.

    Returns DataFrame with columns: gene_symbol, kegg_id, pathway_id, pathway_name,
    is_seed, is_focus_pathway.
    """
    seed_symbols = set(get_gene_symbols())

    # Build kegg_id -> list of pathway_ids
    kegg_to_pathways: dict[str, list[str]] = {}
    for gene_id, pathway_id in gene_pathway_links:
        kegg_to_pathways.setdefault(gene_id, []).append(pathway_id)

    rows = []
    mapped_count = 0
    for symbol in sorted(nodes):
        kegg_id = symbol_to_kegg.get(symbol)
        if not kegg_id:
            continue
        pathways = kegg_to_pathways.get(kegg_id, [])
        if not pathways:
            continue
        mapped_count += 1
        for pw_id in pathways:
            pw_name = pathway_names.get(pw_id, pw_id)
            rows.append({
                "gene_symbol": symbol,
                "kegg_id": kegg_id,
                "pathway_id": pw_id,
                "pathway_name": pw_name,
                "is_seed": symbol in seed_symbols,
                "is_focus_pathway": pw_id in PANDAS_FOCUS_PATHWAYS,
            })

    df = pd.DataFrame(rows)
    logger.info(
        "Mapped %d / %d nodes to KEGG pathways (%d total mappings)",
        mapped_count,
        len(nodes),
        len(df),
    )
    return df


def compute_pathway_enrichment(
    mapping_df: pd.DataFrame,
    network_nodes: set[str],
    symbol_to_kegg: dict[str, str],
    gene_pathway_links: list[tuple[str, str]],
    pathway_names: dict[str, str],
) -> pd.DataFrame:
    """Compute pathway enrichment using Fisher's exact test.

    Tests whether each pathway is over-represented in the network compared
    to the full human genome background.
    """
    if mapping_df.empty:
        return pd.DataFrame()

    # Background: all human genes with KEGG mappings
    all_kegg_genes = set()
    pathway_to_all_genes: dict[str, set[str]] = {}
    for gene_id, pathway_id in gene_pathway_links:
        all_kegg_genes.add(gene_id)
        pathway_to_all_genes.setdefault(pathway_id, set()).add(gene_id)

    # Foreground: network genes with KEGG mappings
    network_kegg_ids = set()
    for sym in network_nodes:
        kid = symbol_to_kegg.get(sym)
        if kid:
            network_kegg_ids.add(kid)

    bg_total = len(all_kegg_genes)
    fg_total = len(network_kegg_ids)

    # For each pathway, compute enrichment
    seed_symbols = set(get_gene_symbols())
    pathway_node_map = mapping_df.groupby("pathway_id")["gene_symbol"].apply(set)
    pathway_ids_in_network = mapping_df["pathway_id"].unique()

    rows = []
    for pw_id in pathway_ids_in_network:
        pw_genes_in_network = pathway_node_map.get(pw_id, set())
        pw_genes_total = pathway_to_all_genes.get(pw_id, set())

        fg_in = len(pw_genes_in_network)
        fg_out = fg_total - fg_in
        bg_in = len(pw_genes_total) - fg_in
        bg_out = bg_total - len(pw_genes_total) - fg_out

        # Ensure non-negative values for the contingency table
        bg_in = max(bg_in, 0)
        bg_out = max(bg_out, 0)

        # Fisher's exact test (one-sided, over-representation)
        _, pvalue = scipy_stats.fisher_exact(
            [[fg_in, fg_out], [bg_in, bg_out]], alternative="greater"
        )

        # Count seed proteins in this pathway
        seeds_in_pathway = pw_genes_in_network & seed_symbols
        pw_name = pathway_names.get(pw_id, pw_id)

        rows.append({
            "pathway_id": pw_id,
            "pathway_name": pw_name,
            "network_genes": fg_in,
            "background_genes": len(pw_genes_total),
            "fold_enrichment": round(
                (fg_in / fg_total) / (len(pw_genes_total) / bg_total), 3
            ) if len(pw_genes_total) > 0 and fg_total > 0 else 0,
            "pvalue": pvalue,
            "seed_proteins_in_pathway": len(seeds_in_pathway),
            "seed_list": ",".join(sorted(seeds_in_pathway)) if seeds_in_pathway else "",
            "is_focus_pathway": pw_id in PANDAS_FOCUS_PATHWAYS,
            "network_gene_list": ",".join(sorted(pw_genes_in_network)),
        })

    enrichment_df = pd.DataFrame(rows)

    # Multiple testing correction (Benjamini-Hochberg)
    if not enrichment_df.empty:
        enrichment_df = enrichment_df.sort_values("pvalue")
        n = len(enrichment_df)
        enrichment_df["rank"] = range(1, n + 1)
        enrichment_df["fdr"] = enrichment_df["pvalue"] * n / enrichment_df["rank"]
        # Enforce monotonicity
        enrichment_df["fdr"] = enrichment_df["fdr"][::-1].cummin()[::-1]
        enrichment_df["fdr"] = enrichment_df["fdr"].clip(upper=1.0)
        enrichment_df = enrichment_df.drop(columns=["rank"])
        enrichment_df = enrichment_df.sort_values("pvalue")

    logger.info(
        "Enrichment analysis: %d pathways tested, %d significant (FDR < 0.05)",
        len(enrichment_df),
        (enrichment_df["fdr"] < 0.05).sum() if not enrichment_df.empty else 0,
    )
    return enrichment_df


def run() -> None:
    """Execute the full KEGG pathway mapping pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Step 1: Get network nodes
    logger.info("Step 1: Loading network nodes")
    network_nodes = get_network_nodes()
    if not network_nodes:
        logger.error("No network nodes found — cannot proceed")
        return

    # Step 2: Download KEGG reference data (3 bulk API calls)
    logger.info("Step 2: Downloading KEGG reference data")
    symbol_to_kegg = download_human_gene_list()
    time.sleep(RATE_LIMIT_DELAY)
    gene_pathway_links = download_gene_pathway_links()
    time.sleep(RATE_LIMIT_DELAY)
    pathway_names = download_pathway_names()

    if not symbol_to_kegg or not gene_pathway_links or not pathway_names:
        logger.error("Failed to download KEGG reference data")
        return

    # Step 3: Map network nodes to pathways
    logger.info("Step 3: Mapping network nodes to KEGG pathways")
    mapping_df = map_nodes_to_pathways(
        network_nodes, symbol_to_kegg, gene_pathway_links, pathway_names
    )

    if mapping_df.empty:
        logger.error("No nodes mapped to pathways")
        return

    mapping_path = DATA_DIR / "kegg_node_pathway_mapping.tsv"
    mapping_df.to_csv(mapping_path, sep="\t", index=False)
    logger.info("Saved node-pathway mapping to %s", mapping_path)

    # Step 4: Pathway enrichment analysis
    logger.info("Step 4: Running pathway enrichment analysis")
    enrichment_df = compute_pathway_enrichment(
        mapping_df, network_nodes, symbol_to_kegg, gene_pathway_links, pathway_names
    )

    if not enrichment_df.empty:
        enrichment_path = DATA_DIR / "kegg_pathway_enrichment.tsv"
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        logger.info("Saved pathway enrichment to %s", enrichment_path)

    # Step 5: Save stats
    seed_symbols = set(get_gene_symbols())
    mapped_nodes = set(mapping_df["gene_symbol"].unique())
    focus_pathways_found = mapping_df[mapping_df["is_focus_pathway"]]
    sig_enriched = (
        enrichment_df[enrichment_df["fdr"] < 0.05] if not enrichment_df.empty
        else pd.DataFrame()
    )

    stats = {
        "total_network_nodes": len(network_nodes),
        "nodes_mapped_to_kegg": len(mapped_nodes),
        "mapping_rate": round(len(mapped_nodes) / len(network_nodes), 4),
        "total_pathways_found": mapping_df["pathway_id"].nunique(),
        "total_node_pathway_mappings": len(mapping_df),
        "focus_pathways_with_network_genes": focus_pathways_found["pathway_id"].nunique()
        if not focus_pathways_found.empty
        else 0,
        "significant_enriched_pathways_fdr05": len(sig_enriched),
        "seed_proteins_mapped": len(mapped_nodes & seed_symbols),
        "focus_pathway_details": {},
    }

    # Add focus pathway details
    for pw_id, pw_name in PANDAS_FOCUS_PATHWAYS.items():
        pw_rows = mapping_df[mapping_df["pathway_id"] == pw_id]
        if not pw_rows.empty:
            pw_genes = set(pw_rows["gene_symbol"])
            pw_seeds = pw_genes & seed_symbols
            enrichment_row = (
                enrichment_df[enrichment_df["pathway_id"] == pw_id]
                if not enrichment_df.empty
                else pd.DataFrame()
            )
            stats["focus_pathway_details"][pw_id] = {
                "name": pw_name,
                "network_genes": len(pw_genes),
                "seed_proteins": sorted(pw_seeds),
                "fdr": round(float(enrichment_row.iloc[0]["fdr"]), 6)
                if not enrichment_row.empty
                else None,
                "fold_enrichment": float(enrichment_row.iloc[0]["fold_enrichment"])
                if not enrichment_row.empty
                else None,
            }

    stats_path = DATA_DIR / "kegg_mapping_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Saved mapping stats to %s", stats_path)

    # Summary
    logger.info("=== KEGG Pathway Mapping Summary ===")
    logger.info(
        "Mapped %d / %d nodes (%.1f%%) to %d KEGG pathways",
        len(mapped_nodes),
        len(network_nodes),
        100 * len(mapped_nodes) / len(network_nodes),
        mapping_df["pathway_id"].nunique(),
    )
    logger.info(
        "Significant enrichment (FDR < 0.05): %d pathways",
        len(sig_enriched),
    )
    if not sig_enriched.empty:
        for _, row in sig_enriched.head(10).iterrows():
            marker = " ***" if row["is_focus_pathway"] else ""
            logger.info(
                "  %s %s: %d genes, fold=%.2f, FDR=%.2e%s",
                row["pathway_id"],
                row["pathway_name"],
                row["network_genes"],
                row["fold_enrichment"],
                row["fdr"],
                marker,
            )


if __name__ == "__main__":
    run()
