"""Query BioGRID and IntAct for supplementary PPI data for PANDAS seed proteins.

Retrieves physical protein-protein interactions from BioGRID (if API key is
available) and IntAct (free, no key required) to supplement the STRING
interaction data. Merges results with existing STRING edge list, deduplicating
by protein pair and annotating source databases.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.biogrid_ppi_query

Output:
    data/pandas_pans/autoantibody_network/supplementary_interactions.tsv
    data/pandas_pans/autoantibody_network/combined_ppi_edgelist.tsv
    data/pandas_pans/autoantibody_network/combined_network_stats.json
"""

import json
import logging
import os
import time
from pathlib import Path

import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/pandas_pans/autoantibody_network")
RATE_LIMIT_DELAY = 1.0

# BioGRID REST API v4
BIOGRID_API_BASE = "https://webservice.thebiogrid.org/interactions"
BIOGRID_API_KEY = os.environ.get("BIOGRID_API_KEY", "")

# IntAct REST API
INTACT_API_BASE = "https://www.ebi.ac.uk/intact/ws/interaction/findInteractions"
INTACT_PAGE_SIZE = 200


def _ensure_output_dir() -> Path:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    return DATA_DIR


def _sorted_edge_key(source: str, target: str) -> tuple[str, str]:
    """Return a normalized (sorted) edge key for deduplication."""
    a, b = sorted([source, target])
    return (a, b)


def query_biogrid(gene_symbols: list[str]) -> pd.DataFrame:
    """Query BioGRID REST API for physical interactions of seed proteins.

    Requires BIOGRID_API_KEY environment variable.
    Returns empty DataFrame if key is not set.
    """
    if not BIOGRID_API_KEY:
        logger.warning("BIOGRID_API_KEY not set — skipping BioGRID queries")
        return pd.DataFrame()

    all_rows: list[dict] = []
    for symbol in gene_symbols:
        params = {
            "accessKey": BIOGRID_API_KEY,
            "format": "json",
            "searchNames": True,
            "geneList": symbol,
            "organismId": 9606,
            "interSpeciesExcluded": True,
            "selfInteractionsExcluded": True,
            "evidenceList": "physical",
            "includeHeader": True,
            "max": 500,
        }
        try:
            resp = requests.get(BIOGRID_API_BASE, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("BioGRID query failed for %s: %s", symbol, e)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        for interaction in data.values():
            all_rows.append({
                "source": interaction.get("OFFICIAL_SYMBOL_A", ""),
                "target": interaction.get("OFFICIAL_SYMBOL_B", ""),
                "source_organism": interaction.get("ORGANISM_A", ""),
                "target_organism": interaction.get("ORGANISM_B", ""),
                "experimental_system": interaction.get("EXPERIMENTAL_SYSTEM", ""),
                "interaction_type": interaction.get("EXPERIMENTAL_SYSTEM_TYPE", ""),
                "pubmed_id": interaction.get("PUBMED_ID", ""),
                "biogrid_id": interaction.get("BIOGRID_INTERACTION_ID", ""),
                "database": "BioGRID",
            })
        logger.info("BioGRID: %d interactions for %s", len(data), symbol)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    return df[df["interaction_type"] == "physical"].reset_index(drop=True)


def query_intact(gene_symbols: list[str]) -> pd.DataFrame:
    """Query IntAct REST API for interactions involving seed proteins.

    IntAct aggregates data from multiple sources including BioGRID.
    No API key required. Uses pagination to retrieve all results.
    """
    all_rows: list[dict] = []

    for symbol in gene_symbols:
        page = 0
        total_for_symbol = 0
        total_elements = 0

        while True:
            url = f"{INTACT_API_BASE}/{symbol}"
            params = {"page": page, "pageSize": INTACT_PAGE_SIZE}
            headers = {"Accept": "application/json"}

            try:
                resp = requests.get(url, params=params, headers=headers, timeout=60)
                if resp.status_code in (404, 400):
                    logger.info("IntAct: no results for %s", symbol)
                    break
                resp.raise_for_status()
                data = resp.json()
            except requests.RequestException as e:
                logger.error("IntAct query failed for %s (page %d): %s", symbol, page, e)
                break

            content = data.get("content", [])
            if not content:
                break

            total_elements = data.get("totalElements", 0)

            for ix in content:
                mol_a = ix.get("moleculeA", "")
                mol_b = ix.get("moleculeB", "")

                # Skip self-interactions
                if mol_a == mol_b:
                    continue

                # Only human-human interactions
                if ix.get("taxIdA") != 9606 or ix.get("taxIdB") != 9606:
                    continue

                # Skip negative interactions
                if ix.get("negative", False):
                    continue

                all_rows.append({
                    "source": mol_a,
                    "target": mol_b,
                    "source_uniprot": ix.get("uniqueIdA", ""),
                    "target_uniprot": ix.get("uniqueIdB", ""),
                    "interaction_type": ix.get("type", ""),
                    "detection_method": ix.get("detectionMethod", ""),
                    "pubmed_id": ix.get("publicationPubmedIdentifier", ""),
                    "mi_score": ix.get("intactMiscore", 0.0),
                    "source_database": ix.get("sourceDatabase", ""),
                    "database": "IntAct",
                })
                total_for_symbol += 1

            # Check if there are more pages
            pageable = data.get("pageable", {})
            current_page = pageable.get("pageNumber", page)
            total_pages = (total_elements + INTACT_PAGE_SIZE - 1) // INTACT_PAGE_SIZE

            if current_page + 1 >= total_pages:
                break
            page += 1
            time.sleep(RATE_LIMIT_DELAY)

        logger.info("IntAct: %d interactions for %s (total in DB: %d)",
                     total_for_symbol, symbol, total_elements)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_rows:
        return pd.DataFrame()

    return pd.DataFrame(all_rows)


def load_string_interactions() -> pd.DataFrame:
    """Load existing STRING interaction data."""
    path = DATA_DIR / "string_interactions.tsv"
    if not path.exists():
        logger.error("STRING interactions file not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    df["database"] = "STRING"
    return df


def merge_and_dedup(
    string_df: pd.DataFrame,
    supplementary_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge STRING and supplementary interactions, deduplicating by protein pair.

    For duplicate pairs, keeps the STRING entry (higher confidence scoring)
    and annotates with all source databases.
    """
    if supplementary_df.empty:
        string_df = string_df.copy()
        string_df["databases"] = "STRING"
        return string_df

    # Build database membership per edge from supplementary data
    supp_db_map: dict[tuple[str, str], set[str]] = {}
    for _, row in supplementary_df.iterrows():
        key = _sorted_edge_key(str(row.get("source", "")), str(row.get("target", "")))
        db = str(row.get("database", "supplementary"))
        supp_db_map.setdefault(key, set()).add(db)

    # Annotate STRING edges with supplementary database info
    string_keys: set[tuple[str, str]] = set()
    databases_list = []
    for _, row in string_df.iterrows():
        key = _sorted_edge_key(str(row.get("source", "")), str(row.get("target", "")))
        string_keys.add(key)
        dbs = {"STRING"}
        if key in supp_db_map:
            dbs.update(supp_db_map[key])
        databases_list.append(",".join(sorted(dbs)))
    string_df = string_df.copy()
    string_df["databases"] = databases_list

    # Add supplementary-only edges (not in STRING)
    new_rows = []
    for _, row in supplementary_df.iterrows():
        key = _sorted_edge_key(str(row.get("source", "")), str(row.get("target", "")))
        if key not in string_keys:
            score = row.get("mi_score") or row.get("confidence_score") or 0
            new_rows.append({
                "source": row["source"],
                "target": row["target"],
                "combined_score": score,
                "database": str(row.get("database", "supplementary")),
                "databases": str(row.get("database", "supplementary")),
            })
            string_keys.add(key)

    if new_rows:
        new_df = pd.DataFrame(new_rows)
        combined = pd.concat([string_df, new_df], ignore_index=True)
    else:
        combined = string_df

    return combined


def compute_combined_stats(
    combined_df: pd.DataFrame,
    seed_symbols: list[str],
) -> dict:
    """Compute stats for the combined network."""
    if combined_df.empty:
        return {"total_edges": 0, "total_nodes": 0}

    all_nodes = set(combined_df["source"]) | set(combined_df["target"])
    seed_set = set(seed_symbols)
    interactors = all_nodes - seed_set

    # Count edges by database source
    db_counts: dict[str, int] = {}
    if "databases" in combined_df.columns:
        for dbs_str in combined_df["databases"]:
            for db in str(dbs_str).split(","):
                db = db.strip()
                if db:
                    db_counts[db] = db_counts.get(db, 0) + 1

    # Edges only in supplementary (not STRING)
    supp_only = 0
    if "databases" in combined_df.columns:
        supp_only = int((~combined_df["databases"].str.contains("STRING", na=False)).sum())

    return {
        "total_edges": len(combined_df),
        "total_nodes": len(all_nodes),
        "seed_proteins": len(seed_set & all_nodes),
        "unique_interactors": len(interactors),
        "database_edge_counts": db_counts,
        "supplementary_only_edges": supp_only,
    }


def run() -> None:
    """Execute supplementary PPI query and merge pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    out_dir = _ensure_output_dir()
    gene_symbols = get_gene_symbols()

    # Step 1: Query BioGRID (if API key available)
    logger.info("Step 1: Querying BioGRID for %d seed proteins", len(gene_symbols))
    biogrid_df = query_biogrid(gene_symbols)
    logger.info("BioGRID: %d interactions retrieved", len(biogrid_df))

    # Step 2: Query IntAct (no key needed)
    logger.info("Step 2: Querying IntAct for %d seed proteins", len(gene_symbols))
    intact_df = query_intact(gene_symbols)
    logger.info("IntAct: %d interactions retrieved", len(intact_df))

    # Step 3: Combine supplementary sources
    supp_frames = [df for df in [biogrid_df, intact_df] if not df.empty]
    if supp_frames:
        supplementary_df = pd.concat(supp_frames, ignore_index=True)
        # Deduplicate within supplementary data
        supplementary_df["_edge_key"] = supplementary_df.apply(
            lambda r: _sorted_edge_key(str(r.get("source", "")), str(r.get("target", ""))),
            axis=1,
        )
        supplementary_df = supplementary_df.drop_duplicates(subset=["_edge_key"])
        supplementary_df = supplementary_df.drop(columns=["_edge_key"])
    else:
        supplementary_df = pd.DataFrame()

    # Save supplementary interactions
    if not supplementary_df.empty:
        supp_path = out_dir / "supplementary_interactions.tsv"
        supplementary_df.to_csv(supp_path, sep="\t", index=False)
        logger.info("Saved %d supplementary interactions to %s",
                     len(supplementary_df), supp_path)

    # Step 4: Load STRING and merge
    logger.info("Step 3: Loading STRING interactions and merging")
    string_df = load_string_interactions()
    if string_df.empty:
        logger.error("No STRING interactions found — cannot merge")
        return

    combined_df = merge_and_dedup(string_df, supplementary_df)

    # Save combined edge list
    combined_path = out_dir / "combined_ppi_edgelist.tsv"
    combined_df.to_csv(combined_path, sep="\t", index=False)
    logger.info("Saved %d combined interactions to %s", len(combined_df), combined_path)

    # Step 5: Compute and save stats
    stats = compute_combined_stats(combined_df, gene_symbols)
    stats_path = out_dir / "combined_network_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info(
        "Combined network: %d nodes, %d edges, %d unique interactors, %d supplementary-only edges",
        stats["total_nodes"],
        stats["total_edges"],
        stats["unique_interactors"],
        stats.get("supplementary_only_edges", 0),
    )


if __name__ == "__main__":
    run()
