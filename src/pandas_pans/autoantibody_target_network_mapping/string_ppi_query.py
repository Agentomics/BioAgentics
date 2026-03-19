"""Query STRING database (v12) for protein-protein interactions of PANDAS seed proteins.

Retrieves first-degree interactors for all seed proteins with combined_score > 700
(confidence > 0.7). Outputs an edge list with interaction types and scores.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.01_string_ppi_query

Output:
    data/pandas_pans/autoantibody_network/string_interactions.tsv
    data/pandas_pans/autoantibody_network/string_network_stats.json
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    SEED_PROTEINS,
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

STRING_API_BASE = "https://string-db.org/api"
SPECIES_HUMAN = 9606
MIN_SCORE = 700  # STRING uses 0-1000 scale; 700 = confidence > 0.7
RATE_LIMIT_DELAY = 1.0  # seconds between API calls


def _ensure_output_dir() -> Path:
    """Create and return the output directory."""
    out = Path("data/pandas_pans/autoantibody_network")
    out.mkdir(parents=True, exist_ok=True)
    return out


def resolve_string_ids(gene_symbols: list[str]) -> dict[str, str]:
    """Resolve gene symbols to STRING identifiers.

    Returns mapping of gene_symbol -> STRING ID (e.g., '9606.ENSP00000327652').
    """
    resolved = {}
    for symbol in gene_symbols:
        url = f"{STRING_API_BASE}/json/get_string_ids"
        params = {
            "identifiers": symbol,
            "species": SPECIES_HUMAN,
            "limit": 1,
            "caller_identity": "bioagentics_pandas_pans",
        }
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            if data:
                resolved[symbol] = data[0]["stringId"]
                logger.info("Resolved %s -> %s", symbol, data[0]["stringId"])
            else:
                logger.warning("No STRING ID found for %s", symbol)
        except requests.RequestException as e:
            logger.error("Failed to resolve %s: %s", symbol, e)
        time.sleep(RATE_LIMIT_DELAY)
    return resolved


def query_interactions(string_ids: list[str]) -> pd.DataFrame:
    """Query STRING network interactions for given STRING identifiers.

    Queries all seed proteins in a single batch request to get their
    interaction network including first-degree interactors.
    """
    url = f"{STRING_API_BASE}/json/network"
    params = {
        "identifiers": "\r".join(string_ids),
        "species": SPECIES_HUMAN,
        "required_score": MIN_SCORE,
        "add_nodes": 50,  # add first-degree interactors (up to 50 per seed)
        "network_type": "functional",
        "caller_identity": "bioagentics_pandas_pans",
    }

    logger.info("Querying STRING network for %d seed proteins...", len(string_ids))
    try:
        resp = requests.get(url, params=params, timeout=120)
        resp.raise_for_status()
        data = resp.json()
    except requests.RequestException as e:
        logger.error("STRING network query failed: %s", e)
        return pd.DataFrame()

    if not data:
        logger.warning("No interactions returned from STRING")
        return pd.DataFrame()

    rows = []
    for edge in data:
        rows.append({
            "source": edge.get("preferredName_A", edge.get("stringId_A", "")),
            "target": edge.get("preferredName_B", edge.get("stringId_B", "")),
            "source_string_id": edge.get("stringId_A", ""),
            "target_string_id": edge.get("stringId_B", ""),
            "combined_score": edge.get("score", 0),
            "nscore": edge.get("nscore", 0),
            "fscore": edge.get("fscore", 0),
            "pscore": edge.get("pscore", 0),
            "ascore": edge.get("ascore", 0),
            "escore": edge.get("escore", 0),
            "dscore": edge.get("dscore", 0),
            "tscore": edge.get("tscore", 0),
        })

    df = pd.DataFrame(rows)
    logger.info("Retrieved %d interactions", len(df))
    return df


def query_interactions_per_seed(
    string_id_map: dict[str, str],
) -> pd.DataFrame:
    """Query interactions for each seed protein individually.

    Fallback approach that queries each seed protein separately
    to ensure complete coverage of first-degree interactors.
    """
    all_frames = []
    for symbol, sid in string_id_map.items():
        url = f"{STRING_API_BASE}/json/network"
        params = {
            "identifiers": sid,
            "species": SPECIES_HUMAN,
            "required_score": MIN_SCORE,
            "add_nodes": 100,
            "network_type": "functional",
            "caller_identity": "bioagentics_pandas_pans",
        }

        logger.info("Querying STRING for %s (%s)...", symbol, sid)
        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("Failed for %s: %s", symbol, e)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        for edge in data:
            all_frames.append({
                "seed_protein": symbol,
                "source": edge.get("preferredName_A", ""),
                "target": edge.get("preferredName_B", ""),
                "source_string_id": edge.get("stringId_A", ""),
                "target_string_id": edge.get("stringId_B", ""),
                "combined_score": edge.get("score", 0),
                "nscore": edge.get("nscore", 0),
                "fscore": edge.get("fscore", 0),
                "pscore": edge.get("pscore", 0),
                "ascore": edge.get("ascore", 0),
                "escore": edge.get("escore", 0),
                "dscore": edge.get("dscore", 0),
                "tscore": edge.get("tscore", 0),
            })

        logger.info("  -> %d edges for %s", len(data), symbol)
        time.sleep(RATE_LIMIT_DELAY)

    if not all_frames:
        return pd.DataFrame()

    df = pd.DataFrame(all_frames)
    # Deduplicate edges (same pair may appear from different seed queries)
    edge_key = df[["source", "target"]].apply(
        lambda r: tuple(sorted([r["source"], r["target"]])), axis=1
    )
    df["_edge_key"] = edge_key
    df = df.drop_duplicates(subset=["_edge_key"]).drop(columns=["_edge_key"])
    return df


def compute_network_stats(
    df: pd.DataFrame, seed_symbols: list[str]
) -> dict:
    """Compute summary statistics for the interaction network."""
    if df.empty:
        return {"total_edges": 0, "total_nodes": 0, "unique_interactors": 0}

    all_nodes = set(df["source"]) | set(df["target"])
    seed_set = set(seed_symbols)
    interactors = all_nodes - seed_set

    # Edges involving at least one seed protein
    seed_edges = df[
        df["source"].isin(seed_set) | df["target"].isin(seed_set)
    ]

    stats = {
        "total_edges": len(df),
        "total_nodes": len(all_nodes),
        "seed_proteins": len(seed_set & all_nodes),
        "unique_interactors": len(interactors),
        "seed_direct_edges": len(seed_edges),
        "mean_combined_score": round(float(df["combined_score"].mean()), 4),
        "min_combined_score": round(float(df["combined_score"].min()), 4),
        "max_combined_score": round(float(df["combined_score"].max()), 4),
        "seed_degree": {
            sym: int(
                ((df["source"] == sym) | (df["target"] == sym)).sum()
            )
            for sym in seed_set
            if sym in all_nodes
        },
    }
    return stats


def run() -> None:
    """Execute the full STRING PPI query pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    out_dir = _ensure_output_dir()
    gene_symbols = get_gene_symbols()

    # Step 1: Resolve STRING IDs
    logger.info("Step 1: Resolving STRING IDs for %d seed proteins", len(gene_symbols))
    string_id_map = resolve_string_ids(gene_symbols)
    logger.info("Resolved %d / %d proteins", len(string_id_map), len(gene_symbols))

    if not string_id_map:
        logger.error("No STRING IDs resolved — cannot proceed")
        return

    # Save ID mapping
    id_map_path = out_dir / "string_id_mapping.json"
    with open(id_map_path, "w") as f:
        json.dump(string_id_map, f, indent=2)
    logger.info("Saved STRING ID mapping to %s", id_map_path)

    # Step 2: Query interactions per seed protein
    logger.info("Step 2: Querying first-degree interactions for each seed protein")
    df = query_interactions_per_seed(string_id_map)

    if df.empty:
        logger.error("No interactions retrieved")
        return

    # Step 3: Save results
    interactions_path = out_dir / "string_interactions.tsv"
    df.to_csv(interactions_path, sep="\t", index=False)
    logger.info("Saved %d interactions to %s", len(df), interactions_path)

    # Step 4: Compute and save stats
    stats = compute_network_stats(df, gene_symbols)
    stats_path = out_dir / "string_network_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("Network stats: %d nodes, %d edges, %d unique interactors",
                stats["total_nodes"], stats["total_edges"],
                stats["unique_interactors"])

    # Check success criterion
    if stats["unique_interactors"] >= 50:
        logger.info("SUCCESS: %d unique interactors (target: >= 50)",
                     stats["unique_interactors"])
    else:
        logger.warning("Below target: %d unique interactors (target: >= 50)",
                        stats["unique_interactors"])


if __name__ == "__main__":
    run()
