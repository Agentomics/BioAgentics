"""Cytokine amplification layer for PANDAS autoantibody network.

Implements the two-hit framework from the research plan:
  Layer 1 = direct autoantibody binding and downstream PPI disruption
  Layer 2 = cytokine-mediated amplification/modification of autoantibody effects

Key cytokines modeled:
  - IL-17A: promotes BBB permeability, gut-oral-brain axis (Matera 2025, PMID 41394880)
  - IFNγ: epigenetic chromatin closing in neurons (Shammas 2026, PMID 41448185)
  - IL-6: central hub cytokine, gatekeeper for inflammatory cascade

Rationale: Anti-NMDAR1 + IL-17 study (Mol Psychiatry 2025, doi: 10.1038/s41380-025-03434-x)
showed autoantibodies alone cause NMDAR hypofunction, but full patient CSF causes neuronal
hyperexcitability — dissociating antibody effects from inflammatory milieu effects.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.cytokine_layer

Output:
    data/pandas_pans/autoantibody_network/cytokine_interactions.tsv
    data/pandas_pans/autoantibody_network/extended_network.tsv
    data/pandas_pans/autoantibody_network/extended_network_stats.json
"""

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

DATA_DIR = Path("data/pandas_pans/autoantibody_network")
STRING_API_BASE = "https://string-db.org/api"
SPECIES_HUMAN = 9606
MIN_SCORE = 700
RATE_LIMIT_DELAY = 1.0


@dataclass(frozen=True)
class CytokineTarget:
    """A cytokine or cytokine receptor in the amplification layer."""

    name: str
    gene_symbol: str
    uniprot_id: str
    role: str  # "cytokine", "receptor", "signaling"
    mechanism: str  # brief description of how it amplifies autoantibody effects
    layer: str = "cytokine_amplification"


# Three key cytokine axes from the research plan
CYTOKINE_TARGETS: list[CytokineTarget] = [
    # IL-17 axis — BBB permeability, gut-oral-brain axis
    CytokineTarget(
        name="Interleukin-17A",
        gene_symbol="IL17A",
        uniprot_id="Q16552",
        role="cytokine",
        mechanism="Promotes BBB permeability enabling autoantibody CNS access; "
        "gut-oral-brain axis (Matera 2025, PMID 41394880)",
    ),
    CytokineTarget(
        name="IL-17 receptor A",
        gene_symbol="IL17RA",
        uniprot_id="Q96F46",
        role="receptor",
        mechanism="Primary receptor for IL-17A signaling",
    ),
    CytokineTarget(
        name="IL-17 receptor C",
        gene_symbol="IL17RC",
        uniprot_id="Q8NAC3",
        role="receptor",
        mechanism="Co-receptor for IL-17A/F signaling",
    ),
    # IFNγ axis — epigenetic chromatin closing
    CytokineTarget(
        name="Interferon gamma",
        gene_symbol="IFNG",
        uniprot_id="P01579",
        role="cytokine",
        mechanism="Drives persistent epigenetic chromatin closing in neurons; "
        "autoantibody + IFNγ may cause qualitatively different damage "
        "(Shammas 2026, PMID 41448185)",
    ),
    CytokineTarget(
        name="IFNγ receptor 1",
        gene_symbol="IFNGR1",
        uniprot_id="P15260",
        role="receptor",
        mechanism="Primary IFNγ receptor subunit; JAK-STAT signaling",
    ),
    CytokineTarget(
        name="IFNγ receptor 2",
        gene_symbol="IFNGR2",
        uniprot_id="P38484",
        role="receptor",
        mechanism="IFNγ receptor accessory subunit",
    ),
    # IL-6 axis — central hub cytokine
    CytokineTarget(
        name="Interleukin-6",
        gene_symbol="IL6",
        uniprot_id="P05231",
        role="cytokine",
        mechanism="Central hub cytokine; gatekeeper for broader inflammatory cascade; "
        "trans-signaling via sIL-6R amplifies neuroinflammation",
    ),
    CytokineTarget(
        name="IL-6 receptor alpha",
        gene_symbol="IL6R",
        uniprot_id="P08887",
        role="receptor",
        mechanism="IL-6 receptor; soluble form (sIL-6R) enables trans-signaling",
    ),
    CytokineTarget(
        name="Glycoprotein 130",
        gene_symbol="IL6ST",
        uniprot_id="P40189",
        role="receptor",
        mechanism="Signal-transducing component shared by IL-6 family cytokines",
    ),
    # Key downstream signaling mediators shared across axes
    CytokineTarget(
        name="JAK1",
        gene_symbol="JAK1",
        uniprot_id="P23458",
        role="signaling",
        mechanism="Janus kinase 1; downstream of IFNγR and IL-6R signaling",
    ),
    CytokineTarget(
        name="JAK2",
        gene_symbol="JAK2",
        uniprot_id="O60674",
        role="signaling",
        mechanism="Janus kinase 2; downstream of IFNγR signaling",
    ),
    CytokineTarget(
        name="STAT1",
        gene_symbol="STAT1",
        uniprot_id="P42224",
        role="signaling",
        mechanism="Signal transducer; IFNγ-STAT1 axis mediates epigenetic changes in neurons",
    ),
    CytokineTarget(
        name="STAT3",
        gene_symbol="STAT3",
        uniprot_id="P40763",
        role="signaling",
        mechanism="Signal transducer; IL-6-STAT3 axis; neuroinflammation amplification",
    ),
    CytokineTarget(
        name="NF-kB p65",
        gene_symbol="RELA",
        uniprot_id="Q04206",
        role="signaling",
        mechanism="NF-κB subunit; convergence node for IL-17 and IL-6 inflammatory signaling",
    ),
]


def get_cytokine_gene_symbols() -> list[str]:
    """Return gene symbols for all cytokine layer proteins."""
    return [c.gene_symbol for c in CYTOKINE_TARGETS]


def get_cytokine_dataframe() -> pd.DataFrame:
    """Return cytokine targets as a DataFrame."""
    return pd.DataFrame([
        {
            "name": c.name,
            "gene_symbol": c.gene_symbol,
            "uniprot_id": c.uniprot_id,
            "role": c.role,
            "mechanism": c.mechanism,
            "layer": c.layer,
        }
        for c in CYTOKINE_TARGETS
    ])


def resolve_string_ids(gene_symbols: list[str]) -> dict[str, str]:
    """Resolve gene symbols to STRING identifiers."""
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


def query_cytokine_interactions(
    cytokine_string_ids: dict[str, str],
    network_gene_symbols: set[str],
) -> pd.DataFrame:
    """Query STRING for interactions between cytokine proteins and existing network.

    Queries each cytokine protein's first-degree interactions, then filters
    to keep only edges that connect to the existing autoantibody network.
    Also retains intra-cytokine-layer edges.
    """
    cytokine_symbols = set(cytokine_string_ids.keys())
    relevant_symbols = network_gene_symbols | cytokine_symbols
    all_rows: list[dict] = []

    for symbol, sid in cytokine_string_ids.items():
        url = f"{STRING_API_BASE}/json/network"
        params = {
            "identifiers": sid,
            "species": SPECIES_HUMAN,
            "required_score": MIN_SCORE,
            "add_nodes": 50,
            "network_type": "functional",
            "caller_identity": "bioagentics_pandas_pans",
        }

        try:
            resp = requests.get(url, params=params, timeout=60)
            resp.raise_for_status()
            data = resp.json()
        except requests.RequestException as e:
            logger.error("STRING query failed for cytokine %s: %s", symbol, e)
            time.sleep(RATE_LIMIT_DELAY)
            continue

        kept = 0
        for edge in data:
            src = edge.get("preferredName_A", "")
            tgt = edge.get("preferredName_B", "")

            # Keep if either end is in the existing network or cytokine layer
            if src in relevant_symbols or tgt in relevant_symbols:
                all_rows.append({
                    "source": src,
                    "target": tgt,
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
                    "layer": "cytokine_amplification",
                    "queried_cytokine": symbol,
                })
                kept += 1

        logger.info("Cytokine %s: %d/%d edges overlap with network",
                     symbol, kept, len(data))
        time.sleep(RATE_LIMIT_DELAY)

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    # Deduplicate by edge pair
    df["_edge_key"] = df.apply(
        lambda r: tuple(sorted([r["source"], r["target"]])), axis=1
    )
    df = df.drop_duplicates(subset=["_edge_key"]).drop(columns=["_edge_key"])
    return df.reset_index(drop=True)


def build_extended_network(
    combined_ppi_path: Path,
    cytokine_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge the autoantibody PPI network with the cytokine amplification layer.

    Annotates each edge with its layer:
      - 'autoantibody_ppi': original PPI network (Layer 1)
      - 'cytokine_amplification': cytokine layer edges (Layer 2)
      - 'cross_layer': edges connecting autoantibody targets to cytokine proteins
    """
    if not combined_ppi_path.exists():
        logger.error("Combined PPI file not found: %s", combined_ppi_path)
        return pd.DataFrame()

    ppi_df = pd.read_csv(combined_ppi_path, sep="\t")
    ppi_df["layer"] = "autoantibody_ppi"

    if cytokine_df.empty:
        return ppi_df

    # Determine which cytokine edges overlap with PPI network
    ppi_keys = set()
    for _, row in ppi_df.iterrows():
        ppi_keys.add(tuple(sorted([str(row["source"]), str(row["target"])])))

    new_rows = []
    upgraded_keys: set[tuple[str, str]] = set()
    for _, row in cytokine_df.iterrows():
        a, b = sorted([str(row["source"]), str(row["target"])])
        key = (a, b)
        if key in ppi_keys:
            upgraded_keys.add(key)
        else:
            new_rows.append(row.to_dict())

    # Update layer annotation for cross-layer edges
    def classify_layer(row: pd.Series) -> str:
        a, b = sorted([str(row["source"]), str(row["target"])])
        if (a, b) in upgraded_keys:
            return "cross_layer"
        return str(row.get("layer", "autoantibody_ppi"))

    ppi_df["layer"] = ppi_df.apply(classify_layer, axis=1)

    if new_rows:
        cytokine_only = pd.DataFrame(new_rows)
        # Ensure consistent columns
        for col in ppi_df.columns:
            if col not in cytokine_only.columns:
                cytokine_only[col] = None
        cytokine_only = cytokine_only[ppi_df.columns]
        return pd.concat([ppi_df, cytokine_only], ignore_index=True)

    return ppi_df


def compute_extended_stats(
    extended_df: pd.DataFrame,
    seed_symbols: list[str],
) -> dict:
    """Compute stats for the extended two-layer network."""
    if extended_df.empty:
        return {"total_edges": 0, "total_nodes": 0}

    all_nodes = set(extended_df["source"]) | set(extended_df["target"])
    seed_set = set(seed_symbols)
    cytokine_set = set(get_cytokine_gene_symbols())

    layer_counts = {}
    if "layer" in extended_df.columns:
        layer_counts = extended_df["layer"].value_counts().to_dict()

    # Cytokine proteins that appear in the network
    cytokine_in_network = cytokine_set & all_nodes

    # Cross-layer connections: edges where one end is a seed protein
    # and the other is a cytokine protein
    cross_connections = 0
    for _, row in extended_df.iterrows():
        s, t = str(row["source"]), str(row["target"])
        if (s in seed_set and t in cytokine_set) or (s in cytokine_set and t in seed_set):
            cross_connections += 1

    return {
        "total_edges": len(extended_df),
        "total_nodes": len(all_nodes),
        "seed_proteins_in_network": len(seed_set & all_nodes),
        "cytokine_proteins_in_network": sorted(cytokine_in_network),
        "unique_interactors": len(all_nodes - seed_set - cytokine_set),
        "layer_edge_counts": {str(k): int(v) for k, v in layer_counts.items()},
        "seed_cytokine_direct_connections": cross_connections,
        "cytokine_axes": {
            "IL17": [s for s in ["IL17A", "IL17RA", "IL17RC"] if s in all_nodes],
            "IFNG": [s for s in ["IFNG", "IFNGR1", "IFNGR2"] if s in all_nodes],
            "IL6": [s for s in ["IL6", "IL6R", "IL6ST"] if s in all_nodes],
            "JAK_STAT": [s for s in ["JAK1", "JAK2", "STAT1", "STAT3"] if s in all_nodes],
            "NFKB": [s for s in ["RELA"] if s in all_nodes],
        },
    }


def run() -> None:
    """Execute cytokine amplification layer pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    cytokine_symbols = get_cytokine_gene_symbols()

    # Step 1: Resolve STRING IDs for cytokine proteins
    logger.info("Step 1: Resolving STRING IDs for %d cytokine proteins", len(cytokine_symbols))
    string_id_map = resolve_string_ids(cytokine_symbols)
    logger.info("Resolved %d / %d cytokine proteins", len(string_id_map), len(cytokine_symbols))

    if not string_id_map:
        logger.error("No STRING IDs resolved for cytokine proteins")
        return

    # Save cytokine ID mapping
    id_path = DATA_DIR / "cytokine_string_id_mapping.json"
    with open(id_path, "w") as f:
        json.dump(string_id_map, f, indent=2)

    # Step 2: Get existing network node set
    combined_path = DATA_DIR / "combined_ppi_edgelist.tsv"
    if combined_path.exists():
        existing = pd.read_csv(combined_path, sep="\t")
        network_symbols = set(existing["source"]) | set(existing["target"])
    else:
        # Fall back to STRING-only
        string_path = DATA_DIR / "string_interactions.tsv"
        if string_path.exists():
            existing = pd.read_csv(string_path, sep="\t")
            network_symbols = set(existing["source"]) | set(existing["target"])
        else:
            logger.error("No PPI network file found")
            return

    logger.info("Existing network has %d unique proteins", len(network_symbols))

    # Step 3: Query cytokine interactions
    logger.info("Step 2: Querying STRING for cytokine layer interactions")
    cytokine_df = query_cytokine_interactions(string_id_map, network_symbols)
    logger.info("Cytokine layer: %d edges", len(cytokine_df))

    # Save cytokine interactions
    if not cytokine_df.empty:
        cyt_path = DATA_DIR / "cytokine_interactions.tsv"
        cytokine_df.to_csv(cyt_path, sep="\t", index=False)
        logger.info("Saved cytokine interactions to %s", cyt_path)

    # Step 4: Build extended two-layer network
    logger.info("Step 3: Building extended two-layer network")
    extended_df = build_extended_network(combined_path, cytokine_df)

    ext_path = DATA_DIR / "extended_network.tsv"
    extended_df.to_csv(ext_path, sep="\t", index=False)
    logger.info("Saved extended network to %s (%d edges)", ext_path, len(extended_df))

    # Step 5: Compute and save stats
    seed_symbols = get_gene_symbols()
    stats = compute_extended_stats(extended_df, seed_symbols)
    stats_path = DATA_DIR / "extended_network_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info("Extended network: %d nodes, %d edges", stats["total_nodes"], stats["total_edges"])
    logger.info("Layer breakdown: %s", stats["layer_edge_counts"])
    logger.info("Cytokine proteins in network: %s", stats["cytokine_proteins_in_network"])
    logger.info("Direct seed-cytokine connections: %d", stats["seed_cytokine_direct_connections"])


if __name__ == "__main__":
    run()
