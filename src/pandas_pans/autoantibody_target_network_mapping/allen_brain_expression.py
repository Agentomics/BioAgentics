"""Allen Human Brain Atlas expression overlay for PANDAS autoantibody network.

Retrieves brain region expression data from the Allen Human Brain Atlas API
for network proteins. Determines enrichment in basal ganglia, prefrontal
cortex, and thalamus — the brain regions implicated in PANDAS.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.allen_brain_expression

Output:
    data/pandas_pans/autoantibody_network/allen_expression_by_region.tsv
    data/pandas_pans/autoantibody_network/allen_region_enrichment.tsv
    data/pandas_pans/autoantibody_network/allen_expression_stats.json
"""

import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

ALLEN_API = "http://api.brain-map.org/api/v2/data/query.json"
DATA_DIR = Path("data/pandas_pans/autoantibody_network")
RATE_LIMIT_DELAY = 0.3

# Allen Human Brain Microarray product ID
HUMAN_MA_PRODUCT_ID = 2

# Donors available for Human Brain Microarray (pre-known)
DONOR_IDS = [15697, 14380, 15496, 9861, 10021, 12876]

# PANDAS-relevant brain regions (top-level structure names from Allen Atlas)
PANDAS_REGIONS = {
    "basal_ganglia": [
        "caudate nucleus",
        "putamen",
        "nucleus accumbens",
        "globus pallidus",
        "claustrum",
        "subthalamic nucleus",
    ],
    "prefrontal_cortex": [
        "orbital gyrus",
        "anterior orbital gyrus",
        "posterior orbital gyrus",
        "medial orbital gyrus",
        "lateral orbital gyrus",
        "gyrus rectus",
        "superior frontal gyrus",
        "middle frontal gyrus",
        "inferior frontal gyrus",
        "frontal pole",
        "prefrontal cortex",
    ],
    "thalamus": [
        "thalamus",
        "dorsal thalamus",
        "ventral thalamus",
        "mediodorsal nucleus",
        "pulvinar",
        "lateral geniculate",
        "medial geniculate",
    ],
    "cerebellum": [
        "cerebellar cortex",
        "cerebellum",
    ],
    "hippocampus": [
        "hippocampal formation",
        "hippocampus",
        "CA1 field",
        "CA2 field",
        "CA3 field",
        "CA4 field",
        "dentate gyrus",
        "subiculum",
        "parahippocampal gyrus",
    ],
}


def _allen_query(criteria: str, num_rows: int = 50, start_row: int = 0) -> dict:
    """Make an Allen API RMA query."""
    try:
        resp = requests.get(
            ALLEN_API,
            params={
                "criteria": criteria,
                "num_rows": num_rows,
                "start_row": start_row,
            },
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Allen API error: %s", e)
        return {"success": False, "msg": str(e)}


def _expression_query(probe_ids: list[int], donor_id: int) -> dict:
    """Query expression values for probes in a donor."""
    probes_str = ",".join(str(p) for p in probe_ids)
    criteria = (
        f"service::human_microarray_expression"
        f"[probes$eq{probes_str}]"
        f"[donors$eq{donor_id}]"
    )
    try:
        resp = requests.get(ALLEN_API, params={"criteria": criteria}, timeout=60)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        logger.error("Expression query failed: %s", e)
        return {"success": False, "msg": str(e)}


def get_network_nodes() -> set[str]:
    """Extract unique gene symbols from the extended network."""
    network_path = DATA_DIR / "extended_network.tsv"
    chunks = pd.read_csv(
        network_path, sep="\t", usecols=["source", "target"], chunksize=5000
    )
    nodes: set[str] = set()
    for chunk in chunks:
        nodes.update(chunk["source"].dropna())
        nodes.update(chunk["target"].dropna())
    return nodes


def resolve_gene_to_probes(gene_symbols: list[str]) -> dict[str, list[dict]]:
    """Resolve gene symbols to Allen probe IDs (batched).

    Returns gene_symbol -> list of {probe_id, gene_id} dicts.
    """
    logger.info("Resolving %d genes to Allen probe IDs...", len(gene_symbols))

    # Download all gene entries from Allen HumanMA in pages
    gene_map: dict[str, int] = {}  # acronym -> gene_id
    page = 0
    page_size = 2000
    while True:
        data = _allen_query(
            f"model::Gene,rma::criteria,products[id$eq{HUMAN_MA_PRODUCT_ID}]",
            num_rows=page_size,
            start_row=page * page_size,
        )
        if not data.get("success") or not data.get("msg"):
            break
        entries = data["msg"]
        if not isinstance(entries, list):
            break
        for g in entries:
            acronym = g.get("acronym", "")
            gid = g.get("id")
            if acronym and gid:
                gene_map[acronym] = gid
        if len(entries) < page_size:
            break
        page += 1
        time.sleep(RATE_LIMIT_DELAY)

    logger.info("Downloaded %d Allen gene entries", len(gene_map))

    # Match our network genes — prioritize seed proteins
    seed_set = set(get_gene_symbols())
    matched_genes = {}
    # Seeds first
    for sym in gene_symbols:
        if sym in gene_map and sym in seed_set:
            matched_genes[sym] = gene_map[sym]
    # Then the rest
    for sym in gene_symbols:
        if sym in gene_map and sym not in matched_genes:
            matched_genes[sym] = gene_map[sym]

    logger.info("Matched %d / %d network genes to Allen Atlas", len(matched_genes), len(gene_symbols))

    # Get probe IDs for matched genes
    # Limit to 300 genes to avoid excessive API calls (~5min instead of ~15min)
    gene_probes: dict[str, list[dict]] = {}
    gene_ids = list(matched_genes.items())[:300]
    logger.info("Querying probes for %d genes (limited from %d)", len(gene_ids), len(matched_genes))

    for i in range(0, len(gene_ids), 50):
        batch = gene_ids[i : i + 50]
        for sym, gid in batch:
            data = _allen_query(
                f"model::Probe,rma::criteria,[gene_id$eq{gid}],products[id$eq{HUMAN_MA_PRODUCT_ID}]",
                num_rows=50,
            )
            if data.get("success") and isinstance(data.get("msg"), list):
                probes = [
                    {"probe_id": p["id"], "gene_id": gid}
                    for p in data["msg"]
                    if isinstance(p, dict) and "id" in p
                ]
                if probes:
                    # Use first (delegate) probe only to avoid redundancy
                    gene_probes[sym] = probes[:1]
            time.sleep(RATE_LIMIT_DELAY)

        logger.info(
            "Resolved probes for %d / %d genes",
            min(i + 50, len(gene_ids)),
            len(gene_ids),
        )

    logger.info("Got probes for %d genes", len(gene_probes))
    return gene_probes


def classify_region(structure_name: str) -> str:
    """Classify a brain structure into PANDAS-relevant region categories."""
    name_lower = structure_name.lower()
    for region, keywords in PANDAS_REGIONS.items():
        for kw in keywords:
            if kw.lower() in name_lower:
                return region
    return "other"


def query_expression_batch(
    gene_probes: dict[str, list[dict]],
    donor_id: int,
) -> tuple[dict, list[dict]]:
    """Query expression for all genes in batches for one donor.

    Returns:
        structure_map: sample_index -> structure info (from first query)
        gene_expressions: list of {gene_symbol, expression_values} dicts
    """
    structure_map: dict = {}
    gene_expressions: list[dict] = []

    # Get structure mapping from first query
    probe_items = list(gene_probes.items())
    first_probe = probe_items[0][1][0]["probe_id"]
    first_result = _expression_query([first_probe], donor_id)

    if not first_result.get("success", True) or "msg" not in first_result:
        return {}, []

    samples = first_result["msg"].get("samples", [])
    structure_map = {
        i: {
            "structure_id": s.get("structure", {}).get("id", ""),
            "structure_name": s.get("structure", {}).get("name", ""),
            "structure_abbrev": s.get("structure", {}).get("abbreviation", ""),
            "top_level_name": s.get("top_level_structure", {}).get("name", ""),
            "top_level_abbrev": s.get("top_level_structure", {}).get("abbreviation", ""),
        }
        for i, s in enumerate(samples)
    }

    # Also save expression for first gene
    probes_data = first_result["msg"].get("probes", [])
    if probes_data:
        expr = probes_data[0].get("expression_level", [])
        gene_expressions.append({
            "gene_symbol": probe_items[0][0],
            "expression_values": [float(v) for v in expr],
        })

    time.sleep(RATE_LIMIT_DELAY)

    # Query remaining genes in batches of 20 probes
    batch_size = 20
    for i in range(1, len(probe_items), batch_size):
        batch = probe_items[i : i + batch_size]
        probe_ids = [items[1][0]["probe_id"] for items in batch]
        sym_map = {items[1][0]["probe_id"]: items[0] for items in batch}

        result = _expression_query(probe_ids, donor_id)
        if result.get("success", True) and "msg" in result:
            for probe_data in result["msg"].get("probes", []):
                pid = probe_data.get("id")
                sym = sym_map.get(pid, probe_data.get("gene-symbol", ""))
                expr = probe_data.get("expression_level", [])
                if expr:
                    gene_expressions.append({
                        "gene_symbol": sym,
                        "expression_values": [float(v) for v in expr],
                    })

        time.sleep(RATE_LIMIT_DELAY)

    return structure_map, gene_expressions


def compute_region_expression(
    structure_map: dict,
    gene_expressions: list[dict],
    seed_symbols: set[str],
) -> pd.DataFrame:
    """Compute mean expression per gene per brain region category.

    Returns DataFrame: gene_symbol, region, mean_expression, z_score, is_seed
    """
    # Classify each sample into region category
    sample_regions = {
        i: classify_region(info["structure_name"])
        for i, info in structure_map.items()
    }

    rows = []
    for ge in gene_expressions:
        sym = ge["gene_symbol"]
        values = ge["expression_values"]
        if not values:
            continue

        # Group expression values by region
        region_values: dict[str, list[float]] = {}
        for i, v in enumerate(values):
            region = sample_regions.get(i, "other")
            region_values.setdefault(region, []).append(v)

        # Compute global stats for z-score
        all_vals = np.array(values, dtype=float)
        global_mean = float(np.mean(all_vals))
        global_std = float(np.std(all_vals))

        for region, vals in region_values.items():
            region_mean = float(np.mean(vals))
            z = (region_mean - global_mean) / global_std if global_std > 0 else 0.0

            rows.append({
                "gene_symbol": sym,
                "region": region,
                "mean_expression": round(region_mean, 4),
                "z_score": round(z, 4),
                "n_samples": len(vals),
                "is_seed": sym in seed_symbols,
            })

    return pd.DataFrame(rows)


def compute_region_enrichment(
    region_df: pd.DataFrame,
    seed_symbols: set[str],
) -> pd.DataFrame:
    """Compute whether the network is enriched in PANDAS-relevant regions.

    For each target region, compare mean expression of network genes in that
    region vs other regions using a paired t-test.
    """
    from scipy import stats as scipy_stats

    target_regions = ["basal_ganglia", "prefrontal_cortex", "thalamus"]
    rows = []

    for target in target_regions:
        target_data = region_df[region_df["region"] == target]
        other_data = region_df[region_df["region"] == "other"]

        if target_data.empty or other_data.empty:
            continue

        # Get genes present in both target and other regions
        common_genes = set(target_data["gene_symbol"]) & set(other_data["gene_symbol"])
        if len(common_genes) < 5:
            continue

        target_means = target_data[target_data["gene_symbol"].isin(common_genes)].set_index(
            "gene_symbol"
        )["mean_expression"]
        other_means = other_data[other_data["gene_symbol"].isin(common_genes)].set_index(
            "gene_symbol"
        )["mean_expression"]

        # Align
        common = sorted(common_genes)
        t_vals = target_means.reindex(common).values
        o_vals = other_means.reindex(common).values

        t_stat, p_val = scipy_stats.ttest_rel(t_vals, o_vals)

        # Seed protein enrichment
        seed_genes = common_genes & seed_symbols
        seed_target_mean = float(
            target_data[target_data["gene_symbol"].isin(seed_genes)]["mean_expression"].mean()
        ) if seed_genes else None
        seed_other_mean = float(
            other_data[other_data["gene_symbol"].isin(seed_genes)]["mean_expression"].mean()
        ) if seed_genes else None

        # Genes with high z-score in target region
        high_z = target_data[target_data["z_score"] > 1.0]["gene_symbol"].tolist()

        rows.append({
            "target_region": target,
            "n_genes": len(common_genes),
            "mean_expression_target": round(float(np.mean(t_vals)), 4),
            "mean_expression_other": round(float(np.mean(o_vals)), 4),
            "expression_ratio": round(float(np.mean(t_vals) / np.mean(o_vals)), 4)
            if np.mean(o_vals) != 0
            else 0,
            "t_statistic": round(float(t_stat), 4),
            "p_value": float(p_val),
            "seed_mean_target": seed_target_mean,
            "seed_mean_other": seed_other_mean,
            "n_seeds_in_region": len(seed_genes),
            "high_z_genes": ",".join(sorted(high_z)[:20]),
            "n_high_z_genes": len(high_z),
        })

    return pd.DataFrame(rows)


def run() -> None:
    """Execute the Allen Brain Atlas expression overlay pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_symbols = set(get_gene_symbols())

    # Step 1: Get network nodes
    logger.info("Step 1: Loading network nodes")
    network_nodes = get_network_nodes()
    logger.info("Found %d network nodes", len(network_nodes))

    # Step 2: Resolve genes to Allen probe IDs
    logger.info("Step 2: Resolving genes to Allen Atlas probe IDs")
    gene_probes = resolve_gene_to_probes(sorted(network_nodes))

    if not gene_probes:
        logger.error("No genes resolved to Allen probes")
        return

    # Step 3: Query expression for one representative donor
    # Using first donor (15697) for efficiency
    donor_id = DONOR_IDS[0]
    logger.info("Step 3: Querying expression data (donor %d)", donor_id)
    structure_map, gene_expressions = query_expression_batch(
        gene_probes, donor_id
    )

    if not gene_expressions:
        logger.error("No expression data retrieved")
        return

    logger.info("Got expression data for %d genes across %d samples",
                len(gene_expressions), len(structure_map))

    # Step 4: Compute region-level expression
    logger.info("Step 4: Computing region-level expression")
    region_df = compute_region_expression(structure_map, gene_expressions, seed_symbols)

    region_path = DATA_DIR / "allen_expression_by_region.tsv"
    region_df.to_csv(region_path, sep="\t", index=False)
    logger.info("Saved region expression to %s", region_path)

    # Step 5: Region enrichment analysis
    logger.info("Step 5: Computing PANDAS region enrichment")
    enrichment_df = compute_region_enrichment(region_df, seed_symbols)

    if not enrichment_df.empty:
        enrichment_path = DATA_DIR / "allen_region_enrichment.tsv"
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        logger.info("Saved enrichment analysis to %s", enrichment_path)

    # Step 6: Save stats
    mapped_genes = set(
        ge["gene_symbol"] for ge in gene_expressions
    )
    seed_mapped = mapped_genes & seed_symbols

    bg_data = region_df[region_df["region"] == "basal_ganglia"]
    bg_enriched = bg_data[bg_data["z_score"] > 0.5] if not bg_data.empty else pd.DataFrame()

    stats = {
        "total_network_nodes": len(network_nodes),
        "genes_with_allen_probes": len(gene_probes),
        "genes_with_expression_data": len(mapped_genes),
        "seed_proteins_mapped": sorted(seed_mapped),
        "donor_used": donor_id,
        "n_brain_samples": len(structure_map),
        "regions_analyzed": list(PANDAS_REGIONS.keys()),
        "basal_ganglia_enriched_genes": len(bg_enriched),
    }

    if not enrichment_df.empty:
        stats["enrichment_results"] = {}
        for _, row in enrichment_df.iterrows():
            stats["enrichment_results"][row["target_region"]] = {
                "n_genes": int(row["n_genes"]),
                "expression_ratio": float(row["expression_ratio"]),
                "p_value": float(row["p_value"]),
                "n_seeds": int(row["n_seeds_in_region"]),
                "n_high_z_genes": int(row["n_high_z_genes"]),
            }

    stats_path = DATA_DIR / "allen_expression_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info("=== Allen Brain Atlas Expression Summary ===")
    logger.info("Genes with expression data: %d / %d network nodes",
                len(mapped_genes), len(network_nodes))
    logger.info("Seed proteins mapped: %s", ", ".join(sorted(seed_mapped)))
    if not enrichment_df.empty:
        for _, row in enrichment_df.iterrows():
            sig = "***" if row["p_value"] < 0.05 else ""
            logger.info(
                "  %s: ratio=%.3f, t=%.2f, p=%.4f, high_z=%d %s",
                row["target_region"],
                row["expression_ratio"],
                row["t_statistic"],
                row["p_value"],
                row["n_high_z_genes"],
                sig,
            )


if __name__ == "__main__":
    run()
