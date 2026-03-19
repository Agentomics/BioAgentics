"""DisGeNET symptom-domain mapping for PANDAS autoantibody network nodes.

Cross-references network proteins with disease-gene associations for
PANDAS-relevant symptom domains using Open Targets Platform API (which
integrates DisGeNET, GWAS Catalog, and other sources).

Symptom domains: OCD (compulsive behavior), tic disorders (motor circuits),
anxiety disorders, eating disorders, cognitive regression, emotional lability.

Usage:
    uv run python -m pandas_pans.autoantibody_target_network_mapping.disgenet_symptom_mapping

Output:
    data/pandas_pans/autoantibody_network/symptom_domain_mapping.tsv
    data/pandas_pans/autoantibody_network/symptom_domain_enrichment.tsv
    data/pandas_pans/autoantibody_network/symptom_mapping_stats.json
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests
from scipy import stats as scipy_stats

from pandas_pans.autoantibody_target_network_mapping.seed_proteins import (
    SEED_PROTEINS,
    get_gene_symbols,
)

logger = logging.getLogger(__name__)

OPENTARGETS_API = "https://api.platform.opentargets.org/api/v4/graphql"
DATA_DIR = Path("data/pandas_pans/autoantibody_network")
RATE_LIMIT_DELAY = 0.2

# PANDAS-relevant symptom domain disease terms
# Each domain maps to EFO/MONDO disease IDs and keyword patterns
SYMPTOM_DOMAINS = {
    "ocd_compulsive": {
        "keywords": [
            "obsessive-compulsive",
            "obsessive compulsive",
            "compulsive behavior",
            "OCD",
        ],
        "description": "OCD and compulsive behaviors",
    },
    "tic_motor": {
        "keywords": [
            "tic disorder",
            "Tourette",
            "motor tic",
            "vocal tic",
            "movement disorder",
            "chorea",
            "dystonia",
            "dyskinesia",
        ],
        "description": "Tic disorders and motor circuits",
    },
    "anxiety": {
        "keywords": [
            "anxiety",
            "generalized anxiety",
            "separation anxiety",
            "panic disorder",
            "phobia",
            "social anxiety",
        ],
        "description": "Anxiety disorders",
    },
    "eating_restriction": {
        "keywords": [
            "eating disorder",
            "anorexia",
            "food restriction",
            "avoidant/restrictive",
            "ARFID",
            "feeding disorder",
        ],
        "description": "Eating disorders and food restriction",
    },
    "cognitive": {
        "keywords": [
            "cognitive",
            "intellectual disability",
            "learning disability",
            "attention deficit",
            "ADHD",
            "developmental delay",
            "mental retardation",
        ],
        "description": "Cognitive regression and attention deficits",
    },
    "emotional_lability": {
        "keywords": [
            "emotional",
            "mood disorder",
            "depression",
            "bipolar",
            "irritability",
            "rage",
            "affective",
        ],
        "description": "Emotional lability and mood disorders",
    },
    "autoimmune_neuropsych": {
        "keywords": [
            "autoimmune",
            "encephalitis",
            "PANDAS",
            "PANS",
            "Sydenham",
            "rheumatic",
            "autoantibody",
            "neuroinflammation",
        ],
        "description": "Autoimmune and neuropsychiatric conditions",
    },
    "dopaminergic": {
        "keywords": [
            "Parkinson",
            "dopamin",
            "substantia nigra",
            "basal ganglia",
            "schizophrenia",
            "psychosis",
        ],
        "description": "Dopaminergic system disorders",
    },
}


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


def resolve_ensembl_ids(gene_symbols: list[str]) -> dict[str, str]:
    """Resolve gene symbols to Ensembl gene IDs using Open Targets.

    Queries Open Targets search endpoint for each gene symbol.
    Returns symbol -> ensembl_id mapping.
    """
    logger.info("Resolving %d gene symbols to Ensembl IDs...", len(gene_symbols))

    query = """
    query searchGene($symbol: String!) {
      search(queryString: $symbol, entityNames: ["target"], page: {index: 0, size: 1}) {
        hits {
          id
          entity
          name
          description
        }
      }
    }
    """

    resolved: dict[str, str] = {}
    batch_size = 50

    for i in range(0, len(gene_symbols), batch_size):
        batch = gene_symbols[i : i + batch_size]
        for sym in batch:
            try:
                resp = requests.post(
                    OPENTARGETS_API,
                    json={"query": query, "variables": {"symbol": sym}},
                    timeout=30,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    hits = data.get("data", {}).get("search", {}).get("hits", [])
                    for hit in hits:
                        if hit.get("entity") == "target":
                            # Verify the symbol matches
                            hit_name = hit.get("name", "")
                            if hit_name.upper() == sym.upper() or sym.upper() in hit.get("description", "").upper():
                                resolved[sym] = hit["id"]
                                break
            except requests.RequestException:
                pass
            time.sleep(RATE_LIMIT_DELAY)

        logger.info(
            "Resolved %d / %d symbols",
            min(i + batch_size, len(gene_symbols)),
            len(gene_symbols),
        )

    logger.info("Resolved %d / %d to Ensembl IDs", len(resolved), len(gene_symbols))
    return resolved


def query_disease_associations_batch(
    ensembl_ids: dict[str, str],
) -> dict[str, list[dict]]:
    """Query Open Targets for disease associations for each gene.

    Returns symbol -> list of {disease_name, disease_id, score} dicts.
    """
    logger.info("Querying disease associations for %d genes...", len(ensembl_ids))

    query = """
    query geneAssociations($ensemblId: String!) {
      target(ensemblId: $ensemblId) {
        approvedSymbol
        associatedDiseases(page: {index: 0, size: 200}) {
          rows {
            disease {
              id
              name
            }
            score
          }
        }
      }
    }
    """

    gene_diseases: dict[str, list[dict]] = {}
    items = list(ensembl_ids.items())

    for i, (sym, eid) in enumerate(items):
        try:
            resp = requests.post(
                OPENTARGETS_API,
                json={"query": query, "variables": {"ensemblId": eid}},
                timeout=30,
            )
            if resp.status_code == 200:
                data = resp.json()
                target = data.get("data", {}).get("target", {})
                rows = target.get("associatedDiseases", {}).get("rows", [])
                diseases = []
                for r in rows:
                    d = r.get("disease", {})
                    diseases.append({
                        "disease_name": d.get("name", ""),
                        "disease_id": d.get("id", ""),
                        "score": r.get("score", 0),
                    })
                if diseases:
                    gene_diseases[sym] = diseases
        except requests.RequestException:
            pass
        time.sleep(RATE_LIMIT_DELAY)

        if (i + 1) % 100 == 0:
            logger.info("Queried %d / %d genes", i + 1, len(items))

    logger.info("Got disease associations for %d genes", len(gene_diseases))
    return gene_diseases


def classify_disease_to_domains(disease_name: str) -> list[str]:
    """Classify a disease name into PANDAS symptom domains."""
    domains = []
    name_lower = disease_name.lower()
    for domain, info in SYMPTOM_DOMAINS.items():
        for kw in info["keywords"]:
            if kw.lower() in name_lower:
                domains.append(domain)
                break
    return domains


def build_symptom_mapping(
    gene_diseases: dict[str, list[dict]],
    seed_symbols: set[str],
) -> pd.DataFrame:
    """Build gene -> symptom domain mapping table."""
    rows = []
    for sym, diseases in gene_diseases.items():
        for d in diseases:
            domains = classify_disease_to_domains(d["disease_name"])
            if domains:
                for domain in domains:
                    rows.append({
                        "gene_symbol": sym,
                        "disease_name": d["disease_name"],
                        "disease_id": d["disease_id"],
                        "association_score": d["score"],
                        "symptom_domain": domain,
                        "is_seed": sym in seed_symbols,
                    })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["symptom_domain", "association_score"], ascending=[True, False])
    return df


def compute_domain_enrichment(
    mapping_df: pd.DataFrame,
    network_genes: set[str],
    gene_diseases: dict[str, list[dict]],
    seed_symbols: set[str],
) -> pd.DataFrame:
    """Compute whether PANDAS symptom domains are enriched in the network."""
    if mapping_df.empty:
        return pd.DataFrame()

    rows = []
    for domain, info in SYMPTOM_DOMAINS.items():
        domain_genes = set(mapping_df[mapping_df["symptom_domain"] == domain]["gene_symbol"])
        seed_in_domain = domain_genes & seed_symbols
        total_genes_with_assoc = len(gene_diseases)

        if not domain_genes:
            continue

        # Mean association score for this domain
        domain_scores = mapping_df[mapping_df["symptom_domain"] == domain]["association_score"]
        mean_score = float(domain_scores.mean())
        max_score = float(domain_scores.max())

        # Top genes
        top_genes = (
            mapping_df[mapping_df["symptom_domain"] == domain]
            .drop_duplicates(subset=["gene_symbol"])
            .nlargest(10, "association_score")
        )

        rows.append({
            "symptom_domain": domain,
            "description": info["description"],
            "n_network_genes": len(domain_genes),
            "pct_of_queried": round(100 * len(domain_genes) / total_genes_with_assoc, 2)
            if total_genes_with_assoc > 0
            else 0,
            "n_seed_proteins": len(seed_in_domain),
            "seed_list": ",".join(sorted(seed_in_domain)),
            "mean_association_score": round(mean_score, 4),
            "max_association_score": round(max_score, 4),
            "top_genes": ",".join(top_genes["gene_symbol"].tolist()),
        })

    return pd.DataFrame(rows).sort_values("n_network_genes", ascending=False)


def run() -> None:
    """Execute the DisGeNET symptom-domain mapping pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(message)s",
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    seed_symbols = set(get_gene_symbols())

    # Step 1: Get network nodes
    logger.info("Step 1: Loading network nodes")
    network_nodes = get_network_nodes()
    # Filter to valid gene symbols (alphanumeric)
    clean_nodes = sorted([
        n for n in network_nodes
        if n.replace("-", "").replace("_", "").isalnum() and " " not in n
    ])
    logger.info("Found %d network nodes (%d clean gene symbols)", len(network_nodes), len(clean_nodes))

    # Step 2: Resolve to Ensembl IDs
    # For efficiency, prioritize seed proteins + high-degree nodes
    # Start with seed proteins, then sample remaining
    priority_genes = list(seed_symbols) + [g for g in clean_nodes if g not in seed_symbols]
    # Limit to first 500 genes to avoid excessive API calls
    genes_to_query = priority_genes[:500]

    logger.info("Step 2: Resolving %d priority genes to Ensembl IDs", len(genes_to_query))
    ensembl_map = resolve_ensembl_ids(genes_to_query)

    if not ensembl_map:
        logger.error("No genes resolved to Ensembl IDs")
        return

    # Step 3: Query disease associations
    logger.info("Step 3: Querying disease associations")
    gene_diseases = query_disease_associations_batch(ensembl_map)

    if not gene_diseases:
        logger.error("No disease associations found")
        return

    # Step 4: Build symptom domain mapping
    logger.info("Step 4: Building symptom domain mapping")
    mapping_df = build_symptom_mapping(gene_diseases, seed_symbols)

    if mapping_df.empty:
        logger.warning("No genes mapped to PANDAS symptom domains")
    else:
        mapping_path = DATA_DIR / "symptom_domain_mapping.tsv"
        mapping_df.to_csv(mapping_path, sep="\t", index=False)
        logger.info("Saved %d symptom mappings to %s", len(mapping_df), mapping_path)

    # Step 5: Domain enrichment
    logger.info("Step 5: Computing domain enrichment")
    enrichment_df = compute_domain_enrichment(
        mapping_df, network_nodes, gene_diseases, seed_symbols
    )

    if not enrichment_df.empty:
        enrichment_path = DATA_DIR / "symptom_domain_enrichment.tsv"
        enrichment_df.to_csv(enrichment_path, sep="\t", index=False)
        logger.info("Saved domain enrichment to %s", enrichment_path)

    # Step 6: Save stats
    mapped_domains = set(mapping_df["symptom_domain"]) if not mapping_df.empty else set()

    stats = {
        "total_network_nodes": len(network_nodes),
        "genes_queried": len(genes_to_query),
        "genes_resolved_to_ensembl": len(ensembl_map),
        "genes_with_disease_associations": len(gene_diseases),
        "symptom_domains_found": len(mapped_domains),
        "total_symptom_mappings": len(mapping_df),
        "seed_proteins_with_symptom_mapping": len(
            set(mapping_df[mapping_df["is_seed"]]["gene_symbol"]) if not mapping_df.empty else set()
        ),
    }

    # Per-domain summary
    if not enrichment_df.empty:
        stats["domain_summary"] = {}
        for _, row in enrichment_df.iterrows():
            stats["domain_summary"][row["symptom_domain"]] = {
                "n_genes": int(row["n_network_genes"]),
                "seed_proteins": row["seed_list"],
                "mean_score": float(row["mean_association_score"]),
                "top_genes": row["top_genes"],
            }

    # Check if different autoantibody targets map to different symptom pathways
    if not mapping_df.empty:
        seed_domain_map = {}
        for seed in seed_symbols:
            seed_data = mapping_df[mapping_df["gene_symbol"] == seed]
            if not seed_data.empty:
                seed_domain_map[seed] = sorted(seed_data["symptom_domain"].unique().tolist())
        stats["seed_symptom_domain_mapping"] = seed_domain_map

    stats_path = DATA_DIR / "symptom_mapping_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    # Summary
    logger.info("=== Symptom Domain Mapping Summary ===")
    logger.info("Genes with disease associations: %d", len(gene_diseases))
    logger.info("Symptom domains covered: %d / %d", len(mapped_domains), len(SYMPTOM_DOMAINS))
    if not enrichment_df.empty:
        for _, row in enrichment_df.iterrows():
            seed_info = f" (seeds: {row['seed_list']})" if row["seed_list"] else ""
            logger.info(
                "  %s: %d genes, mean_score=%.3f%s",
                row["symptom_domain"],
                row["n_network_genes"],
                row["mean_association_score"],
                seed_info,
            )


if __name__ == "__main__":
    run()
