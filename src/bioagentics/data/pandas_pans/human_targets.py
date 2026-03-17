"""Human basal ganglia protein target set for molecular mimicry screening.

Curates proteins enriched in basal ganglia (caudate, putamen, globus pallidus)
from Human Protein Atlas, adds known PANDAS autoantibody targets, and fetches
sequences from UniProt. Outputs FASTA + metadata CSV.

Usage:
    uv run python -m bioagentics.data.pandas_pans.human_targets [--dest DIR] [--force]
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping/human_targets")

TIMEOUT = 120
MAX_RETRIES = 3

UNIPROT_API = "https://rest.uniprot.org"

# Human Protein Atlas API
HPA_API = "https://www.proteinatlas.org/api"

# Basal ganglia brain regions to query from HPA
BASAL_GANGLIA_REGIONS = [
    "caudate",
    "putamen",
    "globus pallidus",
]

# Known PANDAS/PANS autoantibody targets with UniProt IDs.
# These are included regardless of HPA expression to serve as positive controls.
KNOWN_TARGETS: list[dict] = [
    {
        "gene": "DRD1",
        "uniprot_id": "P21728",
        "name": "Dopamine D1 receptor",
        "known_target": True,
        "evidence": "Cross-reactive antibodies in PANDAS sera (Bhattacharyya 2009, Ben-Pazi 2013)",
    },
    {
        "gene": "DRD2",
        "uniprot_id": "P14416",
        "name": "Dopamine D2 receptor",
        "known_target": True,
        "evidence": "Anti-neuronal antibodies in Sydenham chorea and PANDAS (Dale 2012)",
    },
    {
        "gene": "TUBA1A",
        "uniprot_id": "Q71U36",
        "name": "Tubulin alpha-1A chain",
        "known_target": True,
        "evidence": "Anti-tubulin antibodies elevated in PANDAS/SC (Singer 2005)",
    },
    {
        "gene": "TUBB3",
        "uniprot_id": "Q13509",
        "name": "Tubulin beta-3 chain",
        "known_target": True,
        "evidence": "Neuronal tubulin isoform, PANDAS autoantibody target",
    },
    {
        "gene": "CAMK2A",
        "uniprot_id": "Q9UQM7",
        "name": "CaM kinase II alpha",
        "known_target": True,
        "evidence": "CaMKII activation by PANDAS autoantibodies (Bhattacharyya 2009)",
    },
    {
        "gene": "LY6H",
        "uniprot_id": "O94772",
        "name": "Ly-6/neurotoxin-like protein 1",
        "known_target": True,
        "evidence": "Lysoganglioside-related neuronal surface antigen",
    },
    {
        "gene": "DNAL4",
        "uniprot_id": "O96015",
        "name": "Dynein light chain 4, axonemal",
        "known_target": True,
        "evidence": "Neuronal antigen identified in PANDAS sera (Chain 2020)",
    },
    {
        "gene": "GAPDH",
        "uniprot_id": "P04406",
        "name": "Glyceraldehyde-3-phosphate dehydrogenase",
        "known_target": True,
        "evidence": "GAS surface GAPDH mimics human GAPDH; molecular mimicry (Bhattacharyya 2009)",
    },
]


def _request_with_retry(url: str, params: dict | None = None,
                        headers: dict | None = None,
                        accept: str = "application/json") -> requests.Response:
    """HTTP GET with retries and rate-limit handling."""
    hdrs = {"Accept": accept}
    if headers:
        hdrs.update(headers)
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=hdrs, timeout=TIMEOUT)
            if resp.status_code == 429:
                wait = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, waiting %ds", wait)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Request failed (attempt %d), retry in %ds: %s",
                               attempt + 1, wait, e)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded")


def fetch_hpa_brain_enriched() -> list[dict]:
    """Fetch genes enriched in basal ganglia regions from Human Protein Atlas.

    Uses the HPA search API to find proteins with elevated expression in
    caudate, putamen, and globus pallidus.
    """
    all_genes: dict[str, dict] = {}

    for region in BASAL_GANGLIA_REGIONS:
        logger.info("Querying HPA for brain region: %s", region)
        url = f"{HPA_API}/search_download.php"
        params = {
            "search": f"brain_region_expression:{region}",
            "format": "json",
            "columns": "g,up,eg,d",
            "compress": "no",
        }
        try:
            resp = _request_with_retry(url, params=params)
            entries = resp.json() if resp.text.strip() else []
        except Exception:
            # HPA API can be unreliable; fall back to TSV download
            logger.warning("HPA JSON API failed for %s, trying TSV", region)
            entries = []

        for entry in entries:
            gene = entry.get("Gene", entry.get("g", ""))
            if not gene:
                continue
            uniprot_id = entry.get("Uniprot", entry.get("up", ""))
            if gene not in all_genes:
                all_genes[gene] = {
                    "gene": gene,
                    "uniprot_id": uniprot_id,
                    "name": entry.get("Gene description", entry.get("d", "")),
                    "brain_regions": [region],
                    "known_target": False,
                    "evidence": f"HPA brain enrichment: {region}",
                }
            else:
                all_genes[gene]["brain_regions"].append(region)
                all_genes[gene]["evidence"] += f", {region}"

    logger.info("HPA: %d unique genes across %d regions",
                len(all_genes), len(BASAL_GANGLIA_REGIONS))
    return list(all_genes.values())


def fetch_uniprot_brain_proteins() -> list[dict]:
    """Fetch human proteins with basal ganglia annotation from UniProt.

    Alternative/supplementary source to HPA using UniProt tissue annotations.
    """
    logger.info("Querying UniProt for human basal ganglia proteins...")
    url = f"{UNIPROT_API}/uniprotkb/search"

    # Query for human proteins annotated with basal ganglia tissues
    queries = [
        '(organism_id:9606) AND (tissue:"caudate nucleus")',
        '(organism_id:9606) AND (tissue:"putamen")',
        '(organism_id:9606) AND (tissue:"globus pallidus")',
        '(organism_id:9606) AND (tissue:"basal ganglia")',
        '(organism_id:9606) AND (tissue:"striatum")',
        '(organism_id:9606) AND (tissue:"corpus striatum")',
    ]

    all_genes: dict[str, dict] = {}

    for query in queries:
        region = query.split('"')[1]
        resp = _request_with_retry(
            url,
            params={"query": query, "format": "json", "size": 500,
                    "fields": "accession,gene_names,protein_name,cc_tissue_specificity"},
            accept="application/json",
        )
        results = resp.json().get("results", [])
        logger.info("  UniProt tissue:%s => %d proteins", region, len(results))

        for entry in results:
            accession = entry.get("primaryAccession", "")
            genes = entry.get("genes", [])
            gene_name = genes[0]["geneName"]["value"] if genes and "geneName" in genes[0] else ""
            prot_name = entry.get("proteinDescription", {}).get("recommendedName", {})
            full_name = prot_name.get("fullName", {}).get("value", "") if prot_name else ""

            if accession and accession not in all_genes:
                all_genes[accession] = {
                    "gene": gene_name,
                    "uniprot_id": accession,
                    "name": full_name,
                    "brain_regions": [region],
                    "known_target": False,
                    "evidence": f"UniProt tissue annotation: {region}",
                }
            elif accession in all_genes:
                all_genes[accession]["brain_regions"].append(region)

    logger.info("UniProt: %d unique proteins with basal ganglia annotations", len(all_genes))
    return list(all_genes.values())


def merge_targets(hpa_genes: list[dict], uniprot_genes: list[dict]) -> list[dict]:
    """Merge HPA, UniProt, and known targets into a deduplicated list."""
    by_uniprot: dict[str, dict] = {}
    by_gene: dict[str, dict] = {}

    def add(entry: dict) -> None:
        uid = entry.get("uniprot_id", "")
        gene = entry.get("gene", "")
        key = uid if uid else gene
        if not key:
            return

        if uid and uid in by_uniprot:
            existing = by_uniprot[uid]
            # Merge brain regions
            for r in entry.get("brain_regions", []):
                if r not in existing.get("brain_regions", []):
                    existing.setdefault("brain_regions", []).append(r)
            if entry.get("known_target"):
                existing["known_target"] = True
                existing["evidence"] = entry["evidence"]
        elif gene and gene in by_gene:
            existing = by_gene[gene]
            for r in entry.get("brain_regions", []):
                if r not in existing.get("brain_regions", []):
                    existing.setdefault("brain_regions", []).append(r)
            if entry.get("known_target"):
                existing["known_target"] = True
                existing["evidence"] = entry["evidence"]
            if not existing.get("uniprot_id") and uid:
                existing["uniprot_id"] = uid
        else:
            if uid:
                by_uniprot[uid] = entry
            if gene:
                by_gene[gene] = entry

    # Known targets first (highest priority)
    for t in KNOWN_TARGETS:
        t.setdefault("brain_regions", ["basal ganglia"])
        add(t)

    for g in uniprot_genes:
        add(g)

    for g in hpa_genes:
        add(g)

    # Deduplicate: prefer by_uniprot entries
    seen = set()
    merged = []
    for entry in list(by_uniprot.values()) + list(by_gene.values()):
        uid = entry.get("uniprot_id", "")
        gene = entry.get("gene", "")
        key = uid or gene
        if key not in seen:
            seen.add(key)
            if uid:
                seen.add(uid)
            if gene:
                seen.add(gene)
            merged.append(entry)

    return merged


def fetch_sequences(targets: list[dict]) -> dict[str, str]:
    """Fetch protein sequences from UniProt for all targets with UniProt IDs."""
    ids = [t["uniprot_id"] for t in targets if t.get("uniprot_id")]
    if not ids:
        return {}

    logger.info("Fetching sequences for %d proteins from UniProt...", len(ids))
    sequences: dict[str, str] = {}

    # Batch fetch in chunks of 100
    for i in range(0, len(ids), 100):
        batch = ids[i:i + 100]
        query = " OR ".join(f"accession:{acc}" for acc in batch)
        resp = _request_with_retry(
            f"{UNIPROT_API}/uniprotkb/stream",
            params={"query": query, "format": "fasta"},
            accept="text/plain",
        )

        current_id = ""
        current_seq: list[str] = []
        for line in resp.text.splitlines():
            if line.startswith(">"):
                if current_id and current_seq:
                    sequences[current_id] = "".join(current_seq)
                # Parse accession from >sp|P21728|DRD1_HUMAN or >tr|...
                parts = line.split("|")
                current_id = parts[1] if len(parts) >= 2 else line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line.strip())
        if current_id and current_seq:
            sequences[current_id] = "".join(current_seq)

        if i + 100 < len(ids):
            time.sleep(0.5)  # rate limiting

    logger.info("  Retrieved %d/%d sequences", len(sequences), len(ids))
    return sequences


def write_fasta(targets: list[dict], sequences: dict[str, str], dest: Path) -> int:
    """Write target proteins to FASTA file. Returns number of sequences written."""
    count = 0
    with open(dest, "w") as f:
        for t in targets:
            uid = t.get("uniprot_id", "")
            seq = sequences.get(uid, "")
            if not seq:
                continue
            gene = t.get("gene", "unknown")
            name = t.get("name", "")
            known = " [KNOWN_TARGET]" if t.get("known_target") else ""
            regions = ",".join(t.get("brain_regions", []))
            f.write(f">{uid}|{gene}|{regions}{known} {name}\n")
            # Write sequence in 60-char lines
            for j in range(0, len(seq), 60):
                f.write(seq[j:j + 60] + "\n")
            count += 1
    return count


def write_metadata_csv(targets: list[dict], dest: Path) -> None:
    """Write target metadata to CSV."""
    fieldnames = [
        "gene", "uniprot_id", "name", "brain_regions",
        "known_target", "evidence",
    ]
    with open(dest, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for t in targets:
            row = dict(t)
            row["brain_regions"] = ";".join(t.get("brain_regions", []))
            writer.writerow(row)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Curate human basal ganglia protein targets for mimicry screening",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    fasta_path = args.dest / "human_bg_targets.fasta"
    csv_path = args.dest / "target_metadata.csv"

    if not args.force and fasta_path.exists() and csv_path.exists():
        logger.info("Output files already exist. Use --force to regenerate.")
        return

    # Gather targets from multiple sources
    hpa_genes = fetch_hpa_brain_enriched()
    uniprot_genes = fetch_uniprot_brain_proteins()
    targets = merge_targets(hpa_genes, uniprot_genes)
    logger.info("Total targets after merge: %d (known: %d)",
                len(targets),
                sum(1 for t in targets if t.get("known_target")))

    # Fetch sequences
    sequences = fetch_sequences(targets)

    # Write outputs
    n_written = write_fasta(targets, sequences, fasta_path)
    logger.info("FASTA: %s (%d sequences)", fasta_path.name, n_written)

    write_metadata_csv(targets, csv_path)
    logger.info("Metadata: %s (%d entries)", csv_path.name, len(targets))


if __name__ == "__main__":
    main()
