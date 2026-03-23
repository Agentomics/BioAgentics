"""GAS (Group A Streptococcus) proteome download from UniProt.

Downloads complete proteomes for S. pyogenes serotypes associated with
PANDAS/rheumatic fever (M1, M3, M5, M12, M18, M49) from the UniProt REST API.
Outputs per-serotype FASTA, combined FASTA, and metadata CSV.

Usage:
    uv run python -m bioagentics.data.pandas_pans.gas_proteomes [--dest DIR] [--force]
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping/proteomes")

TIMEOUT = 120
MAX_RETRIES = 3

# UniProt REST API base
UNIPROT_API = "https://rest.uniprot.org"

# GAS serotypes → UniProt reference proteome IDs and representative strains.
# M49 includes emm49.8, newly dominant in England 2025-26 season (UK HSA).
SEROTYPES: dict[str, dict] = {
    "M1": {
        "proteome_id": "UP000000750",
        "strain": "SF370 / ATCC 700294",
        "taxonomy_id": 301447,
        "emm_type": "emm1",
        "notes": "Most common invasive GAS globally; rheumatic fever & PANDAS-associated",
    },
    "M3": {
        "proteome_id": "UP000000564",
        "strain": "MGAS315",
        "taxonomy_id": 198466,
        "emm_type": "emm3",
        "notes": "Severe invasive disease; necrotizing fasciitis & streptococcal TSS",
    },
    "M5": {
        "proteome_id": "UP000002591",
        "strain": "Manfredo",
        "taxonomy_id": 392612,
        "emm_type": "emm5",
        "notes": "Classic rheumatic fever serotype",
    },
    "M12": {
        "proteome_id": "UP000002433",
        "strain": "MGAS9429",
        "taxonomy_id": 370553,
        "emm_type": "emm12",
        "notes": "Pharyngitis-associated; PANDAS studies",
    },
    "M18": {
        "proteome_id": "UP000000820",
        "strain": "MGAS8232",
        "taxonomy_id": 186103,
        "emm_type": "emm18",
        "notes": "Rheumatic fever outbreaks; strong rheumatogenic potential",
    },
    "M49": {
        "proteome_id": "UP000001039",
        "strain": "NZ131",
        "taxonomy_id": 471876,
        "emm_type": "emm49",
        "notes": "emm49.8 newly dominant in England 2025-26 (UK HSA); skin & invasive",
    },
}


def _uniprot_request(url: str, params: dict | None = None,
                     accept: str = "text/plain") -> requests.Response:
    """Make a UniProt REST API request with retries and rate-limit handling."""
    headers = {"Accept": accept}
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(url, params=params, headers=headers, timeout=TIMEOUT)
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, waiting %ds", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("UniProt request failed (attempt %d), retry in %ds: %s",
                               attempt + 1, wait, e)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Max retries exceeded for UniProt request")


def _count_fasta_entries(text: str) -> int:
    """Count FASTA entries (lines starting with '>')."""
    return sum(1 for line in text.splitlines() if line.startswith(">"))


def download_proteome_fasta(proteome_id: str, taxonomy_id: int, dest: Path,
                            min_proteins: int = 1500) -> tuple[int, str]:
    """Download a complete proteome as FASTA from UniProt. Returns (protein_count, source).

    Tries three sources in order:
      1. UniProtKB proteome query
      2. UniProtKB organism_id query
      3. UniParc proteome query (for redundant/merged proteomes)

    Each source must yield at least *min_proteins* entries to be accepted
    (S. pyogenes genomes encode ~1700-1900 proteins).

    Skips download if file already exists.
    """
    if dest.exists() and dest.stat().st_size > 0:
        count = _count_fasta_entries(dest.read_text())
        logger.info("  Already exists: %s (%d proteins)", dest.name, count)
        return count, "cached"

    dest.parent.mkdir(parents=True, exist_ok=True)

    # Strategy 1: UniProtKB by proteome ID
    logger.info("  Trying UniProtKB proteome:%s ...", proteome_id)
    resp = _uniprot_request(
        f"{UNIPROT_API}/uniprotkb/stream",
        params={"query": f"(proteome:{proteome_id})", "format": "fasta"},
        accept="text/plain",
    )
    if resp.text.strip():
        count = _count_fasta_entries(resp.text)
        if count >= min_proteins:
            dest.write_text(resp.text)
            logger.info("  Saved %s (%d proteins, source: uniprotkb/proteome)", dest.name, count)
            return count, "uniprotkb/proteome"

    # Strategy 2: UniProtKB by organism taxonomy ID
    logger.info("  Trying UniProtKB organism_id:%d ...", taxonomy_id)
    resp = _uniprot_request(
        f"{UNIPROT_API}/uniprotkb/stream",
        params={"query": f"(organism_id:{taxonomy_id})", "format": "fasta"},
        accept="text/plain",
    )
    if resp.text.strip():
        count = _count_fasta_entries(resp.text)
        if count >= min_proteins:
            dest.write_text(resp.text)
            logger.info("  Saved %s (%d proteins, source: uniprotkb/organism)", dest.name, count)
            return count, "uniprotkb/organism"

    # Strategy 3: UniParc (captures redundant/merged proteomes)
    logger.info("  Trying UniParc upid:%s ...", proteome_id)
    resp = _uniprot_request(
        f"{UNIPROT_API}/uniparc/stream",
        params={"query": f"(upid:{proteome_id})", "format": "fasta"},
        accept="text/plain",
    )
    if resp.text.strip():
        count = _count_fasta_entries(resp.text)
        if count >= min_proteins:
            dest.write_text(resp.text)
            logger.info("  Saved %s (%d proteins, source: uniparc)", dest.name, count)
            return count, "uniparc"

    raise ValueError(
        f"Could not download proteome {proteome_id} (taxid {taxonomy_id}) "
        "from any UniProt source"
    )


def download_all_serotypes(dest_dir: Path, force: bool = False) -> list[dict]:
    """Download proteomes for all configured GAS serotypes.

    Returns list of metadata dicts (one per serotype).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    metadata = []

    for serotype, info in SEROTYPES.items():
        logger.info("Serotype %s (%s) — %s", serotype, info["emm_type"], info["strain"])
        fasta_path = dest_dir / f"gas_{serotype.lower()}.fasta"

        if force and fasta_path.exists():
            fasta_path.unlink()

        protein_count, source = download_proteome_fasta(
            info["proteome_id"], info["taxonomy_id"], fasta_path,
        )

        metadata.append({
            "serotype": serotype,
            "emm_type": info["emm_type"],
            "strain": info["strain"],
            "taxonomy_id": info["taxonomy_id"],
            "proteome_id": info["proteome_id"],
            "protein_count": protein_count,
            "fasta_file": fasta_path.name,
            "source": source,
            "notes": info["notes"],
        })

    return metadata


def build_combined_fasta(dest_dir: Path) -> Path:
    """Concatenate all per-serotype FASTAs + supplements into a single combined file.

    Includes enn_mrp_supplement.fasta if present (Enn/Mrp M-like proteins).
    """
    combined = dest_dir / "gas_combined.fasta"
    parts = sorted(dest_dir.glob("gas_m*.fasta"))
    if not parts:
        raise FileNotFoundError(f"No per-serotype FASTAs found in {dest_dir}")

    with open(combined, "w") as out:
        for p in parts:
            text = p.read_text()
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")

        # Include Enn/Mrp supplement if available
        supplement = dest_dir / "enn_mrp_supplement.fasta"
        if supplement.exists():
            text = supplement.read_text()
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            logger.info("Included Enn/Mrp supplement in combined FASTA")

    total = sum(1 for line in combined.read_text().splitlines() if line.startswith(">"))
    logger.info("Combined FASTA: %s (%d proteins total)", combined.name, total)
    return combined


def write_metadata_csv(metadata: list[dict], dest_dir: Path) -> Path:
    """Write serotype metadata to CSV."""
    csv_path = dest_dir / "serotype_metadata.csv"
    fieldnames = [
        "serotype", "emm_type", "strain", "taxonomy_id",
        "proteome_id", "protein_count", "fasta_file", "source", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    logger.info("Metadata CSV: %s", csv_path.name)
    return csv_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download GAS proteomes from UniProt for molecular mimicry analysis",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Downloading GAS proteomes to %s", args.dest)
    metadata = download_all_serotypes(args.dest, force=args.force)
    build_combined_fasta(args.dest)
    write_metadata_csv(metadata, args.dest)

    total_proteins = sum(m["protein_count"] for m in metadata)
    logger.info("Done: %d serotypes, %d total proteins", len(metadata), total_proteins)


if __name__ == "__main__":
    main()
