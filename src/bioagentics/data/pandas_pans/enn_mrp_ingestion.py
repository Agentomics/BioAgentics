"""Enn/Mrp M-like protein ingestion for GAS molecular mimicry pipeline.

Downloads and integrates Enn (IgA-binding M-like) and Mrp (M-related protein)
sequences into the GAS proteome inputs. These surface proteins are encoded in
the mga regulon alongside emm and play key roles in immune evasion by binding
host immunoglobulins (IgA-Fc binding for Enn, IgG-Fc binding for Mrp).

Reference: Akoolo et al. 2026, Microbiology Spectrum, PMID 41236364.

Not all GAS serotypes carry enn/mrp:
  - M1/SF370: minimal mga locus, lacks mrp and enn
  - M3, M5, M12, M18: typically carry both mrp and enn alleles
  - M49/NZ131: carries ennX (already in reference proteome)

Usage:
    uv run python -m bioagentics.data.pandas_pans.enn_mrp_ingestion [--dest DIR]
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
PROTEOME_DIR = PROJECT_DIR / "proteomes"

TIMEOUT = 60
MAX_RETRIES = 3

UNIPROT_API = "https://rest.uniprot.org"

# Well-characterized Enn/Mrp protein accessions from UniProt.
# These are representative sequences from S. pyogenes isolates with
# confirmed gene annotations for the mga regulon M-like proteins.
#
# Mrp (M-related protein): surface-anchored, binds IgG Fc and IgA Fc.
#   Encoded upstream of emm in the mga regulon in most serotypes.
# Enn (protein Enn): surface-anchored, binds IgA Fc.
#   Encoded downstream of emm in the mga regulon.
#
# Sources: UniProt reviewed/unreviewed entries with gene-level annotation.
# Cross-referenced with Akoolo et al. 2026 serotype characterization.
ENN_MRP_ENTRIES: list[dict] = [
    # Mrp proteins
    {
        "accession": "A0A5S4TFJ9",
        "protein_class": "mrp",
        "gene": "mrp",
        "protein_name": "M-related protein Mrp",
        "serotype_association": "multi-serotype",
        "notes": "Representative Mrp, 415aa, coiled-coil surface protein, IgG-Fc binding",
    },
    {
        "accession": "A0A660A269",
        "protein_class": "mrp",
        "gene": "mrp",
        "protein_name": "M-related protein Mrp",
        "serotype_association": "multi-serotype",
        "notes": "Mrp variant, 388aa, shorter allele found in pharyngitis-associated isolates",
    },
    # Enn proteins
    {
        "accession": "A0A5S4TF66",
        "protein_class": "enn",
        "gene": "enn",
        "protein_name": "M-related protein Enn",
        "serotype_association": "multi-serotype",
        "notes": "Representative Enn, 375aa, IgA Fc-binding, mga regulon",
    },
    {
        "accession": "A0A8B6IYQ9",
        "protein_class": "enn",
        "gene": "enn",
        "protein_name": "M-like protein Enn",
        "serotype_association": "multi-serotype",
        "notes": "Enn variant, 316aa, shorter allele",
    },
]

# M49/NZ131 EnnX is already in the reference proteome (A0A0H3C1S6).
# We track it in metadata but do not re-download it.
EXISTING_ENN_MRP: list[dict] = [
    {
        "accession": "A0A0H3C1S6",
        "protein_class": "enn",
        "gene": "ennX",
        "protein_name": "EnnX protein",
        "serotype_association": "M49",
        "notes": "Already in M49/NZ131 reference proteome (UP000001039)",
        "already_in_proteome": True,
    },
]


def _uniprot_fetch_fasta(accession: str) -> str:
    """Fetch a single protein FASTA from UniProt."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                f"{UNIPROT_API}/uniprotkb/{accession}.fasta",
                timeout=TIMEOUT,
            )
            if resp.status_code == 429:
                retry_after = int(resp.headers.get("Retry-After", 5))
                logger.warning("Rate limited, waiting %ds", retry_after)
                time.sleep(retry_after)
                continue
            resp.raise_for_status()
            return resp.text.strip()
        except requests.RequestException as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Fetch %s failed (attempt %d), retry in %ds: %s",
                               accession, attempt + 1, wait, e)
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Max retries exceeded for {accession}")


def download_enn_mrp(dest_dir: Path, force: bool = False) -> tuple[Path, list[dict]]:
    """Download Enn/Mrp FASTA sequences and return (fasta_path, metadata).

    Creates enn_mrp_supplement.fasta with all downloaded sequences.
    Skips download if file exists (unless force=True).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    fasta_path = dest_dir / "enn_mrp_supplement.fasta"

    if fasta_path.exists() and not force:
        count = sum(1 for line in fasta_path.read_text().splitlines()
                    if line.startswith(">"))
        logger.info("Enn/Mrp supplement already exists: %s (%d sequences)", fasta_path.name, count)
        return fasta_path, _load_existing_metadata(dest_dir)

    fasta_parts: list[str] = []
    metadata: list[dict] = []

    for entry in ENN_MRP_ENTRIES:
        acc = entry["accession"]
        logger.info("Downloading %s (%s, %s)...", acc, entry["protein_class"], entry["gene"])

        fasta_text = _uniprot_fetch_fasta(acc)
        if not fasta_text:
            logger.warning("Empty FASTA for %s, skipping", acc)
            continue

        # Parse length from FASTA
        seq_lines = [l for l in fasta_text.splitlines() if not l.startswith(">")]
        length = sum(len(l.strip()) for l in seq_lines)

        fasta_parts.append(fasta_text)
        metadata.append({
            "accession": acc,
            "protein_class": entry["protein_class"],
            "gene": entry["gene"],
            "protein_name": entry["protein_name"],
            "length": length,
            "serotype_association": entry["serotype_association"],
            "source": "uniprot",
            "already_in_proteome": False,
            "notes": entry["notes"],
        })

        logger.info("  %s: %d aa", acc, length)
        time.sleep(0.5)  # Rate limit courtesy

    # Also record existing Enn/Mrp already in reference proteomes
    for entry in EXISTING_ENN_MRP:
        metadata.append({
            "accession": entry["accession"],
            "protein_class": entry["protein_class"],
            "gene": entry["gene"],
            "protein_name": entry["protein_name"],
            "length": 368,  # A0A0H3C1S6 EnnX
            "serotype_association": entry["serotype_association"],
            "source": "reference_proteome",
            "already_in_proteome": True,
            "notes": entry["notes"],
        })

    # Write supplemental FASTA
    fasta_path.write_text("\n".join(fasta_parts) + "\n")
    fasta_count = len(fasta_parts)
    logger.info("Enn/Mrp supplement: %s (%d new sequences)", fasta_path.name, fasta_count)

    # Write metadata
    _write_metadata(metadata, dest_dir)

    return fasta_path, metadata


def _write_metadata(metadata: list[dict], dest_dir: Path) -> Path:
    """Write Enn/Mrp metadata CSV."""
    csv_path = dest_dir / "enn_mrp_metadata.csv"
    fieldnames = [
        "accession", "protein_class", "gene", "protein_name", "length",
        "serotype_association", "source", "already_in_proteome", "notes",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    logger.info("Enn/Mrp metadata: %s (%d entries)", csv_path.name, len(metadata))
    return csv_path


def _load_existing_metadata(dest_dir: Path) -> list[dict]:
    """Load existing Enn/Mrp metadata CSV."""
    csv_path = dest_dir / "enn_mrp_metadata.csv"
    if not csv_path.exists():
        return []
    with open(csv_path) as f:
        return list(csv.DictReader(f))


def rebuild_combined_fasta(proteome_dir: Path) -> Path:
    """Rebuild the combined FASTA including Enn/Mrp supplement.

    Concatenates per-serotype FASTAs + enn_mrp_supplement.fasta
    into gas_combined.fasta.
    """
    combined = proteome_dir / "gas_combined.fasta"

    # Per-serotype FASTAs
    parts = sorted(proteome_dir.glob("gas_m*.fasta"))
    if not parts:
        raise FileNotFoundError(f"No per-serotype FASTAs found in {proteome_dir}")

    # Enn/Mrp supplement
    supplement = proteome_dir / "enn_mrp_supplement.fasta"

    with open(combined, "w") as out:
        for p in parts:
            text = p.read_text()
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")

        if supplement.exists():
            text = supplement.read_text()
            out.write(text)
            if not text.endswith("\n"):
                out.write("\n")
            logger.info("Included Enn/Mrp supplement in combined FASTA")

    total = sum(1 for line in combined.read_text().splitlines() if line.startswith(">"))
    logger.info("Combined FASTA rebuilt: %s (%d proteins total)", combined.name, total)
    return combined


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download and integrate Enn/Mrp M-like proteins for GAS mimicry pipeline",
    )
    parser.add_argument("--dest", type=Path, default=PROTEOME_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--force", action="store_true",
                        help="Re-download even if files exist")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Downloading Enn/Mrp M-like proteins to %s", args.dest)
    _, metadata = download_enn_mrp(args.dest, force=args.force)

    # Rebuild combined FASTA
    logger.info("Rebuilding combined FASTA with Enn/Mrp supplement...")
    rebuild_combined_fasta(args.dest)

    new_count = sum(1 for m in metadata if not m.get("already_in_proteome"))
    existing_count = sum(1 for m in metadata if m.get("already_in_proteome"))
    logger.info(
        "Done: %d new Enn/Mrp sequences downloaded, %d already in proteomes, %d total tracked",
        new_count, existing_count, len(metadata),
    )


if __name__ == "__main__":
    main()
