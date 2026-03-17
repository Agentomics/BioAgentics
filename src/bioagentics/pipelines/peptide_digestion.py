"""In silico peptide digestion of GAS proteomes for HLA binding prediction.

Reads GAS proteome FASTA files and generates overlapping peptide libraries:
- 8-11mer peptides for MHC-I binding prediction (NetMHCpan)
- 15mer peptides for MHC-II binding prediction (NetMHCIIpan)

Each peptide is tagged with source protein, serotype, and position.
Virulence factor peptides are flagged for prioritization.

Usage:
    uv run python -m bioagentics.pipelines.peptide_digestion [--proteome-dir DIR] [--output-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

from bioagentics.config import DATA_DIR

logger = logging.getLogger(__name__)

PROTEOME_DIR = DATA_DIR / "pandas_pans" / "gas-molecular-mimicry-mapping" / "proteomes"
OUTPUT_DIR = DATA_DIR / "pandas_pans" / "hla-peptide-presentation" / "peptide_libraries"

# MHC-I: 8-11mer, MHC-II: 15mer (core binding is 9mer within the 15mer)
MHC_I_LENGTHS = (8, 9, 10, 11)
MHC_II_LENGTH = 15

# Virulence factors to flag — keywords matched against FASTA header (case-insensitive)
VIRULENCE_KEYWORDS: dict[str, list[str]] = {
    "M_protein": ["emm", "m protein", "mga regulon", "antiphagocytic"],
    "C5a_peptidase": ["scpa", "c5a peptidase", "complement"],
    "streptolysin_O": ["slo", "streptolysin", "thiol-activated cytolysin"],
    "streptokinase": ["ska", "streptokinase", "plasminogen activator"],
    "DNase": ["dnase", "deoxyribonuclease", "spd", "streptodornase"],
    "hyaluronidase": ["hyla", "hyaluronidase", "hyaluronate"],
    "streptolysin_S": ["saga", "sagb", "streptolysin s"],
    "pyrogenic_exotoxin": ["spe", "pyrogenic exotoxin", "superantigen"],
}


@dataclass
class FastaEntry:
    """A single protein from a FASTA file."""
    header: str
    sequence: str
    serotype: str

    @property
    def protein_name(self) -> str:
        """Extract protein name from FASTA header."""
        # UniProt format: >sp|ACCESSION|NAME_SPECIES Description OS=...
        parts = self.header.split("|")
        if len(parts) >= 3:
            desc = parts[2].split(" OS=")[0] if " OS=" in parts[2] else parts[2]
            return desc.strip()
        # Fallback: everything after >
        return self.header.lstrip(">").split()[0]

    @property
    def accession(self) -> str:
        """Extract UniProt accession."""
        parts = self.header.split("|")
        if len(parts) >= 2:
            return parts[1]
        return self.header.lstrip(">").split()[0]

    def virulence_flags(self) -> list[str]:
        """Return list of virulence factor categories matching this protein."""
        header_lower = self.header.lower()
        flags = []
        for category, keywords in VIRULENCE_KEYWORDS.items():
            if any(kw in header_lower for kw in keywords):
                flags.append(category)
        return flags


@dataclass
class Peptide:
    """A single peptide derived from in silico digestion."""
    sequence: str
    length: int
    position_start: int  # 0-based position in source protein
    position_end: int
    source_protein: str
    source_accession: str
    serotype: str
    virulence_flags: list[str] = field(default_factory=list)
    is_virulence_factor: bool = False


def parse_fasta(fasta_path: Path, serotype: str) -> list[FastaEntry]:
    """Parse a FASTA file into FastaEntry objects."""
    entries = []
    current_header = None
    current_seq_parts: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_header is not None:
                    entries.append(FastaEntry(
                        header=current_header,
                        sequence="".join(current_seq_parts),
                        serotype=serotype,
                    ))
                current_header = line
                current_seq_parts = []
            else:
                current_seq_parts.append(line)

    if current_header is not None:
        entries.append(FastaEntry(
            header=current_header,
            sequence="".join(current_seq_parts),
            serotype=serotype,
        ))

    return entries


def digest_protein(entry: FastaEntry, lengths: tuple[int, ...]) -> list[Peptide]:
    """Generate all overlapping peptides of specified lengths from a protein."""
    seq = entry.sequence
    vf = entry.virulence_flags()
    peptides = []

    for length in lengths:
        if len(seq) < length:
            continue
        for i in range(len(seq) - length + 1):
            peptide_seq = seq[i:i + length]
            # Skip peptides with ambiguous amino acids
            if any(aa not in "ACDEFGHIKLMNPQRSTVWY" for aa in peptide_seq):
                continue
            peptides.append(Peptide(
                sequence=peptide_seq,
                length=length,
                position_start=i,
                position_end=i + length,
                source_protein=entry.protein_name,
                source_accession=entry.accession,
                serotype=entry.serotype,
                virulence_flags=vf,
                is_virulence_factor=len(vf) > 0,
            ))

    return peptides


def digest_proteome(fasta_path: Path, serotype: str,
                    mhc_class: str = "I") -> list[Peptide]:
    """Digest an entire proteome into peptides for the specified MHC class.

    Args:
        fasta_path: Path to FASTA file.
        serotype: GAS serotype label (e.g., "M1").
        mhc_class: "I" for 8-11mer or "II" for 15mer.
    """
    entries = parse_fasta(fasta_path, serotype)
    lengths = MHC_I_LENGTHS if mhc_class == "I" else (MHC_II_LENGTH,)

    all_peptides = []
    for entry in entries:
        all_peptides.extend(digest_protein(entry, lengths))

    return all_peptides


def write_peptide_tsv(peptides: list[Peptide], output_path: Path) -> None:
    """Write peptide library to TSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence", "length", "position_start", "position_end",
        "source_protein", "source_accession", "serotype",
        "is_virulence_factor", "virulence_flags",
    ]
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for p in peptides:
            writer.writerow({
                "sequence": p.sequence,
                "length": p.length,
                "position_start": p.position_start,
                "position_end": p.position_end,
                "source_protein": p.source_protein,
                "source_accession": p.source_accession,
                "serotype": p.serotype,
                "is_virulence_factor": p.is_virulence_factor,
                "virulence_flags": ";".join(p.virulence_flags),
            })


def write_summary(all_stats: dict, output_dir: Path) -> None:
    """Write digest summary JSON."""
    with open(output_dir / "digest_summary.json", "w") as f:
        json.dump(all_stats, f, indent=2)


def run_digestion(proteome_dir: Path, output_dir: Path) -> dict:
    """Run full peptide digestion pipeline across all serotype FASTAs.

    Returns summary statistics dict.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    fasta_files = sorted(proteome_dir.glob("gas_m*.fasta"))
    if not fasta_files:
        raise FileNotFoundError(f"No GAS proteome FASTAs found in {proteome_dir}")

    stats: dict = {"serotypes": {}, "totals": {}}
    total_mhc_i = 0
    total_mhc_ii = 0
    total_vf_i = 0
    total_vf_ii = 0

    for fasta_path in fasta_files:
        serotype = fasta_path.stem.replace("gas_", "").upper()
        logger.info("Digesting %s (%s)", serotype, fasta_path.name)

        # MHC-I peptides (8-11mer)
        mhc_i_peptides = digest_proteome(fasta_path, serotype, mhc_class="I")
        mhc_i_out = output_dir / f"{serotype.lower()}_mhc_i_peptides.tsv"
        write_peptide_tsv(mhc_i_peptides, mhc_i_out)

        # MHC-II peptides (15mer)
        mhc_ii_peptides = digest_proteome(fasta_path, serotype, mhc_class="II")
        mhc_ii_out = output_dir / f"{serotype.lower()}_mhc_ii_peptides.tsv"
        write_peptide_tsv(mhc_ii_peptides, mhc_ii_out)

        n_vf_i = sum(1 for p in mhc_i_peptides if p.is_virulence_factor)
        n_vf_ii = sum(1 for p in mhc_ii_peptides if p.is_virulence_factor)

        stats["serotypes"][serotype] = {
            "fasta_file": fasta_path.name,
            "proteins": len(parse_fasta(fasta_path, serotype)),
            "mhc_i_peptides": len(mhc_i_peptides),
            "mhc_ii_peptides": len(mhc_ii_peptides),
            "mhc_i_virulence_factor_peptides": n_vf_i,
            "mhc_ii_virulence_factor_peptides": n_vf_ii,
        }

        total_mhc_i += len(mhc_i_peptides)
        total_mhc_ii += len(mhc_ii_peptides)
        total_vf_i += n_vf_i
        total_vf_ii += n_vf_ii

        logger.info("  %s: %d MHC-I peptides (%d VF), %d MHC-II peptides (%d VF)",
                     serotype, len(mhc_i_peptides), n_vf_i,
                     len(mhc_ii_peptides), n_vf_ii)

    stats["totals"] = {
        "serotypes_processed": len(fasta_files),
        "total_mhc_i_peptides": total_mhc_i,
        "total_mhc_ii_peptides": total_mhc_ii,
        "total_mhc_i_virulence_factor_peptides": total_vf_i,
        "total_mhc_ii_virulence_factor_peptides": total_vf_ii,
    }

    write_summary(stats, output_dir)
    logger.info("Done: %d MHC-I, %d MHC-II peptides across %d serotypes",
                total_mhc_i, total_mhc_ii, len(fasta_files))
    return stats


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="In silico peptide digestion of GAS proteomes for HLA binding prediction",
    )
    parser.add_argument("--proteome-dir", type=Path, default=PROTEOME_DIR,
                        help="Directory containing GAS FASTA files (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR,
                        help="Output directory for peptide libraries (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run_digestion(args.proteome_dir, args.output_dir)


if __name__ == "__main__":
    main()
