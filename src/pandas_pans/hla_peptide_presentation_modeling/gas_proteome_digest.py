"""Phase 2: GAS proteome manifest and in silico peptide digestion.

Reads GAS reference proteomes (FASTA) for serotypes M1, M3, M5, M12, M18, M49,
generates sliding-window peptide libraries for MHC-I (8-11mer) and MHC-II (15mer),
and writes results as streamed TSV files to avoid memory overload.
"""

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "pandas_pans" / "hla-peptide-presentation"

# GAS serotype reference strains
SEROTYPES = {
    "M1": {
        "emm_type": "emm1",
        "strain": "SF370",
        "taxonomy_id": 301447,
        "uniprot_proteome_id": "UP000000750",
    },
    "M3": {
        "emm_type": "emm3",
        "strain": "MGAS315",
        "taxonomy_id": 198466,
        "uniprot_proteome_id": "UP000000564",
    },
    "M5": {
        "emm_type": "emm5",
        "strain": "Manfredo",
        "taxonomy_id": 467705,
        "uniprot_proteome_id": "UP000001004",
    },
    "M12": {
        "emm_type": "emm12",
        "strain": "MGAS9429",
        "taxonomy_id": 370554,
        "uniprot_proteome_id": "UP000002145",
    },
    "M18": {
        "emm_type": "emm18",
        "strain": "MGAS8232",
        "taxonomy_id": 186103,
        "uniprot_proteome_id": "UP000000782",
    },
    "M49": {
        "emm_type": "emm49",
        "strain": "NZ131",
        "taxonomy_id": 471876,
        "uniprot_proteome_id": "UP000001036",
    },
}

VIRULENCE_FACTOR_KEYWORDS = frozenset({
    "M protein", "emm", "streptolysin", "SLO", "C5a peptidase",
    "streptokinase", "DNase", "SpeB", "SpyCEP", "Sic",
})

MHC_I_LENGTHS = range(8, 12)   # 8, 9, 10, 11
MHC_II_LENGTH = 15


@dataclass
class ProteinRecord:
    """A protein from a GAS FASTA file."""

    accession: str
    description: str
    sequence: str
    is_virulence_factor: bool


def parse_fasta_streaming(fasta_path: Path):
    """Yield ProteinRecord objects from a FASTA file without loading all into memory."""
    accession = ""
    description = ""
    seq_parts: list[str] = []

    with open(fasta_path) as f:
        for line in f:
            line = line.rstrip()
            if line.startswith(">"):
                if accession:
                    seq = "".join(seq_parts)
                    is_vf = any(kw.lower() in description.lower() for kw in VIRULENCE_FACTOR_KEYWORDS)
                    yield ProteinRecord(accession, description, seq, is_vf)
                header = line[1:].split(None, 1)
                accession = header[0].split("|")[1] if "|" in header[0] else header[0]
                description = header[1] if len(header) > 1 else ""
                seq_parts = []
            else:
                seq_parts.append(line)

    if accession:
        seq = "".join(seq_parts)
        is_vf = any(kw.lower() in description.lower() for kw in VIRULENCE_FACTOR_KEYWORDS)
        yield ProteinRecord(accession, description, seq, is_vf)


def generate_peptides(sequence: str, lengths: range | tuple[int, ...] | int) -> list[str]:
    """Generate all sliding-window peptides of given length(s) from a sequence."""
    if isinstance(lengths, int):
        lengths = (lengths,)
    peptides = []
    for length in lengths:
        for i in range(len(sequence) - length + 1):
            peptides.append(sequence[i : i + length])
    return peptides


def digest_serotype(
    serotype: str,
    fasta_path: Path,
    output_dir: Path,
    write_chunk_size: int = 50000,
) -> dict:
    """Digest a single serotype's proteome into peptide libraries.

    Streams output to TSV files to minimize memory usage.
    Returns summary statistics.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    mhc_i_path = output_dir / f"{serotype.lower()}_mhc_i_peptides.tsv"
    mhc_ii_path = output_dir / f"{serotype.lower()}_mhc_ii_peptides.tsv"

    header = ["peptide", "length", "protein_accession", "protein_name", "position", "is_virulence_factor"]

    stats = {
        "protein_count": 0,
        "mhc_i_peptides": 0,
        "mhc_ii_peptides": 0,
        "virulence_factor_proteins": 0,
        "vf_mhc_i_peptides": 0,
        "vf_mhc_ii_peptides": 0,
    }

    with (
        open(mhc_i_path, "w", newline="") as f_i,
        open(mhc_ii_path, "w", newline="") as f_ii,
    ):
        w_i = csv.writer(f_i, delimiter="\t")
        w_ii = csv.writer(f_ii, delimiter="\t")
        w_i.writerow(header)
        w_ii.writerow(header)

        buf_i: list[list] = []
        buf_ii: list[list] = []

        for protein in parse_fasta_streaming(fasta_path):
            stats["protein_count"] += 1
            if protein.is_virulence_factor:
                stats["virulence_factor_proteins"] += 1

            # MHC-I peptides (8-11mer)
            for length in MHC_I_LENGTHS:
                for pos in range(len(protein.sequence) - length + 1):
                    pep = protein.sequence[pos : pos + length]
                    buf_i.append([pep, length, protein.accession, protein.description, pos, protein.is_virulence_factor])
                    stats["mhc_i_peptides"] += 1
                    if protein.is_virulence_factor:
                        stats["vf_mhc_i_peptides"] += 1

            # MHC-II peptides (15mer)
            for pos in range(len(protein.sequence) - MHC_II_LENGTH + 1):
                pep = protein.sequence[pos : pos + MHC_II_LENGTH]
                buf_ii.append([pep, MHC_II_LENGTH, protein.accession, protein.description, pos, protein.is_virulence_factor])
                stats["mhc_ii_peptides"] += 1
                if protein.is_virulence_factor:
                    stats["vf_mhc_ii_peptides"] += 1

            # Flush buffers periodically
            if len(buf_i) >= write_chunk_size:
                w_i.writerows(buf_i)
                buf_i.clear()
            if len(buf_ii) >= write_chunk_size:
                w_ii.writerows(buf_ii)
                buf_ii.clear()

        # Flush remaining
        if buf_i:
            w_i.writerows(buf_i)
        if buf_ii:
            w_ii.writerows(buf_ii)

    return stats


def build_proteome_manifest(proteome_dir: Path) -> dict:
    """Build proteome manifest from available FASTA files."""
    manifest = {"serotypes": []}

    for serotype, info in SEROTYPES.items():
        fasta_name = f"gas_{serotype.lower()}.fasta"
        fasta_path = proteome_dir / fasta_name
        if not fasta_path.exists() and fasta_path.is_symlink():
            # Broken symlink
            continue
        if not fasta_path.exists():
            continue

        protein_count = sum(1 for line in open(fasta_path) if line.startswith(">"))
        vf_detected = []
        with open(fasta_path) as f:
            text = f.read()
            for kw in ["M_protein", "C5a_peptidase", "streptolysin_O", "streptokinase", "DNase"]:
                readable = kw.replace("_", " ")
                if readable.lower() in text.lower() or kw.lower() in text.lower():
                    vf_detected.append(kw)

        entry = {
            "serotype": serotype,
            **info,
            "protein_count": protein_count,
            "fasta_file": fasta_name,
            "virulence_factors_detected": vf_detected,
        }
        manifest["serotypes"].append(entry)

    return manifest


def run_digest(output_base: Path | None = None) -> dict:
    """Run full peptide digestion pipeline for all serotypes.

    Returns digest summary dict.
    """
    if output_base is None:
        output_base = DATA_DIR

    proteome_dir = output_base / "gas_proteomes"
    peptide_dir = output_base / "peptide_libraries"
    peptide_dir.mkdir(parents=True, exist_ok=True)

    # Build manifest
    manifest = build_proteome_manifest(proteome_dir)
    manifest_path = proteome_dir / "proteome_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Digest each serotype
    summary = {"serotypes": {}, "totals": {}}
    total_i = 0
    total_ii = 0

    for entry in manifest["serotypes"]:
        serotype = entry["serotype"]
        fasta_path = proteome_dir / entry["fasta_file"]

        if not fasta_path.exists():
            continue

        print(f"Digesting {serotype} ({entry['protein_count']} proteins)...")
        stats = digest_serotype(serotype, fasta_path, peptide_dir)
        summary["serotypes"][serotype] = stats
        total_i += stats["mhc_i_peptides"]
        total_ii += stats["mhc_ii_peptides"]

    summary["totals"] = {
        "serotypes_processed": len(summary["serotypes"]),
        "total_mhc_i_peptides": total_i,
        "total_mhc_ii_peptides": total_ii,
    }

    summary_path = peptide_dir / "digest_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"Digest complete: {total_i:,} MHC-I + {total_ii:,} MHC-II peptides")
    return summary


if __name__ == "__main__":
    run_digest()
