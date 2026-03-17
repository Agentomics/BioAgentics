"""DIAMOND-based sequence mimicry screen: GAS proteome vs human basal ganglia targets.

Runs all-vs-all DIAMOND blastp of GAS proteins against human neuronal proteome.
Filters hits by E-value, alignment length (>=8 aa for epitope relevance),
and percent identity (>=40% over aligned region).

Usage:
    uv run python -m bioagentics.data.pandas_pans.mimicry_screen [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import shutil
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
GAS_FASTA = PROJECT_DIR / "proteomes" / "gas_combined.fasta"
HUMAN_FASTA = PROJECT_DIR / "human_targets" / "human_bg_targets.fasta"
OUTPUT_DIR = PROJECT_DIR / "mimicry_screen"

# Filtering thresholds (from research plan)
MAX_EVALUE = 1e-3
MIN_ALIGNMENT_LENGTH = 8   # amino acids — epitope-relevant length
MIN_PIDENT = 40.0          # percent identity over aligned region


def check_diamond() -> str:
    """Verify DIAMOND is installed and return its path."""
    diamond = shutil.which("diamond")
    if not diamond:
        raise RuntimeError(
            "DIAMOND not found. Install with: brew install diamond (macOS) "
            "or conda install -c bioconda diamond"
        )
    result = subprocess.run([diamond, "version"], capture_output=True, text=True, check=True)
    logger.info("DIAMOND version: %s", result.stdout.strip())
    return diamond


def make_database(diamond: str, fasta: Path, db_path: Path) -> None:
    """Build DIAMOND database from FASTA file."""
    if db_path.with_suffix(".dmnd").exists():
        logger.info("Database already exists: %s.dmnd", db_path)
        return
    logger.info("Building DIAMOND database from %s ...", fasta.name)
    subprocess.run(
        [diamond, "makedb", "--in", str(fasta), "--db", str(db_path)],
        check=True, capture_output=True, text=True,
    )
    logger.info("Database created: %s.dmnd", db_path)


def run_blastp(diamond: str, query: Path, db: Path, output: Path,
               evalue: float = MAX_EVALUE, threads: int = 4) -> Path:
    """Run DIAMOND blastp and output tab-separated results."""
    if output.exists() and output.stat().st_size > 0:
        logger.info("DIAMOND output already exists: %s", output.name)
        return output

    logger.info("Running DIAMOND blastp: %s vs %s ...", query.name, db.name)
    # Output format 6 with extended columns for mimicry analysis
    outfmt_cols = (
        "qseqid sseqid pident length mismatch gapopen "
        "qstart qend sstart send evalue bitscore "
        "qlen slen qcovhsp"
    )
    subprocess.run(
        [
            diamond, "blastp",
            "--query", str(query),
            "--db", str(db),
            "--out", str(output),
            "--outfmt", "6", *outfmt_cols.split(),
            "--evalue", str(evalue),
            "--threads", str(threads),
            "--sensitive",  # more sensitive for short alignments
            "--max-target-seqs", "50",
        ],
        check=True, capture_output=True, text=True,
    )
    logger.info("DIAMOND blastp complete: %s", output.name)
    return output


def load_and_filter(raw_output: Path) -> pd.DataFrame:
    """Load DIAMOND output and apply mimicry-relevant filters."""
    columns = [
        "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
        "qstart", "qend", "sstart", "send", "evalue", "bitscore",
        "qlen", "slen", "qcovhsp",
    ]
    df = pd.read_csv(raw_output, sep="\t", header=None, names=columns)
    logger.info("Raw DIAMOND hits: %d", len(df))

    # Apply mimicry filters
    filtered = df[
        (df["evalue"] <= MAX_EVALUE)
        & (df["length"] >= MIN_ALIGNMENT_LENGTH)
        & (df["pident"] >= MIN_PIDENT)
    ].copy()
    logger.info("After filtering (evalue<=%.0e, length>=%d, pident>=%.0f%%): %d hits",
                MAX_EVALUE, MIN_ALIGNMENT_LENGTH, MIN_PIDENT, len(filtered))

    # Parse human target info from sseqid (format: ACCESSION|GENE|REGIONS...)
    # Note: DIAMOND truncates sseqid at first space, so [KNOWN_TARGET] tag is lost.
    # Cross-reference against known targets list by accession instead.
    from bioagentics.data.pandas_pans.human_targets import KNOWN_TARGETS
    known_accessions = {t["uniprot_id"] for t in KNOWN_TARGETS}

    filtered["human_accession"] = filtered["sseqid"].str.split("|").str[0]
    filtered["human_gene"] = filtered["sseqid"].str.split("|").str[1]
    filtered["known_target"] = filtered["human_accession"].isin(known_accessions)

    # Sort by bitscore descending
    filtered = filtered.sort_values("bitscore", ascending=False)

    return filtered


def run_per_serotype(diamond: str, dest: Path, db_path: Path,
                     threads: int = 4) -> dict[str, pd.DataFrame]:
    """Run mimicry screen for each serotype individually."""
    proteome_dir = PROJECT_DIR / "proteomes"
    per_serotype_results = {}

    for fasta in sorted(proteome_dir.glob("gas_m*.fasta")):
        serotype = fasta.stem.replace("gas_", "").upper()
        output = dest / f"hits_{serotype.lower()}.tsv"
        run_blastp(diamond, fasta, db_path, output, threads=threads)
        if output.exists() and output.stat().st_size > 0:
            df = load_and_filter(output)
            per_serotype_results[serotype] = df
            logger.info("  %s: %d filtered hits", serotype, len(df))

    return per_serotype_results


def summarize_results(combined: pd.DataFrame, per_serotype: dict[str, pd.DataFrame],
                      dest: Path) -> None:
    """Generate summary statistics and reports."""
    # Combined summary
    summary_path = dest / "screen_summary.txt"
    with open(summary_path, "w") as f:
        f.write("GAS Molecular Mimicry Screen — Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total filtered hits (combined): {len(combined)}\n")
        f.write(f"Unique GAS proteins with hits: {combined['qseqid'].nunique()}\n")
        f.write(f"Unique human targets hit: {combined['human_accession'].nunique()}\n")

        known_hits = combined[combined["known_target"]]
        f.write(f"\nKnown PANDAS targets recovered: {known_hits['human_gene'].nunique()}\n")
        if not known_hits.empty:
            for gene in sorted(known_hits["human_gene"].unique()):
                best = known_hits[known_hits["human_gene"] == gene].iloc[0]
                f.write(f"  {gene}: best hit pident={best['pident']:.1f}%, "
                        f"length={best['length']}, evalue={best['evalue']:.1e}\n")

        f.write(f"\nPer-serotype breakdown:\n")
        for sero, df in sorted(per_serotype.items()):
            f.write(f"  {sero}: {len(df)} hits, "
                    f"{df['human_accession'].nunique()} unique human targets\n")

    logger.info("Summary written: %s", summary_path.name)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run DIAMOND-based molecular mimicry screen: GAS vs human basal ganglia",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--threads", type=int, default=4,
                        help="Number of DIAMOND threads (default: 4)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    # Validate inputs exist
    for f in [GAS_FASTA, HUMAN_FASTA]:
        if not f.exists():
            raise FileNotFoundError(
                f"Required input not found: {f}\n"
                "Run gas_proteomes.py and human_targets.py first."
            )

    args.dest.mkdir(parents=True, exist_ok=True)

    # Setup
    diamond = check_diamond()
    db_path = args.dest / "human_targets_db"
    make_database(diamond, HUMAN_FASTA, db_path)

    # Run combined screen
    combined_output = args.dest / "hits_combined.tsv"
    run_blastp(diamond, GAS_FASTA, db_path, combined_output, threads=args.threads)
    combined = load_and_filter(combined_output)

    # Save filtered results
    filtered_path = args.dest / "hits_filtered.tsv"
    combined.to_csv(filtered_path, sep="\t", index=False)
    logger.info("Filtered hits saved: %s (%d rows)", filtered_path.name, len(combined))

    # Run per-serotype analysis
    per_serotype = run_per_serotype(diamond, args.dest, db_path, threads=args.threads)

    # Summary
    summarize_results(combined, per_serotype, args.dest)

    logger.info("Done. Screen results in %s", args.dest)


if __name__ == "__main__":
    main()
