#!/usr/bin/env python3
"""Download species abundance tables from curatedMetagenomicData via R.

Exports pre-processed MetaPhlAn species abundance tables and sample metadata
for external CD cohort validation. Processes one study at a time to stay
within the 8GB RAM constraint.

Target studies:
  - NielsenHB_2014  (MetaHIT, Danish IBD cohort)
  - HallAB_2017     (UK IBD cohort)
  - SchirmerM_2016  (PRISM, Broad IBD cohort)

Usage:
    python 03_download_curated_metagenomic.py [--output-dir DIR] [--studies STUDY1,STUDY2,...]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

# Studies with paired metagenomics relevant to CD external validation
DEFAULT_STUDIES = [
    "NielsenHB_2014",
    "HallAB_2017",
    "SchirmerM_2016",
]

R_SCRIPT = Path(__file__).parent / "export_curated_metagenomic.R"


def check_r_available() -> str:
    """Return path to Rscript or raise."""
    try:
        result = subprocess.run(
            ["Rscript", "--version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # Rscript prints version to stderr
        version = (result.stderr or result.stdout).strip()
        print(f"R found: {version}")
        return "Rscript"
    except FileNotFoundError:
        print("ERROR: Rscript not found. Install R first.", file=sys.stderr)
        sys.exit(1)


def export_study(rscript: str, study: str, output_dir: Path) -> bool:
    """Run the R export script for a single study. Returns True on success."""
    print(f"\n{'='*60}")
    print(f"Exporting: {study}")
    print(f"{'='*60}")

    cmd = [rscript, str(R_SCRIPT), study, str(output_dir)]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,  # 10 min per study
        )
        print(result.stdout)
        if result.returncode != 0:
            print(f"STDERR:\n{result.stderr}", file=sys.stderr)
            print(f"FAILED: {study} (exit code {result.returncode})")
            return False

        # Verify output files exist
        species_file = output_dir / f"{study}_species.tsv"
        metadata_file = output_dir / f"{study}_metadata.tsv"
        if species_file.exists() and metadata_file.exists():
            species_size = species_file.stat().st_size / 1024
            meta_size = metadata_file.stat().st_size / 1024
            print(f"  Species table: {species_size:.1f} KB")
            print(f"  Metadata:      {meta_size:.1f} KB")
            return True
        else:
            print(f"FAILED: Output files not created for {study}")
            return False
    except subprocess.TimeoutExpired:
        print(f"TIMEOUT: {study} exceeded 10 minute limit")
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download species tables from curatedMetagenomicData"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/crohns/microbiome_metabolome_subtyping/external_cohorts"),
        help="Output directory for TSV files",
    )
    parser.add_argument(
        "--studies",
        type=str,
        default=None,
        help="Comma-separated study names (default: all 3 target studies)",
    )
    args = parser.parse_args()

    studies = args.studies.split(",") if args.studies else DEFAULT_STUDIES
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    rscript = check_r_available()

    results: dict[str, bool] = {}
    for study in studies:
        results[study] = export_study(rscript, study.strip(), output_dir)

    print(f"\n{'='*60}")
    print("Summary:")
    for study, success in results.items():
        status = "OK" if success else "FAILED"
        print(f"  {study}: {status}")

    failed = [s for s, ok in results.items() if not ok]
    if failed:
        print(f"\n{len(failed)} study(ies) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        print(f"\nAll {len(results)} studies exported successfully.")


if __name__ == "__main__":
    main()
