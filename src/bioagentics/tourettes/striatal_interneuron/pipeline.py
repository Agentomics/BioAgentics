"""Pipeline entry point for ts-striatal-interneuron-pathology.

Orchestrates the full analysis: data acquisition, QC, integration,
interneuron subtype classification, differential expression, and
target nomination.

Usage:
    uv run python -m bioagentics.tourettes.striatal_interneuron.pipeline [phase]

Phases:
    setup       - Create directories and verify dependencies
    download    - Download reference atlas from GEO
    qc          - Run QC on downloaded datasets
    integrate   - Batch correction and integration
    classify    - Interneuron subtype classification
    de          - Differential expression at subtype resolution
    all         - Run all phases sequentially
"""

from __future__ import annotations

import argparse
import sys

from bioagentics.tourettes.striatal_interneuron.config import (
    DATA_DIR,
    OUTPUT_DIR,
    REFERENCE_DIR,
    ensure_dirs,
)


def phase_setup() -> None:
    """Create project directories and verify dependencies."""
    print("=== Phase: Setup ===")
    ensure_dirs()
    print(f"  Data directory: {DATA_DIR}")
    print(f"  Output directory: {OUTPUT_DIR}")

    # Verify key imports
    deps = {
        "scanpy": "scanpy",
        "anndata": "anndata",
        "harmonypy": "harmonypy",
        "scrublet": "scrublet",
        "leidenalg": "leidenalg",
        "pydeseq2": "pydeseq2",
    }
    missing = []
    for name, module in deps.items():
        try:
            __import__(module)
            print(f"  {name}: OK")
        except ImportError:
            print(f"  {name}: MISSING")
            missing.append(name)

    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: uv add " + " ".join(missing))
        sys.exit(1)

    print("\nSetup complete.")


def phase_download() -> None:
    """Download reference atlas datasets from GEO."""
    print("=== Phase: Download Reference Atlas ===")
    from bioagentics.tourettes.striatal_interneuron.download import download_reference_atlas

    download_reference_atlas(REFERENCE_DIR)


def phase_qc() -> None:
    """Run QC on downloaded datasets."""
    print("=== Phase: QC ===")
    from bioagentics.tourettes.striatal_interneuron.qc_runner import run_qc_pipeline

    run_qc_pipeline()


def phase_integrate() -> None:
    """Batch correction and dataset integration."""
    print("=== Phase: Integration ===")
    from bioagentics.tourettes.striatal_interneuron.integrate import run_integration

    run_integration()


def phase_classify() -> None:
    """Interneuron subtype classification using reference taxonomy."""
    print("=== Phase: Classification ===")
    from bioagentics.tourettes.striatal_interneuron.classify import run_classification

    run_classification()


def phase_de() -> None:
    """Differential expression at interneuron subtype resolution."""
    print("=== Phase: Differential Expression ===")
    from bioagentics.tourettes.striatal_interneuron.de_analysis import run_de

    run_de()


PHASES = {
    "setup": phase_setup,
    "download": phase_download,
    "qc": phase_qc,
    "integrate": phase_integrate,
    "classify": phase_classify,
    "de": phase_de,
}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="TS Striatal Interneuron Pathology Analysis Pipeline"
    )
    parser.add_argument(
        "phase",
        choices=[*PHASES, "all"],
        default="setup",
        nargs="?",
        help="Analysis phase to run (default: setup)",
    )
    args = parser.parse_args(argv)

    if args.phase == "all":
        for name, func in PHASES.items():
            func()
            print()
    else:
        PHASES[args.phase]()


if __name__ == "__main__":
    main()
