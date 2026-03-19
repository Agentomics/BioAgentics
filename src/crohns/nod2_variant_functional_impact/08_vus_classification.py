#!/usr/bin/env python3
"""VUS classification and output visualization.

Applies the trained ensemble model to all NOD2 VUS, ranks by predicted
pathogenicity, and generates structure-colored visualizations.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.08_vus_classification
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.vus_pipeline import run_vus_pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")
OUTPUT_DIR = Path("output/crohns/nod2-variant-functional-impact")


def main():
    """Run VUS classification and visualization pipeline."""
    logger.info("=== VUS Classification & Visualization Pipeline ===")

    results = run_vus_pipeline(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)

    summary = results["summary"]
    logger.info("Total VUS classified: %d", summary["total_vus"])
    logger.info("High confidence predictions: %d", summary["high_confidence"])
    logger.info("Class distribution: %s", summary["class_distribution"])

    # Check target: at least 50 VUS with confidence > 0.8
    if summary["high_confidence"] >= 50:
        logger.info("Target MET: %d high-confidence VUS (target: 50)", summary["high_confidence"])
    else:
        logger.warning(
            "Target NOT MET: %d high-confidence VUS (target: 50). "
            "May need more training data or feature engineering.",
            summary["high_confidence"],
        )

    logger.info("Output files:")
    for name, path in summary["output_files"].items():
        logger.info("  %s: %s", name, path)

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
