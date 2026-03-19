#!/usr/bin/env python3
"""Structural + VarMeter2 feature engineering from AlphaFold.

Computes per-variant: active site distance, rSASA, VarMeter2-style features
(nSASA, mutation energy, pLDDT), and domain annotations.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.04_structural_features
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.varmeter2 import collect_varmeter2_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def main():
    """Run structural + VarMeter2 feature pipeline."""
    logger.info("=== Structural + VarMeter2 Feature Pipeline ===")

    features_df = collect_varmeter2_features(
        variants_path=DATA_DIR / "nod2_variants.tsv",
        output_path=DATA_DIR / "nod2_varmeter2_features.tsv",
    )

    logger.info("VarMeter2 features computed for %d missense variants", len(features_df))

    # Summary statistics
    if not features_df.empty:
        logger.info("  nSASA range: [%.3f, %.3f]", features_df["nsasa"].min(), features_df["nsasa"].max())
        logger.info("  mutation_energy range: [%.3f, %.3f]",
                     features_df["mutation_energy"].min(), features_df["mutation_energy"].max())
        logger.info("  pLDDT range: [%.1f, %.1f]", features_df["plddt"].min(), features_df["plddt"].max())

    return features_df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
