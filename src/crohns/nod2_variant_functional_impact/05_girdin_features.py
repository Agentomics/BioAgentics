#!/usr/bin/env python3
"""Girdin binding domain feature engineering.

Computes girdin interface distance, disruption flag, and binding effect
for each NOD2 variant based on Ghosh et al. JCI 2025.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.05_girdin_features
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.girdin import collect_girdin_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def main():
    """Run girdin feature engineering pipeline."""
    logger.info("=== Girdin Binding Domain Feature Pipeline ===")

    features_df = collect_girdin_features(
        variants_path=DATA_DIR / "nod2_variants.tsv",
        output_path=DATA_DIR / "nod2_girdin_features.tsv",
    )

    logger.info("Girdin features computed for %d variants", len(features_df))

    # Summary
    if not features_df.empty:
        n_disrupts = features_df["disrupts_girdin_domain"].sum()
        binding_effects = features_df["predicted_binding_effect"].value_counts()
        logger.info("  Variants disrupting girdin domain: %d", n_disrupts)
        logger.info("  Binding effect distribution:\n%s", binding_effects.to_string())

    return features_df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
