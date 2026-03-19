#!/usr/bin/env python3
"""AlphaFold structure retrieval and dbNSFP score extraction for NOD2.

Downloads AlphaFold predicted structure (Q9HC29), extracts per-residue
pLDDT, and fetches dbNSFP predictor scores (CADD, REVEL, PolyPhen-2,
SIFT, AlphaMissense) for all NOD2 variants.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.02_structure_scores
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.predictor_scores import collect_predictor_scores
from bioagentics.data.nod2.structure import (
    compute_and_save_structure_features,
    download_alphafold_structure,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def main():
    """Run structure retrieval and score extraction pipeline."""
    logger.info("=== AlphaFold Structure & dbNSFP Score Pipeline ===")

    # Step 1: Download AlphaFold structure
    pdb_path = download_alphafold_structure(DATA_DIR)
    logger.info("AlphaFold PDB: %s", pdb_path)

    # Step 2: Extract per-residue pLDDT and structural features
    struct_df = compute_and_save_structure_features(DATA_DIR / "nod2_structure_features.tsv")
    logger.info("Structure features: %d residues", len(struct_df))

    # Step 3: Fetch dbNSFP predictor scores
    scores_df = collect_predictor_scores(
        variants_path=DATA_DIR / "nod2_variants.tsv",
        output_path=DATA_DIR / "nod2_predictor_scores.tsv",
    )
    logger.info("Predictor scores: %d variants scored", len(scores_df))

    # Summary
    score_cols = ["cadd_phred", "revel", "sift", "alphamissense"]
    for col in score_cols:
        if col in scores_df.columns:
            coverage = scores_df[col].notna().mean() * 100
            logger.info("  %s coverage: %.1f%%", col, coverage)

    return struct_df, scores_df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
