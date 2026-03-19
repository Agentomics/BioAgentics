#!/usr/bin/env python3
"""Conservation feature engineering (PhyloP, GERP++) for NOD2 variants.

Extracts evolutionary conservation scores from dbNSFP annotations:
PhyloP 100-way vertebrate and GERP++ rejected substitution scores.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.03_conservation_features
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def extract_conservation_features(
    scores_path: Path | None = None,
    output_path: Path | None = None,
) -> pd.DataFrame:
    """Extract conservation-specific features from predictor scores.

    PhyloP and GERP++ are fetched as part of the dbNSFP batch query in
    02_structure_scores.py. This script extracts and validates them as a
    dedicated conservation feature table.

    Args:
        scores_path: Path to predictor scores TSV.
        output_path: Path to save conservation features TSV.

    Returns:
        DataFrame with conservation features per variant.
    """
    if scores_path is None:
        scores_path = DATA_DIR / "nod2_predictor_scores.tsv"
    if output_path is None:
        output_path = DATA_DIR / "nod2_conservation.tsv"

    scores_df = pd.read_csv(scores_path, sep="\t")
    logger.info("Loaded %d variants from predictor scores", len(scores_df))

    # Extract conservation columns
    conservation_cols = ["chrom", "pos", "ref", "alt"]
    feature_cols = []

    if "phylop_100way" in scores_df.columns:
        feature_cols.append("phylop_100way")
    if "gerp_rs" in scores_df.columns:
        feature_cols.append("gerp_rs")
    if "phastcons_100way" in scores_df.columns:
        feature_cols.append("phastcons_100way")

    available = conservation_cols + feature_cols
    conservation_df = scores_df[[c for c in available if c in scores_df.columns]].copy()

    # Compute conservation summary score (mean of normalized scores)
    if "phylop_100way" in conservation_df.columns and "gerp_rs" in conservation_df.columns:
        phylop_norm = conservation_df["phylop_100way"].clip(-20, 20) / 20.0
        gerp_norm = conservation_df["gerp_rs"].clip(-12, 6) / 6.0
        conservation_df["conservation_score"] = (phylop_norm + gerp_norm) / 2.0

    # Report coverage
    for col in feature_cols:
        n_valid = conservation_df[col].notna().sum()
        pct = 100 * n_valid / len(conservation_df) if len(conservation_df) > 0 else 0
        logger.info("  %s: %d/%d (%.1f%%) variants have scores", col, n_valid, len(conservation_df), pct)

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conservation_df.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved conservation features to %s (%d variants)", output_path, len(conservation_df))

    return conservation_df


def main():
    """Run conservation feature extraction pipeline."""
    logger.info("=== Conservation Feature Engineering Pipeline ===")
    conservation_df = extract_conservation_features()
    return conservation_df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
