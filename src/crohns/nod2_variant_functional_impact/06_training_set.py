#!/usr/bin/env python3
"""RIPK2/NF-kB interaction features and training set construction.

Computes RIPK2 CARD-CARD proximity features and constructs the GOF/neutral/LOF
training set from ClinVar, Blau syndrome, and SURF data.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.06_training_set
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from bioagentics.data.nod2.training_set import collect_training_set

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")


def add_ripk2_features(training_df: pd.DataFrame) -> pd.DataFrame:
    """Add RIPK2 interaction features from girdin features table.

    The girdin module already computes ripk2_interface_distance
    (proximity to CARD-CARD interaction interface, residues 27-110).
    """
    girdin_path = DATA_DIR / "nod2_girdin_features.tsv"
    if not girdin_path.exists():
        logger.warning("Girdin features not found at %s, skipping RIPK2 features", girdin_path)
        return training_df

    girdin_df = pd.read_csv(girdin_path, sep="\t")

    if "ripk2_interface_distance" not in girdin_df.columns:
        logger.warning("ripk2_interface_distance not in girdin features, skipping")
        return training_df

    # Merge by position
    from bioagentics.data.nod2.varmeter2 import _parse_protein_change

    training_df["residue_pos"] = training_df["hgvs_p"].apply(
        lambda x: (
            _parse_protein_change(str(x))[1]
            if _parse_protein_change(str(x)) is not None
            else _extract_fs_pos(str(x))
        )
    )

    ripk2_cols = girdin_df[["residue_pos", "ripk2_interface_distance"]].drop_duplicates(subset="residue_pos")
    merged = training_df.merge(ripk2_cols, on="residue_pos", how="left")

    # Flag variants in RIPK2 interface (CARD domain, < 10 Angstroms)
    if "ripk2_interface_distance" in merged.columns:
        merged["in_ripk2_interface"] = merged["ripk2_interface_distance"] < 10.0

    n_ripk2 = merged["in_ripk2_interface"].sum() if "in_ripk2_interface" in merged.columns else 0
    logger.info("Variants near RIPK2 interface (<10A): %d", n_ripk2)

    return merged


def _extract_fs_pos(hgvs_p: str) -> int | None:
    """Extract position from frameshift notation."""
    import re
    m = re.search(r"(\d+)fs", hgvs_p)
    return int(m.group(1)) if m else None


def main():
    """Run training set construction pipeline."""
    logger.info("=== RIPK2/NF-kB Features + Training Set Pipeline ===")

    # Build base training set
    training_df = collect_training_set(
        variants_path=DATA_DIR / "nod2_variants.tsv",
        output_path=DATA_DIR / "nod2_training_set.tsv",
    )

    # Add RIPK2 interaction features
    training_df = add_ripk2_features(training_df)

    logger.info("Training set: %d variants", len(training_df))
    if not training_df.empty:
        class_dist = training_df["functional_class"].value_counts()
        logger.info("Class distribution:\n%s", class_dist.to_string())

    return training_df


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
