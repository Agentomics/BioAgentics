#!/usr/bin/env python3
"""NOD2 variant data collection pipeline (ClinVar, gnomAD, IBDGC).

Fetches all known NOD2 variants from ClinVar and gnomAD v4,
merges annotations, and outputs a unified variant table.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.01_data_collection
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.variants import (
    fetch_clinvar_variants,
    fetch_gnomad_variants,
    merge_variants,
    validate_known_variants,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data/crohns/nod2-variant-functional-impact")


def add_domain_location(df):
    """Add domain_location column based on protein position."""
    from bioagentics.data.nod2.structure import get_domain
    from bioagentics.data.nod2.varmeter2 import _parse_protein_change

    def _get_domain(hgvs_p):
        if not hgvs_p or str(hgvs_p) == "nan":
            return ""
        parsed = _parse_protein_change(str(hgvs_p))
        if parsed:
            return get_domain(parsed[1])
        return ""

    df["domain_location"] = df["hgvs_p"].apply(_get_domain)
    return df


def main():
    """Run the full data collection pipeline."""
    output_path = OUTPUT_DIR / "nod2_variants.tsv"

    logger.info("=== NOD2 Variant Data Collection Pipeline ===")

    # Fetch from ClinVar and gnomAD
    clinvar_df = fetch_clinvar_variants()
    gnomad_df = fetch_gnomad_variants()

    logger.info("ClinVar: %d variants, gnomAD: %d variants", len(clinvar_df), len(gnomad_df))

    # Merge
    merged = merge_variants(clinvar_df, gnomad_df)
    logger.info("Merged dataset: %d unique variants", len(merged))

    # Add domain location annotations
    if "hgvs_p" in merged.columns:
        merged = add_domain_location(merged)

    # Validate known pathogenic variants
    if not merged.empty:
        validation = validate_known_variants(merged)
        for name, found in validation.items():
            status = "FOUND" if found else "MISSING"
            logger.info("  %s: %s", name, status)

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, sep="\t", index=False)
    logger.info("Saved %d variants to %s", len(merged), output_path)

    # Summary
    if not merged.empty and "clinvar_significance" in merged.columns:
        sig_counts = merged["clinvar_significance"].value_counts()
        logger.info("ClinVar significance distribution:\n%s", sig_counts.to_string())

    return merged


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
