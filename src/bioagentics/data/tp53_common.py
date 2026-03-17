"""Shared TP53 constants and classification logic.

Used by tp53_hotspot_allele_dependencies scripts to ensure consistent allele
groupings across DepMap and TCGA analyses.

TP53 hotspot clusters (most common missense in TCGA/COSMIC):
- R175H (structural): complete loss of DNA binding, GOF via aggregation
- R248W (contact): direct DNA contact, retains partial structure
- R273H (contact): similar to R248W, different transcriptional targets
- G245S (structural): loop L3 destabilization
- R249S (structural): aflatoxin-associated, HCC enriched
- Y220C (structural): druggable pocket (rezatapopt target)
- R282W (structural): H2 helix destabilization
"""

from __future__ import annotations

import pandas as pd

# TP53 hotspot allele map: protein change -> canonical allele name
TP53_ALLELE_MAP: dict[str, str] = {
    "p.R175H": "R175H",
    "p.R175G": "R175H",  # rare structural variant at same residue
    "p.R175L": "R175H",
    "p.R248W": "R248W",
    "p.R248Q": "R248W",  # same contact residue, group together
    "p.R273H": "R273H",
    "p.R273C": "R273H",  # same contact residue
    "p.R273L": "R273H",
    "p.G245S": "G245S",
    "p.G245D": "G245S",  # same L3 loop residue
    "p.G245C": "G245S",
    "p.R249S": "R249S",
    "p.R249M": "R249S",
    "p.Y220C": "Y220C",
    "p.R282W": "R282W",
}

# Hotspot alleles
HOTSPOT_ALLELES = ["R175H", "R248W", "R273H", "G245S", "R249S", "Y220C", "R282W"]

# Structural vs contact classification
STRUCTURAL_ALLELES = {"R175H", "G245S", "R282W", "Y220C"}
CONTACT_ALLELES = {"R248W", "R273H", "R249S"}

# Priority order for allele classification when a cell line has multiple mutations
ALLELE_PRIORITY = ["R175H", "R248W", "R273H", "G245S", "R249S", "Y220C", "R282W"]

# Variant types considered truncating
TRUNCATING_CONSEQUENCES = {
    "stop_gained",
    "frameshift_variant",
    "splice_acceptor_variant",
    "splice_donor_variant",
}

# VepImpact for missense
MISSENSE_CONSEQUENCES = {
    "missense_variant",
}


def classify_tp53_allele(
    protein_changes: list[str],
    consequences: list[str] | None = None,
) -> tuple[str, str]:
    """Classify TP53 allele from a list of protein changes.

    Returns (allele, allele_class) where:
    - allele: R175H, R248W, ..., other_missense, truncating
    - allele_class: hotspot, other_missense, truncating

    Priority: hotspot alleles by ALLELE_PRIORITY > other_missense > truncating.
    """
    hotspot_alleles = []
    has_missense = False
    has_truncating = False

    for i, pc in enumerate(protein_changes):
        if pd.notna(pc):
            allele = TP53_ALLELE_MAP.get(pc)
            if allele:
                hotspot_alleles.append(allele)
            else:
                has_missense = True

        # Check consequences if available
        if consequences and i < len(consequences):
            cons = str(consequences[i])
            if any(t in cons for t in TRUNCATING_CONSEQUENCES):
                has_truncating = True
            elif any(m in cons for m in MISSENSE_CONSEQUENCES):
                has_missense = True

    # Priority: hotspot > other_missense > truncating
    if hotspot_alleles:
        for preferred in ALLELE_PRIORITY:
            if preferred in hotspot_alleles:
                return preferred, "hotspot"
        return hotspot_alleles[0], "hotspot"

    if has_missense:
        return "other_missense", "other_missense"
    if has_truncating:
        return "truncating", "truncating"

    return "other_missense", "other_missense"


def is_structural(allele: str) -> bool:
    """Check if an allele is a structural mutant."""
    return allele in STRUCTURAL_ALLELES


def is_contact(allele: str) -> bool:
    """Check if an allele is a contact mutant."""
    return allele in CONTACT_ALLELES
