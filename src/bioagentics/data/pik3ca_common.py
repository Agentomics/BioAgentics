"""Shared PIK3CA constants and classification logic.

Used by pik3ca_allele_dependencies scripts to ensure consistent allele
groupings and domain annotations across DepMap and TCGA analyses.

PIK3CA hotspot clusters (~80% of mutations):
- H1047R/L (kinase domain): strongest PI3Kα activity, preferentially activates AKT
- E545K (helical domain): intermediate activity, p85-independent signaling
- E542K (helical domain): similar to E545K, distinct p85 binding effects
- C420R (C2 domain): less common activating mutation
- N345K: emerging hotspot
"""

from __future__ import annotations

import pandas as pd

# PIK3CA hotspot allele map: protein change -> canonical allele name
PIK3CA_ALLELE_MAP = {
    "p.H1047R": "H1047R",
    "p.H1047L": "H1047L",
    "p.H1047Y": "H1047R",  # rare kinase-domain variant, group with H1047R
    "p.E545K": "E545K",
    "p.E545Q": "E545K",  # rare helical variant, group with E545K
    "p.E545G": "E545K",
    "p.E545A": "E545K",
    "p.E542K": "E542K",
    "p.E542V": "E542K",  # rare helical variant
    "p.C420R": "C420R",
    "p.N345K": "N345K",
    "p.Q546K": "E545K",  # adjacent helical hotspot
    "p.Q546R": "E545K",
}

# Map allele -> structural domain
ALLELE_DOMAIN_MAP = {
    "H1047R": "kinase_domain",
    "H1047L": "kinase_domain",
    "E545K": "helical_domain",
    "E542K": "helical_domain",
    "C420R": "c2_domain",
    "N345K": "c2_domain",
    "other_activating": "other",
}

# Priority order for allele classification when a cell line has multiple mutations
ALLELE_PRIORITY = ["H1047R", "H1047L", "E545K", "E542K", "C420R", "N345K"]

# Known activating mutations outside the main hotspots (OncoKB/COSMIC curated)
OTHER_ACTIVATING = {
    "p.R88Q", "p.G118D", "p.K111E", "p.G106V",
    "p.M1043I", "p.M1043V", "p.G1049R", "p.H1047Y",
}

# Variant classifications considered functionally damaging (TCGA MAF format)
DAMAGING_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
}


def classify_pik3ca_allele(protein_changes: list[str]) -> str:
    """Classify PIK3CA allele from a list of protein changes.

    Returns one of: H1047R, H1047L, E545K, E542K, C420R, N345K,
    other_activating, WT.

    Priority: H1047R > H1047L > E545K > E542K > C420R > N345K > other_activating
    """
    hotspot_alleles = []
    has_other_activating = False

    for pc in protein_changes:
        if pd.notna(pc):
            allele = PIK3CA_ALLELE_MAP.get(pc)
            if allele:
                hotspot_alleles.append(allele)
            elif pc in OTHER_ACTIVATING:
                has_other_activating = True

    if not hotspot_alleles:
        if has_other_activating:
            return "other_activating"
        return "other" if protein_changes else "WT"

    for preferred in ALLELE_PRIORITY:
        if preferred in hotspot_alleles:
            return preferred
    return hotspot_alleles[0]


def get_domain(allele: str) -> str:
    """Map an allele classification to its structural domain."""
    return ALLELE_DOMAIN_MAP.get(allele, "other")
