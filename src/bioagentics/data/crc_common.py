"""Shared CRC constants and classification logic.

Used by crc_depmap (cell lines) and crc_tcga (patients) to ensure
consistent driver gene definitions, allele groupings, and annotations.

Key differences from NSCLC:
- G13D is a distinct allele group (not grouped with G13C)
- A146T is a CRC-specific hotspot
- BRAF V600E is mutually exclusive with KRAS; BRAF-mutant lines go in KRAS-WT group
"""

from __future__ import annotations

import pandas as pd

# CRC oncogenic driver genes
DRIVER_GENES = ["KRAS", "APC", "TP53", "PIK3CA", "BRAF", "SMAD4"]

# CRC-specific KRAS hotspot allele map.
# Unlike NSCLC, G13D is separated from G13C and A146T is its own group.
KRAS_ALLELE_MAP = {
    "p.G12D": "G12D",
    "p.G13D": "G13D",
    "p.G12V": "G12V",
    "p.G12C": "G12C",
    "p.G12A": "G12A",
    "p.Q61H": "Q61H",
    "p.A146T": "A146T",
}

# Variant classifications considered functionally damaging (TCGA MAF format)
DAMAGING_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
}

# Known MSI-H CRC cell lines (curated from CCLE/DepMap literature).
# Used as fallback when DepMap metadata lacks explicit MSI annotation.
KNOWN_MSI_H_LINES = {
    "HCT116", "DLD1", "HCT15", "RKO", "LOVO", "LS174T", "SW48",
    "GP5D", "SNU407", "SNUC4", "KM12", "LS411N",
}


def classify_kras_allele(protein_changes: list[str]) -> str:
    """Classify KRAS allele from a list of protein changes.

    Returns one of: G12D, G13D, G12V, G12C, G12A, Q61H, A146T, KRAS_other, WT.
    """
    hotspot_alleles = []
    for pc in protein_changes:
        if pd.notna(pc):
            allele = KRAS_ALLELE_MAP.get(pc)
            if allele:
                hotspot_alleles.append(allele)

    if not hotspot_alleles:
        return "KRAS_other" if protein_changes else "WT"

    # Priority order: most common CRC alleles first
    for preferred in ["G12D", "G13D", "G12V", "G12C", "G12A", "Q61H", "A146T"]:
        if preferred in hotspot_alleles:
            return preferred
    return hotspot_alleles[0]


def classify_msi_status(
    stripped_name: str,
    metadata_msi: str | None = None,
) -> str:
    """Classify MSI status for a CRC cell line.

    Checks DepMap metadata annotation first, falls back to curated list.
    Returns 'MSI-H' or 'MSS'.
    """
    if metadata_msi and str(metadata_msi).upper() in ("MSI", "MSI-H"):
        return "MSI-H"
    if stripped_name in KNOWN_MSI_H_LINES:
        return "MSI-H"
    return "MSS"
