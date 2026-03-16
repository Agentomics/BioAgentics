"""Shared NSCLC constants and classification logic.

Used by both nsclc_depmap (cell lines) and nsclc_tcga (patients) to ensure
consistent driver gene definitions, allele groupings, and subtype labels.
"""

from __future__ import annotations

import pandas as pd

# Oncogenic driver genes to annotate
DRIVER_GENES = [
    "KRAS", "TP53", "STK11", "KEAP1", "EGFR", "ALK",
    "MET", "BRAF", "ROS1", "ERBB2", "NF1", "RB1",
]

# KRAS hotspot codon groupings
KRAS_ALLELE_MAP = {
    "p.G12C": "G12C",
    "p.G12D": "G12D",
    "p.G12V": "G12V",
    "p.G12A": "G12_other",
    "p.G12F": "G12_other",
    "p.G12R": "G12_other",
    "p.G12S": "G12_other",
    "p.G13C": "G13",
    "p.G13D": "G13",
    "p.Q61H": "Q61",
    "p.Q61K": "Q61",
    "p.Q61L": "Q61",
}

# Variant classifications considered functionally damaging (TCGA MAF)
DAMAGING_CLASSIFICATIONS = {
    "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del",
    "Frame_Shift_Ins", "Splice_Site", "In_Frame_Del", "In_Frame_Ins",
    "Translation_Start_Site", "Nonstop_Mutation",
}


def classify_kras_allele(protein_changes: list[str]) -> str:
    """Classify KRAS allele type from a list of protein changes."""
    hotspot_alleles = []
    for pc in protein_changes:
        if pd.notna(pc):
            allele = KRAS_ALLELE_MAP.get(pc)
            if allele:
                hotspot_alleles.append(allele)

    if not hotspot_alleles:
        return "other" if protein_changes else "WT"

    for preferred in ["G12C", "G12D", "G12V"]:
        if preferred in hotspot_alleles:
            return preferred
    return hotspot_alleles[0]


def classify_molecular_subtype(row: pd.Series) -> str:
    """Classify NSCLC molecular subtype from driver mutation flags.

    Expects row to have bool columns: KRAS_mutated, TP53_mutated, STK11_mutated.
    Returns one of: KP, KL, KPL, KOnly, KRAS-WT.
    """
    if not row["KRAS_mutated"]:
        return "KRAS-WT"
    if row["TP53_mutated"] and row["STK11_mutated"]:
        return "KPL"
    if row["TP53_mutated"]:
        return "KP"
    if row["STK11_mutated"]:
        return "KL"
    return "KOnly"
