"""Curated PANS ultra-rare genetic variant gene list organized by pathway axis.

Source: Vettiatil D et al. 2026, "Ultrarare Variants in DNA Damage Repair and
Mitochondrial Genes in Pediatric Acute-Onset Neuropsychiatric Syndrome,"
Dev Neurosci (PMID 41662332). Original discovery cohort from Scientific Reports
s41598-022-15279-3. MBL2 addition from research_director task #335.

The 22-gene set is grouped into five pathway axes:
  1. DDR-cGAS-STING/AIM2 inflammasome (11 genes)
  2. Mitochondrial-innate immunity (2 genes)
  3. Gut-immune (3 genes)
  4. Lectin complement (3 genes)
  5. Chromatin/neuroprotection (3 genes)
"""

from __future__ import annotations

import pandas as pd

# Pathway axis labels
AXIS_DDR = "DDR-cGAS-STING/AIM2 inflammasome"
AXIS_MITO = "Mitochondrial-innate immunity"
AXIS_GUT = "Gut-immune"
AXIS_LECTIN = "Lectin complement"
AXIS_CHROMATIN = "Chromatin/neuroprotection"

# Curated variant gene records.
# variant_type: pathogenic (P), likely pathogenic (LP), or variant of uncertain
# significance (VUS) per Vettiatil et al. 2026 classification.
# functional_annotation: brief summary of known functional impact from published
# data (CADD, PolyPhen-2, SIFT scores where reported).
_VARIANT_RECORDS: list[dict[str, str]] = [
    # ── DDR-cGAS-STING/AIM2 inflammasome ──
    {
        "gene_symbol": "CUX1",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Transcription factor; DDR regulation; haploinsufficiency linked to myeloid malignancy",
    },
    {
        "gene_symbol": "USP45",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Deubiquitinase in base excision repair; CADD>20 damaging",
    },
    {
        "gene_symbol": "PARP14",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "ADP-ribosyltransferase; macrophage polarization and STAT6 signaling; immune-DDR crosstalk",
    },
    {
        "gene_symbol": "UVSSA",
        "pathway_axis": AXIS_DDR,
        "variant_type": "P",
        "functional_annotation": "Transcription-coupled nucleotide excision repair; UV-sensitive syndrome",
    },
    {
        "gene_symbol": "EP300",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Histone acetyltransferase; DDR signaling; PolyPhen-2 probably damaging",
    },
    {
        "gene_symbol": "TREX1",
        "pathway_axis": AXIS_DDR,
        "variant_type": "P",
        "functional_annotation": "3'-5' exonuclease; degrades cytosolic DNA; loss activates cGAS-STING type I IFN; Aicardi-Goutieres",
    },
    {
        "gene_symbol": "SAMHD1",
        "pathway_axis": AXIS_DDR,
        "variant_type": "P",
        "functional_annotation": "dNTPase; innate immunity restrictor; loss activates cGAS-STING; Aicardi-Goutieres",
    },
    {
        "gene_symbol": "STK19",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Serine/threonine kinase; NRAS signaling; DDR-associated; CADD>15",
    },
    {
        "gene_symbol": "PIDD1",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "p53-induced death domain protein; PIDDosome activates caspase-2 in DDR",
    },
    {
        "gene_symbol": "FANCD2",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Fanconi anemia pathway; interstrand crosslink repair; SIFT deleterious",
    },
    {
        "gene_symbol": "RAD54L",
        "pathway_axis": AXIS_DDR,
        "variant_type": "LP",
        "functional_annotation": "Homologous recombination repair; SWI/SNF helicase; CADD>20",
    },
    # ── Mitochondrial-innate immunity ──
    {
        "gene_symbol": "PRKN",
        "pathway_axis": AXIS_MITO,
        "variant_type": "LP",
        "functional_annotation": "E3 ubiquitin ligase; mitophagy; PINK1/Parkin pathway; mitochondrial DAMPs regulation",
    },
    {
        "gene_symbol": "POLG",
        "pathway_axis": AXIS_MITO,
        "variant_type": "LP",
        "functional_annotation": "Mitochondrial DNA polymerase gamma; mtDNA maintenance; mitochondrial dysfunction",
    },
    # ── Gut-immune ──
    {
        "gene_symbol": "LGALS4",
        "pathway_axis": AXIS_GUT,
        "variant_type": "VUS",
        "functional_annotation": "Galectin-4; intestinal epithelial integrity; gut mucosal immunity",
    },
    {
        "gene_symbol": "DUOX2",
        "pathway_axis": AXIS_GUT,
        "variant_type": "LP",
        "functional_annotation": "Dual oxidase 2; gut mucosal ROS defense; IBD-associated; antimicrobial barrier",
    },
    {
        "gene_symbol": "CCR9",
        "pathway_axis": AXIS_GUT,
        "variant_type": "VUS",
        "functional_annotation": "Chemokine receptor; gut-homing T cell trafficking; mucosal immune surveillance",
    },
    # ── Lectin complement ──
    {
        "gene_symbol": "MBL2",
        "pathway_axis": AXIS_LECTIN,
        "variant_type": "LP",
        "functional_annotation": "Mannose-binding lectin; lectin complement pathway initiator; GAS opsonization",
    },
    {
        "gene_symbol": "MASP1",
        "pathway_axis": AXIS_LECTIN,
        "variant_type": "VUS",
        "functional_annotation": "MBL-associated serine protease 1; lectin complement activation",
    },
    {
        "gene_symbol": "MASP2",
        "pathway_axis": AXIS_LECTIN,
        "variant_type": "VUS",
        "functional_annotation": "MBL-associated serine protease 2; cleaves C4/C2 to form C3 convertase",
    },
    # ── Chromatin/neuroprotection ──
    {
        "gene_symbol": "MYT1L",
        "pathway_axis": AXIS_CHROMATIN,
        "variant_type": "P",
        "functional_annotation": "Neuronal transcription factor; chromatin remodeling; neurodevelopmental disorders",
    },
    {
        "gene_symbol": "TEP1",
        "pathway_axis": AXIS_CHROMATIN,
        "variant_type": "LP",
        "functional_annotation": "Telomerase-associated protein 1; vault complex; telomere maintenance",
    },
    {
        "gene_symbol": "ADNP",
        "pathway_axis": AXIS_CHROMATIN,
        "variant_type": "P",
        "functional_annotation": "Activity-dependent neuroprotective protein; chromatin remodeling (SWI/SNF); Helsmoortel-Van der Aa syndrome",
    },
]

PATHWAY_AXES = [AXIS_DDR, AXIS_MITO, AXIS_GUT, AXIS_LECTIN, AXIS_CHROMATIN]

EXPECTED_GENE_COUNT = 22

AXIS_GENE_COUNTS = {
    AXIS_DDR: 11,
    AXIS_MITO: 2,
    AXIS_GUT: 3,
    AXIS_LECTIN: 3,
    AXIS_CHROMATIN: 3,
}


def get_pans_variant_genes() -> pd.DataFrame:
    """Return PANS variant genes as a DataFrame.

    Columns: gene_symbol, pathway_axis, variant_type, functional_annotation.
    """
    return pd.DataFrame(_VARIANT_RECORDS)


def get_pans_gene_symbols() -> list[str]:
    """Return flat list of PANS variant gene symbols."""
    return [r["gene_symbol"] for r in _VARIANT_RECORDS]
