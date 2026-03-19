"""Innate immunity gene module configuration for the innate deficiency model.

Defines gene sets for testing the hypothesis that PANS/PANDAS reflects a primary
deficiency of innate immunity (lectin complement, trained immunity) with secondary
compensatory adaptive immune overshoot.

Modules:
  1. Lectin complement pathway (MBL2, MASP1/2, ficolins, collectins)
  2. Classical innate defense (NK, neutrophil, monocyte effector genes)
  3. Trained immunity markers (epigenetic reprogramming of innate cells)
  4. Adaptive immune reference (T cell, B cell genes for ratio computation)
  5. cGAS-STING pathway (TREX1/SAMHD1 axis for Phase 2 comparison)
  6. Cytokine classification (innate vs adaptive)

Usage::

    from bioagentics.pandas_pans.innate_immunity_modules import (
        INNATE_MODULES,
        ADAPTIVE_MODULES,
        CYTOKINE_CLASSIFICATION,
        get_all_innate_genes,
        get_all_adaptive_genes,
    )
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lectin complement pathway
# ---------------------------------------------------------------------------

LECTIN_COMPLEMENT_GENES: list[str] = [
    "MBL2",    # Mannose-binding lectin — pathway initiator, GAS opsonization
    "MASP1",   # MBL-associated serine protease 1
    "MASP2",   # MBL-associated serine protease 2 — cleaves C4/C2
    "FCN1",    # Ficolin-1 (M-ficolin) — monocyte/granulocyte
    "FCN2",    # Ficolin-2 (L-ficolin) — liver/serum
    "FCN3",    # Ficolin-3 (H-ficolin) — lung/serum
    "COLEC11", # Collectin-11 (CL-K1) — lectin pathway activator
]

# Extended lectin/complement components downstream of the initiation complex
LECTIN_COMPLEMENT_DOWNSTREAM: list[str] = [
    "C2",      # Complement C2 — cleaved by MASP2
    "C4A",     # Complement C4A — cleaved by MASP2
    "C4B",     # Complement C4B
    "C3",      # Complement C3 — central to all complement pathways
    "COLEC10", # Collectin-10 (CL-L1) — forms heteromer with COLEC11
    "MAP1",    # MBL/ficolin-associated protein 1 (MAP19 splice variant of MASP2)
]

# ---------------------------------------------------------------------------
# Classical innate defense genes
# ---------------------------------------------------------------------------

# NK cell effector genes
NK_CELL_GENES: list[str] = [
    "NKG7",    # NK granule protein 7
    "GNLY",    # Granulysin
    "GZMB",    # Granzyme B
    "GZMA",    # Granzyme A
    "PRF1",    # Perforin
    "KLRD1",   # CD94 — NK cell receptor
    "KLRK1",   # NKG2D — activating receptor
    "NCAM1",   # CD56 — NK cell marker
    "FCGR3A",  # CD16 — ADCC mediator
    "FGFBP2",  # CD56dim cytotoxic NK marker
]

# Neutrophil defense genes
NEUTROPHIL_GENES: list[str] = [
    "ELANE",   # Neutrophil elastase
    "MPO",     # Myeloperoxidase
    "CTSG",    # Cathepsin G
    "PRTN3",   # Proteinase 3
    "AZU1",    # Azurocidin
    "DEFA1",   # Defensin alpha 1
    "DEFA3",   # Defensin alpha 3
    "DEFA4",   # Defensin alpha 4
    "LTF",     # Lactoferrin
    "CAMP",    # Cathelicidin (LL-37)
    "LCN2",    # Lipocalin 2 (NGAL)
    "S100A8",  # Calprotectin subunit
    "S100A9",  # Calprotectin subunit
    "S100A12", # EN-RAGE — TLR4 ligand
]

# Monocyte/macrophage innate defense genes
MONOCYTE_DEFENSE_GENES: list[str] = [
    "CD14",    # LPS co-receptor
    "TLR2",    # Gram-positive pathogen recognition
    "TLR4",    # LPS receptor
    "MYD88",   # TLR signaling adaptor
    "IRAK4",   # IL-1R-associated kinase 4
    "TRAF6",   # TNF receptor-associated factor 6
    "NLRP3",   # Inflammasome sensor
    "CASP1",   # Caspase-1 — IL-1b/IL-18 processing
    "PYCARD",  # ASC — inflammasome adaptor
    "LYZ",     # Lysozyme
    "FCN1",    # Ficolin-1 (also in lectin complement)
    "NFKB1",   # NF-kB — innate signaling hub
]

# Pattern recognition receptors (broader innate sensing)
PRR_GENES: list[str] = [
    "TLR1", "TLR2", "TLR4", "TLR6", "TLR7", "TLR8", "TLR9",
    "DDX58",   # RIG-I
    "IFIH1",   # MDA5
    "MAVS",    # Mitochondrial antiviral signaling
    "STING1",  # STING — cGAS-STING pathway
    "CGAS",    # Cyclic GMP-AMP synthase
    "NOD2",    # Nucleotide-binding oligomerization domain 2
    "NLRP3",   # Inflammasome
]

# ---------------------------------------------------------------------------
# Trained immunity markers
# ---------------------------------------------------------------------------

TRAINED_IMMUNITY_GENES: list[str] = [
    # Epigenetic reprogramming markers (Netea et al.)
    "KDM6B",   # H3K27 demethylase — trained immunity activation
    "KAT2A",   # Histone acetyltransferase — H3K27ac mark
    "KAT2B",   # Histone acetyltransferase
    "MTOR",    # mTOR — metabolic reprogramming in trained immunity
    "HIF1A",   # Hypoxia-inducible factor — glycolytic shift
    "AKT1",    # PI3K/AKT — mTOR activation
    # Metabolic markers
    "HK2",     # Hexokinase 2 — glycolysis
    "PFKFB3",  # Phosphofructokinase — glycolysis regulator
    "SLC2A1",  # GLUT1 — glucose transporter
    "ACLY",    # ATP citrate lyase — mevalonate pathway
    "HMGCR",   # HMG-CoA reductase — mevalonate pathway
    # Functional output markers
    "TNF",     # TNF-alpha — enhanced in trained immunity
    "IL6",     # IL-6 — enhanced in trained immunity
    "IL1B",    # IL-1beta — enhanced in trained immunity
]

# ---------------------------------------------------------------------------
# cGAS-STING pathway (Phase 2 comparison — TREX1/SAMHD1 variants)
# ---------------------------------------------------------------------------

CGAS_STING_GENES: list[str] = [
    "CGAS",    # Cyclic GMP-AMP synthase
    "STING1",  # Stimulator of IFN genes
    "TBK1",    # TANK-binding kinase 1
    "IRF3",    # Interferon regulatory factor 3
    "TREX1",   # 3'-5' exonuclease — Aicardi-Goutieres gene
    "SAMHD1",  # dNTPase — Aicardi-Goutieres gene
    "IFNB1",   # IFN-beta — type I interferon output
    "IFNA1",   # IFN-alpha
    "MX1",     # IFN-stimulated gene
    "ISG15",   # IFN-stimulated gene
    "IFIT1",   # IFN-stimulated gene
    "OAS1",    # IFN-stimulated gene
]

# ---------------------------------------------------------------------------
# Adaptive immune reference genes (for innate/adaptive ratio)
# ---------------------------------------------------------------------------

# T cell genes
T_CELL_GENES: list[str] = [
    "CD3D", "CD3E", "CD3G",  # TCR complex
    "CD4",     # CD4+ T cells
    "CD8A",    # CD8+ T cells
    "TRAC",    # TCR alpha constant
    "TRBC1",   # TCR beta constant
    "IL7R",    # CD127 — naive/memory T cells
    "LEF1",    # T cell transcription factor
    "TCF7",    # T cell factor
]

# B cell / immunoglobulin genes
B_CELL_GENES: list[str] = [
    "CD19",    # B cell co-receptor
    "CD79A",   # B cell receptor signaling
    "CD79B",   # B cell receptor signaling
    "MS4A1",   # CD20
    "PAX5",    # B cell transcription factor
    "IGHM",    # IgM heavy chain
    "IGHG1",   # IgG1 heavy chain
    "IGHA1",   # IgA1 heavy chain
    "JCHAIN",  # Joining chain — secretory Ig
    "AICDA",   # Activation-induced cytidine deaminase — class switching
]

# T helper subset signature genes (adaptive immune activation markers)
TH_SIGNATURE_GENES: list[str] = [
    "TBX21",   # Th1 — T-bet
    "GATA3",   # Th2
    "RORC",    # Th17
    "FOXP3",   # Treg
    "BCL6",    # Tfh
    "IFNG",    # Th1 effector cytokine
    "IL4",     # Th2 effector cytokine
    "IL17A",   # Th17 effector cytokine
    "IL21",    # Tfh effector cytokine
    "IL10",    # Treg/regulatory cytokine
]

# ---------------------------------------------------------------------------
# Cytokine classification (innate vs adaptive)
# ---------------------------------------------------------------------------

CYTOKINE_CLASSIFICATION: dict[str, dict[str, list[str]]] = {
    "innate": {
        "genes": [
            "IL1B",   # IL-1beta — inflammasome product
            "TNF",    # TNF-alpha — macrophage/monocyte
            "IL6",    # IL-6 — pleiotropic innate
            "CXCL8",  # IL-8 — neutrophil chemoattractant
            "IL18",   # IL-18 — inflammasome product
            "IL1A",   # IL-1alpha — alarmin
            "CCL2",   # MCP-1 — monocyte recruitment
            "CCL3",   # MIP-1alpha
            "CCL4",   # MIP-1beta
            "CXCL10", # IP-10 — IFN-inducible
        ],
        "proteins": [
            "IL-1β", "TNF-α", "IL-6", "IL-8", "IL-18",
            "IL-1α", "MCP-1", "MIP-1α", "MIP-1β", "IP-10",
        ],
    },
    "adaptive": {
        "genes": [
            "IL4",    # Th2
            "IL13",   # Th2
            "IL5",    # Th2 / eosinophil
            "IFNG",   # Th1 — IFN-gamma
            "IL17A",  # Th17
            "IL17F",  # Th17
            "IL21",   # Tfh
            "IL22",   # Th17/Th22
            "IL2",    # T cell growth factor
            "IL12A",  # DC → Th1 polarization
        ],
        "proteins": [
            "IL-4", "IL-13", "IL-5", "IFN-γ", "IL-17A",
            "IL-17F", "IL-21", "IL-22", "IL-2", "IL-12",
        ],
    },
    "regulatory": {
        "genes": [
            "IL10",   # Treg / regulatory
            "TGFB1",  # TGF-beta1 — immunosuppressive
        ],
        "proteins": [
            "IL-10", "TGF-β1",
        ],
    },
}

# ---------------------------------------------------------------------------
# Aggregated module dictionaries
# ---------------------------------------------------------------------------

INNATE_MODULES: dict[str, list[str]] = {
    "lectin_complement": LECTIN_COMPLEMENT_GENES,
    "lectin_complement_downstream": LECTIN_COMPLEMENT_DOWNSTREAM,
    "nk_cell_effector": NK_CELL_GENES,
    "neutrophil_defense": NEUTROPHIL_GENES,
    "monocyte_defense": MONOCYTE_DEFENSE_GENES,
    "pattern_recognition_receptors": PRR_GENES,
    "trained_immunity": TRAINED_IMMUNITY_GENES,
    "cgas_sting_pathway": CGAS_STING_GENES,
}

ADAPTIVE_MODULES: dict[str, list[str]] = {
    "t_cell_core": T_CELL_GENES,
    "b_cell_core": B_CELL_GENES,
    "th_signatures": TH_SIGNATURE_GENES,
}


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def get_all_innate_genes() -> list[str]:
    """Return deduplicated list of all innate immune module genes."""
    seen: set[str] = set()
    result: list[str] = []
    for genes in INNATE_MODULES.values():
        for g in genes:
            if g not in seen:
                seen.add(g)
                result.append(g)
    return result


def get_all_adaptive_genes() -> list[str]:
    """Return deduplicated list of all adaptive immune module genes."""
    seen: set[str] = set()
    result: list[str] = []
    for genes in ADAPTIVE_MODULES.values():
        for g in genes:
            if g not in seen:
                seen.add(g)
                result.append(g)
    return result


def get_innate_adaptive_ratio_genes() -> tuple[list[str], list[str]]:
    """Return (innate_genes, adaptive_genes) for ratio computation.

    Uses the core modules most relevant for the innate/adaptive ratio:
    - Innate: lectin complement + NK + neutrophil + monocyte defense
    - Adaptive: T cell core + B cell core + Th signatures
    """
    innate = get_all_innate_genes()
    adaptive = get_all_adaptive_genes()
    return innate, adaptive


def get_cytokine_genes(category: str) -> list[str]:
    """Return gene symbols for a cytokine category ('innate', 'adaptive', 'regulatory')."""
    if category not in CYTOKINE_CLASSIFICATION:
        raise ValueError(f"Unknown category: {category}. Use: {list(CYTOKINE_CLASSIFICATION)}")
    return list(CYTOKINE_CLASSIFICATION[category]["genes"])


def get_cytokine_proteins(category: str) -> list[str]:
    """Return protein names for a cytokine category (for matching assay data)."""
    if category not in CYTOKINE_CLASSIFICATION:
        raise ValueError(f"Unknown category: {category}. Use: {list(CYTOKINE_CLASSIFICATION)}")
    return list(CYTOKINE_CLASSIFICATION[category]["proteins"])


def get_lectin_variant_genes() -> list[str]:
    """Return lectin complement genes that have known PANS-associated variants.

    These are the genes from pans_variants.py with loss-of-function variants
    enriched in PANS patients (Vettiatil 2026, FDR=1.12e-5).
    """
    return ["MBL2", "MASP1", "MASP2"]
