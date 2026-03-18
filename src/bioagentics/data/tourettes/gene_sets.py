"""TS risk gene sets for the CSTC circuit expression atlas.

Curated gene sets used across all downstream analyses (AHBA spatial mapping,
BrainSpan developmental trajectories, single-cell deconvolution, WGCNA,
iron pathway profiling, circuit vulnerability scoring).

Gene sets:
1. tsaicg_gwas — TSAICG GWAS risk loci (Yu 2019, Willsey 2024)
2. rare_variant — Established rare variant genes with strong TS evidence
3. iron_homeostasis — Iron metabolism pathway (7T MRI iron depletion study)
4. hippo_signaling — Hippo/WWC1 pathway (Chen 2025 functional validation)
5. ts_combined — Union of all above

Usage:
    from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets
    genes = get_gene_set("tsaicg_gwas")
    for symbol, desc in genes.items():
        print(f"{symbol}: {desc}")
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# TSAICG GWAS risk genes
# From Yu et al. (Am J Hum Genet 2019) meta-analysis (4,819 cases / 9,488
# controls) and Willsey et al. (Cell 2024) updated analysis.
# Genome-wide significant loci + suggestive loci with biological plausibility.
# ---------------------------------------------------------------------------
TSAICG_GWAS: dict[str, str] = {
    "FLT3": "13q12 — receptor tyrosine kinase; first GWS locus (Yu 2019)",
    "MEIS1": "2p14 — homeobox TF, dopamine neuron development (Yu 2019)",
    "PTPRD": "9p24 — receptor protein tyrosine phosphatase delta; also ADHD GWAS",
    "SEMA6D": "15q21 — semaphorin, axon guidance in CSTC circuits",
    "NTN4": "12q22 — netrin-4, axon guidance (Scharf 2013, Yu 2019)",
    "COL27A1": "9q32 — collagen XXVII alpha-1, neural ECM (Scharf 2013)",
    "CADPS2": "7q31 — Ca2+-dependent secretion activator, synaptic vesicle",
    "OPRD1": "1p36 — delta opioid receptor (Yu 2019)",
    "ASH1L": "1q22 — histone methyltransferase H3K36; also de novo variants",
    "CELSR3": "3p21 — planar cell polarity, axon guidance; also de novo variants",
    "MAOA": "Xp11 — monoamine oxidase A, monoamine catabolism",
    "NRXN1": "2p16 — neurexin 1, synaptic adhesion; also rare deletions",
    "CNTNAP2": "7q35 — contactin-associated protein-like 2, axon-glial",
    "NEGR1": "1p31 — neuronal growth regulator 1 (Willsey 2024)",
    "LHX6": "9q33 — LIM homeobox 6, striatal interneuron specification",
    "RBFOX1": "16p13 — RNA-binding Fox-1, neuronal splicing regulator",
}

# ---------------------------------------------------------------------------
# Rare variant genes — strong evidence from family/functional studies
# ---------------------------------------------------------------------------
RARE_VARIANT: dict[str, str] = {
    "SLITRK1": "13q31 — axon guidance; first TS gene (Abelson 2005 Science)",
    "HDC": "15q21 — histidine decarboxylase; W317X (Ercan-Sencicek 2010 NEJM)",
    "NRXN1": "2p16 — neurexin 1; recurrent exonic deletions (Fernandez 2012 PNAS)",
    "CNTN6": "3p26 — contactin-6, neural circuit formation (Paschou 2014)",
    "WWC1": "5q34 — KIBRA/Hippo pathway; W88C functional validation (Chen 2025)",
}

# ---------------------------------------------------------------------------
# Iron homeostasis pathway
# Informed by 7T MRI study (Brain Communications 2025) showing circuit-wide
# iron depletion in caudate, pallidum, STN, thalamus, red nucleus, substantia
# nigra, with D1 receptor correlation to tic severity.
# ---------------------------------------------------------------------------
IRON_HOMEOSTASIS: dict[str, str] = {
    "TF": "3q22 — transferrin; serum iron transport protein",
    "TFRC": "3q29 — transferrin receptor 1 (TFR1); cellular iron uptake",
    "FTH1": "11q13 — ferritin heavy chain 1; intracellular iron storage",
    "FTL": "19q13 — ferritin light chain; intracellular iron storage",
    "ACO1": "9p21 — aconitase 1 / IRP1; iron-responsive element binding",
    "IREB2": "15q25 — iron-responsive element binding protein 2 (IRP2)",
    "HAMP": "19q13 — hepcidin; master regulator of systemic iron homeostasis",
    "SLC40A1": "2q32 — ferroportin; sole cellular iron exporter",
}

# ---------------------------------------------------------------------------
# Hippo signaling pathway
# Informed by WWC1 W88C functional study (Chen 2025 Science Advances)
# showing developmental-stage-specific dopamine dysregulation via
# KIBRA degradation through Hippo pathway.
# ---------------------------------------------------------------------------
HIPPO_SIGNALING: dict[str, str] = {
    "WWC1": "5q34 — KIBRA; Hippo pathway scaffold, TS functional validation",
    "YAP1": "11q22 — Yes-associated protein 1; Hippo effector transcription coactivator",
    "WWTR1": "3q25 — TAZ; YAP1 paralog, Hippo effector",
    "LATS1": "6q25 — large tumor suppressor kinase 1; core Hippo kinase",
    "LATS2": "13q12 — large tumor suppressor kinase 2; core Hippo kinase",
    "STK4": "20q13 — serine/threonine kinase 4 (MST1); upstream Hippo kinase",
    "STK3": "11p15 — serine/threonine kinase 3 (MST2); upstream Hippo kinase",
}

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_GENE_SETS: dict[str, dict[str, str]] = {
    "tsaicg_gwas": TSAICG_GWAS,
    "rare_variant": RARE_VARIANT,
    "iron_homeostasis": IRON_HOMEOSTASIS,
    "hippo_signaling": HIPPO_SIGNALING,
}


def get_gene_set(name: str) -> dict[str, str]:
    """Return a gene set by name (gene symbol -> description).

    The special name ``"ts_combined"`` returns the union of all sets.

    Raises
    ------
    KeyError
        If *name* is not a recognised gene set.
    """
    if name == "ts_combined":
        return _build_combined()
    if name not in _GENE_SETS:
        raise KeyError(
            f"Unknown gene set {name!r}. "
            f"Available: {', '.join(list_gene_sets())}"
        )
    return dict(_GENE_SETS[name])


def list_gene_sets() -> list[str]:
    """Return the names of all available gene sets (including ts_combined)."""
    return sorted([*_GENE_SETS.keys(), "ts_combined"])


def _build_combined() -> dict[str, str]:
    """Build the union of all individual gene sets."""
    combined: dict[str, str] = {}
    for gs in _GENE_SETS.values():
        for symbol, desc in gs.items():
            if symbol not in combined:
                combined[symbol] = desc
    return combined


def gene_symbols(name: str) -> list[str]:
    """Return sorted gene symbols for a given set (convenience helper)."""
    return sorted(get_gene_set(name).keys())
