"""TS risk gene sets and striatal cell-type markers.

Curated gene sets used across all downstream analyses (AHBA spatial mapping,
BrainSpan developmental trajectories, single-cell deconvolution, WGCNA,
iron pathway profiling, circuit vulnerability scoring).

Gene sets (TS risk):
1. tsaicg_gwas — TSAICG GWAS risk loci (Yu 2019, Willsey 2024)
2. rare_variant — Established rare variant genes with strong TS evidence
3. de_novo_variant — De novo variant genes from exome studies
4. iron_homeostasis — Iron metabolism pathway (7T MRI iron depletion study)
5. hippo_signaling — Hippo/WWC1 pathway (Chen 2025 functional validation)
6. hormone_receptors — Gonadal steroid hormone receptors (developmental modulation)
7. ts_combined — Union of all above

Cell-type markers (for deconvolution):
- d1_msn — D1 medium spiny neurons (direct pathway)
- d2_msn — D2 medium spiny neurons (indirect pathway)
- cholinergic_interneuron — Striatal cholinergic interneurons (ChAT+)
- pv_interneuron — Parvalbumin+ GABAergic interneurons
- sst_interneuron — Somatostatin+ GABAergic interneurons
- microglia — Microglial markers
- astrocyte — Astrocyte markers
- oligodendrocyte — Oligodendrocyte markers

Usage:
    from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets
    genes = get_gene_set("tsaicg_gwas")
    for symbol, desc in genes.items():
        print(f"{symbol}: {desc}")

    from bioagentics.data.tourettes.gene_sets import get_celltype_markers
    markers = get_celltype_markers("pv_interneuron")
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
    "BCL11B": "14q32 — BAF chromatin remodeling, striatal MSN differentiation (TSAICG 2024)",
    "NDFIP2": "13q31 — Nedd4 family interacting protein 2, ubiquitin signaling (TSAICG 2024)",
    "RBM26": "13q31 — RNA-binding motif protein 26, post-transcriptional regulation (TSAICG 2024)",
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
# De novo variant genes — from TS exome sequencing studies
# ---------------------------------------------------------------------------
DE_NOVO_VARIANT: dict[str, str] = {
    "PPP5C": "19q13 — protein phosphatase 5C; de novo missense in TS trio studies",
    "EXOC1": "4q12 — exocyst complex component 1; de novo variant, vesicle trafficking",
    "GXYLT1": "12q12 — glucoside xylosyltransferase 1; de novo variant, Notch glycosylation",
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
# Gonadal steroid hormone receptors
# For developmental trajectory modeling: hormonal modulation of dopamine
# signaling during puberty (Phase 4 persistence/remission model).
# ---------------------------------------------------------------------------
HORMONE_RECEPTORS: dict[str, str] = {
    "AR": "Xq12 — androgen receptor; pubertal modulation of dopamine signaling",
    "ESR1": "6q25 — estrogen receptor alpha; neuromodulatory effects on CSTC",
    "ESR2": "14q23 — estrogen receptor beta; neuroprotective, GABAergic modulation",
}

# ===========================================================================
# Striatal cell-type marker genes
# Canonical markers from literature for bulk RNA-seq deconvolution.
# Can be supplemented with Wang et al. 2025 snRNA-seq markers when available.
# ===========================================================================

# ---------------------------------------------------------------------------
# D1-MSN (direct / striatonigral pathway)
# Markers: Gerfen & Surmeier (2011) Annu Rev Neurosci; Lobo et al. (2006)
# ---------------------------------------------------------------------------
D1_MSN_MARKERS: dict[str, str] = {
    "DRD1": "D1 dopamine receptor — defining D1-MSN marker",
    "TAC1": "tachykinin/substance P precursor — D1-MSN neuropeptide",
    "PDYN": "prodynorphin — D1-MSN neuropeptide",
    "CHRM4": "muscarinic M4 receptor — enriched in D1-MSNs",
    "ISL1": "ISL1 transcription factor — striatonigral identity",
}

# ---------------------------------------------------------------------------
# D2-MSN (indirect / striatopallidal pathway)
# Markers: Gerfen & Surmeier (2011); Lobo et al. (2006)
# ---------------------------------------------------------------------------
D2_MSN_MARKERS: dict[str, str] = {
    "DRD2": "D2 dopamine receptor — defining D2-MSN marker",
    "PENK": "proenkephalin — D2-MSN neuropeptide",
    "ADORA2A": "adenosine A2a receptor — D2-MSN enriched",
    "GPR6": "G protein-coupled receptor 6 — D2-MSN enriched",
    "SP9": "Sp9 transcription factor — striatopallidal identity",
}

# ---------------------------------------------------------------------------
# Cholinergic interneurons (ChAT+)
# Markers: Muñoz-Manchado et al. (2018) Cell Reports
# ---------------------------------------------------------------------------
CHOLINERGIC_INTERNEURON_MARKERS: dict[str, str] = {
    "CHAT": "choline acetyltransferase — defining cholinergic marker",
    "SLC5A7": "high-affinity choline transporter (CHT)",
    "SLC18A3": "vesicular acetylcholine transporter (VAChT)",
    "LHX8": "LIM homeobox 8 — cholinergic interneuron TF",
    "GBX2": "gastrulation brain homeobox 2 — cholinergic specification",
}

# ---------------------------------------------------------------------------
# Parvalbumin+ GABAergic interneurons (PV+)
# Markers: Muñoz-Manchado et al. (2018); Kepecs & Bhatt (2019)
# ---------------------------------------------------------------------------
PV_INTERNEURON_MARKERS: dict[str, str] = {
    "PVALB": "parvalbumin — defining PV+ interneuron marker",
    "KCNC1": "Kv3.1 potassium channel — fast-spiking phenotype",
    "KCNC2": "Kv3.2 potassium channel — fast-spiking phenotype",
    "EYA1": "EYA transcriptional coactivator 1 — PV+ identity",
    "TAC3": "tachykinin 3 (neurokinin B) — PV+ interneuron marker",
}

# ---------------------------------------------------------------------------
# SST+ GABAergic interneurons
# Markers: Muñoz-Manchado et al. (2018)
# ---------------------------------------------------------------------------
SST_INTERNEURON_MARKERS: dict[str, str] = {
    "SST": "somatostatin — defining SST+ interneuron marker",
    "NPY": "neuropeptide Y — SST+ co-expressed",
    "NOS1": "neuronal nitric oxide synthase — SST+ co-expressed",
    "CALB2": "calretinin — subset of SST+ interneurons",
}

# ---------------------------------------------------------------------------
# Microglia
# Markers: Butovsky et al. (2014) Nat Neurosci; Bennett et al. (2016)
# ---------------------------------------------------------------------------
MICROGLIA_MARKERS: dict[str, str] = {
    "CX3CR1": "fractalkine receptor — microglial homeostatic marker",
    "P2RY12": "purinergic receptor P2Y12 — homeostatic microglia",
    "TMEM119": "transmembrane protein 119 — microglia-specific",
    "AIF1": "allograft inflammatory factor 1 (Iba1) — microglia/macrophage",
    "CSF1R": "colony stimulating factor 1 receptor — microglial survival",
}

# ---------------------------------------------------------------------------
# Astrocytes
# Markers: Cahoy et al. (2008) J Neurosci; Zhang et al. (2016)
# ---------------------------------------------------------------------------
ASTROCYTE_MARKERS: dict[str, str] = {
    "GFAP": "glial fibrillary acidic protein — astrocyte marker",
    "AQP4": "aquaporin 4 — astrocyte endfeet, water channel",
    "ALDH1L1": "aldehyde dehydrogenase 1L1 — pan-astrocyte marker",
    "S100B": "S100 calcium-binding protein B — astrocyte marker",
    "SLC1A2": "GLT-1/EAAT2 — astrocytic glutamate transporter",
}

# ---------------------------------------------------------------------------
# Oligodendrocytes
# Markers: Zhang et al. (2014) J Neurosci
# ---------------------------------------------------------------------------
OLIGODENDROCYTE_MARKERS: dict[str, str] = {
    "MBP": "myelin basic protein — mature oligodendrocyte marker",
    "PLP1": "proteolipid protein 1 — myelin structural protein",
    "MOG": "myelin oligodendrocyte glycoprotein — mature OL marker",
    "OLIG2": "oligodendrocyte transcription factor 2 — OL lineage",
    "CNP": "2',3'-cyclic nucleotide 3' phosphodiesterase — OL marker",
}

# ---------------------------------------------------------------------------
# Cell-type marker registry
# ---------------------------------------------------------------------------
_CELLTYPE_MARKERS: dict[str, dict[str, str]] = {
    "d1_msn": D1_MSN_MARKERS,
    "d2_msn": D2_MSN_MARKERS,
    "cholinergic_interneuron": CHOLINERGIC_INTERNEURON_MARKERS,
    "pv_interneuron": PV_INTERNEURON_MARKERS,
    "sst_interneuron": SST_INTERNEURON_MARKERS,
    "microglia": MICROGLIA_MARKERS,
    "astrocyte": ASTROCYTE_MARKERS,
    "oligodendrocyte": OLIGODENDROCYTE_MARKERS,
}

# ---------------------------------------------------------------------------
# TS risk gene registry
# ---------------------------------------------------------------------------
_GENE_SETS: dict[str, dict[str, str]] = {
    "tsaicg_gwas": TSAICG_GWAS,
    "rare_variant": RARE_VARIANT,
    "de_novo_variant": DE_NOVO_VARIANT,
    "iron_homeostasis": IRON_HOMEOSTASIS,
    "hippo_signaling": HIPPO_SIGNALING,
    "hormone_receptors": HORMONE_RECEPTORS,
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


# ---------------------------------------------------------------------------
# Cell-type marker accessors
# ---------------------------------------------------------------------------


def get_celltype_markers(name: str) -> dict[str, str]:
    """Return cell-type marker genes by name (gene symbol -> description).

    Raises
    ------
    KeyError
        If *name* is not a recognised cell-type marker set.
    """
    if name not in _CELLTYPE_MARKERS:
        raise KeyError(
            f"Unknown cell-type marker set {name!r}. "
            f"Available: {', '.join(list_celltype_markers())}"
        )
    return dict(_CELLTYPE_MARKERS[name])


def list_celltype_markers() -> list[str]:
    """Return the names of all available cell-type marker sets."""
    return sorted(_CELLTYPE_MARKERS.keys())
