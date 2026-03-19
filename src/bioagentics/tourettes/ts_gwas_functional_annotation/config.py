"""Configuration for ts-gwas-functional-annotation pipeline.

Paths, constants, and TS-relevant gene set definitions.
"""

from __future__ import annotations

from bioagentics.config import REPO_ROOT

# --- Paths ---
PROJECT = "ts-gwas-functional-annotation"
DATA_DIR = REPO_ROOT / "data" / "tourettes" / PROJECT
RAW_DIR = DATA_DIR / "raw"
GWAS_DIR = DATA_DIR / "gwas"
EQTL_DIR = DATA_DIR / "eqtl"
HIC_DIR = DATA_DIR / "hic"
GENE_ANNOT_DIR = DATA_DIR / "gene_annotations"
GENE_SETS_DIR = DATA_DIR / "gene_sets"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / PROJECT
MAPPING_DIR = OUTPUT_DIR / "snp_to_gene"
MAGMA_DIR = OUTPUT_DIR / "magma_results"
GENE_ANALYSIS_DIR = OUTPUT_DIR / "gene_analysis"
ENRICHMENT_DIR = OUTPUT_DIR / "enrichment"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    """Create all project directories."""
    for d in [
        DATA_DIR, RAW_DIR, GWAS_DIR, EQTL_DIR, HIC_DIR, GENE_ANNOT_DIR, GENE_SETS_DIR,
        OUTPUT_DIR, MAPPING_DIR, MAGMA_DIR, GENE_ANALYSIS_DIR, ENRICHMENT_DIR,
        FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# --- GWAS QC thresholds ---
GWAS_QC = {
    "min_maf": 0.01,
    "min_info": 0.6,
    "max_p": 1.0,
    "min_p": 0.0,
    "max_abs_beta": 10.0,
}

# SNP-to-gene mapping window (kb)
POSITIONAL_WINDOW_KB = 10

# GTEx brain tissues for eQTL mapping
GTEX_BRAIN_TISSUES = [
    "Brain_Caudate_basal_ganglia",
    "Brain_Putamen_basal_ganglia",
    "Brain_Nucleus_accumbens_basal_ganglia",
    "Brain_Frontal_Cortex_BA9",
    "Brain_Cortex",
    "Brain_Cerebellum",
    "Brain_Cerebellar_Hemisphere",
    "Brain_Anterior_cingulate_cortex_BA24",
    "Brain_Hippocampus",
    "Brain_Amygdala",
    "Brain_Hypothalamus",
    "Brain_Substantia_nigra",
    "Brain_Spinal_cord_cervical_c-1",
]

# eQTL significance threshold
EQTL_FDR_THRESHOLD = 0.05

# Known TS GWAS candidate genes from prior literature
TS_CANDIDATE_GENES = [
    "FLT3", "MECR", "MEIS1", "SLITRK1", "HDC", "NRXN1",
    "BCL11B", "CELSR3", "CNTN6", "SEMA6D", "NTN4",
]

# Known TS de novo variant genes
TS_DE_NOVO_GENES = [
    "WWC1", "CELSR3", "NIPBL", "FN1", "DNMT3A",
    "OFD1", "PTEN", "HDC",
]
