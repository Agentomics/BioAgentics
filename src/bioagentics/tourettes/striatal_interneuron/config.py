"""Project configuration for ts-striatal-interneuron-pathology.

Paths, constants, and striatal interneuron marker gene definitions.
"""

from __future__ import annotations

from bioagentics.config import REPO_ROOT

# --- Paths ---
PROJECT = "ts-striatal-interneuron-pathology"
DATA_DIR = REPO_ROOT / "data" / "tourettes" / PROJECT
RAW_DIR = DATA_DIR / "raw"
REFERENCE_DIR = DATA_DIR / "reference"
OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / PROJECT
QC_DIR = OUTPUT_DIR / "qc"
INTEGRATION_DIR = OUTPUT_DIR / "integration"
DE_DIR = OUTPUT_DIR / "differential_expression"
FIGURES_DIR = OUTPUT_DIR / "figures"


def ensure_dirs() -> None:
    """Create all project directories."""
    for d in [
        DATA_DIR, RAW_DIR, REFERENCE_DIR,
        OUTPUT_DIR, QC_DIR, INTEGRATION_DIR, DE_DIR, FIGURES_DIR,
    ]:
        d.mkdir(parents=True, exist_ok=True)


# --- GEO Accessions ---
# Reference atlas: human dorsal striatum snRNA-seq, 14 interneuron subclasses
REFERENCE_DATASETS = {
    "GSE151761": {
        "description": "Human dorsal striatum snRNA-seq (Krienen et al. 2024, part 1)",
        "species": "human",
        "tissue": "dorsal striatum",
        "format": "supplementary_h5ad",
    },
    "GSE152058": {
        "description": "Human dorsal striatum snRNA-seq (Krienen et al. 2024, part 2)",
        "species": "human",
        "tissue": "dorsal striatum",
        "format": "supplementary_h5ad",
    },
}

# Wang et al. 2025 TS caudate data (NDA-restricted, accession TBD)
WANG_DATASET = {
    "accession": "TBD",
    "description": "snRNA-seq + snATAC-seq, TS caudate (6 TS, 6 control)",
    "pmid": "39892689",
    "n_ts": 6,
    "n_control": 6,
}


# --- Striatal Interneuron Marker Genes ---
# 14-subclass taxonomy from the reference atlas (Nature Communications 2024)
INTERNEURON_MARKERS: dict[str, list[str]] = {
    # Cholinergic interneurons
    "ChAT_CIN": ["CHAT", "SLC5A7", "ISL1", "SLC18A3", "ACHE", "LHX8"],
    # Parvalbumin interneurons
    "PV_FSI": ["PVALB", "GAD1", "GAD2", "KCNC1", "KCNC2", "SCN1A"],
    # Somatostatin interneurons
    "SST_PLTS": ["SST", "NPY", "NOS1", "PDYN", "TH", "PENK"],
    # Calretinin interneurons
    "CR": ["CALB2", "VIP", "CCK", "CNR1"],
    # NPY-NGF interneurons
    "NPY_NGF": ["NPY", "RELN", "NDNF", "LAMP5"],
    # TH+ interneurons
    "TH": ["TH", "DDC", "SLC6A3"],
    # PTHLH interneurons
    "PTHLH": ["PTHLH", "PVALB", "TAC1"],
}

# Broad striatal cell type markers
STRIATAL_CELL_MARKERS: dict[str, list[str]] = {
    # Medium spiny neurons (D1 and D2)
    "MSN_D1": ["DRD1", "TAC1", "PDYN", "ISL1", "EBFR1"],
    "MSN_D2": ["DRD2", "PENK", "ADORA2A", "GPR6", "SP9"],
    # Interneurons (broad)
    "Interneuron": ["GAD1", "GAD2", "SST", "PVALB", "CHAT", "NPY"],
    # Astrocytes
    "Astrocyte": ["AQP4", "GFAP", "SLC1A2", "SLC1A3", "GJA1"],
    # Oligodendrocytes
    "Oligodendrocyte": ["MBP", "MOG", "PLP1", "OLIG2", "OPALIN"],
    "OPC": ["PDGFRA", "CSPG4", "OLIG1", "SOX10"],
    # Microglia
    "Microglia": ["CX3CR1", "P2RY12", "TMEM119", "CSF1R", "AIF1"],
    # Endothelial / vascular
    "Endothelial": ["CLDN5", "FLT1", "PECAM1", "VWF"],
}

# De novo variant risk genes from TSAICG (lack functional characterization)
DE_NOVO_RISK_GENES = ["PPP5C", "EXOC1", "GXYLT1"]

# --- Analysis Parameters ---
QC_PARAMS = {
    "min_genes": 500,
    "max_pct_mito": 20.0,
    "min_cells": 10,
    "expected_doublet_rate": 0.06,
}

INTEGRATION_PARAMS = {
    "n_top_genes": 3000,
    "n_pcs": 50,
    "harmony_max_iter": 20,
    "batch_key": "dataset",
}

DE_PARAMS = {
    "condition_key": "condition",
    "cell_type_key": "interneuron_subclass",
    "group1_label": "TS",
    "group2_label": "control",
    "min_cells_per_group": 10,
    "fdr_threshold": 0.05,
}
