"""Gene set loader for MSigDB immune/IFNg pathways and curated PANS features.

Loads MSigDB gene sets via gseapy and provides curated candidate feature
gene lists for the transcriptomic biomarker panel classifier:
  - M1/M2 monocyte polarization markers (Rahman SS et al. 2025)
  - IFNg-target genes from HALLMARK (Shammas G et al. 2026)
  - MHC/HLA pathway genes from OCD GWAS
  - KIR+CD8+ T cell genes (IKZF2, TOX)
  - Autoimmune encephalitis classifier features (DN T cells, NK markers)

Usage:
    uv run python -m bioagentics.data.gene_sets [--list] [--set NAME]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "transcriptomic-biomarker-panel"

# ---------------------------------------------------------------------------
# Curated candidate gene sets (from research plan and prior tasks)
# ---------------------------------------------------------------------------

# M1/M2 monocyte polarization markers — Rahman SS et al. 2025 (PMID 41254741)
# M1-like inflammatory monocytes elevated during flare,
# M2-like anti-inflammatory enriched in recovery
MONOCYTE_M1_M2_MARKERS: dict[str, list[str]] = {
    "monocyte_m1_markers": [
        "CD14", "FCGR3A",  # CD16
        "HLA-DRA", "HLA-DRB1",  # HLA-DR
        "CCR2", "TNF", "IL1B", "IL6", "CD80", "CD86",
        "NOS2", "CXCL9", "CXCL10", "CXCL11",
    ],
    "monocyte_m2_markers": [
        "CD163", "MRC1",  # CD206
        "IL10", "TGFB1", "CCL17", "CCL22", "ARG1",
        "PPARG", "CD209", "CLEC10A",
    ],
    "monocyte_surface_panel": [
        "CD14", "FCGR3A", "FCGR3B",  # CD16
        "CD163", "MRC1",  # CD206
        "HLA-DRA", "CCR2",
    ],
}

# KIR+CD8+ T cell genes — from task #337 (striatal interneuron analysis)
KIR_CD8_T_CELL_GENES: list[str] = [
    "IKZF2",  # Helios — regulatory T cell transcription factor
    "TOX",  # T cell exhaustion marker
    "KIR2DL1", "KIR2DL3", "KIR2DL4", "KIR3DL1", "KIR3DL2",
    "CD8A", "CD8B", "GZMB", "PRF1", "IFNG",
]

# Autoimmune encephalitis classifier features — from task #364
# DN (double-negative) T cells and NK markers
AE_CLASSIFIER_FEATURES: list[str] = [
    # DN T cell markers
    "CD3D", "CD3E", "CD3G",  # T cell lineage
    # NK cell markers
    "NCAM1",  # CD56
    "KLRB1",  # CD161
    "KLRD1",  # CD94
    "KLRF1",  # NKp80
    "NKG7",
    "GNLY",
    "NCR1",  # NKp46
    # Shared cytotoxic
    "GZMB", "GZMK", "PRF1",
]

# MHC/HLA pathway genes — from OCD GWAS (task #280)
MHC_HLA_PATHWAY_GENES: list[str] = [
    "HLA-A", "HLA-B", "HLA-C",  # MHC class I
    "HLA-DRA", "HLA-DRB1", "HLA-DRB5",  # MHC class II
    "HLA-DQA1", "HLA-DQB1",
    "HLA-DPA1", "HLA-DPB1",
    "B2M", "TAP1", "TAP2", "TAPBP",  # antigen processing
    "PSMB8", "PSMB9",  # immunoproteasome
    "CIITA",  # MHC-II master regulator
]

# Cunningham Panel target genes — for comparison
CUNNINGHAM_PANEL_GENES: list[str] = [
    "DRD1",  # Dopamine receptor D1
    "DRD2",  # Dopamine receptor D2
    "LYZL4",  # Lysosomal protein (proxy for CaMKII targets)
    "CAMK2A", "CAMK2B",  # CaMKII subunits
    "TUBA1A", "TUBB3",  # Tubulin (anti-tubulin antibody targets)
]


# ---------------------------------------------------------------------------
# MSigDB loader
# ---------------------------------------------------------------------------


def load_msigdb_gene_sets(
    collection: str = "h",
    organism: str = "Human",
) -> dict[str, list[str]]:
    """Load MSigDB gene sets via gseapy.

    Parameters
    ----------
    collection : str
        MSigDB collection. Common: "h" (hallmark), "c2" (curated),
        "c5" (GO), "c7" (immunologic).
    organism : str
        "Human" or "Mouse".

    Returns
    -------
    Dict of gene set name -> list of gene symbols.
    """
    import gseapy

    logger.info("Loading MSigDB collection '%s' for %s...", collection, organism)
    msig = gseapy.Msigdb(organism=organism)
    gene_sets = msig.get_gmt(category=collection)

    if isinstance(gene_sets, dict):
        logger.info("Loaded %d gene sets from MSigDB %s", len(gene_sets), collection)
        return gene_sets

    # gseapy may return a path to a GMT file
    if isinstance(gene_sets, (str, Path)):
        return _read_gmt(Path(gene_sets))

    raise ValueError(f"Unexpected return type from Msigdb.get_gmt: {type(gene_sets)}")


def _read_gmt(path: Path) -> dict[str, list[str]]:
    """Read a GMT file into a dict of gene set name -> gene list."""
    gene_sets: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 3:
                name = parts[0]
                genes = [g for g in parts[2:] if g]
                gene_sets[name] = genes
    return gene_sets


def get_ifng_response_genes(organism: str = "Human") -> list[str]:
    """Get IFNg-responsive gene set (HALLMARK_INTERFERON_GAMMA_RESPONSE).

    Per Shammas G et al. 2026: IFNg drives persistent epigenetic chromatin
    closing in neurons during encephalitis — peripheral leukocyte IFNg-target
    gene expression may serve as proxies for CNS epigenetic damage.
    """
    hallmarks = load_msigdb_gene_sets(collection="h", organism=organism)
    key = "HALLMARK_INTERFERON_GAMMA_RESPONSE"
    if key in hallmarks:
        logger.info("IFNg response gene set: %d genes", len(hallmarks[key]))
        return hallmarks[key]
    # Try case-insensitive match
    for k, v in hallmarks.items():
        if "INTERFERON_GAMMA" in k.upper():
            return v
    raise KeyError(f"{key} not found in hallmark gene sets")


def get_immune_pathway_sets(organism: str = "Human") -> dict[str, list[str]]:
    """Get relevant immune/inflammatory gene sets from MSigDB hallmarks.

    Returns sets for: IFNg response, complement, IL6/JAK/STAT3,
    TNFa/NFkB, inflammatory response.
    """
    hallmarks = load_msigdb_gene_sets(collection="h", organism=organism)

    target_keywords = [
        "INTERFERON_GAMMA", "COMPLEMENT", "IL6_JAK_STAT3",
        "TNFA_SIGNALING", "INFLAMMATORY_RESPONSE", "IL2_STAT5",
        "INTERFERON_ALPHA", "ALLOGRAFT_REJECTION",
    ]

    result: dict[str, list[str]] = {}
    for name, genes in hallmarks.items():
        for kw in target_keywords:
            if kw in name.upper():
                result[name] = genes
                break

    logger.info("Loaded %d immune pathway gene sets", len(result))
    return result


# ---------------------------------------------------------------------------
# Unified loader
# ---------------------------------------------------------------------------


def get_all_candidate_gene_sets(organism: str = "Human") -> dict[str, list[str]]:
    """Return all candidate gene sets for the transcriptomic biomarker panel.

    Combines MSigDB pathway sets with curated gene lists from the research plan.
    """
    gene_sets: dict[str, list[str]] = {}

    # Curated sets (always available, no network needed)
    gene_sets.update(MONOCYTE_M1_M2_MARKERS)
    gene_sets["kir_cd8_t_cell"] = KIR_CD8_T_CELL_GENES
    gene_sets["ae_classifier_features"] = AE_CLASSIFIER_FEATURES
    gene_sets["mhc_hla_pathway"] = MHC_HLA_PATHWAY_GENES
    gene_sets["cunningham_panel"] = CUNNINGHAM_PANEL_GENES

    # MSigDB sets (requires network on first call, cached after)
    try:
        immune_sets = get_immune_pathway_sets(organism)
        gene_sets.update(immune_sets)
    except Exception as exc:
        logger.warning("Could not load MSigDB sets: %s", exc)

    logger.info("Total: %d gene sets loaded", len(gene_sets))
    return gene_sets


def get_curated_gene_sets() -> dict[str, list[str]]:
    """Return only the curated (non-MSigDB) gene sets. No network required."""
    gene_sets: dict[str, list[str]] = {}
    gene_sets.update(MONOCYTE_M1_M2_MARKERS)
    gene_sets["kir_cd8_t_cell"] = KIR_CD8_T_CELL_GENES
    gene_sets["ae_classifier_features"] = AE_CLASSIFIER_FEATURES
    gene_sets["mhc_hla_pathway"] = MHC_HLA_PATHWAY_GENES
    gene_sets["cunningham_panel"] = CUNNINGHAM_PANEL_GENES
    return gene_sets


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Load and display gene sets")
    parser.add_argument("--list", action="store_true", help="List all gene set names")
    parser.add_argument("--set", dest="set_name", help="Print genes in a specific set")
    parser.add_argument("--curated-only", action="store_true", help="Skip MSigDB download")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.curated_only:
        gene_sets = get_curated_gene_sets()
    else:
        gene_sets = get_all_candidate_gene_sets()

    if args.list:
        for name, genes in sorted(gene_sets.items()):
            print(f"  {name}: {len(genes)} genes")
    elif args.set_name:
        if args.set_name in gene_sets:
            genes = gene_sets[args.set_name]
            print(f"{args.set_name} ({len(genes)} genes):")
            for g in sorted(genes):
                print(f"  {g}")
        else:
            print(f"Gene set '{args.set_name}' not found")
    else:
        print(f"Loaded {len(gene_sets)} gene sets")
        for name, genes in sorted(gene_sets.items()):
            print(f"  {name}: {len(genes)} genes")


if __name__ == "__main__":
    main()
