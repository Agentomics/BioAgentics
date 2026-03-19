"""Extract per-patient lectin complement variant status from genetic data.

Uses curated PANS variant data from pans_variants.py (Vettiatil 2026) and
enrichment analysis outputs from pans-genetic-variant-pathway-analysis.

Per-patient WES data with individual genotypes is not yet available in the
repository — this pipeline creates the variant status table from gene-level
annotations and flags the data gap for the human/data_curator role.

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.variant_status_extraction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.pans_variants import (
    AXIS_DDR,
    AXIS_LECTIN,
    AXIS_MITO,
    get_pans_variant_genes,
)
from bioagentics.pandas_pans.innate_immunity_modules import get_lectin_variant_genes

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"
ENRICHMENT_DIR = REPO_ROOT / "output" / "pandas_pans" / "pans-genetic-variant-pathway-analysis"


def extract_lectin_variant_status() -> pd.DataFrame:
    """Extract lectin complement variant gene annotations.

    Returns DataFrame with columns:
        gene_symbol, pathway_axis, variant_type, functional_annotation,
        is_lectin_complement, enrichment_p_value
    """
    variants = get_pans_variant_genes()

    # Mark lectin complement genes
    lectin_genes = set(get_lectin_variant_genes())
    variants["is_lectin_complement"] = variants["gene_symbol"].isin(lectin_genes)

    # Add enrichment p-values from pathway analysis
    enrichment_path = ENRICHMENT_DIR / "enrichment_immune_focused.csv"
    if enrichment_path.exists():
        enrichment = pd.read_csv(enrichment_path)
        lectin_enrich = enrichment[
            enrichment["term"].str.contains("Lectin Pathway", case=False, na=False)
        ]
        if not lectin_enrich.empty:
            best_p = float(lectin_enrich["adj_p_value"].min())
            variants.loc[variants["is_lectin_complement"], "lectin_pathway_fdr"] = best_p
        else:
            variants["lectin_pathway_fdr"] = None
    else:
        variants["lectin_pathway_fdr"] = None

    return variants


def extract_cgas_sting_variant_status() -> pd.DataFrame:
    """Extract cGAS-STING pathway variant genes for Phase 2 comparison.

    Returns subset of PANS variant genes in the DDR-cGAS-STING axis,
    specifically TREX1 and SAMHD1.
    """
    variants = get_pans_variant_genes()
    cgas_genes = {"TREX1", "SAMHD1"}
    return variants[variants["gene_symbol"].isin(cgas_genes)].copy()


def build_variant_burden_table() -> pd.DataFrame:
    """Build variant burden table organized by pathway axis.

    Groups variants by pathway axis and computes burden metrics:
    - n_genes: number of variant genes in axis
    - n_pathogenic: P or LP variants
    - axis_relevance: relevance to innate deficiency hypothesis
    """
    variants = get_pans_variant_genes()

    axis_relevance = {
        "Lectin complement": "direct — lectin complement failure impairs GAS clearance",
        "DDR-cGAS-STING/AIM2 inflammasome": "indirect — cGAS-STING dysregulation affects type I IFN",
        "Mitochondrial-innate immunity": "indirect — mitochondrial DAMPs activate innate immunity",
        "Gut-immune": "supporting — gut barrier affects innate mucosal defense",
        "Chromatin/neuroprotection": "downstream — neuronal vulnerability to immune damage",
    }

    results = []
    for axis in variants["pathway_axis"].unique():
        axis_df = variants[variants["pathway_axis"] == axis]
        n_pathogenic = int(axis_df["variant_type"].isin(["P", "LP"]).sum())
        n_vus = int((axis_df["variant_type"] == "VUS").sum())

        results.append({
            "pathway_axis": axis,
            "n_genes": len(axis_df),
            "n_pathogenic": n_pathogenic,
            "n_vus": n_vus,
            "genes": ", ".join(axis_df["gene_symbol"].tolist()),
            "relevance_to_innate_deficiency": axis_relevance.get(axis, "unknown"),
        })

    return pd.DataFrame(results)


def run_variant_extraction() -> dict[str, Path]:
    """Run full variant status extraction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Lectin complement variant status
    logger.info("=== Lectin complement variant status ===")
    lectin = extract_lectin_variant_status()
    lectin_subset = lectin[lectin["is_lectin_complement"]]
    path = OUTPUT_DIR / "lectin_variant_status.csv"
    lectin.to_csv(path, index=False)
    outputs["all_variants"] = path
    logger.info("Lectin complement variants:\n%s",
                lectin_subset[["gene_symbol", "variant_type", "functional_annotation", "lectin_pathway_fdr"]].to_string(index=False))

    # 2. cGAS-STING variants for comparison
    logger.info("\n=== cGAS-STING variants ===")
    cgas = extract_cgas_sting_variant_status()
    path = OUTPUT_DIR / "cgas_sting_variant_status.csv"
    cgas.to_csv(path, index=False)
    outputs["cgas_sting"] = path
    logger.info("cGAS-STING variants:\n%s",
                cgas[["gene_symbol", "variant_type", "functional_annotation"]].to_string(index=False))

    # 3. Variant burden table
    logger.info("\n=== Variant burden by pathway axis ===")
    burden = build_variant_burden_table()
    path = OUTPUT_DIR / "variant_burden_by_axis.csv"
    burden.to_csv(path, index=False)
    outputs["burden_table"] = path
    logger.info("\n%s", burden[["pathway_axis", "n_genes", "n_pathogenic", "genes"]].to_string(index=False))

    # 4. Note: per-patient genotype data is not yet available
    logger.warning(
        "\nNOTE: Per-patient WES genotype data from Vettiatil 2026 supplementary "
        "tables is not yet in the repository. The variant status table above uses "
        "gene-level annotations. To enable per-patient analysis (tasks 104-105), "
        "the data_curator or human role needs to download and parse the "
        "supplementary variant call table."
    )

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_variant_extraction()
