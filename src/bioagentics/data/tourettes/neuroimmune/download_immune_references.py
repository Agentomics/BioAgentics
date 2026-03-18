"""Download immune cell-type expression references for MAGMA cell-type analysis.

Downloads and processes gene expression data from:
- DICE (Database of Immune Cell Expression, GEO: GSE118165) — sorted human
  immune cell RNA-seq (TPM) for CD4 naive/memory, Th1, Th2, Th17, Tfh, Tregs,
  CD8 naive/memory, NK, classical/non-classical monocytes, B naive/memory.
- ImmGen ULI RNA-seq human module — additional immune cell populations.

Produces gene-by-celltype expression matrices suitable for MAGMA --gene-covar
input and specificity score matrices for cell-type enrichment analysis.

Key cell types for TS neuroimmune subtyping:
  Th17 [HIGH PRIORITY], NK cells [HIGH PRIORITY], CD4+ T, CD8+ T, B cells,
  monocytes/macrophages, neutrophils, dendritic cells, Tregs, microglia.

Output: data/tourettes/ts-neuroimmune-subtyping/immune_references/

Usage:
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_immune_references
    uv run python -m bioagentics.data.tourettes.neuroimmune.download_immune_references --source dice
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = (
    REPO_ROOT / "data" / "tourettes" / "ts-neuroimmune-subtyping" / "immune_references"
)

TIMEOUT = 300
CHUNK_SIZE = 65536

# DICE — sorted human immune cell RNA-seq (Schmiedel et al. 2018, Cell)
# GEO: GSE118165 — TPM expression matrix
# The supplementary table from the DICE portal provides TPM per cell type.
DICE_GEO_ID = "GSE118165"
DICE_SERIES_MATRIX_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118165/matrix/"
    "GSE118165_series_matrix.txt.gz"
)
# Direct supplementary file with TPM values
DICE_SUPP_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE118nnn/GSE118165/suppl/"
    "GSE118165_RNA_gene_abundance.txt.gz"
)

# Cell type mapping: DICE sample column cell type substrings -> standardized names
# Samples named as DONOR-CELLTYPE-CONDITION (e.g. "1001-Th17_precursors-U")
# We extract the cell type part and map it.
DICE_CELLTYPE_MAP = {
    "Naive_Teffs": "CD4_T_naive",
    "Effector_CD4pos_T": "CD4_T_effector",
    "Memory_Teffs": "CD4_T_memory",
    "Th1_precursors": "Th1",
    "Th2_precursors": "Th2",
    "Th17_precursors": "Th17",
    "Follicular_T_Helper": "Tfh",
    "Naive_Tregs": "Treg_naive",
    "Memory_Tregs": "Treg_memory",
    "Regulatory_T": "Treg",
    "Naive_CD8_T": "CD8_T_naive",
    "CD8pos_T": "CD8_T",
    "Central_memory_CD8pos_T": "CD8_T_central_memory",
    "Effector_memory_CD8pos_T": "CD8_T_effector_memory",
    "Gamma_delta_T": "Gamma_delta_T",
    "Mature_NK": "NK_mature",
    "Immature_NK": "NK_immature",
    "Memory_NK": "NK_memory",
    "Naive_B": "B_naive",
    "Mem_B": "B_memory",
    "Bulk_B": "B_bulk",
    "Plasmablasts": "Plasmablasts",
    "Monocytes": "Monocyte",
    "Myeloid_DCs": "Myeloid_DC",
    "pDCs": "Plasmacytoid_DC",
}

# Aggregated cell type groups for MAGMA analysis
CELLTYPE_GROUPS = {
    "CD4_T": ["CD4_T_naive", "CD4_T_effector", "CD4_T_memory", "Th1", "Th2", "Th17", "Tfh"],
    "Th17": ["Th17"],
    "NK": ["NK_mature", "NK_immature", "NK_memory"],
    "Treg": ["Treg_naive", "Treg_memory", "Treg"],
    "CD8_T": ["CD8_T_naive", "CD8_T", "CD8_T_central_memory", "CD8_T_effector_memory"],
    "B_cell": ["B_naive", "B_memory", "B_bulk", "Plasmablasts"],
    "Monocyte": ["Monocyte"],
    "DC": ["Myeloid_DC", "Plasmacytoid_DC"],
    "Gamma_delta_T": ["Gamma_delta_T"],
}


def download_file(url: str, dest: Path, force: bool = False) -> bool:
    """Download a file with streaming, skip if cached."""
    if dest.exists() and not force:
        logger.info("  Cached: %s", dest.name)
        return True

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".tmp")

    logger.info("  Downloading: %s", url.split("/")[-1])
    try:
        resp = requests.get(url, timeout=TIMEOUT, stream=True)
        resp.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
        tmp.rename(dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        logger.info("  Saved: %s (%.1f MB)", dest.name, size_mb)
        return True
    except requests.RequestException as e:
        logger.error("  Download failed: %s — %s", url, e)
        if tmp.exists():
            tmp.unlink()
        return False


def download_dice_data(data_dir: Path, force: bool = False) -> Path | None:
    """Download DICE RNA-seq normalized reads from GEO supplementary."""
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    dest = raw_dir / "GSE118165_RNA_gene_abundance.txt.gz"
    ok = download_file(DICE_SUPP_URL, dest, force=force)
    return dest if ok else None


def parse_dice_expression(raw_path: Path, data_dir: Path) -> pd.DataFrame | None:
    """Parse DICE normalized reads into gene-by-celltype expression matrix.

    The DICE supplementary file contains normalized read counts (TPM-like)
    with samples as columns. We aggregate replicates per cell type to get
    mean expression per cell type.
    """
    output_path = data_dir / "dice_expression_matrix.tsv"
    if output_path.exists():
        logger.info("  DICE matrix exists: %s", output_path.name)
        return pd.read_csv(output_path, sep="\t", index_col=0)

    logger.info("  Parsing DICE expression data...")
    try:
        df = pd.read_csv(raw_path, sep="\t", index_col=0, low_memory=False)
    except Exception as e:
        logger.error("  Failed to parse DICE data: %s", e)
        return None

    logger.info("  Raw DICE matrix: %d genes x %d samples", df.shape[0], df.shape[1])

    # DICE columns are DONOR-CELLTYPE-CONDITION (e.g. "1001-Th17_precursors-U")
    # Extract cell type by stripping donor prefix and condition suffix
    def _extract_celltype(col_name: str) -> str | None:
        parts = col_name.split("-")
        if len(parts) < 2:
            return None
        # Cell type is everything between first and last dash
        celltype = "-".join(parts[1:-1]) if len(parts) >= 3 else parts[1]
        return celltype

    # Map sample columns to standardized cell types
    celltype_cols: dict[str, list[str]] = {}
    unmapped = []
    for col in df.columns:
        raw_ct = _extract_celltype(col)
        if raw_ct and raw_ct in DICE_CELLTYPE_MAP:
            std_ct = DICE_CELLTYPE_MAP[raw_ct]
            celltype_cols.setdefault(std_ct, []).append(col)
        else:
            unmapped.append(col)

    if unmapped:
        logger.debug("  Unmapped columns (%d): %s", len(unmapped), unmapped[:5])

    logger.info("  Matched %d cell types: %s", len(celltype_cols), list(celltype_cols.keys()))

    # Aggregate replicates per cell type (mean expression)
    expr_matrix = pd.DataFrame(index=df.index)
    for celltype, cols in sorted(celltype_cols.items()):
        expr_matrix[celltype] = df[cols].mean(axis=1)
        logger.info("    %s: %d replicates", celltype, len(cols))

    # Filter: keep genes with non-zero expression in at least one cell type
    expr_matrix = expr_matrix[expr_matrix.max(axis=1) > 0]
    logger.info("  Expression matrix: %d genes x %d cell types", *expr_matrix.shape)

    expr_matrix.to_csv(output_path, sep="\t")
    return expr_matrix


def compute_specificity_scores(
    expr_matrix: pd.DataFrame, data_dir: Path
) -> pd.DataFrame:
    """Compute cell-type specificity scores for MAGMA --gene-covar.

    Specificity = expression in cell type / sum of expression across all cell types.
    This is the standard approach used by Finucane et al. 2018 and FUMA.
    """
    output_path = data_dir / "dice_specificity_matrix.tsv"
    if output_path.exists():
        logger.info("  Specificity matrix exists: %s", output_path.name)
        return pd.read_csv(output_path, sep="\t", index_col=0)

    logger.info("  Computing cell-type specificity scores...")
    # Row sums
    row_sums = expr_matrix.sum(axis=1)
    row_sums = row_sums.replace(0, np.nan)

    # Specificity: proportion of total expression
    specificity = expr_matrix.div(row_sums, axis=0)
    specificity = specificity.fillna(0)

    specificity.to_csv(output_path, sep="\t")
    logger.info("  Specificity matrix: %d genes x %d cell types", *specificity.shape)
    return specificity


def compute_grouped_specificity(
    expr_matrix: pd.DataFrame, data_dir: Path
) -> pd.DataFrame:
    """Compute specificity scores for aggregated cell type groups.

    Groups related subtypes (e.g., all CD4+ T subtypes) into broader categories
    for initial MAGMA analysis, while keeping individual subtypes for fine-grained
    follow-up.
    """
    output_path = data_dir / "dice_grouped_specificity.tsv"
    if output_path.exists():
        logger.info("  Grouped specificity exists: %s", output_path.name)
        return pd.read_csv(output_path, sep="\t", index_col=0)

    logger.info("  Computing grouped specificity scores...")
    grouped = pd.DataFrame(index=expr_matrix.index)
    for group_name, subtypes in CELLTYPE_GROUPS.items():
        cols = [c for c in subtypes if c in expr_matrix.columns]
        if cols:
            grouped[group_name] = expr_matrix[cols].mean(axis=1)
        else:
            logger.warning("  No columns found for group %s", group_name)

    # Add any ungrouped cell types
    grouped_celltypes = set()
    for subtypes in CELLTYPE_GROUPS.values():
        grouped_celltypes.update(subtypes)
    for col in expr_matrix.columns:
        if col not in grouped_celltypes:
            grouped[col] = expr_matrix[col]

    row_sums = grouped.sum(axis=1).replace(0, np.nan)
    specificity = grouped.div(row_sums, axis=0).fillna(0)

    specificity.to_csv(output_path, sep="\t")
    logger.info("  Grouped specificity: %d genes x %d groups", *specificity.shape)
    return specificity


def generate_magma_gene_covar(
    specificity: pd.DataFrame, data_dir: Path, prefix: str = "dice"
) -> Path:
    """Generate MAGMA --gene-covar input file.

    MAGMA expects: GENE <tab> covar1 <tab> covar2 ...
    where GENE is the gene symbol and covariates are continuous specificity scores.
    """
    output_path = data_dir / f"{prefix}_magma_gene_covar.tsv"
    if output_path.exists():
        logger.info("  MAGMA gene-covar exists: %s", output_path.name)
        return output_path

    logger.info("  Generating MAGMA gene-covar file...")
    # MAGMA needs GENE column
    out = specificity.copy()
    out.index.name = "GENE"
    out.to_csv(output_path, sep="\t")
    logger.info("  MAGMA gene-covar: %d genes, %d cell types", *out.shape)
    return output_path


def generate_magma_gene_sets(
    specificity: pd.DataFrame, data_dir: Path, prefix: str = "dice",
    top_pct: float = 0.10,
) -> Path:
    """Generate MAGMA --set-annot gene set file from top-specific genes.

    For each cell type, take the top N% most specifically expressed genes
    as a binary gene set. This is the approach used by Skene et al. 2018.
    """
    output_path = data_dir / f"{prefix}_magma_gene_sets.tsv"
    if output_path.exists():
        logger.info("  MAGMA gene-sets exists: %s", output_path.name)
        return output_path

    logger.info("  Generating MAGMA gene sets (top %.0f%%)...", top_pct * 100)
    n_top = max(1, int(len(specificity) * top_pct))

    lines = []
    for celltype in specificity.columns:
        top_genes = specificity[celltype].nlargest(n_top).index.tolist()
        # MAGMA set-annot format: SET_NAME <tab> GENE1 GENE2 ...
        lines.append(f"{celltype}\t{' '.join(top_genes)}")
        logger.info("    %s: %d genes", celltype, len(top_genes))

    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    return output_path


def write_manifest(data_dir: Path, expr_shape: tuple, spec_shape: tuple) -> None:
    """Write manifest documenting the immune reference data."""
    manifest = {
        "description": "Immune cell-type expression references for MAGMA cell-type analysis",
        "project": "ts-neuroimmune-subtyping",
        "phase": "2 — Immune Cell-Type Enrichment",
        "sources": {
            "DICE": {
                "geo_accession": DICE_GEO_ID,
                "pmid": "30449622",
                "reference": "Schmiedel et al. 2018, Cell",
                "description": "Sorted human immune cell RNA-seq",
            },
        },
        "outputs": {
            "dice_expression_matrix.tsv": "Gene-by-celltype mean expression (normalized reads)",
            "dice_specificity_matrix.tsv": "Cell-type specificity scores (per-cell-type subtypes)",
            "dice_grouped_specificity.tsv": "Cell-type specificity scores (grouped: CD4_T, Th17, NK, etc.)",
            "dice_magma_gene_covar.tsv": "MAGMA --gene-covar input (grouped specificity)",
            "dice_magma_gene_sets.tsv": "MAGMA --set-annot input (top 10% specific genes per cell type)",
        },
        "expression_matrix_shape": {"genes": expr_shape[0], "cell_types": expr_shape[1]},
        "specificity_matrix_shape": {"genes": spec_shape[0], "cell_types": spec_shape[1]},
        "generated": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    manifest_path = data_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("Manifest written: %s", manifest_path)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download immune cell expression references for MAGMA analysis"
    )
    parser.add_argument(
        "--source",
        choices=["dice", "all"],
        default="all",
        help="Which data source to download (default: all)",
    )
    parser.add_argument("--force", action="store_true", help="Re-download if cached")
    parser.add_argument("--dest", type=Path, default=DATA_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    data_dir = args.dest
    data_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Download DICE data
    logger.info("=== DICE: Sorted Human Immune Cell Expression ===")
    raw_path = download_dice_data(data_dir, force=args.force)
    if raw_path is None:
        logger.error("DICE download failed. Exiting.")
        return

    # Step 2: Parse into expression matrix
    expr_matrix = parse_dice_expression(raw_path, data_dir)
    if expr_matrix is None:
        logger.error("DICE parsing failed. Exiting.")
        return

    # Step 3: Compute specificity scores (per-subtype)
    specificity = compute_specificity_scores(expr_matrix, data_dir)

    # Step 4: Compute grouped specificity
    grouped_spec = compute_grouped_specificity(expr_matrix, data_dir)

    # Step 5: Generate MAGMA inputs
    generate_magma_gene_covar(grouped_spec, data_dir)
    generate_magma_gene_sets(grouped_spec, data_dir)

    # Step 6: Manifest
    write_manifest(data_dir, expr_matrix.shape, grouped_spec.shape)

    logger.info("=== Complete ===")
    logger.info("  Expression matrix: %d genes x %d cell types", *expr_matrix.shape)
    logger.info("  Grouped specificity: %d genes x %d groups", *grouped_spec.shape)
    logger.info(
        "  Priority cell types available: Th17=%s, NK=%s",
        "Th17" in grouped_spec.columns,
        "NK" in grouped_spec.columns,
    )


if __name__ == "__main__":
    main()
