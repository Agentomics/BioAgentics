"""Extract lectin complement gene expression from IVIG scRNA-seq and bulk data.

Processes Han VX et al. scRNA-seq (BD Rhapsody TCM files) and bulk RNA-seq
to quantify lectin complement pathway expression across cell types and
conditions (PANS pre-IVIG, PANS post-IVIG, controls).

Memory-safe: processes one TCM file at a time, extracting only target genes.

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.lectin_complement_extraction
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from bioagentics.config import DATA_DIR, REPO_ROOT
from bioagentics.pandas_pans.innate_immunity_modules import LECTIN_COMPLEMENT_GENES

logger = logging.getLogger(__name__)

RAW_DIR = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "raw"
BULK_DIR = DATA_DIR / "pandas_pans" / "ivig-mechanism-single-cell-analysis" / "bulk"
OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"

# Lineage markers for basic cell type assignment from raw TCM
# Minimal set to avoid requiring full annotation pipeline
LINEAGE_MARKERS: dict[str, list[str]] = {
    "Monocyte": ["CD14", "LYZ", "CST3", "S100A8", "S100A9", "FCN1"],
    "T_cell": ["CD3D", "CD3E", "CD3G", "TRAC"],
    "B_cell": ["CD79A", "MS4A1", "CD19"],
    "NK_cell": ["NKG7", "GNLY", "KLRD1", "NCAM1"],
    "Neutrophil": ["FCGR3B", "CXCR2", "S100A12"],
    "DC": ["FLT3", "IRF8", "LILRA4"],
}


def _parse_sample_info(filename: str) -> dict[str, str]:
    """Extract patient, condition, and GSM from TCM filename."""
    name = Path(filename).stem.replace(".tsv", "")
    parts = name.split("_")
    gsm = parts[0]

    if "Control" in name:
        patient = f"Control_{parts[2]}"
        condition = "control"
    else:
        patient_num = parts[2]
        patient = f"Patient_{patient_num}"
        condition = parts[3].lower()  # "pre" or "post"

    return {"gsm": gsm, "patient": patient, "condition": condition}


def _load_tcm_target_genes(
    tcm_path: Path,
    target_genes: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Load only target genes from a TCM file (memory-efficient).

    Returns (expression_df, found_genes) where expression_df has
    genes as rows, cell barcodes as columns.
    """
    # Read gene names column first
    gene_col = pd.read_csv(tcm_path, sep="\t", usecols=[0], compression="gzip")
    gene_names = gene_col.iloc[:, 0].tolist()

    # Find row indices for target genes
    target_set = set(target_genes)
    row_indices = []
    found_genes = []
    for i, g in enumerate(gene_names):
        if g in target_set:
            row_indices.append(i)
            found_genes.append(g)

    if not row_indices:
        return pd.DataFrame(), found_genes

    # Read only the target rows using skiprows
    all_rows = set(range(len(gene_names)))
    skip = sorted(all_rows - set(row_indices))
    # skiprows in read_csv is 0-indexed and includes header, so add 1
    skip_with_header = [s + 1 for s in skip]

    df = pd.read_csv(tcm_path, sep="\t", skiprows=skip_with_header, compression="gzip")
    df = df.set_index(df.columns[0])

    return df, found_genes


def _assign_lineage(cell_expr: pd.DataFrame, gene_names: list[str]) -> pd.Series:
    """Assign basic lineage to cells based on marker scoring.

    Simple max-score assignment: for each cell, compute mean expression
    of each lineage's markers, assign to the lineage with highest score.
    """
    gene_set = set(gene_names)
    scores = {}

    for lineage, markers in LINEAGE_MARKERS.items():
        available = [m for m in markers if m in gene_set]
        if not available:
            scores[lineage] = pd.Series(0.0, index=cell_expr.columns)
            continue

        mask = cell_expr.index.isin(available)
        if mask.any():
            scores[lineage] = cell_expr.loc[mask].mean(axis=0)
        else:
            scores[lineage] = pd.Series(0.0, index=cell_expr.columns)

    score_df = pd.DataFrame(scores)
    # Assign to max-scoring lineage; "Unassigned" if all zero
    max_scores = score_df.max(axis=1)
    assignments = score_df.idxmax(axis=1)
    assignments[max_scores == 0] = "Unassigned"
    return assignments


def extract_from_scrna(
    target_genes: list[str] | None = None,
    raw_dir: Path | None = None,
) -> pd.DataFrame:
    """Extract target gene expression per cell type per condition from raw TCM files.

    Processes each TCM file sequentially to stay within 8GB RAM.

    Returns DataFrame with columns:
        gene, cell_type, condition, patient, mean_expr, total_expr,
        n_cells, pct_expressing
    """
    if raw_dir is None:
        raw_dir = RAW_DIR
    if target_genes is None:
        target_genes = LECTIN_COMPLEMENT_GENES

    # Collect all genes we need: target genes + lineage markers for annotation
    all_markers = []
    for markers in LINEAGE_MARKERS.values():
        all_markers.extend(markers)
    genes_to_load = list(set(target_genes + all_markers))

    tcm_files = sorted(raw_dir.glob("*_TCM.tsv.gz"))
    if not tcm_files:
        logger.warning("No TCM files found in %s", raw_dir)
        return pd.DataFrame()

    logger.info("Processing %d TCM files for %d target genes", len(tcm_files), len(target_genes))

    all_results: list[dict] = []

    for tcm_path in tcm_files:
        sample_info = _parse_sample_info(tcm_path.name)
        logger.info("  Processing %s (%s, %s)", tcm_path.name, sample_info["patient"], sample_info["condition"])

        # Load target genes + markers
        expr_df, found_genes = _load_tcm_target_genes(tcm_path, genes_to_load)
        if expr_df.empty:
            logger.warning("  No target genes found in %s", tcm_path.name)
            continue

        n_cells = expr_df.shape[1]
        logger.info("  Loaded %d genes x %d cells", len(found_genes), n_cells)

        # Assign cell types
        cell_types = _assign_lineage(expr_df, found_genes)

        # Aggregate per cell_type for target genes only
        target_set = set(target_genes)
        target_found = [g for g in found_genes if g in target_set]

        for gene in target_found:
            if gene not in expr_df.index:
                continue
            gene_expr = expr_df.loc[gene]

            for ct in cell_types.unique():
                ct_mask = cell_types == ct
                ct_expr = gene_expr[ct_mask]
                n_ct = int(ct_mask.sum())
                if n_ct == 0:
                    continue

                all_results.append({
                    "gene": gene,
                    "cell_type": ct,
                    "condition": sample_info["condition"],
                    "patient": sample_info["patient"],
                    "gsm": sample_info["gsm"],
                    "mean_expr": float(ct_expr.mean()),
                    "total_expr": float(ct_expr.sum()),
                    "n_cells": n_ct,
                    "pct_expressing": float((ct_expr > 0).sum() / n_ct * 100),
                })

        # Free memory
        del expr_df, cell_types
        logger.info("  Done with %s", sample_info["patient"])

    result_df = pd.DataFrame(all_results)
    if not result_df.empty:
        result_df = result_df.sort_values(["gene", "cell_type", "condition", "patient"])
    return result_df


def extract_from_bulk(
    target_genes: list[str] | None = None,
    bulk_path: Path | None = None,
) -> pd.DataFrame:
    """Extract target gene expression from bulk RNA-seq counts.

    Returns DataFrame with columns:
        gene, sample, condition, raw_count
    """
    if bulk_path is None:
        bulk_path = BULK_DIR / "GSE278678_pans_ivig_counts.csv.gz"
    if target_genes is None:
        target_genes = LECTIN_COMPLEMENT_GENES

    if not bulk_path.exists():
        logger.warning("Bulk counts file not found: %s", bulk_path)
        return pd.DataFrame()

    df = pd.read_csv(bulk_path)
    target_set = set(target_genes)
    target_df = df[df["SYMBOL"].isin(list(target_set))].copy()

    if target_df.empty:
        logger.warning("No target genes found in bulk data")
        return pd.DataFrame()

    # Melt to long format
    sample_cols = [c for c in df.columns if c not in ("Unnamed: 0", "ENTREZID", "SYMBOL")]
    melted = target_df.melt(
        id_vars=["SYMBOL", "ENTREZID"],
        value_vars=sample_cols,
        var_name="sample",
        value_name="raw_count",
    )
    melted = melted.rename(columns={"SYMBOL": "gene"})

    # Assign condition based on sample name
    def _assign_condition(sample: str) -> str:
        if sample.startswith("Control"):
            return "control"
        elif sample.startswith("PreIVIG"):
            return "pre"
        elif sample.startswith("PostIVIG"):
            return "post"
        elif sample.startswith("Secondpost"):
            return "second_post"
        return "unknown"

    melted["condition"] = melted["sample"].apply(_assign_condition)
    melted = melted[["gene", "sample", "condition", "raw_count"]].sort_values(by=["gene", "condition", "sample"])

    return melted


def compute_bulk_summary(bulk_long: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics per gene per condition from bulk long format."""
    summary = (
        bulk_long.groupby(["gene", "condition"])["raw_count"]
        .agg(["mean", "std", "median", "min", "max", "count"])
        .reset_index()
    )
    summary.columns = ["gene", "condition", "mean_count", "std_count", "median_count", "min_count", "max_count", "n_samples"]
    return summary.sort_values(["gene", "condition"])


def run_extraction() -> dict[str, Path]:
    """Run full lectin complement extraction pipeline.

    Returns dict of output file paths.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # 1. Bulk RNA-seq extraction
    logger.info("=== Extracting from bulk RNA-seq ===")
    bulk_long = extract_from_bulk()
    if not bulk_long.empty:
        bulk_path = OUTPUT_DIR / "lectin_complement_bulk_expression.csv"
        bulk_long.to_csv(bulk_path, index=False)
        outputs["bulk_expression"] = bulk_path
        logger.info("Saved bulk expression: %s (%d rows)", bulk_path, len(bulk_long))

        bulk_summary = compute_bulk_summary(bulk_long)
        summary_path = OUTPUT_DIR / "lectin_complement_bulk_summary.csv"
        bulk_summary.to_csv(summary_path, index=False)
        outputs["bulk_summary"] = summary_path
        logger.info("Saved bulk summary: %s", summary_path)

    # 2. scRNA-seq extraction (all innate genes for broader analysis)
    logger.info("=== Extracting from scRNA-seq (lectin complement) ===")
    scrna_results = extract_from_scrna(target_genes=LECTIN_COMPLEMENT_GENES)
    if not scrna_results.empty:
        scrna_path = OUTPUT_DIR / "lectin_complement_scrna_expression.csv"
        scrna_results.to_csv(scrna_path, index=False)
        outputs["scrna_expression"] = scrna_path
        logger.info("Saved scRNA expression: %s (%d rows)", scrna_path, len(scrna_results))

    logger.info("=== Extraction complete: %d output files ===", len(outputs))
    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_extraction()
