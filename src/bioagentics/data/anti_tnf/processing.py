"""Data processing pipeline for anti-TNF response prediction.

Downloads GEO transcriptomic datasets, extracts expression data and clinical
metadata, maps probe IDs to gene symbols, and builds per-study expression
matrices with a unified clinical metadata table.

Usage:
    uv run python -m bioagentics.data.anti_tnf.processing [--output-dir PATH] [--cache-dir PATH]
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import GEOparse
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.anti_tnf.datasets import DATASETS, EXPECTED_RESPONSE_GENES

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = REPO_ROOT / "output" / "crohns" / "anti-tnf-response-prediction" / "processed"
DEFAULT_CACHE = REPO_ROOT / "data" / "crohns" / "anti-tnf-response-prediction" / "geo_cache"

# GEO series matrix files are already RMA/quantile-normalized by original authors.
# We use these pre-processed values rather than re-running RMA from CEL files,
# which requires R/Bioconductor (affy/oligo). This is standard practice.

# Datasets with confirmed usable anti-TNF response annotations.
# GSE100833 is ustekinumab (not anti-TNF) — excluded.
# GSE57945 (RISK) has no response annotation in GEO metadata — excluded until
# follow-up data can be obtained separately.
USABLE_DATASETS = ["GSE16879", "GSE12251", "GSE73661"]


def download_geo_dataset(accession: str, cache_dir: Path) -> GEOparse.GEOTypes.GSE:
    """Download a GEO dataset series matrix file."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s to %s", accession, cache_dir)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(cache_dir), silent=True)
    return gse


def _parse_chars(gsm) -> dict[str, str]:
    """Parse GEO sample characteristics into a flat dict."""
    chars: dict[str, str] = {}
    for key, values in gsm.metadata.items():
        if key.startswith("characteristics_ch"):
            for v in values:
                if ":" in v:
                    k, _, val = v.partition(":")
                    chars[k.strip().lower()] = val.strip()
                else:
                    chars[v.strip().lower()] = v.strip()
    return chars


def _extract_gse16879(gse) -> pd.DataFrame:
    """GSE16879 (Arijs 2009): response to infliximab: Yes/No, with timepoint.

    Filters to pre-treatment CD samples only.
    """
    records = []
    for gsm_name, gsm in gse.gsms.items():
        chars = _parse_chars(gsm)
        disease = chars.get("disease", "unknown").lower()
        response_raw = chars.get("response to infliximab", "").lower()
        timepoint_raw = chars.get(
            "before or after first infliximab treatment", ""
        ).lower()

        if response_raw == "yes":
            response = "responder"
        elif response_raw == "no":
            response = "non_responder"
        else:
            response = "control"

        is_pre = "before" in timepoint_raw
        records.append({
            "sample_id": gsm_name,
            "study": "GSE16879",
            "response_status": response,
            "drug": "infliximab",
            "tissue_type": chars.get("tissue", "mucosal_biopsy").lower(),
            "timepoint": "pre_treatment" if is_pre else "post_treatment",
            "prior_biologic_exposure": "unknown",
            "disease": disease,
        })

    df = pd.DataFrame(records)
    # Keep only pre-treatment CD patients with known response
    cd_pattern = r"(?i)(crohn|^cd$)"
    mask = (
        (df["timepoint"] == "pre_treatment")
        & (df["disease"].str.contains(cd_pattern, regex=True, na=False))
        & (df["response_status"].isin(["responder", "non_responder"]))
    )
    filtered = df[mask].copy()
    logger.info(
        "GSE16879: %d/%d samples kept (pre-treatment CD with response)",
        len(filtered),
        len(df),
    )
    return filtered


def _extract_gse12251(gse) -> pd.DataFrame:
    """GSE12251 (Arijs 2009): WK8RSPHM: Yes/No (week 8 histological response).

    All samples are pre-treatment colonic biopsies.
    """
    records = []
    for gsm_name, gsm in gse.gsms.items():
        chars = _parse_chars(gsm)
        wk8 = chars.get("wk8rsphm", "").lower()

        if wk8 == "yes":
            response = "responder"
        elif wk8 == "no":
            response = "non_responder"
        else:
            response = "unknown"

        records.append({
            "sample_id": gsm_name,
            "study": "GSE12251",
            "response_status": response,
            "drug": "infliximab",
            "tissue_type": "colonic_biopsy",
            "timepoint": "pre_treatment",
            "prior_biologic_exposure": "unknown",
            "disease": "crohn's disease",
        })

    df = pd.DataFrame(records)
    filtered = df[df["response_status"].isin(["responder", "non_responder"])].copy()
    logger.info(
        "GSE12251: %d/%d samples with response annotation",
        len(filtered),
        len(df),
    )
    return filtered


def _extract_gse73661(gse) -> pd.DataFrame:
    """GSE73661 (Haberman 2014): Mayo endoscopic subscore with timepoint.

    Anti-TNF response derived from Mayo subscore: responder if score drops
    to 0-1 by week 4-6 from baseline >= 2. Keeps IFX-treated patients at W0.
    """
    records = []
    for gsm_name, gsm in gse.gsms.items():
        chars = _parse_chars(gsm)
        therapy = chars.get("induction therapy_maintenance therapy", "").upper()
        week = chars.get("week (w)", "").upper()
        mayo = chars.get("mayo endoscopic subscore", "")
        patient_id = chars.get("study individual number", "unknown")

        records.append({
            "sample_id": gsm_name,
            "study": "GSE73661",
            "therapy": therapy,
            "week": week,
            "mayo": mayo,
            "patient_id": patient_id,
            "tissue_type": "colonic_biopsy",
            "disease": "ibd",
        })

    df = pd.DataFrame(records)

    # Filter to IFX-treated patients
    ifx_mask = df["therapy"].str.contains("IFX", na=False)
    ifx_patients = df[ifx_mask]["patient_id"].unique()
    df_ifx = df[df["patient_id"].isin(ifx_patients)].copy()

    if df_ifx.empty:
        logger.warning("GSE73661: no IFX-treated patients found")
        return pd.DataFrame(columns=[
            "sample_id", "study", "response_status", "drug", "tissue_type",
            "timepoint", "prior_biologic_exposure", "disease",
        ])

    # Derive response from Mayo subscore changes
    df_ifx["mayo_num"] = pd.to_numeric(df_ifx["mayo"], errors="coerce")

    # Get baseline (W0) and follow-up Mayo scores per patient
    baseline = df_ifx[df_ifx["week"] == "W0"].set_index("patient_id")["mayo_num"]
    # Best follow-up score at any post-baseline timepoint
    followup = df_ifx[df_ifx["week"] != "W0"].groupby("patient_id")["mayo_num"].min()

    patient_response = {}
    for pid in ifx_patients:
        base = baseline.get(pid)
        fu = followup.get(pid)
        if pd.notna(base) and pd.notna(fu):
            # Responder: baseline >= 2 and follow-up <= 1
            if base >= 2 and fu <= 1:
                patient_response[pid] = "responder"
            elif base >= 2:
                patient_response[pid] = "non_responder"
            else:
                patient_response[pid] = "unknown"  # mild at baseline
        else:
            patient_response[pid] = "unknown"

    # Keep only W0 (baseline) samples with known response
    w0_mask = df_ifx["week"] == "W0"
    result_records = []
    for _, row in df_ifx[w0_mask].iterrows():
        resp = patient_response.get(row["patient_id"], "unknown")
        result_records.append({
            "sample_id": row["sample_id"],
            "study": "GSE73661",
            "response_status": resp,
            "drug": "infliximab",
            "tissue_type": "colonic_biopsy",
            "timepoint": "pre_treatment",
            "prior_biologic_exposure": "unknown",
            "disease": "ibd",
        })

    result = pd.DataFrame(result_records)
    known = result[result["response_status"].isin(["responder", "non_responder"])]
    logger.info(
        "GSE73661: %d IFX patients at W0, %d with derived response",
        len(result),
        len(known),
    )
    return known


# Dataset-specific metadata extractors
_METADATA_EXTRACTORS = {
    "GSE16879": _extract_gse16879,
    "GSE12251": _extract_gse12251,
    "GSE73661": _extract_gse73661,
}


def extract_metadata(gse, accession: str) -> pd.DataFrame:
    """Extract clinical metadata using dataset-specific parser."""
    extractor = _METADATA_EXTRACTORS.get(accession)
    if extractor is None:
        logger.warning("%s: no metadata extractor, returning empty", accession)
        return pd.DataFrame(columns=[
            "sample_id", "study", "response_status", "drug", "tissue_type",
            "timepoint", "prior_biologic_exposure", "disease",
        ])
    return extractor(gse)


def get_platform_annotation(gse: GEOparse.GEOTypes.GSE) -> pd.DataFrame | None:
    """Extract probe-to-gene symbol mapping from platform annotation.

    Returns a DataFrame with columns: probe_id, gene_symbol
    """
    for gpl_name, gpl in gse.gpls.items():
        table = gpl.table
        if table is None or table.empty:
            continue

        # Find the gene symbol column (check in priority order)
        gene_col = None
        for candidate in ["Gene Symbol", "gene_symbol", "Symbol", "gene symbol"]:
            if candidate in table.columns:
                gene_col = candidate
                break

        if gene_col is None:
            for col in table.columns:
                cl = col.lower()
                if "gene" in cl and "symbol" in cl:
                    gene_col = col
                    break

        if gene_col is None and "gene_assignment" in table.columns:
            gene_col = "gene_assignment"

        if gene_col is None:
            logger.warning("No gene symbol column found in %s. Columns: %s", gpl_name, list(table.columns))
            continue

        # Find probe ID column
        id_col = "ID"
        if id_col not in table.columns:
            id_col = table.columns[0]

        annotation = table[[id_col, gene_col]].copy()
        annotation.columns = ["probe_id", "gene_symbol"]

        # Parse gene symbol from gene_assignment format if needed
        if gene_col.lower().replace("_", " ") == "gene assignment":
            def _parse_gene_assignment(val):
                if pd.isna(val) or val == "---":
                    return ""
                parts = str(val).split("//")
                if len(parts) >= 2:
                    return parts[1].strip()
                return ""
            annotation["gene_symbol"] = annotation["gene_symbol"].apply(_parse_gene_assignment)

        # Clean gene symbols
        annotation["gene_symbol"] = (
            annotation["gene_symbol"]
            .astype(str)
            .str.strip()
            .str.upper()
            .replace({"": np.nan, "---": np.nan, "NAN": np.nan, "NA": np.nan})
        )
        annotation = annotation.dropna(subset=["gene_symbol"])

        # Handle multi-gene probes (take first gene)
        annotation["gene_symbol"] = annotation["gene_symbol"].str.split(r"[;/\\|]").str[0].str.strip()
        annotation = annotation[annotation["gene_symbol"].str.len() > 0]

        logger.info(
            "%s: %d probes mapped to %d unique genes",
            gpl_name,
            len(annotation),
            annotation["gene_symbol"].nunique(),
        )
        return annotation

    return None


def extract_expression(gse: GEOparse.GEOTypes.GSE) -> pd.DataFrame:
    """Extract expression matrix from GEO dataset.

    Returns: DataFrame with probes as rows and samples as columns.
    """
    # Pivot the GSM tables
    frames = {}
    for gsm_name, gsm in gse.gsms.items():
        tbl = gsm.table
        if tbl is not None and not tbl.empty:
            frames[gsm_name] = tbl.set_index("ID_REF")["VALUE"]

    if not frames:
        logger.error("No expression data found in GSMs")
        return pd.DataFrame()

    expr = pd.DataFrame(frames)
    expr.index.name = "probe_id"

    # Convert to numeric
    expr = expr.apply(pd.to_numeric, errors="coerce")

    logger.info("Expression matrix: %d probes x %d samples", *expr.shape)
    return expr


def map_probes_to_genes(
    expr: pd.DataFrame, annotation: pd.DataFrame
) -> pd.DataFrame:
    """Map probe IDs to gene symbols and collapse multi-probe genes.

    For genes mapped by multiple probes, keeps the probe with highest mean
    expression (standard practice for microarray analysis).

    Returns: gene-level expression matrix (genes x samples).
    """
    # Merge expression with annotation
    expr_reset = expr.reset_index()
    expr_reset.columns = ["probe_id"] + list(expr.columns)
    merged = expr_reset.merge(annotation, on="probe_id", how="inner")

    if merged.empty:
        logger.error("No probes matched annotation")
        return pd.DataFrame()

    sample_cols = [c for c in merged.columns if c not in ("probe_id", "gene_symbol")]

    # Compute mean expression per probe
    merged["_mean_expr"] = merged[sample_cols].mean(axis=1)

    # For each gene, keep the probe with highest mean expression
    idx = merged.groupby("gene_symbol")["_mean_expr"].idxmax()
    gene_expr = merged.loc[idx].drop(columns=["probe_id", "_mean_expr"]).set_index("gene_symbol")

    # Filter out non-gene symbols (numeric-only, too short)
    valid_mask = gene_expr.index.to_series().apply(
        lambda s: len(s) >= 2 and not s.isnumeric() and re.match(r"^[A-Z]", s) is not None
    )
    gene_expr = gene_expr[valid_mask]

    logger.info("Gene-level matrix: %d genes x %d samples", *gene_expr.shape)
    return gene_expr


def process_dataset(
    accession: str,
    cache_dir: Path,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """Download, process, and save a single GEO dataset.

    Returns (expression_df, metadata_df) or None on failure.
    """
    logger.info("=== Processing %s ===", accession)

    # Download
    try:
        gse = download_geo_dataset(accession, cache_dir)
    except Exception:
        logger.exception("Failed to download %s", accession)
        return None

    # Extract expression
    expr = extract_expression(gse)
    if expr.empty:
        logger.error("%s: empty expression matrix", accession)
        return None

    # Extract metadata
    metadata = extract_metadata(gse, accession)
    logger.info(
        "%s: %d samples, response distribution: %s",
        accession,
        len(metadata),
        metadata["response_status"].value_counts().to_dict(),
    )

    # Keep only samples in both expression and metadata
    common_samples = list(set(expr.columns) & set(metadata["sample_id"]))
    if not common_samples:
        logger.error("%s: no samples in common between expression and metadata", accession)
        return None

    expr = expr[common_samples]
    metadata = metadata[metadata["sample_id"].isin(common_samples)]

    # Map probes to genes
    annotation = get_platform_annotation(gse)
    if annotation is None or annotation.empty:
        logger.error("%s: no platform annotation available", accession)
        return None

    gene_expr = map_probes_to_genes(expr, annotation)
    if gene_expr.empty:
        logger.error("%s: gene mapping produced empty matrix", accession)
        return None

    # Save per-study files
    output_dir.mkdir(parents=True, exist_ok=True)
    expr_path = output_dir / f"{accession}_expression.csv"
    meta_path = output_dir / f"{accession}_metadata.csv"
    gene_expr.to_csv(expr_path)
    metadata.to_csv(meta_path, index=False)
    logger.info("%s: saved expression (%s) and metadata (%s)", accession, expr_path.name, meta_path.name)

    return gene_expr, metadata


def qc_check(
    expr: pd.DataFrame, metadata: pd.DataFrame, accession: str
) -> list[str]:
    """Run QC checks on a processed dataset. Returns list of issues."""
    issues = []

    # Gene count
    n_genes = len(expr)
    if n_genes < 10000:
        issues.append(f"{accession}: low gene count ({n_genes})")
    elif n_genes > 30000:
        issues.append(f"{accession}: unusually high gene count ({n_genes})")

    # Check for all-NA columns
    na_cols = expr.columns[expr.isna().all()].tolist()
    if na_cols:
        issues.append(f"{accession}: {len(na_cols)} all-NA samples")

    # Check for expected response genes
    found_genes = set(expr.index) & set(EXPECTED_RESPONSE_GENES)
    if len(found_genes) < 5:
        issues.append(
            f"{accession}: only {len(found_genes)}/{len(EXPECTED_RESPONSE_GENES)} "
            f"expected response genes found"
        )

    # Check response annotation
    n_resp = (metadata["response_status"] == "responder").sum()
    n_nonresp = (metadata["response_status"] == "non_responder").sum()
    n_unknown = (metadata["response_status"] == "unknown").sum()
    if n_resp == 0 and n_nonresp == 0:
        issues.append(f"{accession}: no response annotations found")
    elif n_unknown > 0:
        issues.append(f"{accession}: {n_unknown} samples with unknown response")

    return issues


def run_pipeline(
    output_dir: Path = DEFAULT_OUTPUT,
    cache_dir: Path = DEFAULT_CACHE,
    accessions: list[str] | None = None,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Run the full data processing pipeline.

    Returns:
        study_expressions: dict of accession -> gene expression DataFrame
        combined_metadata: merged clinical metadata for all studies
    """
    if accessions is None:
        accessions = list(USABLE_DATASETS)

    study_expressions: dict[str, pd.DataFrame] = {}
    all_metadata: list[pd.DataFrame] = []
    all_issues: list[str] = []

    for accession in accessions:
        result = process_dataset(accession, cache_dir, output_dir)
        if result is None:
            logger.error("Skipping %s due to processing failure", accession)
            continue

        expr, metadata = result
        issues = qc_check(expr, metadata, accession)
        all_issues.extend(issues)

        study_expressions[accession] = expr
        all_metadata.append(metadata)

    if not all_metadata:
        logger.error("No datasets processed successfully")
        return {}, pd.DataFrame()

    combined_metadata = pd.concat(all_metadata, ignore_index=True)

    # Save combined metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    combined_metadata.to_csv(output_dir / "combined_metadata.csv", index=False)

    # Save summary
    _print_summary(study_expressions, combined_metadata, all_issues)

    return study_expressions, combined_metadata


def _print_summary(
    study_expressions: dict[str, pd.DataFrame],
    combined_metadata: pd.DataFrame,
    issues: list[str],
) -> None:
    """Print processing summary."""
    print("\n" + "=" * 60)
    print("Anti-TNF Data Processing Summary")
    print("=" * 60)

    for accession, expr in study_expressions.items():
        meta = combined_metadata[combined_metadata["study"] == accession]
        n_resp = (meta["response_status"] == "responder").sum()
        n_nonresp = (meta["response_status"] == "non_responder").sum()
        n_unk = (meta["response_status"] == "unknown").sum()
        print(
            f"  {accession}: {expr.shape[1]} samples, {expr.shape[0]} genes | "
            f"R={n_resp} NR={n_nonresp} Unk={n_unk}"
        )

    # Common genes across studies
    if len(study_expressions) > 1:
        gene_sets = [set(df.index) for df in study_expressions.values()]
        common = gene_sets[0]
        for gs in gene_sets[1:]:
            common &= gs
        print(f"\n  Common genes across {len(study_expressions)} studies: {len(common)}")

        # Check expected response genes
        found = common & set(EXPECTED_RESPONSE_GENES)
        print(f"  Expected response genes in common set: {len(found)}/{len(EXPECTED_RESPONSE_GENES)}")
        if found:
            print(f"    Found: {', '.join(sorted(found))}")
        missing = set(EXPECTED_RESPONSE_GENES) - common
        if missing:
            print(f"    Missing: {', '.join(sorted(missing))}")

    print(f"\n  Total samples: {len(combined_metadata)}")
    print(f"  Response distribution:")
    for status, count in combined_metadata["response_status"].value_counts().items():
        print(f"    {status}: {count}")

    if issues:
        print(f"\n  QC Issues ({len(issues)}):")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print("\n  No QC issues found.")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Process GEO transcriptomic datasets for anti-TNF response prediction"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output directory for processed data",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE,
        help="Cache directory for downloaded GEO files",
    )
    parser.add_argument(
        "--accessions",
        nargs="+",
        default=None,
        help="Specific accessions to process (default: all)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    run_pipeline(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        accessions=args.accessions,
    )


if __name__ == "__main__":
    main()
