"""Download and preprocess GEO datasets for CRC liquid biopsy panel.

Downloads:
  - GSE149282: cfDNA methylation profiles (CRC patients vs healthy controls)
  - GSE164191: Circulating protein biomarker panels in CRC screening
  - GSE48684: Tissue methylation (CRC vs adjacent normal, Illumina 450K)

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_geo GSE149282 [--force]
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_geo GSE164191 [--force]
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_geo GSE48684 [--force]
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_geo all [--force]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"


def _download_gse(accession: str, dest_dir: Path):
    """Download a GEO Series and return the GEOparse GSE object."""
    import GEOparse

    logger.info("Downloading %s from GEO...", accession)
    gse = GEOparse.get_GEO(geo=accession, destdir=str(dest_dir), silent=True)
    return gse


def _extract_sample_metadata(gse) -> pd.DataFrame:
    """Extract sample metadata from a GEOparse GSE object."""
    records: list[dict] = []
    for gsm_name, gsm in gse.gsms.items():
        meta = gsm.metadata
        rec: dict = {
            "gsm": gsm_name,
            "title": meta.get("title", [""])[0],
            "source": meta.get("source_name_ch1", [""])[0],
            "organism": meta.get("organism_ch1", [""])[0],
            "platform": meta.get("platform_id", [""])[0],
        }
        for ch in meta.get("characteristics_ch1", []):
            if ":" in ch:
                key, _, val = ch.partition(":")
                key = key.strip().lower().replace(" ", "_")
                rec[key] = val.strip()
            else:
                rec.setdefault("characteristics", [])
                if isinstance(rec["characteristics"], list):
                    rec["characteristics"].append(ch.strip())
        if isinstance(rec.get("characteristics"), list):
            rec["characteristics"] = "; ".join(rec["characteristics"])
        records.append(rec)
    return pd.DataFrame(records).set_index("gsm")


def _extract_methylation_table(gse) -> pd.DataFrame | None:
    """Extract methylation beta values from a GEOparse GSE object.

    Looks for beta values in sample tables. Returns CpG x samples matrix
    or None if no methylation data found in tables.
    """
    sample_data = {}
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table is None or table.empty:
            continue
        # Look for columns that contain beta values
        beta_col = None
        for col in table.columns:
            cl = col.lower()
            if "beta" in cl or "value" in cl or "methylat" in cl:
                beta_col = col
                break
        if beta_col is None and "VALUE" in table.columns:
            beta_col = "VALUE"
        if beta_col is None:
            continue

        # Use ID_REF as probe index
        id_col = "ID_REF" if "ID_REF" in table.columns else table.columns[0]
        series = table.set_index(id_col)[beta_col]
        series = pd.to_numeric(series, errors="coerce")
        sample_data[gsm_name] = series

    if not sample_data:
        return None

    matrix = pd.DataFrame(sample_data)
    logger.info("  Extracted methylation matrix: %d probes x %d samples", *matrix.shape)
    return matrix


def download_gse149282(dest_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download GSE149282: cfDNA methylation in CRC patients vs controls.

    Returns dict with 'metadata' and 'methylation' DataFrames.
    """
    meta_path = dest_dir / "gse149282_metadata.parquet"
    meth_path = dest_dir / "gse149282_cfdna_methylation.parquet"

    if meta_path.exists() and meth_path.exists() and not force:
        logger.info("Loading cached GSE149282 data")
        return {
            "metadata": pd.read_parquet(meta_path),
            "methylation": pd.read_parquet(meth_path),
        }

    gse = _download_gse("GSE149282", dest_dir / "raw")
    metadata = _extract_sample_metadata(gse)

    # Classify samples as CRC or control based on metadata
    metadata["condition"] = metadata.apply(_classify_crc_condition, axis=1)
    logger.info("GSE149282 conditions: %s", metadata["condition"].value_counts().to_dict())

    # Extract methylation data
    meth = _extract_methylation_table(gse)
    if meth is None:
        logger.warning("No per-sample methylation tables in GSE149282; checking supplementary...")
        meth = _try_supplementary_methylation(gse, dest_dir / "raw")

    if meth is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        meth.to_parquet(meth_path)
    else:
        logger.warning("Could not extract methylation matrix from GSE149282")
        # Save empty placeholder
        meth = pd.DataFrame()
        meth.to_parquet(meth_path)

    metadata.to_parquet(meta_path)
    return {"metadata": metadata, "methylation": meth}


def download_gse164191(dest_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download GSE164191: circulating protein biomarkers in CRC screening.

    Returns dict with 'metadata' and 'protein_data' DataFrames.
    """
    meta_path = dest_dir / "gse164191_metadata.parquet"
    protein_path = dest_dir / "gse164191_protein_biomarkers.parquet"

    if meta_path.exists() and protein_path.exists() and not force:
        logger.info("Loading cached GSE164191 data")
        return {
            "metadata": pd.read_parquet(meta_path),
            "protein_data": pd.read_parquet(protein_path),
        }

    gse = _download_gse("GSE164191", dest_dir / "raw")
    metadata = _extract_sample_metadata(gse)
    metadata["condition"] = metadata.apply(_classify_crc_condition, axis=1)
    logger.info("GSE164191 conditions: %s", metadata["condition"].value_counts().to_dict())

    # Extract protein expression data from sample tables
    protein_data = _extract_expression_table(gse)
    if protein_data is None:
        logger.warning("No per-sample tables in GSE164191; checking supplementary...")
        protein_data = _try_supplementary_expression(gse, dest_dir / "raw")

    if protein_data is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        protein_data.to_parquet(protein_path)
    else:
        logger.warning("Could not extract protein data from GSE164191")
        protein_data = pd.DataFrame()
        protein_data.to_parquet(protein_path)

    metadata.to_parquet(meta_path)
    return {"metadata": metadata, "protein_data": protein_data}


def download_gse48684(dest_dir: Path, force: bool = False) -> dict[str, pd.DataFrame]:
    """Download GSE48684: tissue methylation in CRC vs normal (450K).

    Returns dict with 'metadata' and 'methylation' DataFrames.
    """
    meta_path = dest_dir / "gse48684_metadata.parquet"
    meth_path = dest_dir / "gse48684_tissue_methylation.parquet"

    if meta_path.exists() and meth_path.exists() and not force:
        logger.info("Loading cached GSE48684 data")
        return {
            "metadata": pd.read_parquet(meta_path),
            "methylation": pd.read_parquet(meth_path),
        }

    gse = _download_gse("GSE48684", dest_dir / "raw")
    metadata = _extract_sample_metadata(gse)
    metadata["condition"] = metadata.apply(_classify_tissue_condition, axis=1)
    logger.info("GSE48684 conditions: %s", metadata["condition"].value_counts().to_dict())

    meth = _extract_methylation_table(gse)
    if meth is None:
        logger.warning("No per-sample methylation tables in GSE48684; checking supplementary...")
        meth = _try_supplementary_methylation(gse, dest_dir / "raw")

    if meth is not None:
        dest_dir.mkdir(parents=True, exist_ok=True)
        meth.to_parquet(meth_path)
    else:
        logger.warning("Could not extract methylation matrix from GSE48684")
        meth = pd.DataFrame()
        meth.to_parquet(meth_path)

    metadata.to_parquet(meta_path)
    return {"metadata": metadata, "methylation": meth}


def _classify_crc_condition(row: pd.Series) -> str:
    """Classify a sample as CRC, adenoma, or control from metadata."""
    text = " ".join(str(v).lower() for v in row.values if pd.notna(v))
    if any(
        term in text
        for term in ("colorectal cancer", "colon cancer", "rectal cancer", "crc", "carcinoma", "tumor")
    ):
        return "CRC"
    if "adenoma" in text:
        return "adenoma"
    if any(term in text for term in ("healthy", "control", "normal")):
        return "control"
    return "unknown"


def _classify_tissue_condition(row: pd.Series) -> str:
    """Classify tissue sample as tumor, adjacent normal, or normal."""
    text = " ".join(str(v).lower() for v in row.values if pd.notna(v))
    if any(t in text for t in ("tumor", "cancer", "carcinoma", "malignant")):
        return "tumor"
    if any(t in text for t in ("adjacent", "matched normal")):
        return "adjacent_normal"
    if "normal" in text:
        return "normal"
    return "unknown"


def _extract_expression_table(gse) -> pd.DataFrame | None:
    """Extract expression/protein data from GEOparse sample tables."""
    sample_data = {}
    for gsm_name, gsm in gse.gsms.items():
        table = gsm.table
        if table is None or table.empty:
            continue
        val_col = None
        for col in table.columns:
            cl = col.lower()
            if "value" in cl or "signal" in cl or "intensity" in cl:
                val_col = col
                break
        if val_col is None and "VALUE" in table.columns:
            val_col = "VALUE"
        if val_col is None:
            continue

        id_col = "ID_REF" if "ID_REF" in table.columns else table.columns[0]
        series = table.set_index(id_col)[val_col]
        series = pd.to_numeric(series, errors="coerce")
        sample_data[gsm_name] = series

    if not sample_data:
        return None

    matrix = pd.DataFrame(sample_data)
    logger.info("  Extracted expression matrix: %d features x %d samples", *matrix.shape)
    return matrix


def _try_supplementary_methylation(gse, dest_dir: Path) -> pd.DataFrame | None:
    """Try to extract methylation data from supplementary files."""
    return _try_supplementary_matrix(gse, dest_dir, data_type="methylation")


def _try_supplementary_expression(gse, dest_dir: Path) -> pd.DataFrame | None:
    """Try to extract expression data from supplementary files."""
    return _try_supplementary_matrix(gse, dest_dir, data_type="expression")


def _try_supplementary_matrix(gse, _dest_dir: Path, data_type: str) -> pd.DataFrame | None:
    """Try to load a data matrix from GEO supplementary files."""
    import gzip
    import io

    import requests

    # Check for series-level supplementary files
    supp_files = gse.metadata.get("supplementary_file", [])
    for url in supp_files:
        if not url or url == "NONE":
            continue
        fname = url.split("/")[-1].lower()
        if any(
            ext in fname
            for ext in (".txt.gz", ".tsv.gz", ".csv.gz", ".matrix.gz", "_matrix_")
        ):
            logger.info("  Trying supplementary file: %s", url)
            try:
                resp = requests.get(url, timeout=300)
                resp.raise_for_status()
                raw = resp.content
                if fname.endswith(".gz"):
                    with gzip.open(io.BytesIO(raw), "rt") as f:
                        df = pd.read_csv(f, sep="\t", index_col=0)
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0)

                if df.shape[0] > 100 and df.shape[1] > 2:
                    logger.info(
                        "  Loaded supplementary %s: %d x %d",
                        data_type,
                        df.shape[0],
                        df.shape[1],
                    )
                    return df
            except Exception as e:
                logger.warning("  Failed to load %s: %s", url, e)
                continue
    return None


DATASETS = {
    "GSE149282": download_gse149282,
    "GSE164191": download_gse164191,
    "GSE48684": download_gse48684,
}


def main():
    parser = argparse.ArgumentParser(description="Download GEO datasets for CRC liquid biopsy")
    parser.add_argument("dataset", choices=[*DATASETS.keys(), "all"], help="GEO accession or 'all'")
    parser.add_argument("--dest", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    datasets = DATASETS.keys() if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        logger.info("=== Processing %s ===", ds)
        result = DATASETS[ds](args.dest, force=args.force)
        for key, df in result.items():
            print(f"  {ds} {key}: {df.shape}")


if __name__ == "__main__":
    main()
