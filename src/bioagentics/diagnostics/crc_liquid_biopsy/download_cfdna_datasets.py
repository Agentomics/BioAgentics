"""Download and inspect GSE97923 and GSE149438 cfDNA datasets for CRC staging validation.

GSE97923: 773 CRC cfDNA samples (bisulfite-seq). Check for stage annotations.
GSE149438: 300 plasma samples, multi-GI cancer cfDNA (targeted bisulfite-seq).

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_cfdna_datasets inspect-gse97923
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_cfdna_datasets inspect-gse149438
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_cfdna_datasets download-gse97923 [--force]
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_cfdna_datasets download-gse149438 [--force]
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
    dest_dir.mkdir(parents=True, exist_ok=True)
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


def inspect_gse97923() -> pd.DataFrame:
    """Download GSE97923 SOFT file and inspect metadata for stage annotations."""
    dest = DATA_DIR / "gse97923" / "raw"
    gse = _download_gse("GSE97923", dest)

    metadata = _extract_sample_metadata(gse)
    n_samples = len(metadata)
    logger.info("GSE97923: %d samples", n_samples)
    logger.info("Columns: %s", list(metadata.columns))

    # Check for stage-related columns
    stage_cols = [
        c for c in metadata.columns
        if any(term in c.lower() for term in ("stage", "tnm", "ajcc", "duke", "grade"))
    ]
    logger.info("Stage-related columns: %s", stage_cols)

    # Print sample characteristics overview
    for col in metadata.columns:
        unique = metadata[col].nunique()
        if unique <= 20:
            logger.info("  %s (%d unique): %s", col, unique, metadata[col].value_counts().to_dict())
        else:
            logger.info("  %s (%d unique): [too many to list]", col, unique)

    # Save metadata
    out_path = DATA_DIR / "gse97923" / "gse97923_metadata.parquet"
    metadata.to_parquet(out_path)
    logger.info("Saved metadata to %s", out_path)

    # Also save as CSV for easy inspection
    csv_path = DATA_DIR / "gse97923" / "gse97923_metadata.csv"
    metadata.to_csv(csv_path)
    logger.info("Saved metadata CSV to %s", csv_path)

    return metadata


def inspect_gse149438() -> pd.DataFrame:
    """Download GSE149438 SOFT file and inspect metadata."""
    dest = DATA_DIR / "gse149438" / "raw"
    gse = _download_gse("GSE149438", dest)

    metadata = _extract_sample_metadata(gse)
    n_samples = len(metadata)
    logger.info("GSE149438: %d samples", n_samples)
    logger.info("Columns: %s", list(metadata.columns))

    # Check for stage-related columns
    stage_cols = [
        c for c in metadata.columns
        if any(term in c.lower() for term in ("stage", "tnm", "ajcc", "duke", "grade", "cancer"))
    ]
    logger.info("Stage/cancer-related columns: %s", stage_cols)

    for col in metadata.columns:
        unique = metadata[col].nunique()
        if unique <= 20:
            logger.info("  %s (%d unique): %s", col, unique, metadata[col].value_counts().to_dict())
        else:
            logger.info("  %s (%d unique): [too many to list]", col, unique)

    # List supplementary files
    supp = gse.metadata.get("supplementary_file", [])
    logger.info("Supplementary files: %s", supp)

    # Save metadata
    out_path = DATA_DIR / "gse149438" / "gse149438_metadata.parquet"
    metadata.to_parquet(out_path)
    logger.info("Saved metadata to %s", out_path)

    csv_path = DATA_DIR / "gse149438" / "gse149438_metadata.csv"
    metadata.to_csv(csv_path)
    logger.info("Saved metadata CSV to %s", csv_path)

    return metadata


def download_gse97923(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download full GSE97923 dataset including methylation data."""
    meta_path = DATA_DIR / "gse97923" / "gse97923_metadata.parquet"
    meth_path = DATA_DIR / "gse97923" / "gse97923_cfdna_methylation.parquet"

    if meta_path.exists() and meth_path.exists() and not force:
        logger.info("Loading cached GSE97923 data")
        return {
            "metadata": pd.read_parquet(meta_path),
            "methylation": pd.read_parquet(meth_path),
        }

    dest = DATA_DIR / "gse97923" / "raw"
    gse = _download_gse("GSE97923", dest)

    metadata = _extract_sample_metadata(gse)
    metadata["condition"] = metadata.apply(_classify_crc_condition, axis=1)
    logger.info("GSE97923 conditions: %s", metadata["condition"].value_counts().to_dict())
    metadata.to_parquet(meta_path)

    # Try to extract methylation data from supplementary files
    meth = _try_supplementary_matrix(gse, dest)
    if meth is not None:
        meth.to_parquet(meth_path)
    else:
        logger.warning("No methylation matrix found in GSE97923 supplementary files")
        pd.DataFrame().to_parquet(meth_path)

    return {"metadata": metadata, "methylation": meth or pd.DataFrame()}


def download_gse149438(force: bool = False) -> dict[str, pd.DataFrame]:
    """Download full GSE149438 dataset."""
    meta_path = DATA_DIR / "gse149438" / "gse149438_metadata.parquet"
    meth_path = DATA_DIR / "gse149438" / "gse149438_cfdna_methylation.parquet"

    if meta_path.exists() and meth_path.exists() and not force:
        logger.info("Loading cached GSE149438 data")
        return {
            "metadata": pd.read_parquet(meta_path),
            "methylation": pd.read_parquet(meth_path),
        }

    dest = DATA_DIR / "gse149438" / "raw"
    gse = _download_gse("GSE149438", dest)

    metadata = _extract_sample_metadata(gse)
    metadata["condition"] = metadata.apply(_classify_crc_condition, axis=1)
    logger.info("GSE149438 conditions: %s", metadata["condition"].value_counts().to_dict())
    metadata.to_parquet(meta_path)

    meth = _try_supplementary_matrix(gse, dest)
    if meth is not None:
        meth.to_parquet(meth_path)
    else:
        logger.warning("No methylation matrix found in GSE149438 supplementary files")
        pd.DataFrame().to_parquet(meth_path)

    return {"metadata": metadata, "methylation": meth or pd.DataFrame()}


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


def _try_supplementary_matrix(gse, _dest_dir: Path) -> pd.DataFrame | None:
    """Try to load a data matrix from GEO supplementary files."""
    import gzip
    import io

    import requests

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
                resp = requests.get(url, timeout=300, stream=True)
                resp.raise_for_status()
                # Check content-length to avoid loading huge files into memory
                content_length = int(resp.headers.get("content-length", 0))
                if content_length > 500_000_000:  # 500MB limit
                    logger.warning("  Skipping %s: too large (%d bytes)", url, content_length)
                    continue
                raw = resp.content
                if fname.endswith(".gz"):
                    with gzip.open(io.BytesIO(raw), "rt") as f:
                        df = pd.read_csv(f, sep="\t", index_col=0)
                else:
                    df = pd.read_csv(io.BytesIO(raw), sep="\t", index_col=0)

                if df.shape[0] > 100 and df.shape[1] > 2:
                    logger.info("  Loaded supplementary data: %d x %d", df.shape[0], df.shape[1])
                    return df
            except Exception as e:
                logger.warning("  Failed to load %s: %s", url, e)
                continue
    return None


def main():
    parser = argparse.ArgumentParser(description="Download cfDNA datasets for CRC staging validation")
    parser.add_argument(
        "command",
        choices=["inspect-gse97923", "inspect-gse149438", "download-gse97923", "download-gse149438"],
        help="Command to run",
    )
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if args.command == "inspect-gse97923":
        metadata = inspect_gse97923()
        print(f"\nGSE97923: {len(metadata)} samples, {len(metadata.columns)} metadata fields")
        print(f"Columns: {list(metadata.columns)}")
    elif args.command == "inspect-gse149438":
        metadata = inspect_gse149438()
        print(f"\nGSE149438: {len(metadata)} samples, {len(metadata.columns)} metadata fields")
        print(f"Columns: {list(metadata.columns)}")
    elif args.command == "download-gse97923":
        result = download_gse97923(force=args.force)
        for key, df in result.items():
            print(f"  GSE97923 {key}: {df.shape}")
    elif args.command == "download-gse149438":
        result = download_gse149438(force=args.force)
        for key, df in result.items():
            print(f"  GSE149438 {key}: {df.shape}")


if __name__ == "__main__":
    main()
