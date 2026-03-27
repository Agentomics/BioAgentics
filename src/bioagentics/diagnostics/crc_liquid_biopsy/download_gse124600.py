"""Download GSE124600 MCTA-Seq cfDNA methylation data for CRC validation.

GSE124600: 340 samples total from plasma cfDNA (CRC + healthy) and tissue.
Best independent cfDNA CRC validation cohort with stage annotations
(Stage I:42, II:101, III:58, IV:7).

Also maps our priority CpGs to genomic coordinates for cross-referencing
with MCTA-Seq CpG island coverage.

Output:
    data/diagnostics/crc-liquid-biopsy-panel/gse124600/
        gse124600_metadata.parquet
        gse124600_supplementary/  (raw files)

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_gse124600 [--force]
"""

from __future__ import annotations

import argparse
import gzip
import logging
import shutil
from pathlib import Path

import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel" / "gse124600"

# Priority CpGs from our panel for cross-referencing
PRIORITY_CPGS = {
    "cg22418909": {"gene": "SEPT9", "chr": "chr17", "pos": 77373470},
    "cg20611276": {"gene": "VIM", "chr": "chr10", "pos": 17268770},
    "cg06132028": {"gene": "BMP3", "chr": "chr4", "pos": 81032540},
}


def download_metadata(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download GSE124600 metadata via GEOparse."""
    meta_path = dest_dir / "gse124600_metadata.parquet"
    if meta_path.exists() and not force:
        logger.info("Loading cached metadata from %s", meta_path)
        return pd.read_parquet(meta_path)

    import GEOparse

    dest_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(exist_ok=True)

    logger.info("Downloading GSE124600 from GEO...")
    gse = GEOparse.get_GEO(geo="GSE124600", destdir=str(raw_dir), silent=True)

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
        # Extract supplementary file URLs
        supp = meta.get("supplementary_file", [])
        rec["supplementary_files"] = "; ".join(supp) if supp else ""
        records.append(rec)

    metadata = pd.DataFrame(records).set_index("gsm")

    # Identify sample types and stages
    logger.info("GSE124600: %d samples", len(metadata))
    logger.info("Columns: %s", list(metadata.columns))

    # Check for condition/stage columns
    for col in metadata.columns:
        unique = metadata[col].nunique()
        if unique < 20:
            logger.info("  %s (%d unique): %s", col, unique,
                        dict(metadata[col].value_counts().head(10)))

    metadata.to_parquet(meta_path)
    logger.info("Saved metadata to %s", meta_path)
    return metadata


def download_supplementary_files(
    metadata: pd.DataFrame, dest_dir: Path, force: bool = False
) -> list[Path]:
    """Download supplementary methylation files from GEO.

    Downloads series-level supplementary files which typically contain
    the processed methylation matrix for MCTA-Seq data.
    """
    supp_dir = dest_dir / "gse124600_supplementary"
    supp_dir.mkdir(parents=True, exist_ok=True)

    # GEO series supplementary files URL pattern
    base_url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE124nnn/GSE124600/suppl/"

    # Try to list available files
    logger.info("Checking supplementary files at GEO FTP...")
    downloaded = []

    # Common supplementary file patterns for MCTA-Seq
    filenames = [
        "GSE124600_processed_data.txt.gz",
        "GSE124600_RAW.tar",
        "GSE124600_methylation_matrix.txt.gz",
        "GSE124600_MCTA_Seq_data.txt.gz",
    ]

    for fname in filenames:
        url = base_url + fname
        dest_path = supp_dir / fname
        if dest_path.exists() and not force:
            logger.info("Already have %s", fname)
            downloaded.append(dest_path)
            continue

        logger.info("Trying to download %s...", url)
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                with open(dest_path, "wb") as f:
                    shutil.copyfileobj(resp.raw, f)
                logger.info("Downloaded %s (%.1f MB)", fname, dest_path.stat().st_size / 1e6)
                downloaded.append(dest_path)
            else:
                logger.info("  Not found (HTTP %d)", resp.status_code)
        except requests.RequestException as e:
            logger.warning("  Failed: %s", e)

    # Also try to download per-sample supplementary files from metadata
    # but only if no series-level files were found
    if not downloaded:
        logger.info("No series-level files found. Checking per-sample files...")
        sample_urls = []
        for _gsm, row in metadata.iterrows():
            supp = row.get("supplementary_files", "")
            if supp:
                for url in supp.split("; "):
                    url = url.strip()
                    if url:
                        sample_urls.append(url)

        logger.info("Found %d per-sample supplementary URLs", len(sample_urls))
        # Download first few to inspect format
        for url in sample_urls[:5]:
            fname = url.split("/")[-1]
            dest_path = supp_dir / fname
            if dest_path.exists() and not force:
                downloaded.append(dest_path)
                continue
            try:
                resp = requests.get(url, stream=True, timeout=120)
                if resp.status_code == 200:
                    with open(dest_path, "wb") as f:
                        shutil.copyfileobj(resp.raw, f)
                    logger.info("Downloaded %s (%.1f KB)", fname, dest_path.stat().st_size / 1e3)
                    downloaded.append(dest_path)
            except requests.RequestException as e:
                logger.warning("Failed to download %s: %s", fname, e)

    return downloaded


def map_priority_cpgs() -> pd.DataFrame:
    """Map priority CpGs to genomic coordinates for MCTA-Seq cross-reference.

    MCTA-Seq targets CpG islands (CGIs) rather than individual CpG sites.
    This maps our 450K-based CpG IDs to chromosomal positions so we can
    check if they fall within MCTA-Seq-covered CGIs.
    """
    records = []
    for cpg_id, info in PRIORITY_CPGS.items():
        records.append({
            "cpg_id": cpg_id,
            "gene": info["gene"],
            "chr": info["chr"],
            "position": info["pos"],
            "note": f"450K probe in {info['gene']} promoter region",
        })

    df = pd.DataFrame(records)
    logger.info("Priority CpG coordinate mapping:")
    for _, row in df.iterrows():
        logger.info("  %s (%s): %s:%d", row["cpg_id"], row["gene"], row["chr"], row["position"])

    return df


def run_download(
    data_dir: Path = DATA_DIR,
    force: bool = False,
) -> dict:
    """Download GSE124600 data and metadata."""
    # 1. Download and parse metadata
    metadata = download_metadata(data_dir, force=force)

    # 2. Download supplementary files
    downloaded = download_supplementary_files(metadata, data_dir, force=force)

    # 3. Map priority CpGs
    cpg_map = map_priority_cpgs()
    cpg_map.to_parquet(data_dir / "priority_cpg_coordinates.parquet")

    # 4. Summarize
    n_total = len(metadata)

    # Try to identify CRC vs healthy and stages
    condition_col = None
    stage_col = None
    for col in metadata.columns:
        vals = metadata[col].str.lower().fillna("")
        if vals.str.contains("cancer|crc|tumor|carcinoma").any() and vals.str.contains("normal|healthy|control").any():
            condition_col = col
        if vals.str.contains("stage|i|ii|iii|iv").sum() > 5:
            if "stage" in col.lower() or "tnm" in col.lower():
                stage_col = col

    summary = {
        "accession": "GSE124600",
        "n_total_samples": n_total,
        "metadata_columns": list(metadata.columns),
        "files_downloaded": [str(f.name) for f in downloaded],
        "condition_column": condition_col,
        "stage_column": stage_col,
        "priority_cpgs": list(PRIORITY_CPGS.keys()),
    }

    if condition_col:
        summary["condition_distribution"] = dict(metadata[condition_col].value_counts())
    if stage_col:
        summary["stage_distribution"] = dict(metadata[stage_col].value_counts())

    return summary


def main():
    parser = argparse.ArgumentParser(description="Download GSE124600 cfDNA methylation data")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    summary = run_download(args.data_dir, force=args.force)

    print(f"\n=== GSE124600 Download Summary ===")
    print(f"Total samples: {summary['n_total_samples']}")
    print(f"Files downloaded: {len(summary['files_downloaded'])}")
    if summary.get("condition_distribution"):
        print(f"Conditions: {summary['condition_distribution']}")
    if summary.get("stage_distribution"):
        print(f"Stages: {summary['stage_distribution']}")
    print(f"Priority CpGs mapped: {summary['priority_cpgs']}")


if __name__ == "__main__":
    main()
