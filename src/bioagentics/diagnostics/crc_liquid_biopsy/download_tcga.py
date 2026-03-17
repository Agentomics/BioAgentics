"""Download and preprocess TCGA-COAD/READ methylation and clinical data.

Downloads 450K methylation beta values from UCSC Xena (GDC hub) pre-compiled
matrices, and clinical metadata from the GDC REST API. Merges COAD and READ
into a single cohort and outputs parquet files.

Output:
    data/diagnostics/crc-liquid-biopsy-panel/tcga_methylation.parquet
    data/diagnostics/crc-liquid-biopsy-panel/tcga_clinical.parquet

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_tcga [--force]
"""

from __future__ import annotations

import argparse
import gzip
import io
import json
import logging
from pathlib import Path

import pandas as pd
import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel"

# UCSC Xena GDC hub URLs for TCGA 450K methylation data
XENA_BASE = "https://gdc-hub.s3.us-east-1.amazonaws.com/download"
METHYLATION_URLS = {
    "COAD": f"{XENA_BASE}/TCGA-COAD.methylation450.tsv.gz",
    "READ": f"{XENA_BASE}/TCGA-READ.methylation450.tsv.gz",
}

# GDC REST API for clinical data
GDC_API = "https://api.gdc.cancer.gov"


def _download_gz_tsv(url: str, desc: str) -> pd.DataFrame:
    """Download a gzipped TSV from a URL and return as DataFrame."""
    logger.info("Downloading %s from %s ...", desc, url)
    resp = requests.get(url, timeout=300, stream=True)
    resp.raise_for_status()

    raw = resp.content
    with gzip.open(io.BytesIO(raw), "rt") as f:
        df = pd.read_csv(f, sep="\t", index_col=0)
    logger.info("  %s: %d rows x %d cols", desc, df.shape[0], df.shape[1])
    return df


def download_methylation(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download and merge TCGA-COAD + TCGA-READ 450K methylation data.

    Returns a DataFrame with CpG sites as rows and samples as columns.
    Beta values range 0-1.
    """
    cache_path = dest_dir / "tcga_methylation.parquet"
    if cache_path.exists() and not force:
        logger.info("Loading cached methylation data from %s", cache_path)
        return pd.read_parquet(cache_path)

    dfs = []
    for cohort, url in METHYLATION_URLS.items():
        df = _download_gz_tsv(url, f"TCGA-{cohort} methylation450")
        dfs.append(df)

    # Merge on shared CpG probes (intersection of rows)
    merged = pd.concat(dfs, axis=1, join="inner")
    # Drop any fully-NA CpG rows
    merged = merged.dropna(how="all")
    logger.info(
        "Merged methylation matrix: %d CpG sites x %d samples",
        merged.shape[0],
        merged.shape[1],
    )

    dest_dir.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(cache_path)
    logger.info("Saved methylation data to %s", cache_path)
    return merged


def _fetch_gdc_clinical(project: str) -> list[dict]:
    """Fetch clinical data from GDC API for a TCGA project."""
    fields = [
        "submitter_id",
        "demographic.gender",
        "demographic.race",
        "demographic.vital_status",
        "demographic.days_to_death",
        "diagnoses.age_at_diagnosis",
        "diagnoses.ajcc_pathologic_stage",
        "diagnoses.tumor_stage",
        "diagnoses.days_to_last_follow_up",
        "diagnoses.primary_diagnosis",
        "samples.sample_type",
        "samples.submitter_id",
    ]
    filters = {
        "op": "=",
        "content": {"field": "project.project_id", "value": project},
    }
    all_hits = []
    size = 500
    offset = 0
    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "size": size,
            "from": offset,
            "format": "JSON",
        }
        resp = requests.get(f"{GDC_API}/cases", params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()["data"]
        hits = data["hits"]
        if not hits:
            break
        all_hits.extend(hits)
        offset += size
        if offset >= data["pagination"]["total"]:
            break
    logger.info("  GDC %s: %d cases", project, len(all_hits))
    return all_hits


def _parse_gdc_cases(hits: list[dict], cohort: str) -> pd.DataFrame:
    """Parse GDC case records into a flat DataFrame."""
    rows = []
    for case in hits:
        demo = case.get("demographic", {}) or {}
        diags = case.get("diagnoses", []) or []
        diag = diags[0] if diags else {}
        samples = case.get("samples", []) or []

        # Get sample submitter IDs (prefer primary tumor)
        sample_ids = []
        for s in samples:
            if "tumor" in (s.get("sample_type") or "").lower():
                sample_ids.append(s.get("submitter_id", ""))

        # Age is stored in days in GDC, convert to years
        age_days = diag.get("age_at_diagnosis")
        age_years = round(age_days / 365.25) if age_days is not None else None

        row = {
            "case_id": case.get("submitter_id", ""),
            "cohort": cohort,
            "sex": demo.get("gender"),
            "race": demo.get("race"),
            "vital_status": demo.get("vital_status"),
            "days_to_death": demo.get("days_to_death"),
            "age_at_diagnosis": age_years,
            "stage": diag.get("ajcc_pathologic_stage") or diag.get("tumor_stage"),
            "days_to_last_followup": diag.get("days_to_last_follow_up"),
            "primary_diagnosis": diag.get("primary_diagnosis"),
            "sample_ids": ";".join(sample_ids) if sample_ids else None,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def download_clinical(dest_dir: Path, force: bool = False) -> pd.DataFrame:
    """Download and merge TCGA-COAD + TCGA-READ clinical data from GDC API.

    Returns a DataFrame with one row per case containing stage, age, sex,
    vital status, and other clinical annotations.
    """
    cache_path = dest_dir / "tcga_clinical.parquet"
    if cache_path.exists() and not force:
        logger.info("Loading cached clinical data from %s", cache_path)
        return pd.read_parquet(cache_path)

    logger.info("Downloading clinical data from GDC API...")
    dfs = []
    for cohort in ("TCGA-COAD", "TCGA-READ"):
        hits = _fetch_gdc_clinical(cohort)
        df = _parse_gdc_cases(hits, cohort.split("-")[1])
        dfs.append(df)

    clinical = pd.concat(dfs, ignore_index=True)
    clinical = clinical.set_index("case_id")

    # Add numeric stage
    clinical["stage_numeric"] = clinical["stage"].apply(_parse_stage_number)
    logger.info("Merged clinical data: %d cases", len(clinical))

    dest_dir.mkdir(parents=True, exist_ok=True)
    clinical.to_parquet(cache_path)
    logger.info("Saved clinical data to %s", cache_path)
    return clinical


def _parse_stage_number(stage_str) -> int | None:
    """Parse AJCC stage string to numeric stage (1-4)."""
    if pd.isna(stage_str):
        return None
    s = str(stage_str).upper().strip()
    if "IV" in s or "STAGE IV" in s:
        return 4
    if "III" in s or "STAGE III" in s:
        return 3
    if "II" in s or "STAGE II" in s:
        return 2
    if "I" in s or "STAGE I" in s:
        return 1
    return None


def main():
    parser = argparse.ArgumentParser(description="Download TCGA-COAD/READ methylation + clinical data")
    parser.add_argument("--dest", type=Path, default=DATA_DIR)
    parser.add_argument("--force", action="store_true", help="Re-download even if cached")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    meth = download_methylation(args.dest, force=args.force)
    clin = download_clinical(args.dest, force=args.force)

    print(f"\nMethylation matrix: {meth.shape[0]} CpG sites x {meth.shape[1]} samples")
    print(f"Clinical data: {len(clin)} samples")

    if "stage_numeric" in clin.columns:
        print("\nStage distribution:")
        print(clin["stage_numeric"].value_counts().sort_index())


if __name__ == "__main__":
    main()
