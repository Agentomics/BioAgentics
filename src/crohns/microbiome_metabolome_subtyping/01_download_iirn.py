#!/usr/bin/env python3
"""Download IIRN paired metabolomics + metagenomics data (Sci Rep 2026).

Study: Sheba Medical Center IIRN cohort — 80 CD + 43 HC subjects with
paired fecal 16S, shotgun metagenomics, and fecal/serum LC-MS metabolomics.

Data sources:
- 16S amplicon: PRJNA1053872 (NCBI SRA)
- Shotgun metagenomics: PRJNA1057679 (NCBI SRA)
- Metabolomics: Supplementary Dataset S1 (doi:10.1038/s41598-026-38558-9)
- Analysis code: github.com/ShebaMicrobiomeCenter/IIRN_metabolomics

This script downloads SRA run accessions and metadata via NCBI Entrez,
and fetches supplementary metabolomics tables from the publisher.

Usage:
    uv run python -m crohns.microbiome_metabolome_subtyping.01_download_iirn
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

OUTPUT_DIR = (
    REPO_ROOT / "data" / "crohns" / "microbiome-metabolome-subtyping" / "iirn_metabolomics"
)

# NCBI BioProject accessions
BIOPROJECT_16S = "PRJNA1053872"
BIOPROJECT_SHOTGUN = "PRJNA1057679"

# NCBI Entrez API for fetching SRA run info
ENTREZ_SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
ENTREZ_FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
SRA_RUNINFO_URL = "https://trace.ncbi.nlm.nih.gov/Traces/sra/sra.cgi"

# Supplementary data URL pattern for Nature Scientific Reports
# The supplementary dataset is typically available as an Excel file
SUPPL_BASE = "https://static-content.springer.com/esm/art%3A10.1038%2Fs41598-026-38558-9/MediaObjects"

# GitHub repo for analysis code
GITHUB_REPO_URL = "https://github.com/ShebaMicrobiomeCenter/IIRN_metabolomics"

MAX_RETRIES = 4
INITIAL_BACKOFF_S = 2.0
TIMEOUT_S = 120


# --- Helpers -----------------------------------------------------------------

def _download_with_retry(
    url: str,
    dest: Path,
    *,
    retries: int = MAX_RETRIES,
    backoff: float = INITIAL_BACKOFF_S,
    headers: dict[str, str] | None = None,
) -> bool:
    """Download *url* to *dest* with exponential backoff.

    Returns True on success, False if all retries exhausted or 404.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, timeout=TIMEOUT_S, stream=True, headers=headers or {},
            )
            if resp.status_code == 404:
                logger.warning("404 Not Found: %s", url)
                return False
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
            return True
        except (requests.RequestException, OSError) as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d for %s failed (%s). Retrying in %.1fs ...",
                attempt, retries, url, exc, wait,
            )
            time.sleep(wait)
    return False


def _fetch_sra_runinfo(bioproject: str, dest: Path) -> bool:
    """Fetch SRA run info CSV for a BioProject via NCBI Entrez.

    Saves a CSV with columns: Run, spots, bases, avgLength, size_MB,
    download_path, Experiment, SRAStudy, BioProject, Sample, etc.
    """
    if dest.exists():
        logger.info("SRA run info already exists: %s", dest)
        return True

    logger.info("Fetching SRA run info for %s ...", bioproject)

    # Step 1: Search for SRA entries linked to this BioProject
    search_params = {
        "db": "sra",
        "term": f"{bioproject}[BioProject]",
        "retmax": 1000,
        "retmode": "json",
    }
    try:
        resp = requests.get(ENTREZ_SEARCH_URL, params=search_params, timeout=TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        id_list = data.get("esearchresult", {}).get("idlist", [])
        count = int(data.get("esearchresult", {}).get("count", 0))
        logger.info("Found %d SRA entries for %s", count, bioproject)

        if not id_list:
            logger.warning("No SRA entries found for %s", bioproject)
            return False
    except (requests.RequestException, ValueError) as exc:
        logger.error("Entrez search failed for %s: %s", bioproject, exc)
        return False

    # Step 2: Fetch run info via SRA run selector
    # Use the ENA API as a more reliable alternative for getting run accessions
    run_info_url = (
        f"https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={bioproject}"
        f"&result=read_run"
        f"&fields=run_accession,experiment_accession,sample_accession,"
        f"instrument_model,library_strategy,library_source,library_layout,"
        f"read_count,base_count,fastq_ftp,fastq_bytes,fastq_md5,"
        f"sample_alias,sample_title"
        f"&format=tsv"
        f"&limit=0"
    )

    try:
        resp = requests.get(run_info_url, timeout=TIMEOUT_S)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(resp.text)
        # Count lines (minus header)
        n_runs = len(resp.text.strip().split("\n")) - 1
        logger.info("Saved %d run records to %s", n_runs, dest)
        return True
    except (requests.RequestException, OSError) as exc:
        logger.error("Failed to fetch run info for %s: %s", bioproject, exc)
        return False


def _download_supplementary_metabolomics(dest_dir: Path) -> bool:
    """Download supplementary metabolomics data from the publisher.

    Tries multiple common supplementary file patterns used by
    Nature Scientific Reports.
    """
    suppl_dir = dest_dir / "metabolomics"
    suppl_dir.mkdir(parents=True, exist_ok=True)

    # Try common supplementary file naming patterns
    candidate_filenames = [
        "41598_2026_38558_MOESM2_ESM.xlsx",
        "41598_2026_38558_MOESM2_ESM.csv",
        "41598_2026_38558_MOESM1_ESM.xlsx",
        "41598_2026_38558_MOESM3_ESM.xlsx",
    ]

    downloaded_any = False
    for filename in candidate_filenames:
        url = f"{SUPPL_BASE}/{filename}"
        dest_file = suppl_dir / filename
        if dest_file.exists():
            logger.info("Supplementary file already exists: %s", dest_file.name)
            downloaded_any = True
            continue

        logger.info("Trying supplementary URL: %s", url)
        if _download_with_retry(url, dest_file):
            logger.info("Downloaded supplementary file: %s", dest_file.name)
            downloaded_any = True

    return downloaded_any


def _save_download_manifest(dest_dir: Path) -> None:
    """Save a manifest file documenting the data sources."""
    manifest = dest_dir / "MANIFEST.md"
    manifest.write_text(
        "# IIRN Metabolomics Dataset\n\n"
        "## Source\n"
        "Sheba Medical Center IIRN cohort (Sci Rep 2026)\n"
        "DOI: 10.1038/s41598-026-38558-9\n\n"
        "## Cohort\n"
        "- 80 Crohn's disease patients\n"
        "- 43 healthy controls\n"
        "- Paired fecal 16S, shotgun metagenomics, fecal+serum LC-MS metabolomics\n\n"
        "## Data Accessions\n"
        f"- 16S amplicon: {BIOPROJECT_16S} (NCBI SRA)\n"
        f"- Shotgun metagenomics: {BIOPROJECT_SHOTGUN} (NCBI SRA)\n"
        "- Metabolomics: Supplementary Dataset S1\n"
        f"- Analysis code: {GITHUB_REPO_URL}\n\n"
        "## Files\n"
        f"- sra_runinfo_16s.tsv: SRA run accessions for {BIOPROJECT_16S}\n"
        f"- sra_runinfo_shotgun.tsv: SRA run accessions for {BIOPROJECT_SHOTGUN}\n"
        "- metabolomics/: Supplementary metabolomics data files\n\n"
        "## Notes\n"
        "- Raw FASTQ files are NOT downloaded automatically (too large).\n"
        "  Use `fasterq-dump` with the run accessions in the TSV files.\n"
        "- 6,602 fecal microbe-metabolite correlations reported in study.\n"
    )
    logger.info("Saved download manifest: %s", manifest)


# --- Main --------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory: %s", OUTPUT_DIR.resolve())

    success = True

    # 1. Fetch 16S SRA run info
    logger.info("--- Step 1: Fetching 16S run info (%s) ---", BIOPROJECT_16S)
    runinfo_16s = OUTPUT_DIR / "sra_runinfo_16s.tsv"
    if not _fetch_sra_runinfo(BIOPROJECT_16S, runinfo_16s):
        logger.error("Failed to fetch 16S run info for %s", BIOPROJECT_16S)
        success = False

    # Be polite to NCBI — pause between requests
    time.sleep(1)

    # 2. Fetch shotgun metagenomics SRA run info
    logger.info("--- Step 2: Fetching shotgun run info (%s) ---", BIOPROJECT_SHOTGUN)
    runinfo_shotgun = OUTPUT_DIR / "sra_runinfo_shotgun.tsv"
    if not _fetch_sra_runinfo(BIOPROJECT_SHOTGUN, runinfo_shotgun):
        logger.error("Failed to fetch shotgun run info for %s", BIOPROJECT_SHOTGUN)
        success = False

    # 3. Download supplementary metabolomics data
    logger.info("--- Step 3: Downloading supplementary metabolomics ---")
    if not _download_supplementary_metabolomics(OUTPUT_DIR):
        logger.warning(
            "Could not download supplementary metabolomics. "
            "Try manually from: doi:10.1038/s41598-026-38558-9"
        )
        # Not fatal — supplementary URLs may vary

    # 4. Save manifest
    _save_download_manifest(OUTPUT_DIR)

    if success:
        logger.info("IIRN data download complete. Run accessions saved.")
        logger.info(
            "To download raw FASTQs, use:\n"
            "  fasterq-dump --split-files <RUN_ACCESSION>\n"
            "with accessions from the TSV files."
        )
    else:
        logger.error("Some downloads failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
