#!/usr/bin/env python3
"""Download pediatric CD multi-omics data from EMBL-EBI (Nature Comms Med 2025).

Study: 58 pediatric CD patients (27 remission, 31 active) with integrated
6-omics: fecal bacteria, fecal metabolites, fecal proteins, plasma metabolites,
fecal fungi, urine metabolites.

Data sources:
- Metabolomics: MTBLS9877 (EMBL-EBI MetaboLights)
- 16S/ITS amplicon: PRJEB74164 (ENA)
- Proteomics: PXD062519 (MassIVE / PRIDE)

This script fetches metadata and file lists from each repository.
Raw sequencing data is not downloaded directly (too large); instead,
run accessions and FTP paths are saved for selective download.

Usage:
    uv run python -m crohns.microbiome_metabolome_subtyping.02_download_pediatric_multiomics
"""

from __future__ import annotations

import json
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
    REPO_ROOT
    / "data"
    / "crohns"
    / "microbiome-metabolome-subtyping"
    / "pediatric_multiomics"
)

# MetaboLights study accession
MTBLS_ACCESSION = "MTBLS9877"
MTBLS_API_URL = f"https://www.ebi.ac.uk/metabolights/ws/studies/{MTBLS_ACCESSION}"
MTBLS_FTP_BASE = (
    f"ftp://ftp.ebi.ac.uk/pub/databases/metabolights/studies/public/{MTBLS_ACCESSION}"
)

# ENA BioProject for 16S/ITS amplicon sequencing
ENA_BIOPROJECT = "PRJEB74164"

# PRIDE/MassIVE proteomics dataset
PXD_ACCESSION = "PXD062519"
PRIDE_API_URL = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{PXD_ACCESSION}"
PRIDE_FILES_URL = (
    f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{PXD_ACCESSION}/files"
)

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
) -> bool:
    """Download *url* to *dest* with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S, stream=True)
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


def _fetch_json_with_retry(
    url: str,
    *,
    retries: int = MAX_RETRIES,
    backoff: float = INITIAL_BACKOFF_S,
    headers: dict[str, str] | None = None,
) -> dict | list | None:
    """Fetch JSON from *url* with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(
                url, timeout=TIMEOUT_S, headers=headers or {},
            )
            if resp.status_code == 404:
                logger.warning("404 Not Found: %s", url)
                return None
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d for %s failed (%s). Retrying in %.1fs ...",
                attempt, retries, url, exc, wait,
            )
            time.sleep(wait)
    return None


def fetch_metabolights_metadata(dest_dir: Path) -> bool:
    """Fetch MetaboLights study metadata and file list for MTBLS9877."""
    mtbls_dir = dest_dir / "metabolomics"
    mtbls_dir.mkdir(parents=True, exist_ok=True)

    # Fetch study descriptor
    descriptor_file = mtbls_dir / "study_descriptor.json"
    if descriptor_file.exists():
        logger.info("MetaboLights descriptor already exists: %s", descriptor_file)
    else:
        logger.info("Fetching MetaboLights study descriptor for %s ...", MTBLS_ACCESSION)
        data = _fetch_json_with_retry(
            MTBLS_API_URL,
            headers={"Accept": "application/json"},
        )
        if data is not None:
            descriptor_file.write_text(json.dumps(data, indent=2))
            logger.info("Saved study descriptor: %s", descriptor_file)
        else:
            logger.warning("Could not fetch MetaboLights descriptor")

    # Fetch file list from MetaboLights
    files_file = mtbls_dir / "file_list.json"
    if files_file.exists():
        logger.info("MetaboLights file list already exists: %s", files_file)
    else:
        logger.info("Fetching MetaboLights file list ...")
        file_data = _fetch_json_with_retry(
            f"{MTBLS_API_URL}/files?include_sub_dir=true",
            headers={"Accept": "application/json"},
        )
        if file_data is not None:
            files_file.write_text(json.dumps(file_data, indent=2))
            # Count files
            if isinstance(file_data, list):
                logger.info("Found %d files in MetaboLights study", len(file_data))
            elif isinstance(file_data, dict):
                study_files = file_data.get("study", [])
                logger.info("Found %d files in MetaboLights study", len(study_files))
            logger.info("Saved file list: %s", files_file)
        else:
            logger.warning("Could not fetch MetaboLights file list")

    # Try to download sample metadata (ISA-Tab format)
    for isa_file in ["s_study.txt", "i_Investigation.txt", "a_assay.txt"]:
        isa_dest = mtbls_dir / isa_file
        if isa_dest.exists():
            logger.info("ISA-Tab file already exists: %s", isa_file)
            continue
        # MetaboLights provides ISA-Tab files via the study FTP
        isa_url = (
            f"https://www.ebi.ac.uk/metabolights/ws/studies/"
            f"{MTBLS_ACCESSION}/download/{isa_file}"
        )
        logger.info("Trying to download ISA-Tab: %s", isa_file)
        _download_with_retry(isa_url, isa_dest)

    return True


def fetch_ena_run_info(dest_dir: Path) -> bool:
    """Fetch ENA run info for 16S/ITS amplicon data (PRJEB74164)."""
    ena_dir = dest_dir / "amplicon"
    ena_dir.mkdir(parents=True, exist_ok=True)

    runinfo_file = ena_dir / "ena_runinfo.tsv"
    if runinfo_file.exists():
        logger.info("ENA run info already exists: %s", runinfo_file)
        return True

    logger.info("Fetching ENA run info for %s ...", ENA_BIOPROJECT)

    run_info_url = (
        f"https://www.ebi.ac.uk/ena/portal/api/filereport"
        f"?accession={ENA_BIOPROJECT}"
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
        runinfo_file.write_text(resp.text)
        n_runs = len(resp.text.strip().split("\n")) - 1
        logger.info("Saved %d run records to %s", n_runs, runinfo_file)
        return True
    except (requests.RequestException, OSError) as exc:
        logger.error("Failed to fetch ENA run info: %s", exc)
        return False


def fetch_pride_metadata(dest_dir: Path) -> bool:
    """Fetch PRIDE project metadata for proteomics data (PXD062519)."""
    prot_dir = dest_dir / "proteomics"
    prot_dir.mkdir(parents=True, exist_ok=True)

    # Fetch project metadata
    meta_file = prot_dir / "pride_project.json"
    if meta_file.exists():
        logger.info("PRIDE project metadata already exists: %s", meta_file)
    else:
        logger.info("Fetching PRIDE project metadata for %s ...", PXD_ACCESSION)
        data = _fetch_json_with_retry(
            PRIDE_API_URL,
            headers={"Accept": "application/json"},
        )
        if data is not None:
            meta_file.write_text(json.dumps(data, indent=2))
            logger.info("Saved PRIDE project metadata: %s", meta_file)
        else:
            logger.warning("Could not fetch PRIDE metadata for %s", PXD_ACCESSION)

    # Fetch file list
    files_file = prot_dir / "pride_files.json"
    if files_file.exists():
        logger.info("PRIDE file list already exists: %s", files_file)
    else:
        logger.info("Fetching PRIDE file list ...")
        file_data = _fetch_json_with_retry(
            f"{PRIDE_FILES_URL}?pageSize=500",
            headers={"Accept": "application/json"},
        )
        if file_data is not None:
            files_file.write_text(json.dumps(file_data, indent=2))
            logger.info("Saved PRIDE file list: %s", files_file)
        else:
            logger.warning("Could not fetch PRIDE file list")

    return True


def _save_download_manifest(dest_dir: Path) -> None:
    """Save a manifest file documenting the data sources."""
    manifest = dest_dir / "MANIFEST.md"
    manifest.write_text(
        "# Pediatric CD Multi-Omics Dataset\n\n"
        "## Source\n"
        "Nature Communications Medicine 2025\n"
        "Integrated multi-omics of feces, plasma and urine can describe\n"
        "and differentiate pediatric active Crohn's Disease from remission.\n\n"
        "## Cohort\n"
        "- 58 pediatric Crohn's disease patients\n"
        "  - 27 remission\n"
        "  - 31 active disease\n"
        "- 6 omics modalities: fecal bacteria (40%), fecal metabolites (22%),\n"
        "  fecal proteins (16%), plasma metabolites (12%), fecal fungi (6%),\n"
        "  urine metabolites (4%)\n\n"
        "## Data Accessions\n"
        f"- Metabolomics: {MTBLS_ACCESSION} (EMBL-EBI MetaboLights)\n"
        f"- 16S/ITS amplicon: {ENA_BIOPROJECT} (ENA)\n"
        f"- Proteomics: {PXD_ACCESSION} (PRIDE/MassIVE)\n\n"
        "## Directory Structure\n"
        "- metabolomics/: MetaboLights study metadata and ISA-Tab files\n"
        "- amplicon/: ENA run accessions for 16S/ITS sequencing\n"
        "- proteomics/: PRIDE project metadata and file list\n\n"
        "## Notes\n"
        "- Raw sequencing data is NOT downloaded automatically (too large).\n"
        "  Use run accessions from amplicon/ena_runinfo.tsv for selective download.\n"
        "- Proteomics raw files can be downloaded from PRIDE using accessions\n"
        "  in proteomics/pride_files.json.\n"
        "- Feature importance in integrated model: fecal bacteria 40%,\n"
        "  fecal metabolites 22% — our metagenomics + metabolomics focus\n"
        "  captures top two modalities (62% combined).\n"
    )
    logger.info("Saved download manifest: %s", manifest)


# --- Main --------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory: %s", OUTPUT_DIR.resolve())

    success = True

    # 1. MetaboLights metabolomics metadata
    logger.info("--- Step 1: MetaboLights metabolomics (%s) ---", MTBLS_ACCESSION)
    if not fetch_metabolights_metadata(OUTPUT_DIR):
        logger.error("Failed to fetch MetaboLights metadata")
        success = False

    time.sleep(1)

    # 2. ENA 16S/ITS amplicon run info
    logger.info("--- Step 2: ENA amplicon run info (%s) ---", ENA_BIOPROJECT)
    if not fetch_ena_run_info(OUTPUT_DIR):
        logger.error("Failed to fetch ENA run info")
        success = False

    time.sleep(1)

    # 3. PRIDE proteomics metadata
    logger.info("--- Step 3: PRIDE proteomics metadata (%s) ---", PXD_ACCESSION)
    if not fetch_pride_metadata(OUTPUT_DIR):
        logger.error("Failed to fetch PRIDE metadata")
        success = False

    # 4. Save manifest
    _save_download_manifest(OUTPUT_DIR)

    if success:
        logger.info("Pediatric multi-omics metadata download complete.")
    else:
        logger.error("Some downloads failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
