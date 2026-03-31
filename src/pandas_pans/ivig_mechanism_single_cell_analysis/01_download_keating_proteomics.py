#!/usr/bin/env python3
"""Download Keating et al. TMTpro proteomics/phosphoproteomics from PRIDE PXD064184.

Study: NTI164 (medicinal cannabis) treatment of paediatric PANS.
14 children with chronic-relapsing PANS, 12-week open-label trial.
TMTpro 18-plex: 6 PANS pre-treatment, 6 PANS post-treatment, 6 matched controls.
PBMCs. Q Exactive Plus LC-MS/MS. MaxQuant v1.6.7.0 processed.
Associated publication: PMID 41513541, Neurotherapeutics 2026.

Downloads processed MaxQuant output files (not raw MS data).
Raw files total >30GB and are not needed for multi-omics integration.

Usage:
    uv run python -m pandas_pans.ivig_mechanism_single_cell_analysis.01_download_keating_proteomics
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

OUTPUT_DIR = REPO_ROOT / "data" / "pandas_pans" / "keating_proteomics"

PXD_ACCESSION = "PXD064184"
PRIDE_API_URL = f"https://www.ebi.ac.uk/pride/ws/archive/v2/projects/{PXD_ACCESSION}"
PRIDE_FTP_HTTPS = (
    f"https://ftp.pride.ebi.ac.uk/pride/data/archive/2026/03/{PXD_ACCESSION}"
)

# MaxQuant processed files to download (filename -> description)
TARGET_FILES = {
    "proteinGroups.txt": "Protein-level quantification (TMTpro 18-plex)",
    "Phospho_STY_Sites.txt": "Phosphorylation site quantification",
    "peptides.txt": "Peptide-level quantification",
    "parameters.txt": "MaxQuant search parameters",
    "summary.txt": "MaxQuant run summary",
    "Oxidation_M_Sites.txt": "Oxidation modification sites",
    "Deamidation_NQ_Sites.txt": "Deamidation modification sites",
}

MAX_RETRIES = 4
INITIAL_BACKOFF_S = 2.0
TIMEOUT_S = 300  # 5 min for larger files


# --- Helpers -----------------------------------------------------------------

def _download_with_retry(
    url: str,
    dest: Path,
    *,
    retries: int = MAX_RETRIES,
    backoff: float = INITIAL_BACKOFF_S,
) -> bool:
    """Download *url* to *dest* with exponential backoff, streaming."""
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
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("Downloaded %s (%.1f MB)", dest.name, size_mb)
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
) -> dict | list | None:
    """Fetch JSON from *url* with exponential backoff."""
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S)
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


# --- Steps -------------------------------------------------------------------

def fetch_project_metadata(dest_dir: Path) -> bool:
    """Fetch PRIDE project metadata."""
    meta_file = dest_dir / "pride_project.json"
    if meta_file.exists():
        logger.info("Project metadata already exists: %s", meta_file)
        return True

    logger.info("Fetching PRIDE project metadata for %s ...", PXD_ACCESSION)
    data = _fetch_json_with_retry(PRIDE_API_URL)
    if data is not None:
        meta_file.write_text(json.dumps(data, indent=2))
        logger.info("Saved project metadata: %s", meta_file)
        return True
    logger.error("Could not fetch PRIDE metadata for %s", PXD_ACCESSION)
    return False


def download_maxquant_outputs(dest_dir: Path) -> bool:
    """Download processed MaxQuant output files."""
    mqout_dir = dest_dir / "maxquant_output"
    mqout_dir.mkdir(parents=True, exist_ok=True)

    all_ok = True
    for filename, description in TARGET_FILES.items():
        dest = mqout_dir / filename
        if dest.exists():
            logger.info("Already downloaded: %s", filename)
            continue
        url = f"{PRIDE_FTP_HTTPS}/{filename}"
        logger.info("Downloading %s (%s) ...", filename, description)
        if not _download_with_retry(url, dest):
            logger.error("Failed to download %s", filename)
            all_ok = False
        time.sleep(1)  # polite delay between downloads
    return all_ok


def save_manifest(dest_dir: Path) -> None:
    """Save a manifest documenting the data source."""
    manifest = dest_dir / "MANIFEST.md"
    manifest.write_text(
        "# Keating et al. TMTpro Proteomics/Phosphoproteomics\n\n"
        "## Source\n"
        "- **PRIDE:** PXD064184\n"
        "- **Publication:** PMID 41513541, Neurotherapeutics 2026\n"
        "- **PI:** Dr. Russell Dale, Kids Neuroscience Centre, Sydney\n\n"
        "## Study Design\n"
        "- 14 paediatric PANS patients, 12-week open-label NTI164 trial\n"
        "- TMTpro 18-plex: 6 pre-treatment, 6 post-treatment, 6 matched controls\n"
        "- PBMCs, Q Exactive Plus LC-MS/MS\n"
        "- MaxQuant v1.6.7.0 processed\n\n"
        "## Files\n"
        "- `pride_project.json` — PRIDE project metadata\n"
        "- `maxquant_output/proteinGroups.txt` — Protein quantification\n"
        "- `maxquant_output/Phospho_STY_Sites.txt` — Phosphosite quantification\n"
        "- `maxquant_output/peptides.txt` — Peptide quantification\n"
        "- `maxquant_output/parameters.txt` — MaxQuant parameters\n"
        "- `maxquant_output/summary.txt` — Run summary\n"
        "- `maxquant_output/Oxidation_M_Sites.txt` — Oxidation sites\n"
        "- `maxquant_output/Deamidation_NQ_Sites.txt` — Deamidation sites\n\n"
        "## Complementary Datasets\n"
        "- GEO GSE278678/GSE278679: Transcriptomics (same study)\n"
        "- GEO GSE301611/GSE299764: Additional omics\n\n"
        "## Notes\n"
        "- Raw MS files (>30GB) are NOT downloaded. Available from PRIDE if needed.\n"
        "- Peak list files (.apl) also not downloaded.\n"
        "- proteinGroups.txt and Phospho_STY_Sites.txt are the primary files\n"
        "  for multi-omics integration with the IVIG scRNA-seq pipeline.\n"
    )
    logger.info("Saved manifest: %s", manifest)


# --- Main --------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory: %s", OUTPUT_DIR.resolve())

    success = True

    # 1. Fetch project metadata
    logger.info("--- Step 1: PRIDE project metadata ---")
    if not fetch_project_metadata(OUTPUT_DIR):
        success = False

    time.sleep(1)

    # 2. Download MaxQuant processed outputs
    logger.info("--- Step 2: MaxQuant processed outputs ---")
    if not download_maxquant_outputs(OUTPUT_DIR):
        success = False

    # 3. Save manifest
    save_manifest(OUTPUT_DIR)

    if success:
        logger.info("Keating proteomics download complete.")
    else:
        logger.error("Some downloads failed. Check logs above.")
        sys.exit(1)


if __name__ == "__main__":
    main()
