#!/usr/bin/env python3
"""Download E-MTAB-7604 (PANTS study) processed count files from EBI BioStudies Fire.

Downloads 44 per-sample count files (gene symbol + count, ~680KB each) and the
SDRF metadata file into data/crohns/anti-tnf-response-prediction/E-MTAB-7604/.

Usage:
    uv run python -m crohns.anti_tnf_response_prediction.01_download_emtab7604
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

# --- Configuration -----------------------------------------------------------

BASE_URL = (
    "https://ftp.ebi.ac.uk/biostudies/fire/E-MTAB-/604/E-MTAB-7604/Files"
)

SAMPLE_IDS = [
    "GC049971", "GC049992", "GC049997", "GC049998",
    "GC050011", "GC050016", "GC050024", "GC050040",
    "GC050130", "GC050131", "GC050149", "GC050152",
    "GC050158", "GC050162", "GC050173", "GC050174",
    "GC050193", "GC050242", "GC050260", "GC050265",
    "GC050272", "GC050281", "GC050282", "GC050312",
    "GC050318", "GC050320", "GC050321", "GC050322",
    "GC058827", "GC058833", "GC058835", "GC058840",
    "GC058849", "GC058851", "GC058859", "GC058865",
    "GC058866", "GC058872", "GC058874", "GC058889",
    "GC058897", "GC058904", "GC058910", "GC058911",
]

# Filename pattern for count files — EBI Fire typically stores them as
# {sample_id}.genes.results.tsv or {sample_id}.count. We try multiple
# suffixes and use whichever succeeds first.
COUNT_SUFFIXES = [
    ".genes.results.tsv",
    ".count",
    ".counts.tsv",
    ".tsv",
    ".txt",
]

SDRF_FILENAME = "E-MTAB-7604.sdrf.txt"

OUTPUT_DIR = Path("data/crohns/anti-tnf-response-prediction/E-MTAB-7604")

MAX_RETRIES = 4
INITIAL_BACKOFF_S = 2.0
TIMEOUT_S = 60


# --- Helpers -----------------------------------------------------------------

def _download_with_retry(
    url: str,
    dest: Path,
    *,
    retries: int = MAX_RETRIES,
    backoff: float = INITIAL_BACKOFF_S,
) -> bool:
    """Download *url* to *dest* with exponential backoff.

    Returns True on success, False if all retries are exhausted.
    """
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT_S, stream=True)
            if resp.status_code == 404:
                return False  # not found — try next suffix
            resp.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with open(dest, "wb") as fh:
                for chunk in resp.iter_content(chunk_size=64 * 1024):
                    fh.write(chunk)
            return True
        except (requests.RequestException, OSError) as exc:
            wait = backoff * (2 ** (attempt - 1))
            logger.warning(
                "Attempt %d/%d for %s failed (%s). Retrying in %.1fs …",
                attempt, retries, url, exc, wait,
            )
            time.sleep(wait)
    return False


def _discover_suffix(sample_id: str) -> str | None:
    """Probe the first sample to find which file suffix EBI uses."""
    for suffix in COUNT_SUFFIXES:
        url = f"{BASE_URL}/{sample_id}{suffix}"
        try:
            resp = requests.head(url, timeout=TIMEOUT_S, allow_redirects=True)
            if resp.status_code == 200:
                logger.info("Discovered file suffix: %s", suffix)
                return suffix
        except requests.RequestException:
            continue
    return None


# --- Main --------------------------------------------------------------------

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Target directory: %s", OUTPUT_DIR.resolve())

    # 1. Download SDRF metadata
    sdrf_url = f"{BASE_URL}/{SDRF_FILENAME}"
    sdrf_dest = OUTPUT_DIR / SDRF_FILENAME
    if sdrf_dest.exists():
        logger.info("SDRF already exists: %s", sdrf_dest)
    else:
        logger.info("Downloading SDRF: %s", sdrf_url)
        if not _download_with_retry(sdrf_url, sdrf_dest):
            logger.error(
                "Failed to download SDRF after %d retries. "
                "EBI Fire may be unreachable — try again later or download "
                "from ENA accession ERP113396.",
                MAX_RETRIES,
            )
            sys.exit(1)
        logger.info("SDRF saved to %s", sdrf_dest)

    # 2. Discover count file suffix
    logger.info("Probing file suffix for count files …")
    suffix = _discover_suffix(SAMPLE_IDS[0])
    if suffix is None:
        logger.error(
            "Could not determine count file suffix. EBI Fire may be down. "
            "Alternative: download raw FASTQs from ENA (ERP113396)."
        )
        sys.exit(1)

    # 3. Download count files
    downloaded = 0
    skipped = 0
    failed: list[str] = []

    for sid in SAMPLE_IDS:
        dest = OUTPUT_DIR / f"{sid}{suffix}"
        if dest.exists():
            logger.info("Already exists, skipping: %s", dest.name)
            skipped += 1
            continue

        url = f"{BASE_URL}/{sid}{suffix}"
        logger.info("Downloading %s …", dest.name)
        if _download_with_retry(url, dest):
            downloaded += 1
        else:
            logger.error("Failed: %s", sid)
            failed.append(sid)

    # 4. Summary
    total = len(SAMPLE_IDS)
    logger.info(
        "Done. Downloaded: %d, Skipped (existing): %d, Failed: %d / %d",
        downloaded, skipped, len(failed), total,
    )
    if failed:
        logger.error("Failed samples: %s", ", ".join(failed))
        sys.exit(1)


if __name__ == "__main__":
    main()
