"""Download GSE149438 CRC and Normal sample methylation files.

Downloads individual per-sample methylation files (bsmap_meth.txt.gz)
for CRC patients and Normal controls from GEO.

Usage:
    uv run python -m bioagentics.diagnostics.crc_liquid_biopsy.download_gse149438_samples [--limit N]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import requests

from bioagentics.config import REPO_ROOT

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data" / "diagnostics" / "crc-liquid-biopsy-panel" / "gse149438"
SAMPLES_DIR = DATA_DIR / "samples"


def download_samples(limit: int | None = None) -> int:
    """Download CRC + Normal sample methylation files from GSE149438.

    Returns number of files successfully downloaded.
    """
    urls_path = DATA_DIR / "download_urls.json"
    if not urls_path.exists():
        logger.error("download_urls.json not found. Run inspect-gse149438 first.")
        return 0

    with open(urls_path) as f:
        url_list = json.load(f)

    if limit:
        url_list = url_list[:limit]

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    skipped = 0
    failed = 0

    for i, (gsm, source, url) in enumerate(url_list):
        fname = url.split("/")[-1]
        dest = SAMPLES_DIR / fname

        if dest.exists():
            skipped += 1
            continue

        logger.info("[%d/%d] Downloading %s (%s, %s)...", i + 1, len(url_list), gsm, source, fname)
        try:
            resp = requests.get(url, timeout=300, stream=True)
            resp.raise_for_status()

            # Stream to disk to avoid memory issues
            with open(dest, "wb") as out:
                for chunk in resp.iter_content(chunk_size=65536):
                    out.write(chunk)

            size_mb = dest.stat().st_size / 1024 / 1024
            downloaded += 1
            logger.info("  Saved %s (%.1f MB)", fname, size_mb)

            # Brief pause between downloads to be polite to GEO servers
            time.sleep(0.5)

        except Exception as e:
            logger.warning("  Failed to download %s: %s", gsm, e)
            if dest.exists():
                dest.unlink()
            failed += 1

    logger.info(
        "Done: %d downloaded, %d skipped (cached), %d failed out of %d total",
        downloaded, skipped, failed, len(url_list),
    )
    return downloaded + skipped


def main():
    parser = argparse.ArgumentParser(description="Download GSE149438 CRC+Normal sample files")
    parser.add_argument("--limit", type=int, help="Download only first N files (for testing)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    download_samples(limit=args.limit)


if __name__ == "__main__":
    main()
