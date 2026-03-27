"""Download Wang DLPFC snRNA-seq data from GEO GSE311334.

Tourette disorder DLPFC snRNA-seq companion data from Wang group.
9 samples (5 TD, 4 controls). Files:
  - GSE311334_counts_matrix_24149.csv.gz (261MB)
  - GSE311334_metadata_24149.csv.gz (14MB)

Saves to: data/tourettes/cstc-circuit-expression-atlas/wang_dlpfc_snrnaseq/
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE311nnn/GSE311334/suppl"
DATA_DIR = Path("data/tourettes/cstc-circuit-expression-atlas/wang_dlpfc_snrnaseq")

FILES = [
    {
        "name": "counts_matrix",
        "url": f"{GEO_BASE}/GSE311334_counts_matrix_24149.csv.gz",
        "filename": "GSE311334_counts_matrix_24149.csv.gz",
        "size_mb": 261,
        "description": "Gene x cell counts matrix (24149 cells)",
    },
    {
        "name": "metadata",
        "url": f"{GEO_BASE}/GSE311334_metadata_24149.csv.gz",
        "filename": "GSE311334_metadata_24149.csv.gz",
        "size_mb": 14,
        "description": "Cell metadata with annotations (24149 cells)",
    },
]

CHUNK_SIZE = 8192


def _stream_download(url: str, dest: Path) -> str:
    """Stream-download a file and return its SHA-256 hash."""
    sha = hashlib.sha256()
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                sha.update(chunk)
                downloaded += len(chunk)
                if total and downloaded % (10 * 1024 * 1024) < CHUNK_SIZE:
                    pct = downloaded * 100 // total
                    logger.info("  %s: %d%% (%dMB / %dMB)", dest.name, pct,
                                downloaded // (1024 * 1024), total // (1024 * 1024))
    return sha.hexdigest()


def download_all(force: bool = False) -> dict[str, Path]:
    """Download all Wang DLPFC files. Returns dict of name -> local path."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    results = {}

    for entry in FILES:
        dest = DATA_DIR / entry["filename"]
        if dest.exists() and not force:
            logger.info("Already exists: %s (skipping)", dest)
            results[entry["name"]] = dest
            continue

        logger.info("Downloading %s (%dMB) ...", entry["name"], entry["size_mb"])
        sha = _stream_download(entry["url"], dest)
        logger.info("  Saved: %s (sha256: %s)", dest, sha[:16])

        # Write checksum sidecar
        checksum_file = dest.with_suffix(dest.suffix + ".sha256")
        checksum_file.write_text(f"{sha}  {entry['filename']}\n")

        results[entry["name"]] = dest

    return results


def verify_downloads() -> bool:
    """Verify all expected files exist."""
    ok = True
    for entry in FILES:
        dest = DATA_DIR / entry["filename"]
        if not dest.exists():
            logger.error("Missing: %s", dest)
            ok = False
        else:
            size_mb = dest.stat().st_size / (1024 * 1024)
            logger.info("OK: %s (%.1fMB)", dest.name, size_mb)
    return ok


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    paths = download_all()
    if verify_downloads():
        logger.info("All files downloaded successfully.")
    else:
        logger.error("Some files are missing!")
        raise SystemExit(1)
