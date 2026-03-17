"""Download GEO datasets for CD fibrosis drug repurposing.

Downloads expression data from NCBI GEO for three key datasets:
- GSE16879: Mucosal biopsies from CD patients (stricturing vs inflammatory phenotypes)
- GSE57945: RISK cohort — ileal gene expression, progressors vs non-progressors
- GSE275144: CTHRC1+ fibroblast scRNA-seq reference

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.geo
    uv run python -m bioagentics.data.cd_fibrosis.geo --datasets GSE16879 GSE57945
    uv run python -m bioagentics.data.cd_fibrosis.geo --skip-supplementary
"""

from __future__ import annotations

import argparse
import gzip
import sys
from pathlib import Path

import requests
from tqdm import tqdm

from bioagentics.config import REPO_ROOT

DEFAULT_DEST = REPO_ROOT / "data" / "crohns" / "cd-fibrosis-drug-repurposing" / "geo"
TIMEOUT = 120
CHUNK_SIZE = 1024 * 64

# GEO datasets needed for CD fibrosis signatures
DATASETS: dict[str, dict] = {
    "GSE16879": {
        "description": "CD mucosal biopsies — stricturing (B2) vs inflammatory (B1) phenotypes",
        "platform": "GPL570",  # Affymetrix HG-U133 Plus 2.0
        "use": "Bulk tissue fibrosis signature derivation (B2 vs B1 DE)",
    },
    "GSE57945": {
        "description": "RISK cohort — ileal gene expression, progressors vs non-progressors",
        "platform": "GPL570",  # Affymetrix HG-U133 Plus 2.0
        "use": "Bulk tissue fibrosis signature (fibrosis progressors)",
    },
    "GSE275144": {
        "description": "CTHRC1+ fibroblast scRNA-seq from CD stricture tissue",
        "platform": "scRNA-seq",
        "use": "Cell-type-resolved fibroblast fibrosis signature",
    },
}

# NCBI GEO FTP/HTTP base URLs
GEO_BASE = "https://ftp.ncbi.nlm.nih.gov/geo/series"
SOFT_BASE = "https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi"


def _geo_ftp_prefix(gse: str) -> str:
    """Get the GEO FTP directory prefix for a GSE accession.

    GSE16879 -> GSE16nnn/GSE16879
    """
    numeric = gse.replace("GSE", "")
    prefix = numeric[:-3] if len(numeric) > 3 else ""
    return f"GSE{prefix}nnn/{gse}"


def download_series_matrix(gse: str, dest_dir: Path) -> Path:
    """Download the Series Matrix file for a GEO dataset.

    The Series Matrix contains the normalized expression matrix and
    sample metadata (phenotype, disease status, etc.).
    """
    prefix = _geo_ftp_prefix(gse)
    url = f"{GEO_BASE}/{prefix}/matrix/{gse}_series_matrix.txt.gz"

    dest_file = dest_dir / f"{gse}_series_matrix.txt.gz"
    if dest_file.exists():
        print(f"  {dest_file.name} already exists, skipping")
        return dest_file

    print(f"  Downloading {gse} series matrix...")
    _download_file(url, dest_file)
    return dest_file


def download_soft(gse: str, dest_dir: Path) -> Path:
    """Download the SOFT format file with full metadata."""
    prefix = _geo_ftp_prefix(gse)
    url = f"{GEO_BASE}/{prefix}/soft/{gse}_family.soft.gz"

    dest_file = dest_dir / f"{gse}_family.soft.gz"
    if dest_file.exists():
        print(f"  {dest_file.name} already exists, skipping")
        return dest_file

    print(f"  Downloading {gse} SOFT file...")
    _download_file(url, dest_file)
    return dest_file


def download_supplementary(gse: str, dest_dir: Path) -> list[Path]:
    """Download supplementary files (e.g. processed count matrices for scRNA-seq).

    For scRNA-seq datasets like GSE275144, the key data is in supplementary files
    (count matrices, cell annotations) rather than the series matrix.
    """
    prefix = _geo_ftp_prefix(gse)
    suppl_url = f"{GEO_BASE}/{prefix}/suppl/"

    print(f"  Checking supplementary files for {gse}...")
    try:
        resp = requests.get(suppl_url, timeout=TIMEOUT)
        resp.raise_for_status()
    except requests.HTTPError:
        print(f"  No supplementary directory found for {gse}")
        return []

    # Parse file listing from the HTML directory index
    files = _parse_ftp_listing(resp.text)
    if not files:
        print(f"  No supplementary files found")
        return []

    print(f"  Found {len(files)} supplementary files")
    downloaded = []
    suppl_dir = dest_dir / "supplementary"
    suppl_dir.mkdir(exist_ok=True)

    for fname in files:
        dest_file = suppl_dir / fname
        if dest_file.exists():
            print(f"    {fname} already exists, skipping")
            downloaded.append(dest_file)
            continue

        file_url = f"{suppl_url}{fname}"
        print(f"    Downloading {fname}...")
        try:
            _download_file(file_url, dest_file)
            downloaded.append(dest_file)
        except requests.HTTPError as e:
            print(f"    Failed: {e}", file=sys.stderr)

    return downloaded


def _parse_ftp_listing(html: str) -> list[str]:
    """Extract filenames from an NCBI FTP HTML directory listing."""
    files = []
    for line in html.splitlines():
        # NCBI FTP listings have <a href="filename"> tags
        if 'href="' in line and not line.strip().startswith(".."):
            start = line.find('href="') + 6
            end = line.find('"', start)
            if end > start:
                name = line[start:end]
                # Skip directory links and parent
                if not name.endswith("/") and name not in (".", ".."):
                    files.append(name)
    return files


def _download_file(url: str, dest: Path) -> None:
    """Download a file with progress bar."""
    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=dest.name) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))


def parse_series_matrix_samples(matrix_path: Path) -> dict[str, dict]:
    """Parse sample metadata from a series matrix file.

    Returns a dict of sample_id -> {title, source, characteristics, ...}.
    Useful for identifying which samples are stricturing vs inflammatory.
    """
    metadata: dict[str, dict] = {}
    sample_ids: list[str] = []

    opener = gzip.open if str(matrix_path).endswith(".gz") else open
    with opener(matrix_path, "rt") as f:
        for line in f:
            line = line.strip()
            if line.startswith("!Sample_geo_accession"):
                sample_ids = [s.strip('"') for s in line.split("\t")[1:]]
                for sid in sample_ids:
                    metadata[sid] = {}
            elif line.startswith("!Sample_title"):
                values = [s.strip('"') for s in line.split("\t")[1:]]
                for sid, val in zip(sample_ids, values):
                    metadata[sid]["title"] = val
            elif line.startswith("!Sample_source_name"):
                values = [s.strip('"') for s in line.split("\t")[1:]]
                for sid, val in zip(sample_ids, values):
                    metadata[sid]["source"] = val
            elif line.startswith("!Sample_characteristics"):
                values = [s.strip('"') for s in line.split("\t")[1:]]
                for sid, val in zip(sample_ids, values):
                    chars = metadata[sid].setdefault("characteristics", [])
                    chars.append(val)
            elif line.startswith("!series_matrix_table_begin"):
                break  # Stop at the expression data

    return metadata


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download GEO datasets for CD fibrosis drug repurposing"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=[*DATASETS, "all"],
        default=["all"],
        help="Datasets to download (default: all)",
    )
    parser.add_argument(
        "--skip-supplementary",
        action="store_true",
        help="Skip downloading supplementary files",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    selected = list(DATASETS) if "all" in args.datasets else args.datasets
    args.dest.mkdir(parents=True, exist_ok=True)

    print(f"CD Fibrosis Drug Repurposing — GEO Data Download")
    print(f"Destination: {args.dest}\n")

    failed = []
    for gse in selected:
        info = DATASETS[gse]
        print(f"--- {gse}: {info['description']} ---")
        print(f"    Platform: {info['platform']}")
        print(f"    Use: {info['use']}")

        gse_dir = args.dest / gse
        gse_dir.mkdir(exist_ok=True)

        try:
            # Always download series matrix (expression + metadata)
            download_series_matrix(gse, gse_dir)

            # Download SOFT for detailed metadata
            download_soft(gse, gse_dir)

            # Supplementary files (important for scRNA-seq)
            if not args.skip_supplementary:
                download_supplementary(gse, gse_dir)

        except (requests.RequestException, OSError) as e:
            print(f"  ERROR: {e}", file=sys.stderr)
            failed.append(gse)

        print()

    if failed:
        print(f"Failed datasets: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("All GEO downloads complete.")


if __name__ == "__main__":
    main()
