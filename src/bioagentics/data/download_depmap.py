"""Download DepMap 25Q3 datasets into data/depmap/25q3/.

Usage:
    uv run python -m bioagentics.data.download_depmap
    uv run python -m bioagentics.data.download_depmap --list
    uv run python -m bioagentics.data.download_depmap --files CRISPRGeneDependency.csv Model.csv
    uv run python -m bioagentics.data.download_depmap --include-prism
    uv run python -m bioagentics.data.download_depmap --force
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import requests
from tqdm import tqdm

API_BASE = "https://depmap.org/portal/download/api"
DEFAULT_RELEASE = "DepMap Public 25Q3"
DEFAULT_PRISM_RELEASE = "PRISM Primary Repurposing DepMap Public 24Q2"

# Key datasets for cancer research pipeline (from data_curator assessment)
DEFAULT_FILES = [
    "CRISPRGeneDependency.csv",
    "CRISPRGeneEffect.csv",
    "OmicsExpressionProteinCodingGenesTPMLogp1.csv",
    "OmicsSomaticMutations.csv",
    "OmicsCNGene.csv",
    "Model.csv",
]

PRISM_FILES = [
    "Repurposing_Public_24Q2_Extended_Primary_Data_Matrix.csv",
    "Repurposing_Public_24Q2_Treatment_Meta_Data.csv",
    "Repurposing_Public_24Q2_Cell_Line_Meta_Data.csv",
]

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = REPO_ROOT / "data" / "depmap" / "25q3"

TIMEOUT = 60
CHUNK_SIZE = 1024 * 64  # 64 KB


def list_release_files(release: str) -> list[dict]:
    """List available files for a DepMap release.

    The API returns {"table": [...], ...}. We filter the table entries
    to those matching the requested release name.
    """
    resp = requests.get(
        f"{API_BASE}/downloads",
        params={"release": release},
        timeout=TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    # API returns a dict with a "table" key containing file entries
    table = data.get("table", [])
    return [entry for entry in table if entry.get("releaseName") == release]


def resolve_download_url(entry: dict) -> str:
    """Resolve the download URL from a table entry.

    DepMap portal files use relative URLs; PRISM/Figshare files use absolute URLs.
    """
    url = entry.get("downloadUrl", "")
    if url.startswith("/"):
        return f"https://depmap.org{url}"
    return url


def build_url_map(release_files: list[dict]) -> dict[str, str]:
    """Build a mapping of fileName -> resolved download URL from release metadata."""
    url_map: dict[str, str] = {}
    for entry in release_files:
        name = entry.get("fileName", "")
        if name:
            url_map[name] = resolve_download_url(entry)
    return url_map


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_from_url(
    url: str,
    file_name: str,
    dest_dir: Path,
    *,
    force: bool = False,
) -> Path:
    """Download a file from a resolved URL."""
    dest = dest_dir / file_name
    if dest.exists() and not force:
        print(f"  Skipping {file_name} (already exists)")
        return dest

    with requests.get(url, stream=True, timeout=TIMEOUT) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(dest, "wb") as f,
            tqdm(total=total or None, unit="B", unit_scale=True, desc=file_name) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))
    return dest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Download DepMap 25Q3 datasets for cancer research",
    )
    parser.add_argument(
        "--release",
        default=DEFAULT_RELEASE,
        help=f"DepMap release name (default: {DEFAULT_RELEASE})",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available files for the release and exit",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        metavar="FILE",
        help="Specific files to download (default: core datasets)",
    )
    parser.add_argument(
        "--include-prism",
        action="store_true",
        help="Also download PRISM drug response data",
    )
    parser.add_argument(
        "--prism-release",
        default=DEFAULT_PRISM_RELEASE,
        help=f"PRISM release name (default: {DEFAULT_PRISM_RELEASE})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if files already exist",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    if args.list:
        print(f"Fetching file list for release: {args.release}")
        try:
            files = list_release_files(args.release)
        except requests.HTTPError as e:
            print(f"Error fetching file list: {e}", file=sys.stderr)
            sys.exit(1)
        for entry in files:
            name = entry.get("fileName", "unknown")
            size = entry.get("fileSize", entry.get("size", "?"))
            print(f"  {name}  ({size})")
        return

    args.dest.mkdir(parents=True, exist_ok=True)

    # --- Resolve download URLs from release metadata ---
    files_to_download = args.files or DEFAULT_FILES[:]

    print(f"Fetching release metadata for {args.release}...")
    try:
        release_table = list_release_files(args.release)
    except requests.HTTPError as e:
        print(f"Error fetching release metadata: {e}", file=sys.stderr)
        sys.exit(1)
    url_map = build_url_map(release_table)
    print(f"  Found {len(url_map)} files in release")

    print(f"\nDownloading {len(files_to_download)} files from {args.release}")
    print(f"Destination: {args.dest}\n")

    failed = []
    for file_name in files_to_download:
        url = url_map.get(file_name)
        if not url:
            print(f"  FAILED {file_name}: not found in {args.release}", file=sys.stderr)
            failed.append(file_name)
            continue
        try:
            download_from_url(url, file_name, args.dest, force=args.force)
        except requests.HTTPError as e:
            print(f"  FAILED {file_name}: {e}", file=sys.stderr)
            failed.append(file_name)
        except requests.ConnectionError as e:
            print(f"  CONNECTION ERROR {file_name}: {e}", file=sys.stderr)
            failed.append(file_name)

    # --- Download PRISM files from separate release ---
    if args.include_prism and not args.files:
        print(f"\nFetching release metadata for {args.prism_release}...")
        try:
            prism_table = list_release_files(args.prism_release)
        except requests.HTTPError as e:
            print(f"Error fetching PRISM release metadata: {e}", file=sys.stderr)
            failed.extend(PRISM_FILES)
            prism_table = []

        prism_url_map = build_url_map(prism_table)

        print(f"\nDownloading {len(PRISM_FILES)} PRISM files from {args.prism_release}")
        print(f"Destination: {args.dest}\n")

        for file_name in PRISM_FILES:
            url = prism_url_map.get(file_name)
            if not url:
                print(f"  FAILED {file_name}: not found in {args.prism_release}", file=sys.stderr)
                failed.append(file_name)
                continue
            try:
                download_from_url(url, file_name, args.dest, force=args.force)
            except requests.HTTPError as e:
                print(f"  FAILED {file_name}: {e}", file=sys.stderr)
                failed.append(file_name)
            except requests.ConnectionError as e:
                print(f"  CONNECTION ERROR {file_name}: {e}", file=sys.stderr)
                failed.append(file_name)

    print()
    if failed:
        print(f"{len(failed)} file(s) failed: {', '.join(failed)}")
        sys.exit(1)
    else:
        total = len(files_to_download)
        if args.include_prism and not args.files:
            total += len(PRISM_FILES)
        print(f"All {total} files downloaded successfully.")


if __name__ == "__main__":
    main()
