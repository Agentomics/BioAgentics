"""Download TCGA data from GDC for a configurable cancer type.

Uses the GDC REST API (https://api.gdc.cancer.gov/) to query and download
open-access RNA-seq expression, somatic mutations, copy number, and clinical
data. Generates GDC-compatible manifests for reproducibility.

Usage:
    uv run python -m bioagentics.data.download_tcga
    uv run python -m bioagentics.data.download_tcga --cancer-type COAD
    uv run python -m bioagentics.data.download_tcga --manifest-only
    uv run python -m bioagentics.data.download_tcga --data-types expression mutations
    uv run python -m bioagentics.data.download_tcga --limit 10
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import requests
from tqdm import tqdm

GDC_API = "https://api.gdc.cancer.gov"
DEFAULT_CANCER_TYPE = "BRCA"

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DEST = REPO_ROOT / "data" / "tcga"

TIMEOUT = 60
CHUNK_SIZE = 1024 * 64

# GDC data type configurations for open-access TCGA data
DATA_TYPES: dict[str, dict] = {
    "expression": {
        "label": "RNA-seq Gene Expression (STAR Counts)",
        "data_category": "Transcriptome Profiling",
        "data_type": "Gene Expression Quantification",
        "workflow_type": "STAR - Counts",
    },
    "mutations": {
        "label": "Somatic Mutations (MAF)",
        "data_category": "Simple Nucleotide Variation",
        "data_type": "Masked Somatic Mutation",
        "workflow_type": None,
    },
    "copy_number": {
        "label": "Copy Number (Gene Level)",
        "data_category": "Copy Number Variation",
        "data_type": "Gene Level Copy Number",
        "workflow_type": None,
    },
}

CLINICAL_FIELDS = [
    "case_id",
    "submitter_id",
    "demographic.gender",
    "demographic.vital_status",
    "demographic.days_to_death",
    "demographic.year_of_birth",
    "demographic.race",
    "demographic.ethnicity",
    "diagnoses.primary_diagnosis",
    "diagnoses.tumor_stage",
    "diagnoses.days_to_last_follow_up",
    "diagnoses.age_at_diagnosis",
    "diagnoses.tissue_or_organ_of_origin",
]


def _build_filter(
    project_id: str,
    data_category: str,
    data_type: str,
    workflow_type: str | None = None,
) -> dict:
    """Build a GDC API filter object."""
    content = [
        {"op": "=", "content": {"field": "cases.project.project_id", "value": project_id}},
        {"op": "=", "content": {"field": "data_category", "value": data_category}},
        {"op": "=", "content": {"field": "data_type", "value": data_type}},
        {"op": "=", "content": {"field": "access", "value": "open"}},
    ]
    if workflow_type:
        content.append(
            {"op": "=", "content": {"field": "analysis.workflow_type", "value": workflow_type}}
        )
    return {"op": "and", "content": content}


def query_files(project_id: str, cfg: dict, page_size: int = 1000) -> list[dict]:
    """Query GDC for files matching the data type configuration.

    Handles pagination automatically.
    """
    filt = _build_filter(
        project_id, cfg["data_category"], cfg["data_type"], cfg.get("workflow_type"),
    )
    fields = "file_id,file_name,file_size,md5sum,data_category,data_type"
    all_hits: list[dict] = []
    offset = 0

    while True:
        params = {
            "filters": json.dumps(filt),
            "fields": fields,
            "size": page_size,
            "from": offset,
            "format": "json",
        }
        resp = requests.get(f"{GDC_API}/files", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["data"]
        all_hits.extend(data["hits"])

        total = data["pagination"]["total"]
        if len(all_hits) >= total:
            break
        offset += page_size

    return all_hits


def query_clinical_cases(project_id: str, page_size: int = 500) -> list[dict]:
    """Query structured clinical data from the GDC cases API."""
    filt = {"op": "=", "content": {"field": "project.project_id", "value": project_id}}
    all_cases: list[dict] = []
    offset = 0

    while True:
        params = {
            "filters": json.dumps(filt),
            "fields": ",".join(CLINICAL_FIELDS),
            "size": page_size,
            "from": offset,
            "format": "json",
        }
        resp = requests.get(f"{GDC_API}/cases", params=params, timeout=TIMEOUT)
        resp.raise_for_status()
        data = resp.json()["data"]
        all_cases.extend(data["hits"])

        if len(all_cases) >= data["pagination"]["total"]:
            break
        offset += page_size

    return all_cases


def save_manifest(files: list[dict], path: Path) -> None:
    """Save a GDC Data Transfer Tool-compatible manifest."""
    with open(path, "w") as f:
        f.write("id\tfilename\tmd5\tsize\tstate\n")
        for entry in files:
            f.write(
                f"{entry['file_id']}\t{entry['file_name']}\t"
                f"{entry['md5sum']}\t{entry['file_size']}\tlive\n"
            )


def md5_file(path: Path) -> str:
    """Compute MD5 hex digest of a file."""
    h = hashlib.md5(usedforsecurity=False)
    with open(path, "rb") as f:
        while True:
            chunk = f.read(CHUNK_SIZE)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def download_gdc_file(
    file_id: str,
    file_name: str,
    dest_dir: Path,
    *,
    expected_md5: str | None = None,
    force: bool = False,
) -> Path:
    """Download a single file from GDC by UUID."""
    dest = dest_dir / file_name

    if dest.exists() and not force:
        if expected_md5 and md5_file(dest) == expected_md5:
            return dest
        if not expected_md5:
            return dest

    url = f"{GDC_API}/data/{file_id}"
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

    if expected_md5:
        actual = md5_file(dest)
        if actual != expected_md5:
            dest.unlink()
            raise ValueError(
                f"MD5 mismatch for {file_name}: expected {expected_md5}, got {actual}"
            )

    return dest


def _format_size(size_bytes: int | float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download TCGA data from GDC")
    parser.add_argument(
        "--cancer-type",
        default=DEFAULT_CANCER_TYPE,
        help="TCGA cancer type code (default: BRCA)",
    )
    parser.add_argument(
        "--data-types",
        nargs="+",
        choices=[*DATA_TYPES, "clinical", "all"],
        default=["all"],
        help="Data types to download (default: all)",
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Generate manifests without downloading files",
    )
    parser.add_argument("--force", action="store_true", help="Re-download existing files")
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max files to download per data type (0 = all)",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Destination directory (default: {DEFAULT_DEST})",
    )
    args = parser.parse_args(argv)

    project_id = f"TCGA-{args.cancer_type}"
    dest_base = args.dest / args.cancer_type.lower()
    dest_base.mkdir(parents=True, exist_ok=True)

    selected = (
        [*DATA_TYPES, "clinical"] if "all" in args.data_types else args.data_types
    )

    print(f"Project: {project_id}")
    print(f"Destination: {dest_base}\n")

    failed_types: list[str] = []

    # Download genomic data types
    for dtype in selected:
        if dtype == "clinical":
            continue
        cfg = DATA_TYPES[dtype]
        print(f"--- {cfg['label']} ---")

        try:
            files = query_files(project_id, cfg)
        except requests.HTTPError as e:
            print(f"  Error querying {dtype}: {e}", file=sys.stderr)
            failed_types.append(dtype)
            continue

        if not files:
            print(f"  No open-access files found")
            continue

        total_size = sum(f["file_size"] for f in files)
        print(f"  Found {len(files)} files ({_format_size(total_size)})")

        subdir = dest_base / dtype
        subdir.mkdir(parents=True, exist_ok=True)

        manifest_path = subdir / "manifest.tsv"
        save_manifest(files, manifest_path)
        print(f"  Manifest: {manifest_path}")

        if args.manifest_only:
            print()
            continue

        to_download = files[: args.limit] if args.limit else files
        if args.limit:
            print(f"  Downloading {len(to_download)}/{len(files)} files (--limit {args.limit})")
        else:
            print(f"  Downloading {len(to_download)} files...")

        download_failed = 0
        for entry in to_download:
            try:
                download_gdc_file(
                    entry["file_id"],
                    entry["file_name"],
                    subdir,
                    expected_md5=entry.get("md5sum"),
                    force=args.force,
                )
            except (requests.RequestException, ValueError, OSError) as e:
                print(f"  FAILED {entry['file_name']}: {e}", file=sys.stderr)
                download_failed += 1

        if download_failed:
            print(f"  {download_failed}/{len(to_download)} files failed")
            failed_types.append(dtype)
        else:
            print(f"  Done ({len(to_download)} files)")
        print()

    # Clinical data via cases API (structured JSON, more usable than XML supplements)
    if "clinical" in selected:
        print("--- Clinical Data (Cases API) ---")
        try:
            cases = query_clinical_cases(project_id)
            clinical_dir = dest_base / "clinical"
            clinical_dir.mkdir(parents=True, exist_ok=True)
            clinical_path = clinical_dir / "cases_clinical.json"
            with open(clinical_path, "w") as f:
                json.dump(cases, f, indent=2)
            print(f"  Saved {len(cases)} cases to {clinical_path}")
        except (requests.RequestException, OSError) as e:
            print(f"  Error: {e}", file=sys.stderr)
            failed_types.append("clinical")
        print()

    if failed_types:
        print(f"Some data types had failures: {', '.join(failed_types)}")
        sys.exit(1)
    else:
        print("Done.")


if __name__ == "__main__":
    main()
