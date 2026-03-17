"""GDC API client for TCGA diagnostic WSI metadata retrieval.

Queries the GDC portal for diagnostic whole-slide image file metadata
for MSI-relevant cancer types (COAD, READ, UCEC, STAD). Outputs manifest
CSVs with file_id, case_id, cancer_type, and download URLs.
"""

import json
import logging
import time
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

# MSI-relevant TCGA cancer types
MSI_CANCER_TYPES = ["TCGA-COAD", "TCGA-READ", "TCGA-UCEC", "TCGA-STAD"]

# Rate limiting
REQUEST_DELAY = 0.5  # seconds between API calls


def _gdc_request(
    endpoint: str,
    filters: dict,
    fields: list[str],
    size: int = 1000,
    from_: int = 0,
) -> dict:
    """Make a paginated request to the GDC API."""
    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": size,
        "from": from_,
    }
    resp = requests.get(endpoint, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _paginate_gdc(
    endpoint: str,
    filters: dict,
    fields: list[str],
    page_size: int = 500,
) -> list[dict]:
    """Paginate through all GDC API results."""
    all_hits = []
    from_ = 0

    while True:
        data = _gdc_request(endpoint, filters, fields, size=page_size, from_=from_)
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            break
        all_hits.extend(hits)
        total = data["data"]["pagination"]["total"]
        logger.info(f"  fetched {len(all_hits)}/{total} records")
        if len(all_hits) >= total:
            break
        from_ += page_size
        time.sleep(REQUEST_DELAY)

    return all_hits


def query_diagnostic_wsi_metadata(
    project_ids: list[str] | None = None,
) -> pd.DataFrame:
    """Query GDC for diagnostic WSI file metadata.

    Args:
        project_ids: TCGA project IDs to query. Defaults to MSI-relevant types.

    Returns:
        DataFrame with columns: file_id, file_name, case_id, case_submitter_id,
        project_id, cancer_type, data_format, file_size, md5sum, download_url.
    """
    if project_ids is None:
        project_ids = MSI_CANCER_TYPES

    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": project_ids,
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Slide Image",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "experimental_strategy",
                    "value": "Diagnostic Slide",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": "SVS",
                },
            },
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "md5sum",
        "data_format",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
    ]

    logger.info(f"Querying GDC for diagnostic WSIs: {project_ids}")
    hits = _paginate_gdc(GDC_FILES_ENDPOINT, filters, fields)

    records = []
    for hit in hits:
        cases = hit.get("cases", [])
        if not cases:
            continue
        case = cases[0]
        project_id = case.get("project", {}).get("project_id", "")
        # Extract short cancer type (e.g., COAD from TCGA-COAD)
        cancer_type = project_id.replace("TCGA-", "") if project_id else ""

        records.append(
            {
                "file_id": hit["file_id"],
                "file_name": hit.get("file_name", ""),
                "case_id": case["case_id"],
                "case_submitter_id": case.get("submitter_id", ""),
                "project_id": project_id,
                "cancer_type": cancer_type,
                "data_format": hit.get("data_format", ""),
                "file_size": hit.get("file_size", 0),
                "md5sum": hit.get("md5sum", ""),
                "download_url": f"{GDC_DATA_ENDPOINT}/{hit['file_id']}",
            }
        )

    df = pd.DataFrame(records)
    if not df.empty:
        df = df.sort_values(["cancer_type", "case_submitter_id"]).reset_index(drop=True)
    logger.info(
        f"Found {len(df)} diagnostic WSIs across {df['cancer_type'].nunique() if not df.empty else 0} cancer types"
    )
    return df


def save_manifest(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "tcga_diagnostic_wsi_manifest.csv",
) -> Path:
    """Save WSI manifest CSV to the specified directory.

    Args:
        df: DataFrame from query_diagnostic_wsi_metadata.
        output_dir: Directory to save the manifest.
        filename: Output filename.

    Returns:
        Path to the saved manifest file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename
    df.to_csv(output_path, index=False)
    logger.info(f"Saved manifest: {output_path} ({len(df)} files)")
    return output_path


def save_per_cancer_manifests(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save separate manifest CSVs per cancer type.

    Returns:
        List of paths to saved manifest files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cancer_type, group in df.groupby("cancer_type"):
        filename = f"tcga_{cancer_type.lower()}_diagnostic_wsi_manifest.csv"
        path = output_dir / filename
        group.to_csv(path, index=False)
        logger.info(f"  {cancer_type}: {len(group)} WSIs -> {path}")
        paths.append(path)
    return paths


def generate_gdc_download_manifest(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "gdc_download_manifest.txt",
) -> Path:
    """Generate a GDC-compatible download manifest (TSV format).

    This file can be used with the GDC Data Transfer Tool for bulk downloads.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    manifest_df = df[["file_id", "file_name", "md5sum", "file_size"]].copy()
    manifest_df.columns = ["id", "filename", "md5", "size"]
    manifest_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved GDC download manifest: {output_path} ({len(manifest_df)} files)")
    return output_path


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the WSI manifest."""
    if df.empty:
        print("No WSIs found.")
        return

    print(f"\nTotal diagnostic WSIs: {len(df)}")
    print(f"Unique cases: {df['case_id'].nunique()}")
    print(f"Total size: {df['file_size'].sum() / 1e9:.1f} GB")
    print("\nPer cancer type:")
    for cancer_type, group in df.groupby("cancer_type"):
        print(
            f"  {cancer_type}: {len(group)} WSIs, "
            f"{group['case_id'].nunique()} cases, "
            f"{group['file_size'].sum() / 1e9:.1f} GB"
        )


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path("data/diagnostics/pathology-fm-msi-prescreening/manifests")

    if "--dry-run" in sys.argv:
        print("Dry run: would query GDC for diagnostic WSI metadata")
        print(f"  Cancer types: {MSI_CANCER_TYPES}")
        print(f"  Output directory: {output_dir}")
        sys.exit(0)

    df = query_diagnostic_wsi_metadata()
    print_summary(df)

    if not df.empty:
        save_manifest(df, output_dir)
        save_per_cancer_manifests(df, output_dir)
        generate_gdc_download_manifest(df, output_dir)
        print(f"\nManifests saved to {output_dir}")
