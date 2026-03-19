"""CPTAC external validation data preparation pipeline.

Queries the GDC portal for CPTAC WSI metadata and MSIsensor2 MSI labels
for the external validation cohort (COAD, UCEC, LSCC, LUAD). This cohort
is completely held out from training and used for Phase 5 validation.

CPTAC projects on GDC:
- CPTAC-2: COAD, UCEC (via TCGA)
- CPTAC-3: LSCC, LUAD, and others

The same tiling and feature extraction pipeline as TCGA is used; this
module handles only the CPTAC-specific metadata retrieval and MSI label
curation (MSIsensor2 scores instead of MANTIS/PCR).
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

# CPTAC projects on GDC for the validation cohort
CPTAC_PROJECTS = ["CPTAC-2", "CPTAC-3"]

# CPTAC cancer types of interest for MSI validation
CPTAC_CANCER_TYPES = ["COAD", "UCEC", "LSCC", "LUAD"]

# MSIsensor2 threshold for MSI-H classification
# MSIsensor2 scores >= 20% are considered MSI-H (Li et al., 2020)
MSISENSOR2_MSI_H_THRESHOLD = 20.0
MSISENSOR2_MSI_L_THRESHOLD = 10.0  # Between 10-20 is MSI-L/indeterminate

# Rate limiting
REQUEST_DELAY = 0.5


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


def _extract_cancer_type(project_id: str, primary_site: str) -> str:
    """Map CPTAC project + primary site to cancer type abbreviation.

    CPTAC projects contain multiple cancer types, so we use primary_site
    to determine the specific cancer type.
    """
    site_lower = primary_site.lower() if primary_site else ""

    if "colon" in site_lower or "rectum" in site_lower or "colorectal" in site_lower:
        return "COAD"
    if "uterus" in site_lower or "endometri" in site_lower or "corpus uteri" in site_lower:
        return "UCEC"
    if "lung" in site_lower and "squamous" in site_lower:
        return "LSCC"
    if "lung" in site_lower:
        # Default lung to LUAD unless squamous is specified
        return "LUAD"
    if "bronchus" in site_lower:
        return "LUAD"

    # Fallback: return cleaned primary site
    return primary_site.upper().replace(" ", "_") if primary_site else "UNKNOWN"


def query_cptac_wsi_metadata(
    project_ids: list[str] | None = None,
    cancer_types: list[str] | None = None,
) -> pd.DataFrame:
    """Query GDC for CPTAC WSI file metadata.

    Args:
        project_ids: CPTAC project IDs to query. Defaults to CPTAC-2, CPTAC-3.
        cancer_types: Filter to specific cancer types. Defaults to all CPTAC types.

    Returns:
        DataFrame with columns: file_id, file_name, case_id, case_submitter_id,
        project_id, cancer_type, primary_site, data_format, file_size, md5sum,
        download_url.
    """
    if project_ids is None:
        project_ids = CPTAC_PROJECTS
    if cancer_types is None:
        cancer_types = CPTAC_CANCER_TYPES

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
                "op": "in",
                "content": {
                    "field": "data_format",
                    "value": ["SVS", "TIFF"],
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
        "experimental_strategy",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
        "cases.diagnoses.primary_diagnosis",
        "cases.diagnoses.tissue_or_organ_of_origin",
    ]

    logger.info(f"Querying GDC for CPTAC WSIs: {project_ids}")
    hits = _paginate_gdc(GDC_FILES_ENDPOINT, filters, fields)

    records = []
    for hit in hits:
        cases = hit.get("cases", [])
        if not cases:
            continue
        case = cases[0]
        project_id = case.get("project", {}).get("project_id", "")

        # Extract primary site from diagnoses
        diagnoses = case.get("diagnoses", [])
        primary_site = ""
        if diagnoses:
            primary_site = diagnoses[0].get("tissue_or_organ_of_origin", "")

        cancer_type = _extract_cancer_type(project_id, primary_site)

        # Filter to requested cancer types
        if cancer_type not in cancer_types:
            continue

        records.append(
            {
                "file_id": hit["file_id"],
                "file_name": hit.get("file_name", ""),
                "case_id": case["case_id"],
                "case_submitter_id": case.get("submitter_id", ""),
                "project_id": project_id,
                "cancer_type": cancer_type,
                "primary_site": primary_site,
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
        f"Found {len(df)} CPTAC WSIs across "
        f"{df['cancer_type'].nunique() if not df.empty else 0} cancer types"
    )
    return df


def classify_msi_from_msisensor2(score: float | None) -> str:
    """Classify MSI status from MSIsensor2 score.

    Thresholds from Li et al., 2020:
    - MSI-H: score >= 20%
    - MSI-L: 10% <= score < 20% (indeterminate)
    - MSS: score < 10%
    """
    if score is None or pd.isna(score):
        return "unknown"
    if score >= MSISENSOR2_MSI_H_THRESHOLD:
        return "MSI-H"
    if score >= MSISENSOR2_MSI_L_THRESHOLD:
        return "MSI-L"
    return "MSS"


def load_cptac_msi_annotations(
    annotations_file: str | Path,
) -> pd.DataFrame:
    """Load CPTAC MSI annotations from a file containing MSIsensor2 scores.

    Expected format: CSV/TSV with columns for case/sample ID and MSIsensor2 score.
    The function auto-detects common column naming conventions.

    Args:
        annotations_file: Path to annotations file.

    Returns:
        DataFrame with columns: case_id, msisensor2_score, msi_status.
    """
    path = Path(annotations_file)
    if path.suffix == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, sep="\t")

    # Normalize column names
    col_map = {}
    for col in df.columns:
        lower = col.lower()
        if "case" in lower and "id" in lower:
            col_map[col] = "case_id"
        elif "submitter" in lower or "barcode" in lower or "sample" in lower or "patient" in lower:
            if "case_id" not in col_map.values():
                col_map[col] = "case_id"
        elif "msisensor" in lower and "score" in lower:
            col_map[col] = "msisensor2_score"
        elif lower in ("msisensor2", "msisensor_score", "msi_score"):
            col_map[col] = "msisensor2_score"

    df = df.rename(columns=col_map)

    if "case_id" not in df.columns:
        raise ValueError(
            f"Missing case identifier column. Available: {list(df.columns)}"
        )

    if "msisensor2_score" in df.columns:
        df["msisensor2_score"] = pd.to_numeric(df["msisensor2_score"], errors="coerce")
        df["msi_status"] = df["msisensor2_score"].apply(classify_msi_from_msisensor2)
    else:
        logger.warning("No MSIsensor2 score column found; MSI status will be unknown")
        df["msisensor2_score"] = None
        df["msi_status"] = "unknown"

    return df


def curate_cptac_labels(
    wsi_metadata: pd.DataFrame,
    annotations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Curate MSI labels for CPTAC cohort by merging WSI metadata with MSI annotations.

    Args:
        wsi_metadata: From query_cptac_wsi_metadata().
        annotations_df: From load_cptac_msi_annotations(). If None, all labels
            will be 'unknown'.

    Returns:
        DataFrame with columns: case_id, case_submitter_id, project_id,
        cancer_type, msi_status, msisensor2_score, file_id.
    """
    df = wsi_metadata.copy()

    if annotations_df is not None and not annotations_df.empty:
        ann = annotations_df.copy()

        # Try merging on case_submitter_id first (CPTAC IDs may match submitter IDs)
        if "case_id" in ann.columns:
            # Try direct merge on case_submitter_id
            merged = df.merge(
                ann[["case_id", "msisensor2_score", "msi_status"]],
                left_on="case_submitter_id",
                right_on="case_id",
                how="left",
                suffixes=("", "_ann"),
            )
            if "case_id_ann" in merged.columns:
                merged = merged.drop(columns=["case_id_ann"])

            # For unmatched rows, try matching on case_id
            unmatched = merged["msi_status"].isna()
            if unmatched.any():
                fallback = df[unmatched].merge(
                    ann[["case_id", "msisensor2_score", "msi_status"]],
                    on="case_id",
                    how="left",
                    suffixes=("", "_fb"),
                )
                for col in ["msisensor2_score", "msi_status"]:
                    fb_col = f"{col}_fb" if f"{col}_fb" in fallback.columns else col
                    if fb_col in fallback.columns:
                        merged.loc[unmatched, col] = fallback[fb_col].values

            df = merged
        else:
            df["msisensor2_score"] = None
            df["msi_status"] = "unknown"
    else:
        df["msisensor2_score"] = None
        df["msi_status"] = "unknown"

    # Fill remaining NaN statuses
    df["msi_status"] = df["msi_status"].fillna("unknown")

    return df


def save_cptac_manifest(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "cptac_wsi_manifest.csv",
) -> Path:
    """Save CPTAC WSI manifest CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)
    logger.info(f"Saved CPTAC manifest: {path} ({len(df)} files)")
    return path


def save_cptac_labels(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "cptac_msi_labels.csv",
) -> Path:
    """Save curated CPTAC MSI labels to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)

    if not df.empty and "msi_status" in df.columns:
        counts = df["msi_status"].value_counts()
        logger.info(f"Saved CPTAC labels: {path}")
        logger.info(f"  MSI status distribution: {counts.to_dict()}")
        if "cancer_type" in df.columns:
            for ct in sorted(df["cancer_type"].unique()):
                ct_counts = df[df["cancer_type"] == ct]["msi_status"].value_counts().to_dict()
                logger.info(f"  {ct}: {ct_counts}")

    return path


def save_per_cancer_manifests(
    df: pd.DataFrame,
    output_dir: str | Path,
) -> list[Path]:
    """Save separate manifest CSVs per cancer type."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for cancer_type, group in df.groupby("cancer_type"):
        filename = f"cptac_{cancer_type.lower()}_wsi_manifest.csv"
        path = output_dir / filename
        group.to_csv(path, index=False)
        logger.info(f"  {cancer_type}: {len(group)} WSIs -> {path}")
        paths.append(path)
    return paths


def generate_gdc_download_manifest(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "cptac_gdc_download_manifest.txt",
) -> Path:
    """Generate a GDC-compatible download manifest for CPTAC WSIs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / filename

    manifest_df = df[["file_id", "file_name", "md5sum", "file_size"]].copy()
    manifest_df.columns = ["id", "filename", "md5", "size"]
    manifest_df.to_csv(output_path, sep="\t", index=False)
    logger.info(f"Saved CPTAC GDC download manifest: {output_path} ({len(manifest_df)} files)")
    return output_path


def print_summary(df: pd.DataFrame) -> None:
    """Print a summary of the CPTAC WSI metadata."""
    if df.empty:
        print("No CPTAC WSIs found.")
        return

    print(f"\nTotal CPTAC WSIs: {len(df)}")
    print(f"Unique cases: {df['case_id'].nunique()}")
    print(f"Total size: {df['file_size'].sum() / 1e9:.1f} GB")
    print("\nPer cancer type:")
    for cancer_type, group in df.groupby("cancer_type"):
        print(
            f"  {cancer_type}: {len(group)} WSIs, "
            f"{group['case_id'].nunique()} cases, "
            f"{group['file_size'].sum() / 1e9:.1f} GB"
        )

    if "msi_status" in df.columns:
        print(f"\nMSI status distribution:")
        for status, count in df["msi_status"].value_counts().items():
            pct = 100 * count / len(df)
            print(f"  {status}: {count} ({pct:.1f}%)")


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    output_dir = Path("data/diagnostics/pathology-fm-msi-prescreening/cptac")

    if "--dry-run" in sys.argv:
        print("Dry run: would query GDC for CPTAC WSI metadata")
        print(f"  Projects: {CPTAC_PROJECTS}")
        print(f"  Cancer types: {CPTAC_CANCER_TYPES}")
        print(f"  Output directory: {output_dir}")
        sys.exit(0)

    df = query_cptac_wsi_metadata()
    print_summary(df)

    if not df.empty:
        save_cptac_manifest(df, output_dir)
        save_per_cancer_manifests(df, output_dir)
        generate_gdc_download_manifest(df, output_dir)
        print(f"\nManifests saved to {output_dir}")
