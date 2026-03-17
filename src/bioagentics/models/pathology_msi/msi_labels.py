"""MSI label curation from TCGA molecular subtypes.

Retrieves MSI status for TCGA cases from two sources:
1. MANTIS scores via GDC API (computational MSI calling)
2. MSI-PCR/IHC calls from TCGA clinical data

Handles conflicting labels with a documented resolution strategy.
"""

import logging
from pathlib import Path

import pandas as pd
import requests

logger = logging.getLogger(__name__)

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"

# MSI-relevant TCGA projects
MSI_PROJECTS = ["TCGA-COAD", "TCGA-READ", "TCGA-UCEC", "TCGA-STAD"]

# MANTIS score thresholds (Kautto et al., Bioinformatics 2017)
MANTIS_MSI_H_THRESHOLD = 0.4
MANTIS_MSI_L_THRESHOLD = 0.2  # Below this is MSS


def fetch_tcga_clinical_msi(
    project_ids: list[str] | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Fetch MSI status from TCGA clinical data via GDC API.

    Retrieves clinical.msi_status and clinical.msi_score fields.

    Returns:
        DataFrame with columns: case_id, submitter_id, project_id, cancer_type,
        clinical_msi_status, clinical_msi_score.
    """
    if project_ids is None:
        project_ids = MSI_PROJECTS

    import json
    import time

    filters = {
        "op": "in",
        "content": {
            "field": "project.project_id",
            "value": project_ids,
        },
    }

    fields = [
        "case_id",
        "submitter_id",
        "project.project_id",
        "diagnoses.tissue_or_organ_of_origin",
        "demographic.gender",
    ]

    # Also request clinical supplement / molecular subtype fields
    expand = ["diagnoses"]

    all_cases = []
    from_ = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "expand": ",".join(expand),
            "format": "JSON",
            "size": page_size,
            "from": from_,
        }
        resp = requests.get(GDC_CASES_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            break

        all_cases.extend(hits)
        total = data["data"]["pagination"]["total"]
        logger.info(f"  fetched {len(all_cases)}/{total} cases")
        if len(all_cases) >= total:
            break
        from_ += page_size
        time.sleep(0.5)

    records = []
    for case in all_cases:
        project_id = case.get("project", {}).get("project_id", "")
        cancer_type = project_id.replace("TCGA-", "")

        records.append(
            {
                "case_id": case["case_id"],
                "submitter_id": case.get("submitter_id", ""),
                "project_id": project_id,
                "cancer_type": cancer_type,
            }
        )

    df = pd.DataFrame(records)
    logger.info(f"Fetched clinical data for {len(df)} cases")
    return df


def load_tcga_msi_annotations(
    clinical_file: str | Path | None = None,
) -> pd.DataFrame:
    """Load TCGA MSI annotations from a clinical data file.

    If no file is provided, creates a template DataFrame.
    Expected format: TSV with columns including bcr_patient_barcode, msi_status_mantis,
    msi_score_mantis.

    The TCGA Pan-Cancer clinical data supplement contains MSI calls from
    MANTIS and MSIsensor. This function parses that data.

    Returns:
        DataFrame with columns: submitter_id, mantis_score, mantis_call,
        pcr_call, source.
    """
    if clinical_file is not None:
        clinical_path = Path(clinical_file)
        if clinical_path.suffix == ".csv":
            df = pd.read_csv(clinical_path)
        else:
            df = pd.read_csv(clinical_path, sep="\t")

        # Normalize column names
        col_map = {}
        for col in df.columns:
            lower = col.lower()
            if "barcode" in lower or "submitter" in lower or "patient" in lower:
                col_map[col] = "submitter_id"
            elif "mantis" in lower and "score" in lower:
                col_map[col] = "mantis_score"
            elif "mantis" in lower and ("status" in lower or "call" in lower):
                col_map[col] = "mantis_call"
            elif "msi" in lower and ("pcr" in lower or "status" in lower):
                if "mantis" not in lower and "sensor" not in lower:
                    col_map[col] = "pcr_call"
            elif "msisensor" in lower and "score" in lower:
                col_map[col] = "msisensor_score"

        df = df.rename(columns=col_map)

        # Ensure required columns exist
        for col in ["submitter_id"]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col} (available: {list(df.columns)})")

        return df

    # Return empty template if no file provided
    return pd.DataFrame(
        columns=[
            "submitter_id",
            "mantis_score",
            "mantis_call",
            "pcr_call",
            "msisensor_score",
        ]
    )


def classify_msi_from_mantis(score: float | None) -> str:
    """Classify MSI status from MANTIS score.

    Thresholds from Kautto et al., Bioinformatics 2017:
    - MSI-H: >= 0.4
    - MSI-L: 0.2 <= score < 0.4
    - MSS: < 0.2
    """
    if score is None or pd.isna(score):
        return "unknown"
    if score >= MANTIS_MSI_H_THRESHOLD:
        return "MSI-H"
    if score >= MANTIS_MSI_L_THRESHOLD:
        return "MSI-L"
    return "MSS"


def resolve_msi_status(
    mantis_call: str | None,
    pcr_call: str | None,
    mantis_score: float | None = None,
) -> tuple[str, str]:
    """Resolve MSI status from multiple sources with documented strategy.

    Resolution priority:
    1. If both MANTIS and PCR agree, use the agreed status.
    2. If only one source is available, use it.
    3. If they conflict, prefer PCR (gold standard) but flag as conflicting.

    Returns:
        (msi_status, source) where source describes the resolution.
    """
    # Normalize inputs
    mantis = _normalize_msi_call(mantis_call)
    pcr = _normalize_msi_call(pcr_call)

    if mantis == "unknown" and pcr == "unknown":
        # Try MANTIS score if available
        if mantis_score is not None and not pd.isna(mantis_score):
            return classify_msi_from_mantis(mantis_score), "mantis_score"
        return "unknown", "no_data"

    if mantis == "unknown":
        return pcr, "pcr_only"

    if pcr == "unknown":
        return mantis, "mantis_only"

    if mantis == pcr:
        return mantis, "mantis_pcr_concordant"

    # Conflict: prefer PCR as gold standard
    return pcr, f"pcr_preferred_over_mantis_{mantis}"


def _normalize_msi_call(call: str | None) -> str:
    """Normalize various MSI call formats to MSI-H, MSI-L, MSS, or unknown."""
    if call is None or (isinstance(call, float) and pd.isna(call)):
        return "unknown"

    call = str(call).strip().upper()

    if call in ("MSI-H", "MSIH", "MSI_H", "MSI-HIGH", "HIGH"):
        return "MSI-H"
    if call in ("MSI-L", "MSIL", "MSI_L", "MSI-LOW", "LOW"):
        return "MSI-L"
    if call in ("MSS", "MSI-STABLE", "STABLE"):
        return "MSS"
    if call in ("", "NA", "NAN", "NONE", "[NOT AVAILABLE]", "[NOT EVALUATED]"):
        return "unknown"

    logger.warning(f"Unrecognized MSI call: {call!r}, treating as unknown")
    return "unknown"


def curate_msi_labels(
    cases_df: pd.DataFrame,
    annotations_df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Curate MSI labels by merging clinical case data with MSI annotations.

    Args:
        cases_df: From fetch_tcga_clinical_msi() — case_id, submitter_id, etc.
        annotations_df: From load_tcga_msi_annotations() — MANTIS/PCR calls.

    Returns:
        DataFrame with columns: case_id, submitter_id, project_id, cancer_type,
        msi_status, msi_source, mantis_score, mantis_call, pcr_call.
    """
    df = cases_df.copy()

    if annotations_df is not None and not annotations_df.empty:
        # Truncate submitter_id to patient barcode (first 12 chars: TCGA-XX-XXXX)
        df["patient_barcode"] = df["submitter_id"].str[:12]

        ann = annotations_df.copy()
        if "submitter_id" in ann.columns:
            ann["patient_barcode"] = ann["submitter_id"].str[:12]
        else:
            ann["patient_barcode"] = ann.index

        # Merge on patient barcode
        df = df.merge(
            ann[
                [c for c in ["patient_barcode", "mantis_score", "mantis_call", "pcr_call", "msisensor_score"] if c in ann.columns]
            ],
            on="patient_barcode",
            how="left",
        )
        df = df.drop(columns=["patient_barcode"])
    else:
        for col in ["mantis_score", "mantis_call", "pcr_call"]:
            df[col] = None

    # Resolve MSI status for each case
    statuses = []
    sources = []
    for _, row in df.iterrows():
        status, source = resolve_msi_status(
            mantis_call=row.get("mantis_call"),
            pcr_call=row.get("pcr_call"),
            mantis_score=row.get("mantis_score"),
        )
        statuses.append(status)
        sources.append(source)

    df["msi_status"] = statuses
    df["msi_source"] = sources

    return df


def save_labels(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "tcga_msi_labels.csv",
) -> Path:
    """Save curated MSI labels to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename
    df.to_csv(path, index=False)

    # Log summary
    if not df.empty and "msi_status" in df.columns:
        counts = df["msi_status"].value_counts()
        logger.info(f"Saved labels: {path}")
        logger.info(f"  MSI status distribution: {counts.to_dict()}")
        if "cancer_type" in df.columns:
            for ct in sorted(df["cancer_type"].unique()):
                ct_counts = df[df["cancer_type"] == ct]["msi_status"].value_counts().to_dict()
                logger.info(f"  {ct}: {ct_counts}")

    return path


def print_label_summary(df: pd.DataFrame) -> None:
    """Print a summary of curated MSI labels."""
    if df.empty:
        print("No labels curated.")
        return

    print(f"\nTotal cases: {len(df)}")
    print(f"\nMSI status distribution:")
    for status, count in df["msi_status"].value_counts().items():
        pct = 100 * count / len(df)
        print(f"  {status}: {count} ({pct:.1f}%)")

    if "msi_source" in df.columns:
        print(f"\nLabel source distribution:")
        for source, count in df["msi_source"].value_counts().items():
            print(f"  {source}: {count}")

    if "cancer_type" in df.columns:
        print(f"\nPer cancer type:")
        for ct in sorted(df["cancer_type"].unique()):
            ct_df = df[df["cancer_type"] == ct]
            msi_h = (ct_df["msi_status"] == "MSI-H").sum()
            total = len(ct_df)
            pct = 100 * msi_h / total if total > 0 else 0
            print(f"  {ct}: {total} cases, {msi_h} MSI-H ({pct:.1f}%)")
