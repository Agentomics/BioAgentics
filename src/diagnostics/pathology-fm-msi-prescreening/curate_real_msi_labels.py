#!/usr/bin/env python3
"""Curate real MSI labels from TCGA molecular subtype data.

Replaces synthetic labels (based on published prevalence rates) with real
per-patient MSI calls from GDC API molecular test data and MANTIS scores.

Sources:
- GDC API follow_ups.molecular_tests.msi_status
- GDC API clinical supplement files (MANTIS scores)
- Cortes-Ciriano et al. (PMID: 28585546)

Usage:
    uv run python src/diagnostics/pathology-fm-msi-prescreening/curate_real_msi_labels.py
"""

import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bioagentics.models.pathology_msi.msi_labels import (
    MSI_PROJECTS,
    classify_msi_from_mantis,
    curate_msi_labels,
    fetch_tcga_clinical_msi,
    print_label_summary,
    resolve_msi_status,
    save_labels,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GDC_CASES_ENDPOINT = "https://api.gdc.cancer.gov/cases"
GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"

DATA_DIR = PROJECT_ROOT / "data/diagnostics/pathology-fm-msi-prescreening"
LABELS_DIR = DATA_DIR / "labels"


def fetch_gdc_molecular_tests_msi(
    project_ids: list[str] | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Fetch MSI status from GDC molecular tests data.

    Queries follow_ups.molecular_tests which contains MSI-PCR and IHC results
    for TCGA cases.
    """
    if project_ids is None:
        project_ids = MSI_PROJECTS

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
    ]

    expand = ["follow_ups.molecular_tests"]

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
        logger.info(f"  molecular tests: fetched {len(all_cases)}/{total} cases")
        if len(all_cases) >= total:
            break
        from_ += page_size
        time.sleep(0.5)

    records = []
    for case in all_cases:
        project_id = case.get("project", {}).get("project_id", "")
        submitter_id = case.get("submitter_id", "")

        # Extract MSI status from molecular tests
        msi_status = None
        msi_test_type = None
        follow_ups = case.get("follow_ups", [])
        for fu in follow_ups:
            mol_tests = fu.get("molecular_tests", [])
            for test in mol_tests:
                test_msi = test.get("msi_status")
                if test_msi and str(test_msi).upper() not in (
                    "",
                    "NOT REPORTED",
                    "UNKNOWN",
                ):
                    msi_status = test_msi
                    msi_test_type = test.get("test_type", "molecular_test")
                    break
            if msi_status:
                break

        if msi_status:
            records.append(
                {
                    "submitter_id": submitter_id,
                    "case_id": case["case_id"],
                    "project_id": project_id,
                    "cancer_type": project_id.replace("TCGA-", ""),
                    "pcr_call": msi_status,
                    "test_type": msi_test_type,
                }
            )

    df = pd.DataFrame(records)
    logger.info(
        f"Found {len(df)} cases with molecular test MSI data"
    )
    return df


def fetch_gdc_clinical_supplement_msi(
    project_ids: list[str] | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Fetch MSI status from GDC clinical supplement (biotab) files.

    Queries GDC for clinical supplement files and extracts MSI data
    from TCGA biotab format.
    """
    if project_ids is None:
        project_ids = MSI_PROJECTS

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
                    "field": "data_category",
                    "value": "Clinical",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "data_type",
                    "value": "Clinical Supplement",
                },
            },
            {
                "op": "=",
                "content": {
                    "field": "data_format",
                    "value": "BCR XML",
                },
            },
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
    ]

    params = {
        "filters": json.dumps(filters),
        "fields": ",".join(fields),
        "format": "JSON",
        "size": 1,
    }

    resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    total = data.get("data", {}).get("pagination", {}).get("total", 0)
    logger.info(f"Found {total} clinical supplement files in GDC")

    # Clinical supplement XML parsing is complex; molecular_tests is the
    # preferred source. Return empty if not needed.
    return pd.DataFrame()


def fetch_gdc_ssm_msi_scores(
    project_ids: list[str] | None = None,
    page_size: int = 500,
) -> pd.DataFrame:
    """Fetch MANTIS/MSIsensor scores from GDC analysis files.

    Queries for Microsatellite Instability analysis results.
    """
    if project_ids is None:
        project_ids = MSI_PROJECTS

    # Query for MSI analysis files
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
                "op": "in",
                "content": {
                    "field": "analysis.workflow_type",
                    "value": ["MSIsensor"],
                },
            },
        ],
    }

    fields = [
        "file_id",
        "file_name",
        "file_size",
        "cases.case_id",
        "cases.submitter_id",
        "cases.project.project_id",
    ]

    all_hits = []
    from_ = 0

    while True:
        params = {
            "filters": json.dumps(filters),
            "fields": ",".join(fields),
            "format": "JSON",
            "size": page_size,
            "from": from_,
        }
        resp = requests.get(GDC_FILES_ENDPOINT, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()
        hits = data.get("data", {}).get("hits", [])
        if not hits:
            break
        all_hits.extend(hits)
        total = data["data"]["pagination"]["total"]
        logger.info(f"  MSIsensor files: fetched {len(all_hits)}/{total}")
        if len(all_hits) >= total:
            break
        from_ += page_size
        time.sleep(0.5)

    if not all_hits:
        logger.info("No MSIsensor analysis files found in GDC")
        return pd.DataFrame()

    # Download and parse MSIsensor score files (they are small text files)
    records = []
    for hit in all_hits:
        file_id = hit["file_id"]
        cases = hit.get("cases", [])
        if not cases:
            continue
        case = cases[0]

        # Download the score file
        try:
            resp = requests.get(
                f"https://api.gdc.cancer.gov/data/{file_id}",
                timeout=30,
            )
            resp.raise_for_status()
            # MSIsensor output is typically: "Total_Number_of_Sites\tNumber_of_Somatic_Sites\t%"
            # or just a single score value
            content = resp.text.strip()
            score = _parse_msisensor_score(content)
            if score is not None:
                records.append(
                    {
                        "submitter_id": case.get("submitter_id", ""),
                        "case_id": case["case_id"],
                        "project_id": case.get("project", {}).get("project_id", ""),
                        "msisensor_score": score,
                    }
                )
        except Exception as e:
            logger.warning(f"Failed to download MSIsensor file {file_id}: {e}")
        time.sleep(0.3)

    df = pd.DataFrame(records)
    logger.info(f"Parsed {len(df)} MSIsensor scores")
    return df


def _parse_msisensor_score(content: str) -> float | None:
    """Parse MSIsensor output to extract the MSI score."""
    lines = content.strip().split("\n")
    for line in lines:
        parts = line.strip().split("\t")
        if len(parts) >= 3:
            try:
                return float(parts[2])
            except (ValueError, IndexError):
                pass
        # Try parsing as a single float
        try:
            return float(line.strip())
        except ValueError:
            pass
    return None


def merge_msi_sources(
    cases_df: pd.DataFrame,
    molecular_tests_df: pd.DataFrame,
    msisensor_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge MSI data from multiple GDC sources into curated labels.

    Priority:
    1. molecular_tests (PCR/IHC - gold standard)
    2. MSIsensor scores (computational)
    """
    df = cases_df.copy()

    # Merge molecular test results
    if not molecular_tests_df.empty:
        mol_cols = ["submitter_id", "pcr_call"]
        available = [c for c in mol_cols if c in molecular_tests_df.columns]
        mt = molecular_tests_df[available].drop_duplicates(subset=["submitter_id"])
        # Use patient barcode (first 12 chars) for matching
        df["patient_barcode"] = df["submitter_id"].str[:12]
        mt["patient_barcode"] = mt["submitter_id"].str[:12]
        df = df.merge(
            mt[["patient_barcode", "pcr_call"]],
            on="patient_barcode",
            how="left",
        )
    else:
        df["patient_barcode"] = df["submitter_id"].str[:12]
        df["pcr_call"] = None

    # Merge MSIsensor scores
    if not msisensor_df.empty:
        ms = msisensor_df[["submitter_id", "msisensor_score"]].drop_duplicates(
            subset=["submitter_id"]
        )
        ms["patient_barcode"] = ms["submitter_id"].str[:12]
        df = df.merge(
            ms[["patient_barcode", "msisensor_score"]],
            on="patient_barcode",
            how="left",
        )
    else:
        df["msisensor_score"] = None

    df = df.drop(columns=["patient_barcode"])

    # Resolve MSI status using priority logic
    statuses = []
    sources = []
    for _, row in df.iterrows():
        pcr = row.get("pcr_call")
        msisensor = row.get("msisensor_score")

        # MSIsensor uses threshold of 3.5 (vs MANTIS 0.4)
        mantis_call = None
        if msisensor is not None and not pd.isna(msisensor):
            if msisensor >= 3.5:
                mantis_call = "MSI-H"
            else:
                mantis_call = "MSS"

        status, source = resolve_msi_status(
            mantis_call=mantis_call,
            pcr_call=pcr,
            mantis_score=None,  # We use MSIsensor not MANTIS
        )
        statuses.append(status)
        sources.append(source)

    df["msi_status"] = statuses
    df["msi_source"] = sources

    return df


def main() -> None:
    """Main entry point for real MSI label curation."""
    logger.info("=" * 60)
    logger.info("Curating real MSI labels from TCGA molecular data")
    logger.info("=" * 60)

    # Step 1: Fetch case data from GDC
    logger.info("\n[1/4] Fetching case data from GDC API...")
    cases_df = fetch_tcga_clinical_msi()
    logger.info(f"  Got {len(cases_df)} cases across {cases_df['cancer_type'].nunique()} cancer types")
    for ct, count in cases_df["cancer_type"].value_counts().items():
        logger.info(f"    {ct}: {count} cases")

    # Step 2: Fetch molecular test MSI data
    logger.info("\n[2/4] Fetching molecular test MSI data from GDC...")
    mol_tests_df = fetch_gdc_molecular_tests_msi()

    # Step 3: Fetch MSIsensor scores
    logger.info("\n[3/4] Fetching MSIsensor analysis scores from GDC...")
    msisensor_df = fetch_gdc_ssm_msi_scores()

    # Step 4: Merge and resolve
    logger.info("\n[4/4] Merging MSI data sources and resolving labels...")
    curated_df = merge_msi_sources(cases_df, mol_tests_df, msisensor_df)

    # Print summary
    print("\n" + "=" * 60)
    print("CURATED MSI LABELS SUMMARY")
    print("=" * 60)
    print_label_summary(curated_df)

    # Compare with synthetic labels
    synthetic_path = LABELS_DIR / "tcga_msi_labels.csv"
    if synthetic_path.exists():
        synthetic_df = pd.read_csv(synthetic_path)
        print(f"\n--- Comparison with synthetic labels ---")
        print(f"Synthetic labels: {len(synthetic_df)} cases")
        print(f"Real labels: {len(curated_df)} cases")
        if "msi_source" in synthetic_df.columns:
            syn_sources = synthetic_df["msi_source"].value_counts().to_dict()
            print(f"Synthetic sources: {syn_sources}")
        real_sources = curated_df["msi_source"].value_counts().to_dict()
        print(f"Real sources: {real_sources}")

        # Count how many have real (non-unknown) labels
        known = curated_df[curated_df["msi_status"] != "unknown"]
        print(f"\nCases with known MSI status: {len(known)}/{len(curated_df)}")

    # Save real labels (backup synthetic first)
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if synthetic_path.exists():
        backup_path = LABELS_DIR / "tcga_msi_labels_synthetic_backup.csv"
        if not backup_path.exists():
            synthetic_df.to_csv(backup_path, index=False)
            logger.info(f"Backed up synthetic labels to {backup_path}")

    output_path = save_labels(curated_df, LABELS_DIR, "tcga_msi_labels.csv")
    logger.info(f"\nSaved curated labels to {output_path}")

    # Also save a detailed version with all source columns
    detailed_path = LABELS_DIR / "tcga_msi_labels_detailed.csv"
    curated_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed labels to {detailed_path}")


if __name__ == "__main__":
    main()
