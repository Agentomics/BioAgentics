#!/usr/bin/env python3
"""Curate real MSI labels from TCGA molecular subtype data.

Replaces synthetic labels (based on published prevalence rates) with real
per-patient MSI calls from cBioPortal TCGA PanCanAtlas data (MANTIS scores
and MSIsensor scores).

Sources:
- cBioPortal TCGA PanCanAtlas studies (MANTIS + MSIsensor scores)
- GDC API for case metadata
- Cortes-Ciriano et al. (PMID: 28585546)

Usage:
    uv run python src/diagnostics/pathology-fm-msi-prescreening/curate_real_msi_labels.py
"""

import logging
import sys
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bioagentics.models.pathology_msi.msi_labels import (
    MANTIS_MSI_H_THRESHOLD,
    fetch_tcga_clinical_msi,
    print_label_summary,
    save_labels,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data/diagnostics/pathology-fm-msi-prescreening"
LABELS_DIR = DATA_DIR / "labels"

# cBioPortal TCGA PanCanAtlas study IDs
CBIO_STUDIES = {
    "coadread_tcga_pan_can_atlas_2018": ["COAD", "READ"],
    "ucec_tcga_pan_can_atlas_2018": ["UCEC"],
    "stad_tcga_pan_can_atlas_2018": ["STAD"],
}

# MSIsensor threshold (Niu et al., Bioinformatics 2014)
MSISENSOR_MSI_H_THRESHOLD = 3.5


def fetch_cbioportal_msi_scores() -> pd.DataFrame:
    """Fetch MANTIS and MSIsensor scores from cBioPortal TCGA PanCanAtlas.

    Returns DataFrame with: patient_id, mantis_score, msisensor_score, study_id
    """
    all_records = []

    for study_id, cancer_types in CBIO_STUDIES.items():
        logger.info(f"  Fetching {study_id}...")
        url = f"https://www.cbioportal.org/api/studies/{study_id}/clinical-data"
        params = {"clinicalDataType": "SAMPLE", "projection": "DETAILED"}
        resp = requests.get(
            url,
            params=params,
            headers={"accept": "application/json"},
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # Group by patient
        patient_data: dict[str, dict] = {}
        for entry in data:
            attr_id = entry.get("clinicalAttributeId", "")
            patient_id = entry.get("patientId", "")
            value = entry.get("value", "")

            if attr_id not in ("MSI_SCORE_MANTIS", "MSI_SENSOR_SCORE"):
                continue

            if patient_id not in patient_data:
                patient_data[patient_id] = {
                    "patient_id": patient_id,
                    "study_id": study_id,
                }

            if attr_id == "MSI_SCORE_MANTIS":
                try:
                    patient_data[patient_id]["mantis_score"] = float(value)
                except (ValueError, TypeError):
                    pass
            elif attr_id == "MSI_SENSOR_SCORE":
                try:
                    patient_data[patient_id]["msisensor_score"] = float(value)
                except (ValueError, TypeError):
                    pass

        records = list(patient_data.values())
        logger.info(
            f"    {study_id}: {len(records)} patients with MSI scores"
        )
        all_records.extend(records)

    df = pd.DataFrame(all_records)
    logger.info(f"  Total: {len(df)} patients with MSI scores from cBioPortal")
    return df


def classify_mantis_binary(score: float) -> str:
    """Classify MSI status from MANTIS score using binary threshold.

    MANTIS (Kautto et al., Bioinformatics 2017) uses a single threshold of 0.4:
    - MSI-H: >= 0.4
    - MSS: < 0.4

    Note: MSI-L is defined by PCR-based testing (Bethesda markers), not by
    computational scores. MANTIS scores in the 0.3-0.4 range are MSS, not MSI-L.
    """
    if score >= MANTIS_MSI_H_THRESHOLD:
        return "MSI-H"
    return "MSS"


def classify_msi_dual_score(
    mantis_score: float | None,
    msisensor_score: float | None,
) -> tuple[str, str]:
    """Classify MSI status using both MANTIS and MSIsensor scores.

    Uses binary classification (MSI-H vs MSS) for both computational scores.

    Resolution priority:
    1. If both scores agree, use concordant call
    2. If they disagree, prefer MANTIS (used more in literature)
    3. If only one score available, use it

    Returns (msi_status, source).
    """
    mantis_call = None
    msisensor_call = None

    if mantis_score is not None and not pd.isna(mantis_score):
        mantis_call = classify_mantis_binary(mantis_score)

    if msisensor_score is not None and not pd.isna(msisensor_score):
        msisensor_call = "MSI-H" if msisensor_score >= MSISENSOR_MSI_H_THRESHOLD else "MSS"

    if mantis_call is None and msisensor_call is None:
        return "unknown", "no_data"

    if mantis_call is None:
        return msisensor_call, "msisensor_only"  # type: ignore[return-value]

    if msisensor_call is None:
        return mantis_call, "mantis_only"

    # Both available
    if mantis_call == msisensor_call:
        return mantis_call, "mantis_msisensor_concordant"

    # Disagreement: MANTIS is more widely used; flag the conflict
    return mantis_call, f"mantis_preferred_over_msisensor_{msisensor_call}"


def merge_cases_with_msi(
    cases_df: pd.DataFrame,
    msi_scores_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge GDC case metadata with cBioPortal MSI scores."""
    df = cases_df.copy()

    # Create patient barcode for matching (first 12 chars of submitter_id)
    df["patient_barcode"] = df["submitter_id"].str[:12]

    if not msi_scores_df.empty:
        scores = msi_scores_df.copy()
        # patient_id in cBioPortal is the TCGA barcode (e.g., TCGA-AA-3562)
        scores["patient_barcode"] = scores["patient_id"].str[:12]

        # Deduplicate (some patients may appear in multiple samples)
        scores = scores.drop_duplicates(subset=["patient_barcode"], keep="first")

        merge_cols = ["patient_barcode"]
        score_cols = [c for c in ["mantis_score", "msisensor_score"] if c in scores.columns]
        df = df.merge(
            scores[merge_cols + score_cols],
            on="patient_barcode",
            how="left",
        )
    else:
        df["mantis_score"] = None
        df["msisensor_score"] = None

    df = df.drop(columns=["patient_barcode"])

    # Classify MSI status
    statuses = []
    sources = []
    for _, row in df.iterrows():
        status, source = classify_msi_dual_score(
            mantis_score=row.get("mantis_score"),
            msisensor_score=row.get("msisensor_score"),
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

    # Step 1: Fetch case metadata from GDC
    logger.info("\n[1/3] Fetching case metadata from GDC API...")
    cases_df = fetch_tcga_clinical_msi()
    logger.info(f"  {len(cases_df)} cases across {cases_df['cancer_type'].nunique()} cancer types")
    for ct, count in cases_df["cancer_type"].value_counts().items():
        logger.info(f"    {ct}: {count}")

    # Step 2: Fetch MSI scores from cBioPortal
    logger.info("\n[2/3] Fetching MSI scores from cBioPortal PanCanAtlas...")
    msi_scores_df = fetch_cbioportal_msi_scores()

    # Step 3: Merge and classify
    logger.info("\n[3/3] Merging case data with MSI scores...")
    curated_df = merge_cases_with_msi(cases_df, msi_scores_df)

    # Print summary
    print("\n" + "=" * 60)
    print("CURATED REAL MSI LABELS SUMMARY")
    print("=" * 60)
    print_label_summary(curated_df)

    # Score distribution summary
    has_mantis = curated_df["mantis_score"].notna().sum()
    has_msisensor = curated_df["msisensor_score"].notna().sum()
    print(f"\nScore coverage:")
    print(f"  MANTIS scores: {has_mantis}/{len(curated_df)}")
    print(f"  MSIsensor scores: {has_msisensor}/{len(curated_df)}")

    # Compare with previous labels
    synthetic_backup = LABELS_DIR / "tcga_msi_labels_synthetic_backup.csv"
    current_labels = LABELS_DIR / "tcga_msi_labels.csv"

    if current_labels.exists():
        prev_df = pd.read_csv(current_labels)
        prev_sources = prev_df["msi_source"].value_counts().to_dict()
        is_synthetic = any("synthetic" in str(s) for s in prev_sources)

        if is_synthetic:
            print(f"\n--- Replacing synthetic labels ---")
            print(f"Previous: {len(prev_df)} cases (source: {prev_sources})")

    # Backup synthetic labels if this is the first real curation
    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    if current_labels.exists() and not synthetic_backup.exists():
        prev_df = pd.read_csv(current_labels)
        if any("synthetic" in str(s) for s in prev_df["msi_source"].unique()):
            prev_df.to_csv(synthetic_backup, index=False)
            logger.info(f"Backed up synthetic labels to {synthetic_backup}")

    # Save curated labels
    output_path = save_labels(curated_df, LABELS_DIR, "tcga_msi_labels.csv")
    logger.info(f"Saved curated labels to {output_path}")

    # Save detailed version with all score columns
    detailed_path = LABELS_DIR / "tcga_msi_labels_detailed.csv"
    curated_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed labels to {detailed_path}")

    # Summary stats for journal
    known = curated_df[curated_df["msi_status"] != "unknown"]
    msi_h = (curated_df["msi_status"] == "MSI-H").sum()
    print(f"\nFinal: {len(known)}/{len(curated_df)} cases with real MSI labels")
    print(f"  MSI-H: {msi_h}, MSS: {(curated_df['msi_status'] == 'MSS').sum()}")
    print(f"  MSI-L: {(curated_df['msi_status'] == 'MSI-L').sum()}")
    print(f"  Unknown: {(curated_df['msi_status'] == 'unknown').sum()}")


if __name__ == "__main__":
    main()
