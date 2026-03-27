#!/usr/bin/env python3
"""Curate a 30-slide TCGA-COAD pilot cohort with balanced MSI-H/MSS labels.

Queries GDC API for diagnostic WSI file UUIDs, cross-references with
cBioPortal MSI labels, and selects 15 MSI-H + 15 MSS cases.

Output: data/diagnostics/pathology-fm-msi-prescreening/pilot_cohort_manifest.json

Usage:
    uv run python src/diagnostics/pathology-fm-msi-prescreening/curate_pilot_cohort.py
"""

import json
import logging
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from bioagentics.models.pathology_msi.gdc_client import query_diagnostic_wsi_metadata

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = PROJECT_ROOT / "data/diagnostics/pathology-fm-msi-prescreening"
LABELS_PATH = DATA_DIR / "labels/tcga_msi_labels_detailed.csv"
OUTPUT_PATH = DATA_DIR / "pilot_cohort_manifest.json"

N_MSI_H = 15
N_MSS = 15


def main() -> None:
    logger.info("=" * 60)
    logger.info("Curating 30-slide TCGA-COAD MSI pilot cohort")
    logger.info("=" * 60)

    # Step 1: Load existing MSI labels
    logger.info("\n[1/3] Loading MSI labels...")
    labels = pd.read_csv(LABELS_PATH)
    coad_labels = labels[labels["cancer_type"] == "COAD"].copy()
    logger.info(f"  COAD cases with labels: {len(coad_labels)}")
    logger.info(f"  MSI-H: {(coad_labels['msi_status'] == 'MSI-H').sum()}")
    logger.info(f"  MSS: {(coad_labels['msi_status'] == 'MSS').sum()}")

    # Step 2: Query GDC for COAD diagnostic WSI file UUIDs
    logger.info("\n[2/3] Querying GDC for COAD diagnostic WSI file UUIDs...")
    wsi_df = query_diagnostic_wsi_metadata(project_ids=["TCGA-COAD"])
    logger.info(f"  Found {len(wsi_df)} diagnostic WSIs")

    # Match WSIs to labels by case_id
    # labels have case_id (GDC UUID), WSI manifest has case_id (GDC UUID)
    merged = wsi_df.merge(
        coad_labels[["case_id", "submitter_id", "msi_status", "mantis_score", "msisensor_score"]],
        on="case_id",
        how="inner",
        suffixes=("_wsi", "_label"),
    )
    # Resolve submitter_id conflict if any
    if "submitter_id_wsi" in merged.columns:
        merged["submitter_id"] = merged["submitter_id_label"]
        merged = merged.drop(columns=["submitter_id_wsi", "submitter_id_label"])

    logger.info(f"  WSIs matched to MSI labels: {len(merged)}")
    logger.info(f"  MSI-H with WSI: {(merged['msi_status'] == 'MSI-H').sum()}")
    logger.info(f"  MSS with WSI: {(merged['msi_status'] == 'MSS').sum()}")

    # Step 3: Select balanced cohort
    logger.info("\n[3/3] Selecting balanced pilot cohort...")

    # Filter to known MSI status only
    msi_h = merged[merged["msi_status"] == "MSI-H"].copy()
    mss = merged[merged["msi_status"] == "MSS"].copy()

    if len(msi_h) < N_MSI_H:
        logger.warning(f"  Only {len(msi_h)} MSI-H available (need {N_MSI_H})")
    if len(mss) < N_MSS:
        logger.warning(f"  Only {len(mss)} MSS available (need {N_MSS})")

    # Prefer cases with both MANTIS scores; sort by MANTIS to get confident calls
    msi_h = msi_h.sort_values("mantis_score", ascending=False, na_position="last")
    mss = mss.sort_values("mantis_score", ascending=True, na_position="last")

    selected_h = msi_h.head(N_MSI_H)
    selected_s = mss.head(N_MSS)
    selected = pd.concat([selected_h, selected_s], ignore_index=True)

    # Build manifest
    manifest = []
    for _, row in selected.iterrows():
        entry = {
            "case_id": row["case_id"],
            "file_uuid": row["file_id"],
            "patient_id": row.get("submitter_id", row.get("case_submitter_id", "")),
            "msi_label": row["msi_status"],
            "mantis_score": round(float(row["mantis_score"]), 4) if pd.notna(row["mantis_score"]) else None,
            "msisensor_score": round(float(row["msisensor_score"]), 4) if pd.notna(row["msisensor_score"]) else None,
            "file_name": row.get("file_name", ""),
            "file_size_bytes": int(row.get("file_size", 0)),
        }
        manifest.append(entry)

    # Save
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"\nSaved pilot cohort manifest: {OUTPUT_PATH}")
    logger.info(f"  Total slides: {len(manifest)}")
    logger.info(f"  MSI-H: {sum(1 for e in manifest if e['msi_label'] == 'MSI-H')}")
    logger.info(f"  MSS: {sum(1 for e in manifest if e['msi_label'] == 'MSS')}")

    total_gb = sum(e["file_size_bytes"] for e in manifest) / 1e9
    logger.info(f"  Total download size: {total_gb:.1f} GB")

    # Print manifest summary
    print("\n--- Pilot Cohort Summary ---")
    for e in manifest:
        score_str = f"MANTIS={e['mantis_score']}" if e["mantis_score"] else "no MANTIS"
        print(f"  {e['patient_id']}: {e['msi_label']} ({score_str})")


if __name__ == "__main__":
    main()
