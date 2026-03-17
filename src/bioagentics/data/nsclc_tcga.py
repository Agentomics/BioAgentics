"""Classify TCGA NSCLC patients into molecular subtypes.

Reads TCGA LUAD and LUSC MAF files, identifies driver mutations,
and classifies patients into KP/KL/KPL/KOnly/KRAS-WT subtypes
with KRAS allele annotation.

Usage:
    uv run python -m bioagentics.data.nsclc_tcga
    uv run python -m bioagentics.data.nsclc_tcga --dest data/tcga/nsclc_patient_subtypes.csv
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.nsclc_common import (
    DAMAGING_CLASSIFICATIONS,
    DRIVER_GENES,
    classify_kras_allele,
    classify_molecular_subtype,
)

DEFAULT_TCGA_DIR = REPO_ROOT / "data" / "tcga"


def extract_patient_id(barcode: str) -> str:
    """Extract TCGA patient ID from sample barcode (first 3 segments)."""
    parts = barcode.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else barcode


def load_tcga_maf_dir(maf_dir: Path) -> pd.DataFrame:
    """Load all MAF.gz files from a directory into one DataFrame."""
    files = sorted(maf_dir.glob("*.maf.gz"))
    if not files:
        raise FileNotFoundError(f"No .maf.gz files in {maf_dir}")

    cols = ["Hugo_Symbol", "Tumor_Sample_Barcode", "Variant_Classification", "HGVSp_Short"]
    frames = []
    for f in files:
        with gzip.open(f, "rt") as fh:
            df = pd.read_csv(fh, sep="\t", comment="#", usecols=lambda c: c in cols)
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


def classify_patients(
    tcga_dir: Path,
    cancer_types: tuple[str, ...] = ("luad", "lusc"),
) -> pd.DataFrame:
    """Classify TCGA NSCLC patients by driver mutation and KRAS allele.

    Returns a DataFrame with one row per patient, containing:
    - patient_id, cancer_type, sample_barcode
    - Per-driver mutation status (bool) and protein change
    - KRAS_allele, molecular_subtype
    """
    all_muts = []
    for ct in cancer_types:
        maf_dir = tcga_dir / ct / "mutations"
        if not maf_dir.exists():
            print(f"  Warning: {maf_dir} not found, skipping")
            continue
        print(f"  Loading {ct.upper()} mutations...")
        df = load_tcga_maf_dir(maf_dir)
        df["cancer_type"] = ct.upper()
        all_muts.append(df)

    muts = pd.concat(all_muts, ignore_index=True)
    muts["patient_id"] = muts["Tumor_Sample_Barcode"].apply(extract_patient_id)

    # Filter to damaging mutations in driver genes
    driver_muts = muts[
        (muts["Hugo_Symbol"].isin(DRIVER_GENES))
        & (muts["Variant_Classification"].isin(DAMAGING_CLASSIFICATIONS))
    ].copy()

    # Get unique patients with their cancer type and sample barcode
    patient_info = (
        muts.groupby("patient_id")
        .agg(cancer_type=("cancer_type", "first"),
             sample_barcode=("Tumor_Sample_Barcode", "first"))
        .reset_index()
    )

    # Build per-patient driver annotation
    for gene in DRIVER_GENES:
        gene_muts = driver_muts[driver_muts["Hugo_Symbol"] == gene]
        mutated_patients = set(gene_muts["patient_id"])
        patient_info[f"{gene}_mutated"] = patient_info["patient_id"].isin(mutated_patients)

        # Protein change
        pc = gene_muts.groupby("patient_id")["HGVSp_Short"].apply(
            lambda x: ";".join(x.dropna().unique())
        )
        patient_info[f"{gene}_protein_change"] = patient_info["patient_id"].map(pc).fillna("")

    # KRAS allele classification
    kras_muts = driver_muts[driver_muts["Hugo_Symbol"] == "KRAS"]
    kras_alleles = kras_muts.groupby("patient_id")["HGVSp_Short"].apply(list)

    def get_allele(pid: str) -> str:
        if pid not in kras_alleles.index:
            return "WT"
        return classify_kras_allele(kras_alleles[pid])

    patient_info["KRAS_allele"] = [get_allele(p) for p in patient_info["patient_id"]]

    # Molecular subtype classification
    patient_info["molecular_subtype"] = patient_info.apply(classify_molecular_subtype, axis=1)

    return patient_info


def merge_clinical(patients: pd.DataFrame, tcga_dir: Path) -> pd.DataFrame:
    """Merge clinical data if available."""
    for ct in patients["cancer_type"].unique():
        clin_path = tcga_dir / ct.lower() / "clinical" / "cases_clinical.json"
        if not clin_path.exists():
            continue

        with open(clin_path) as f:
            cases = json.load(f)

        clin_rows = []
        for case in cases:
            row = {"submitter_id": case.get("submitter_id", "")}
            demo = case.get("demographic", {})
            if isinstance(demo, list):
                demo = demo[0] if demo else {}
            row["gender"] = demo.get("gender")
            row["vital_status"] = demo.get("vital_status")
            row["days_to_death"] = demo.get("days_to_death")
            diag = case.get("diagnoses", [])
            if isinstance(diag, list) and diag:
                d = diag[0]
                row["tumor_stage"] = d.get("tumor_stage")
                row["age_at_diagnosis"] = d.get("age_at_diagnosis")
            clin_rows.append(row)

        clin_df = pd.DataFrame(clin_rows)
        # submitter_id format matches patient_id (TCGA-XX-XXXX)
        patients = patients.merge(
            clin_df, left_on="patient_id", right_on="submitter_id",
            how="left", suffixes=("", f"_clin_{ct}"),
        )
        patients = patients.drop(columns=["submitter_id"], errors="ignore")

    return patients


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Classify TCGA NSCLC patients into molecular subtypes",
    )
    parser.add_argument(
        "--tcga-dir", type=Path, default=DEFAULT_TCGA_DIR,
        help=f"TCGA data directory (default: {DEFAULT_TCGA_DIR})",
    )
    parser.add_argument(
        "--dest", type=Path, default=DEFAULT_TCGA_DIR / "nsclc_patient_subtypes.csv",
        help="Output CSV path",
    )
    args = parser.parse_args(argv)

    print("Classifying TCGA NSCLC patients...")
    patients = classify_patients(args.tcga_dir)

    # Try merging clinical data
    patients = merge_clinical(patients, args.tcga_dir)

    print(f"\nTotal patients: {len(patients)}")
    print(f"\nBy cancer type:")
    print(patients["cancer_type"].value_counts().to_string())
    print(f"\nMolecular subtypes:")
    print(patients["molecular_subtype"].value_counts().to_string())

    kras_mut = patients[patients["KRAS_mutated"]]
    print(f"\nKRAS alleles ({len(kras_mut)} KRAS-mutant):")
    print(kras_mut["KRAS_allele"].value_counts().to_string())

    # Subtype proportions among KRAS-mutant
    if len(kras_mut) > 0:
        print(f"\nSubtype proportions among KRAS-mutant:")
        subtype_pct = kras_mut["molecular_subtype"].value_counts(normalize=True) * 100
        print(subtype_pct.round(1).to_string())

    patients.to_csv(args.dest, index=False)
    print(f"\nSaved to {args.dest}")


if __name__ == "__main__":
    main()
