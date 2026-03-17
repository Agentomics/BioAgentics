"""Stratified train/validation/test splits for MSI classification.

Creates 70/15/15 splits stratified by cancer type and MSI status.
Ensures no patient-level leakage between splits.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedShuffleSplit

logger = logging.getLogger(__name__)

DEFAULT_SEED = 42
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
TEST_FRAC = 0.15


def _extract_patient_id(submitter_id: str) -> str:
    """Extract patient-level ID from TCGA submitter barcode.

    TCGA barcodes: TCGA-XX-XXXX-01A-... The patient ID is the first 3 fields.
    Multiple samples/slides from the same patient share this prefix.
    """
    parts = submitter_id.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:3])
    return submitter_id


def create_stratified_splits(
    labels_df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = DEFAULT_SEED,
    case_id_col: str = "case_id",
    submitter_id_col: str = "submitter_id",
    cancer_type_col: str = "cancer_type",
    msi_status_col: str = "msi_status",
) -> pd.DataFrame:
    """Create stratified train/val/test splits.

    Stratifies by the combination of cancer_type and msi_status.
    Groups by patient_id to prevent leakage.

    Args:
        labels_df: DataFrame with case-level MSI labels.
        train_frac: Fraction for training set.
        val_frac: Fraction for validation set.
        test_frac: Fraction for test set.
        seed: Random seed for reproducibility.
        case_id_col: Column name for case ID.
        submitter_id_col: Column name for submitter ID.
        cancer_type_col: Column name for cancer type.
        msi_status_col: Column name for MSI status.

    Returns:
        DataFrame with added 'split' and 'patient_id' columns.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "Fractions must sum to 1"

    df = labels_df.copy()

    # Filter out unknown MSI status
    known_mask = df[msi_status_col].isin(["MSI-H", "MSI-L", "MSS"])
    n_unknown = (~known_mask).sum()
    if n_unknown > 0:
        logger.warning(f"Excluding {n_unknown} cases with unknown MSI status")
    df = df[known_mask].reset_index(drop=True)

    # Extract patient IDs to prevent leakage
    df["patient_id"] = df[submitter_id_col].apply(_extract_patient_id)

    # Create stratification key combining cancer type + MSI status
    df["strat_key"] = df[cancer_type_col] + "_" + df[msi_status_col]

    # Deduplicate to patient level for splitting
    patient_df = df.groupby("patient_id").first().reset_index()

    # Two-stage split: first train vs (val+test), then val vs test
    rng = np.random.RandomState(seed)
    test_val_frac = val_frac + test_frac

    # Stage 1: split off test+val
    try:
        splitter1 = StratifiedShuffleSplit(
            n_splits=1, test_size=test_val_frac, random_state=rng,
        )
        train_idx, testval_idx = next(
            splitter1.split(patient_df, patient_df["strat_key"])
        )
    except ValueError:
        # If stratification fails (too few samples in a stratum), fall back
        logger.warning("Stratified split failed, falling back to non-stratified")
        n = len(patient_df)
        indices = rng.permutation(n)
        n_train = int(n * train_frac)
        train_idx = indices[:n_train]
        testval_idx = indices[n_train:]

    train_patients = set(patient_df.iloc[train_idx]["patient_id"])
    testval_patients = patient_df.iloc[testval_idx]

    # Stage 2: split val from test
    val_of_testval = val_frac / test_val_frac
    try:
        splitter2 = StratifiedShuffleSplit(
            n_splits=1, test_size=1 - val_of_testval, random_state=rng,
        )
        val_idx, test_idx = next(
            splitter2.split(testval_patients, testval_patients["strat_key"])
        )
    except ValueError:
        n = len(testval_patients)
        indices = rng.permutation(n)
        n_val = int(n * val_of_testval)
        val_idx = indices[:n_val]
        test_idx = indices[n_val:]

    val_patients = set(testval_patients.iloc[val_idx]["patient_id"])
    test_patients = set(testval_patients.iloc[test_idx]["patient_id"])

    # Assign splits
    def _assign_split(patient_id: str) -> str:
        if patient_id in train_patients:
            return "train"
        if patient_id in val_patients:
            return "val"
        if patient_id in test_patients:
            return "test"
        return "unknown"

    df["split"] = df["patient_id"].apply(_assign_split)
    df = df.drop(columns=["strat_key"])

    return df


def validate_splits(df: pd.DataFrame) -> dict:
    """Validate that splits have no patient leakage and proper stratification.

    Returns:
        Dict with validation results and statistics.
    """
    results = {"valid": True, "issues": []}

    # Check no patient leakage
    train_patients = set(df[df["split"] == "train"]["patient_id"])
    val_patients = set(df[df["split"] == "val"]["patient_id"])
    test_patients = set(df[df["split"] == "test"]["patient_id"])

    train_val = train_patients & val_patients
    train_test = train_patients & test_patients
    val_test = val_patients & test_patients

    if train_val:
        results["valid"] = False
        results["issues"].append(f"Train/val overlap: {len(train_val)} patients")
    if train_test:
        results["valid"] = False
        results["issues"].append(f"Train/test overlap: {len(train_test)} patients")
    if val_test:
        results["valid"] = False
        results["issues"].append(f"Val/test overlap: {len(val_test)} patients")

    # Check split proportions
    total = len(df)
    for split_name in ["train", "val", "test"]:
        count = (df["split"] == split_name).sum()
        frac = count / total if total > 0 else 0
        results[f"{split_name}_count"] = count
        results[f"{split_name}_frac"] = round(frac, 3)

    # Check per-cancer-type MSI-H proportions
    if "cancer_type" in df.columns and "msi_status" in df.columns:
        strat_stats = {}
        for ct in sorted(df["cancer_type"].unique()):
            ct_df = df[df["cancer_type"] == ct]
            ct_stats = {}
            for split_name in ["train", "val", "test"]:
                split_df = ct_df[ct_df["split"] == split_name]
                if len(split_df) > 0:
                    msi_h_frac = (split_df["msi_status"] == "MSI-H").mean()
                    ct_stats[split_name] = round(msi_h_frac, 3)
            strat_stats[ct] = ct_stats
        results["msi_h_fractions"] = strat_stats

    return results


def save_splits(
    df: pd.DataFrame,
    output_dir: str | Path,
    filename: str = "tcga_msi_splits.csv",
) -> Path:
    """Save split assignments to CSV."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / filename

    cols = [c for c in ["case_id", "submitter_id", "patient_id", "split",
                         "cancer_type", "msi_status"] if c in df.columns]
    df[cols].to_csv(path, index=False)
    logger.info(f"Saved splits: {path} ({len(df)} cases)")
    return path


def print_split_summary(df: pd.DataFrame) -> None:
    """Print a summary of the split assignments."""
    total = len(df)
    print(f"\nTotal cases: {total}")
    for split_name in ["train", "val", "test"]:
        count = (df["split"] == split_name).sum()
        pct = 100 * count / total if total > 0 else 0
        print(f"  {split_name}: {count} ({pct:.1f}%)")

    if "cancer_type" in df.columns and "msi_status" in df.columns:
        print("\nPer cancer type MSI-H rates:")
        for ct in sorted(df["cancer_type"].unique()):
            ct_df = df[df["cancer_type"] == ct]
            print(f"  {ct}:")
            for split_name in ["train", "val", "test"]:
                split_df = ct_df[ct_df["split"] == split_name]
                if len(split_df) > 0:
                    msi_h = (split_df["msi_status"] == "MSI-H").sum()
                    pct = 100 * msi_h / len(split_df)
                    print(f"    {split_name}: {len(split_df)} cases, {msi_h} MSI-H ({pct:.1f}%)")
