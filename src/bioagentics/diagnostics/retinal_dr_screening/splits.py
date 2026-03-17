"""Stratified train/val/test splitting with dataset-of-origin preservation.

Features:
  1. Stratified splits by DR grade ensuring class balance
  2. Dataset-of-origin preserved for cross-population evaluation
  3. Patient-level grouping to prevent leakage (both eyes in same split)
  4. Hold-out external validation from complete datasets (e.g. Messidor-2)
  5. Reproducible split assignments saved to CSV

Usage:
    uv run python -m bioagentics.diagnostics.retinal_dr_screening.splits \\
        --catalog data/diagnostics/smartphone-retinal-dr-screening/processed/catalog_processed.csv \\
        --holdout messidor2
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from bioagentics.diagnostics.retinal_dr_screening.config import (
    DATA_DIR,
    RANDOM_SEED,
    TEST_RATIO,
    TRAIN_RATIO,
    VAL_RATIO,
)

logger = logging.getLogger(__name__)


def extract_patient_id(row: pd.Series) -> str:
    """Extract patient ID from image filename to group both eyes.

    Conventions:
    - EyePACS: "{patient_id}_{left|right}.jpeg"
    - APTOS: unique per image (treat each as separate patient)
    - IDRiD: "IDRiD_{number}.jpg"
    - Messidor-2: varies
    - ODIR: "{patient_id}_{left|right}_..."

    Falls back to filename stem if no pattern matches.
    """
    filename = str(row.get("original_filename", ""))
    dataset = str(row.get("dataset_source", ""))

    if dataset == "eyepacs":
        # EyePACS format: "12345_left.jpeg" or "12345_right.jpeg"
        match = re.match(r"^(\d+)_(left|right)", filename)
        if match:
            return f"eyepacs_{match.group(1)}"

    if dataset == "odir5k":
        # ODIR format: "{id}_{left|right}.jpg"
        match = re.match(r"^(\d+)_(left|right)", filename)
        if match:
            return f"odir5k_{match.group(1)}"

    # Default: use dataset + filename stem as unique patient ID
    stem = Path(filename).stem
    return f"{dataset}_{stem}"


def create_splits(
    catalog: pd.DataFrame,
    holdout_datasets: list[str] | None = None,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Create stratified train/val/test splits with external holdout.

    Args:
        catalog: DataFrame with columns: image_path, dr_grade, dataset_source, ...
        holdout_datasets: Dataset names to hold out entirely for external validation.
        train_ratio: Fraction for training set.
        val_ratio: Fraction for validation set.
        test_ratio: Fraction for test set.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with added 'split' column: 'train', 'val', 'test', or 'external_val'.
    """
    if holdout_datasets is None:
        holdout_datasets = ["messidor2"]

    df = catalog.copy()
    df["split"] = ""

    # Step 1: Separate holdout datasets for external validation
    holdout_mask = df["dataset_source"].isin(holdout_datasets)
    df.loc[holdout_mask, "split"] = "external_val"

    # Step 2: Extract patient IDs for the remaining data
    remaining = df[~holdout_mask].copy()

    if remaining.empty:
        logger.warning("No data remaining after holdout — all data is external_val")
        return df

    remaining["patient_id"] = remaining.apply(extract_patient_id, axis=1)

    # Step 3: Get unique patients with their DR grades (use max grade per patient)
    patient_grades = remaining.groupby("patient_id")["dr_grade"].max().reset_index()
    patient_grades.columns = ["patient_id", "max_grade"]

    # Step 4: Stratified split at patient level
    # Use StratifiedGroupKFold to get train vs (val+test), then split val+test
    rng = np.random.default_rng(seed)

    n_patients = len(patient_grades)
    test_val_ratio = val_ratio + test_ratio
    n_folds = max(2, round(1.0 / test_val_ratio))

    # First split: train vs (val+test)
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    patients_array = patient_grades["patient_id"].values
    grades_array = patient_grades["max_grade"].values

    # Use patient_id as both X and group (each patient is its own group at this level)
    train_idx, test_val_idx = next(
        sgkf.split(patients_array, grades_array, groups=patients_array)
    )

    train_patients = set(patients_array[train_idx])
    test_val_patients = list(patients_array[test_val_idx])

    # Second split: val vs test (within test_val set)
    test_val_grades = patient_grades.iloc[test_val_idx]
    val_fraction = val_ratio / test_val_ratio

    # Simple stratified split for val/test
    val_patients = set()
    test_patients = set()

    # Group by grade and split each
    for grade in sorted(test_val_grades["max_grade"].unique()):
        grade_patients = np.array(
            test_val_grades[test_val_grades["max_grade"] == grade]["patient_id"].tolist()
        )
        rng.shuffle(grade_patients)
        n_val = max(1, int(len(grade_patients) * val_fraction))
        val_patients.update(grade_patients[:n_val])
        test_patients.update(grade_patients[n_val:])

    # Step 5: Assign splits back to images
    patient_split_map = {}
    for p in train_patients:
        patient_split_map[p] = "train"
    for p in val_patients:
        patient_split_map[p] = "val"
    for p in test_patients:
        patient_split_map[p] = "test"

    for idx, row in remaining.iterrows():
        pid = row["patient_id"]
        df.loc[idx, "split"] = patient_split_map.get(pid, "train")

    # Drop temporary column
    if "patient_id" in df.columns:
        df = df.drop(columns=["patient_id"])

    # Log split statistics
    _log_split_stats(df)

    return df


def _log_split_stats(df: pd.DataFrame) -> None:
    """Log split distribution statistics."""
    for split in ["train", "val", "test", "external_val"]:
        subset = df[df["split"] == split]
        if subset.empty:
            continue
        grade_counts = subset["dr_grade"].value_counts().sort_index()
        sources = subset["dataset_source"].value_counts()
        logger.info(
            "%s: %d images, grades: %s, sources: %s",
            split,
            len(subset),
            grade_counts.to_dict(),
            sources.to_dict(),
        )


def save_splits(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Save split assignments to CSV."""
    if path is None:
        path = DATA_DIR / "splits.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    logger.info("Splits saved to %s", path)
    return path


def load_split(
    splits_path: Path,
    split: str,
    gradable_only: bool = True,
) -> pd.DataFrame:
    """Load a specific split from the splits CSV.

    Args:
        splits_path: Path to splits CSV.
        split: One of 'train', 'val', 'test', 'external_val'.
        gradable_only: If True, filter to only gradable images.

    Returns:
        Filtered DataFrame for the requested split.
    """
    df = pd.read_csv(splits_path)
    subset = df[df["split"] == split].copy()

    if gradable_only and "is_gradable" in subset.columns:
        before = len(subset)
        subset = subset[subset["is_gradable"]].copy()
        filtered = before - len(subset)
        if filtered > 0:
            logger.info("Filtered %d ungradable images from %s split", filtered, split)

    return subset


# ── CLI ──


def main():
    parser = argparse.ArgumentParser(description="Create stratified splits for DR screening")
    parser.add_argument(
        "--catalog",
        type=Path,
        default=DATA_DIR / "processed" / "catalog_processed.csv",
        help="Path to processed catalog CSV",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DATA_DIR / "splits.csv",
        help="Output path for splits CSV",
    )
    parser.add_argument(
        "--holdout",
        nargs="+",
        default=["messidor2"],
        help="Datasets to hold out for external validation",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    catalog = pd.read_csv(args.catalog)
    df = create_splits(catalog, holdout_datasets=args.holdout, seed=args.seed)
    save_splits(df, args.output)

    print(f"\nSplit summary:")
    for split in ["train", "val", "test", "external_val"]:
        n = len(df[df["split"] == split])
        print(f"  {split}: {n} images ({100 * n / len(df):.1f}%)")


if __name__ == "__main__":
    main()
