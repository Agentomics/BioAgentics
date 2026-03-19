"""MIMIC-IV hourly vitals and labs extraction pipeline.

Extracts hourly feature vectors from MIMIC-IV tables:
- 6 vitals (HR, MAP, SBP, RR, SpO2, temp) from chartevents
- 20 labs from labevents
- Demographics (age, sex, ethnicity) from patients/admissions

Resamples to hourly resolution with forward-fill.
Output: per-admission hourly DataFrames saved as parquet.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.diagnostics.sepsis.config import (
    DATA_DIR,
    ETHNICITY_MAP,
    LAB_FEATURES,
    LAB_ITEMIDS,
    OUTPUT_DIR,
    TEMP_FAHRENHEIT_ITEMIDS,
    VITAL_FEATURES,
    VITAL_ITEMIDS,
)

logger = logging.getLogger(__name__)


def load_mimic_table(data_dir: Path, table_name: str) -> pd.DataFrame:
    """Load a MIMIC-IV table from parquet or CSV."""
    parquet_path = data_dir / f"{table_name}.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    csv_path = data_dir / f"{table_name}.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    csv_gz_path = data_dir / f"{table_name}.csv.gz"
    if csv_gz_path.exists():
        return pd.read_csv(csv_gz_path)
    raise FileNotFoundError(f"Table {table_name} not found in {data_dir}")


def extract_demographics(data_dir: Path) -> pd.DataFrame:
    """Extract age, sex, ethnicity per admission from patients/admissions tables.

    Returns DataFrame indexed by (subject_id, hadm_id) with columns:
    age, sex, ethnicity.
    """
    patients = load_mimic_table(data_dir, "patients")
    admissions = load_mimic_table(data_dir, "admissions")

    merged = admissions.merge(patients, on="subject_id", how="left")

    # Compute age at admission
    if "anchor_age" in merged.columns and "anchor_year" in merged.columns:
        admit_year = pd.to_datetime(merged["admittime"]).dt.year
        merged["age"] = merged["anchor_age"] + (admit_year - merged["anchor_year"])
    elif "dob" in merged.columns:
        merged["age"] = (
            pd.to_datetime(merged["admittime"]) - pd.to_datetime(merged["dob"])
        ).dt.days / 365.25
    else:
        merged["age"] = np.nan

    # Sex: map to binary
    merged["sex"] = merged["gender"].map({"M": 0, "F": 1}).astype("Int64")

    # Ethnicity: simplify using map
    race_col = "race" if "race" in merged.columns else "ethnicity"
    merged["ethnicity"] = merged[race_col].map(ETHNICITY_MAP).fillna("other")

    result = merged[["subject_id", "hadm_id", "age", "sex", "ethnicity"]].copy()
    result = result.set_index(["subject_id", "hadm_id"])
    return result


def extract_vitals(data_dir: Path, admissions: pd.DataFrame) -> pd.DataFrame:
    """Extract vital signs from chartevents, resample to hourly.

    Parameters
    ----------
    data_dir : Path to MIMIC-IV data directory.
    admissions : DataFrame with subject_id, hadm_id, admittime, dischtime.

    Returns
    -------
    DataFrame with columns: subject_id, hadm_id, hours_in, + vital feature columns.
    """
    chartevents = load_mimic_table(data_dir, "chartevents")

    # Build itemid -> feature name mapping
    itemid_to_feature: dict[int, str] = {}
    for feature, itemids in VITAL_ITEMIDS.items():
        for iid in itemids:
            itemid_to_feature[iid] = feature

    all_itemids = list(itemid_to_feature.keys())
    chart_filtered = chartevents[chartevents["itemid"].isin(all_itemids)].copy()
    chart_filtered["feature"] = chart_filtered["itemid"].map(itemid_to_feature)
    chart_filtered["charttime"] = pd.to_datetime(chart_filtered["charttime"])
    chart_filtered["valuenum"] = pd.to_numeric(
        chart_filtered["valuenum"], errors="coerce"
    )
    chart_filtered = chart_filtered.dropna(subset=["valuenum"])

    # Convert Fahrenheit temperatures to Celsius
    f_mask = chart_filtered["itemid"].isin(TEMP_FAHRENHEIT_ITEMIDS)
    chart_filtered.loc[f_mask, "valuenum"] = (
        chart_filtered.loc[f_mask, "valuenum"] - 32
    ) * 5 / 9

    # Merge with admissions to get admittime
    admissions_slim = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    admissions_slim["admittime"] = pd.to_datetime(admissions_slim["admittime"])
    chart_filtered = chart_filtered.merge(
        admissions_slim, on=["subject_id", "hadm_id"], how="inner"
    )

    # Compute hours since admission
    chart_filtered["hours_in"] = (
        chart_filtered["charttime"] - chart_filtered["admittime"]
    ).dt.total_seconds() / 3600
    chart_filtered["hour_bin"] = chart_filtered["hours_in"].round().astype(int)

    # Pivot: take median per hour per feature per admission
    grouped = (
        chart_filtered.groupby(["subject_id", "hadm_id", "hour_bin", "feature"])[
            "valuenum"
        ]
        .median()
        .reset_index()
    )
    pivoted = grouped.pivot_table(
        index=["subject_id", "hadm_id", "hour_bin"],
        columns="feature",
        values="valuenum",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"hour_bin": "hours_in"})

    # Ensure all vital columns exist
    for feat in VITAL_FEATURES:
        if feat not in pivoted.columns:
            pivoted[feat] = np.nan

    return pivoted


def extract_labs(data_dir: Path, admissions: pd.DataFrame) -> pd.DataFrame:
    """Extract lab values from labevents, resample to hourly.

    Returns DataFrame with columns: subject_id, hadm_id, hours_in, + lab feature columns.
    """
    labevents = load_mimic_table(data_dir, "labevents")

    itemid_to_feature: dict[int, str] = {}
    for feature, itemids in LAB_ITEMIDS.items():
        for iid in itemids:
            itemid_to_feature[iid] = feature

    all_itemids = list(itemid_to_feature.keys())
    lab_filtered = labevents[labevents["itemid"].isin(all_itemids)].copy()
    lab_filtered["feature"] = lab_filtered["itemid"].map(itemid_to_feature)
    lab_filtered["charttime"] = pd.to_datetime(lab_filtered["charttime"])
    lab_filtered["valuenum"] = pd.to_numeric(
        lab_filtered["valuenum"], errors="coerce"
    )
    lab_filtered = lab_filtered.dropna(subset=["valuenum"])

    admissions_slim = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    admissions_slim["admittime"] = pd.to_datetime(admissions_slim["admittime"])
    lab_filtered = lab_filtered.merge(
        admissions_slim, on=["subject_id", "hadm_id"], how="inner"
    )

    lab_filtered["hours_in"] = (
        lab_filtered["charttime"] - lab_filtered["admittime"]
    ).dt.total_seconds() / 3600
    lab_filtered["hour_bin"] = lab_filtered["hours_in"].round().astype(int)

    grouped = (
        lab_filtered.groupby(["subject_id", "hadm_id", "hour_bin", "feature"])[
            "valuenum"
        ]
        .median()
        .reset_index()
    )
    pivoted = grouped.pivot_table(
        index=["subject_id", "hadm_id", "hour_bin"],
        columns="feature",
        values="valuenum",
    ).reset_index()
    pivoted.columns.name = None
    pivoted = pivoted.rename(columns={"hour_bin": "hours_in"})

    for feat in LAB_FEATURES:
        if feat not in pivoted.columns:
            pivoted[feat] = np.nan

    return pivoted


def merge_and_resample(
    vitals: pd.DataFrame,
    labs: pd.DataFrame,
    demographics: pd.DataFrame,
    max_hours: int = 336,
) -> pd.DataFrame:
    """Merge vitals and labs into a single hourly feature matrix with forward-fill.

    Parameters
    ----------
    vitals : Hourly vitals DataFrame.
    labs : Hourly labs DataFrame.
    demographics : Demographics indexed by (subject_id, hadm_id).
    max_hours : Maximum ICU hours to include (default 336 = 14 days).

    Returns
    -------
    DataFrame with one row per admission-hour, all features forward-filled.
    """
    key_cols = ["subject_id", "hadm_id", "hours_in"]
    merged = vitals.merge(labs, on=key_cols, how="outer")

    # Filter to reasonable range
    merged = merged[(merged["hours_in"] >= 0) & (merged["hours_in"] <= max_hours)]

    # Reindex to fill missing hours with NaN, then forward-fill per admission
    result_frames = []
    for (sid, hid), group in merged.groupby(["subject_id", "hadm_id"]):
        min_h = int(group["hours_in"].min())
        max_h = int(group["hours_in"].max())
        full_hours = pd.DataFrame({"hours_in": range(min_h, max_h + 1)})
        filled = full_hours.merge(group, on="hours_in", how="left")
        filled["subject_id"] = sid
        filled["hadm_id"] = hid

        # Forward-fill features (not IDs)
        feat_cols = VITAL_FEATURES + LAB_FEATURES
        existing_feats = [c for c in feat_cols if c in filled.columns]
        filled[existing_feats] = filled[existing_feats].ffill()
        result_frames.append(filled)

    if not result_frames:
        return pd.DataFrame()

    result = pd.concat(result_frames, ignore_index=True)

    # Attach demographics
    demo_reset = demographics.reset_index()
    result = result.merge(demo_reset, on=["subject_id", "hadm_id"], how="left")

    return result


def run_extraction(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Run the full extraction pipeline.

    Returns the merged hourly feature matrix.
    """
    output_path = output_dir / "hourly_features.parquet"
    if output_path.exists() and not force:
        logger.info("Loading cached hourly features from %s", output_path)
        return pd.read_parquet(output_path)

    logger.info("Extracting data from %s", data_dir)

    # Load admissions for join keys
    admissions = load_mimic_table(data_dir, "admissions")

    demographics = extract_demographics(data_dir)
    logger.info("Extracted demographics for %d admissions", len(demographics))

    vitals = extract_vitals(data_dir, admissions)
    logger.info("Extracted vitals: %d rows", len(vitals))

    labs = extract_labs(data_dir, admissions)
    logger.info("Extracted labs: %d rows", len(labs))

    result = merge_and_resample(vitals, labs, demographics)
    logger.info("Merged hourly matrix: %d rows, %d columns", *result.shape)

    output_dir.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path, index=False)
    logger.info("Saved to %s", output_path)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract hourly vitals/labs from MIMIC-IV"
    )
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    result = run_extraction(args.data_dir, args.output_dir, force=args.force)

    print(f"\n=== Extraction Complete ===")
    print(f"Rows: {len(result)}")
    print(f"Columns: {list(result.columns)}")
    n_admissions = result.groupby(["subject_id", "hadm_id"]).ngroups
    print(f"Admissions: {n_admissions}")


if __name__ == "__main__":
    main()
