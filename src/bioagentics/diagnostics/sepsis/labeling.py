"""Sepsis-3 onset labeling module.

Implements Sepsis-3 criteria:
- Suspected infection: earliest of (antibiotics order, culture order)
- SOFA score >= 2
- Onset time: when both criteria first met

Produces per-admission binary label with onset timestamp.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.diagnostics.sepsis.config import DATA_DIR, OUTPUT_DIR, SOFA_THRESHOLD
from bioagentics.diagnostics.sepsis.data_extraction import load_mimic_table

logger = logging.getLogger(__name__)


def compute_sofa_respiratory(pao2: pd.Series, fio2: pd.Series) -> pd.Series:
    """Respiratory SOFA component from PaO2/FiO2 ratio."""
    ratio = pao2 / fio2.replace(0, np.nan)
    score = pd.Series(0, index=ratio.index, dtype=int)
    score = score.where(ratio.isna() | (ratio >= 400), 1)
    score = score.where(ratio.isna() | (ratio >= 300) | (score > 1), 1)
    score = score.where(ratio.isna() | (ratio >= 200) | (score > 2), 2)
    score = score.where(ratio.isna() | (ratio >= 100) | (score > 3), 3)
    score = score.where(ratio.isna() | (ratio >= 100), 4)
    return score


def compute_sofa_coagulation(platelets: pd.Series) -> pd.Series:
    """Coagulation SOFA component from platelet count (10^3/uL)."""
    score = pd.Series(0, index=platelets.index, dtype=int)
    mask = platelets.notna()
    score[mask & (platelets < 150)] = 1
    score[mask & (platelets < 100)] = 2
    score[mask & (platelets < 50)] = 3
    score[mask & (platelets < 20)] = 4
    return score


def compute_sofa_liver(bilirubin: pd.Series) -> pd.Series:
    """Liver SOFA component from total bilirubin (mg/dL)."""
    score = pd.Series(0, index=bilirubin.index, dtype=int)
    mask = bilirubin.notna()
    score[mask & (bilirubin >= 1.2)] = 1
    score[mask & (bilirubin >= 2.0)] = 2
    score[mask & (bilirubin >= 6.0)] = 3
    score[mask & (bilirubin >= 12.0)] = 4
    return score


def compute_sofa_cardiovascular(map_val: pd.Series) -> pd.Series:
    """Cardiovascular SOFA component from MAP (simplified, no vasopressors).

    Full SOFA cardiovascular scoring requires vasopressor data. This simplified
    version uses MAP only: MAP < 70 = 1 point.
    """
    score = pd.Series(0, index=map_val.index, dtype=int)
    mask = map_val.notna()
    score[mask & (map_val < 70)] = 1
    return score


def compute_sofa_renal(creatinine: pd.Series) -> pd.Series:
    """Renal SOFA component from creatinine (mg/dL)."""
    score = pd.Series(0, index=creatinine.index, dtype=int)
    mask = creatinine.notna()
    score[mask & (creatinine >= 1.2)] = 1
    score[mask & (creatinine >= 2.0)] = 2
    score[mask & (creatinine >= 3.5)] = 3
    score[mask & (creatinine >= 5.0)] = 4
    return score


def compute_sofa_scores(hourly_features: pd.DataFrame) -> pd.Series:
    """Compute total SOFA score from hourly feature matrix.

    Uses available components: respiratory, coagulation, liver, cardiovascular, renal.
    CNS (GCS) component is omitted as it requires separate extraction.

    Returns Series of SOFA scores aligned with hourly_features index.
    """
    components = []

    # Respiratory
    if "pao2" in hourly_features.columns and "fio2" in hourly_features.columns:
        components.append(
            compute_sofa_respiratory(hourly_features["pao2"], hourly_features["fio2"])
        )

    # Coagulation
    if "platelets" in hourly_features.columns:
        components.append(compute_sofa_coagulation(hourly_features["platelets"]))

    # Liver
    if "bilirubin_total" in hourly_features.columns:
        components.append(compute_sofa_liver(hourly_features["bilirubin_total"]))

    # Cardiovascular
    if "map" in hourly_features.columns:
        components.append(compute_sofa_cardiovascular(hourly_features["map"]))

    # Renal
    if "creatinine" in hourly_features.columns:
        components.append(compute_sofa_renal(hourly_features["creatinine"]))

    if not components:
        return pd.Series(0, index=hourly_features.index, dtype=int)

    return sum(components)


def extract_suspected_infection_times(
    data_dir: Path,
) -> pd.DataFrame:
    """Extract suspected infection times from prescriptions and microbiologyevents.

    Suspected infection = earliest of (antibiotics order, culture order).

    Returns DataFrame with columns: subject_id, hadm_id, infection_time.
    """
    infection_times = []

    # Try prescriptions (antibiotics)
    try:
        prescriptions = load_mimic_table(data_dir, "prescriptions")
        # Common antibiotic drug types in MIMIC-IV
        abx_keywords = [
            "cillin",
            "cephalosporin",
            "mycin",
            "cycline",
            "floxacin",
            "sulfa",
            "metronidazole",
            "vancomycin",
            "meropenem",
            "imipenem",
            "azithromycin",
            "levofloxacin",
            "ciprofloxacin",
            "piperacillin",
            "ceftriaxone",
            "cefepime",
            "ampicillin",
            "amoxicillin",
            "doxycycline",
            "trimethoprim",
            "linezolid",
            "daptomycin",
        ]
        drug_col = "drug" if "drug" in prescriptions.columns else "drug_name_generic"
        if drug_col in prescriptions.columns:
            pattern = "|".join(abx_keywords)
            abx = prescriptions[
                prescriptions[drug_col].str.lower().str.contains(pattern, na=False)
            ].copy()
            time_col = "starttime" if "starttime" in abx.columns else "startdate"
            abx["abx_time"] = pd.to_datetime(abx[time_col])
            abx_earliest = (
                abx.groupby(["subject_id", "hadm_id"])["abx_time"].min().reset_index()
            )
            infection_times.append(abx_earliest.rename(columns={"abx_time": "time"}))
    except FileNotFoundError:
        logger.warning("prescriptions table not found, skipping antibiotics")

    # Try microbiologyevents (cultures)
    try:
        micro = load_mimic_table(data_dir, "microbiologyevents")
        time_col = "charttime" if "charttime" in micro.columns else "chartdate"
        micro["culture_time"] = pd.to_datetime(micro[time_col])
        culture_earliest = (
            micro.groupby(["subject_id", "hadm_id"])["culture_time"]
            .min()
            .reset_index()
        )
        infection_times.append(
            culture_earliest.rename(columns={"culture_time": "time"})
        )
    except FileNotFoundError:
        logger.warning("microbiologyevents table not found, skipping cultures")

    if not infection_times:
        return pd.DataFrame(columns=["subject_id", "hadm_id", "infection_time"])

    combined = pd.concat(infection_times, ignore_index=True)
    # Take earliest time per admission
    result = combined.groupby(["subject_id", "hadm_id"])["time"].min().reset_index()
    result = result.rename(columns={"time": "infection_time"})
    return result


def label_sepsis(
    hourly_features: pd.DataFrame,
    infection_times: pd.DataFrame,
    admissions: pd.DataFrame,
    sofa_threshold: int = SOFA_THRESHOLD,
    window_hours: float = 48.0,
) -> pd.DataFrame:
    """Label admissions with Sepsis-3 onset.

    Sepsis onset = first time SOFA >= threshold within `window_hours` of
    suspected infection time.

    Parameters
    ----------
    hourly_features : Hourly feature matrix with subject_id, hadm_id, hours_in.
    infection_times : DataFrame with subject_id, hadm_id, infection_time.
    admissions : Admissions table with admittime.
    sofa_threshold : SOFA score threshold (default 2).
    window_hours : Hours around infection time to check SOFA (default 48).

    Returns
    -------
    DataFrame with columns: subject_id, hadm_id, sepsis_label (0/1),
    sepsis_onset_hour (hours from admission, NaN if no sepsis).
    """
    sofa = compute_sofa_scores(hourly_features)
    hourly_features = hourly_features.copy()
    hourly_features["sofa"] = sofa

    # Merge infection times
    admissions_slim = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    admissions_slim["admittime"] = pd.to_datetime(admissions_slim["admittime"])
    infection = infection_times.copy()
    infection["infection_time"] = pd.to_datetime(infection["infection_time"])
    infection = infection.merge(admissions_slim, on=["subject_id", "hadm_id"])
    infection["infection_hour"] = (
        infection["infection_time"] - infection["admittime"]
    ).dt.total_seconds() / 3600

    labels = []
    all_admissions = hourly_features.groupby(["subject_id", "hadm_id"])

    for (sid, hid), group in all_admissions:
        inf_row = infection[
            (infection["subject_id"] == sid) & (infection["hadm_id"] == hid)
        ]

        if inf_row.empty:
            # No suspected infection -> no sepsis
            labels.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "sepsis_label": 0,
                    "sepsis_onset_hour": np.nan,
                }
            )
            continue

        inf_hour = inf_row["infection_hour"].iloc[0]

        # Check SOFA within window around infection
        window_mask = (group["hours_in"] >= inf_hour - window_hours) & (
            group["hours_in"] <= inf_hour + window_hours
        )
        window_data = group[window_mask]

        sofa_met = window_data[window_data["sofa"] >= sofa_threshold]
        if sofa_met.empty:
            labels.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "sepsis_label": 0,
                    "sepsis_onset_hour": np.nan,
                }
            )
        else:
            onset_hour = sofa_met["hours_in"].min()
            labels.append(
                {
                    "subject_id": sid,
                    "hadm_id": hid,
                    "sepsis_label": 1,
                    "sepsis_onset_hour": onset_hour,
                }
            )

    return pd.DataFrame(labels)


def run_labeling(
    data_dir: Path = DATA_DIR,
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Run the full labeling pipeline.

    Requires hourly_features.parquet from data extraction step.
    """
    output_path = output_dir / "sepsis_labels.parquet"
    if output_path.exists() and not force:
        logger.info("Loading cached labels from %s", output_path)
        return pd.read_parquet(output_path)

    # Load hourly features
    features_path = output_dir / "hourly_features.parquet"
    if not features_path.exists():
        raise FileNotFoundError(
            f"Run data extraction first: {features_path} not found"
        )
    hourly_features = pd.read_parquet(features_path)

    admissions = load_mimic_table(data_dir, "admissions")
    infection_times = extract_suspected_infection_times(data_dir)
    logger.info("Found infection times for %d admissions", len(infection_times))

    labels = label_sepsis(hourly_features, infection_times, admissions)

    n_sepsis = labels["sepsis_label"].sum()
    n_total = len(labels)
    logger.info(
        "Labeled %d admissions: %d sepsis (%.1f%%)",
        n_total,
        n_sepsis,
        100 * n_sepsis / max(n_total, 1),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    labels.to_parquet(output_path, index=False)
    logger.info("Saved labels to %s", output_path)

    return labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Label admissions with Sepsis-3")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    labels = run_labeling(args.data_dir, args.output_dir, force=args.force)

    print(f"\n=== Labeling Complete ===")
    print(f"Total admissions: {len(labels)}")
    print(f"Sepsis: {labels['sepsis_label'].sum()}")
    print(f"No sepsis: {(labels['sepsis_label'] == 0).sum()}")


if __name__ == "__main__":
    main()
