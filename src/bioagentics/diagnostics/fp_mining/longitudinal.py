"""Longitudinal outcome tracker for MIMIC-IV FP patients.

For patients flagged as false positives in ehr-sepsis predictions:
1. Look up subsequent admissions/encounters
2. Check if patient later developed sepsis or severe infection
3. Calculate time-to-event from FP flag to actual sepsis diagnosis

This is the core validation step for the hypothesis that FP predictions
contain pre-clinical disease signal.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.diagnostics.fp_mining.extract import ExtractionResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/diagnostics/false-positive-biomarker-mining/longitudinal")


def build_patient_timeline(
    admissions: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    """Build a longitudinal timeline of admissions with sepsis labels.

    Args:
        admissions: DataFrame with at minimum: subject_id, hadm_id, admittime.
        labels: DataFrame with: subject_id, hadm_id, sepsis_label, sepsis_onset_hour.

    Returns:
        DataFrame with admissions ordered by time per patient, annotated
        with sepsis labels.
    """
    required_adm = {"subject_id", "hadm_id", "admittime"}
    missing = required_adm - set(admissions.columns)
    if missing:
        raise ValueError(f"Admissions missing required columns: {missing}")

    timeline = admissions[["subject_id", "hadm_id", "admittime"]].copy()
    timeline["admittime"] = pd.to_datetime(timeline["admittime"])

    # Merge sepsis labels
    if "sepsis_label" in labels.columns:
        label_cols = ["subject_id", "hadm_id", "sepsis_label"]
        if "sepsis_onset_hour" in labels.columns:
            label_cols.append("sepsis_onset_hour")
        timeline = timeline.merge(labels[label_cols], on=["subject_id", "hadm_id"], how="left")
        timeline["sepsis_label"] = timeline["sepsis_label"].fillna(0).astype(int)
    else:
        timeline["sepsis_label"] = 0

    timeline = timeline.sort_values(["subject_id", "admittime"]).reset_index(drop=True)
    return timeline


def track_fp_outcomes(
    fp_result: ExtractionResult,
    timeline: pd.DataFrame,
) -> pd.DataFrame:
    """Track longitudinal outcomes for false positive patients.

    For each FP patient, finds subsequent admissions and checks whether
    sepsis developed later.

    Args:
        fp_result: ExtractionResult containing false positive predictions.
        timeline: Patient timeline from build_patient_timeline().

    Returns:
        DataFrame with columns:
            - subject_id: Patient identifier
            - fp_hadm_id: Admission where FP occurred
            - fp_admittime: Time of FP admission
            - subsequent_admissions: Count of later admissions
            - developed_sepsis: Whether sepsis occurred in any later admission
            - time_to_sepsis_days: Days from FP to first sepsis (NaN if none)
            - n_subsequent_sepsis: Number of subsequent sepsis episodes
    """
    fp = fp_result.false_positives

    if "sample_id" not in fp.columns:
        raise ValueError("FP data must include sample_id column")

    # Parse subject_id from composite sample_id (format: "{subject_id}_{hadm_id}")
    fp_patients = fp["sample_id"].copy()
    if fp_patients.str.contains("_").any():
        parts = fp_patients.str.rsplit("_", n=1, expand=True)
        fp_ids = pd.DataFrame({
            "subject_id": parts[0],
            "fp_hadm_id": parts[1],
        })
    else:
        fp_ids = pd.DataFrame({
            "subject_id": fp_patients,
            "fp_hadm_id": fp_patients,
        })

    # Ensure subject_id types match
    fp_ids["subject_id"] = fp_ids["subject_id"].astype(str)
    timeline = timeline.copy()
    timeline["subject_id"] = timeline["subject_id"].astype(str)
    timeline["hadm_id"] = timeline["hadm_id"].astype(str)

    outcomes = []
    for _, row in fp_ids.iterrows():
        sid = row["subject_id"]
        fp_hadm = row["fp_hadm_id"]

        # Get all admissions for this patient
        patient_timeline = timeline[timeline["subject_id"] == sid].sort_values("admittime")

        if len(patient_timeline) == 0:
            outcomes.append({
                "subject_id": sid,
                "fp_hadm_id": fp_hadm,
                "fp_admittime": pd.NaT,
                "subsequent_admissions": 0,
                "developed_sepsis": False,
                "time_to_sepsis_days": np.nan,
                "n_subsequent_sepsis": 0,
            })
            continue

        # Find the FP admission
        fp_adm = patient_timeline[patient_timeline["hadm_id"] == fp_hadm]
        if len(fp_adm) == 0:
            fp_time = patient_timeline["admittime"].iloc[0]
        else:
            fp_time = fp_adm["admittime"].iloc[0]

        # Find subsequent admissions
        subsequent = patient_timeline[patient_timeline["admittime"] > fp_time]
        n_subsequent = len(subsequent)

        # Check for subsequent sepsis
        subsequent_sepsis = subsequent[subsequent["sepsis_label"] == 1]
        developed = len(subsequent_sepsis) > 0

        time_to_sepsis = np.nan
        if developed:
            first_sepsis_time = subsequent_sepsis["admittime"].iloc[0]
            time_to_sepsis = (first_sepsis_time - fp_time).total_seconds() / 86400.0

        outcomes.append({
            "subject_id": sid,
            "fp_hadm_id": fp_hadm,
            "fp_admittime": fp_time,
            "subsequent_admissions": n_subsequent,
            "developed_sepsis": developed,
            "time_to_sepsis_days": time_to_sepsis,
            "n_subsequent_sepsis": len(subsequent_sepsis),
        })

    return pd.DataFrame(outcomes)


def compute_hazard_ratio(
    fp_outcomes: pd.DataFrame,
    tn_outcomes: pd.DataFrame,
) -> dict:
    """Compute hazard ratio for sepsis in FP vs TN patients.

    Uses simple incidence rate ratio as a proxy for hazard ratio when
    survival analysis libraries are unavailable.

    Args:
        fp_outcomes: Longitudinal outcomes for FP patients.
        tn_outcomes: Longitudinal outcomes for TN patients.

    Returns:
        Dict with incidence rates and ratio.
    """
    fp_rate = fp_outcomes["developed_sepsis"].mean() if len(fp_outcomes) > 0 else 0.0
    tn_rate = tn_outcomes["developed_sepsis"].mean() if len(tn_outcomes) > 0 else 0.0

    ratio = fp_rate / max(tn_rate, 1e-10)

    result = {
        "fp_sepsis_rate": float(fp_rate),
        "tn_sepsis_rate": float(tn_rate),
        "incidence_rate_ratio": float(ratio),
        "n_fp": len(fp_outcomes),
        "n_tn": len(tn_outcomes),
        "n_fp_developed": int(fp_outcomes["developed_sepsis"].sum()) if len(fp_outcomes) > 0 else 0,
        "n_tn_developed": int(tn_outcomes["developed_sepsis"].sum()) if len(tn_outcomes) > 0 else 0,
        "hypothesis_supported": ratio >= 2.0,  # >= 2x baseline incidence
    }

    logger.info(
        "FP sepsis rate=%.3f, TN sepsis rate=%.3f, IRR=%.2f (hypothesis %s)",
        fp_rate,
        tn_rate,
        ratio,
        "SUPPORTED" if result["hypothesis_supported"] else "not supported",
    )

    return result


def compute_time_to_event_stats(outcomes: pd.DataFrame) -> dict:
    """Compute time-to-event statistics for patients who developed sepsis.

    Returns:
        Dict with median, mean, and percentile time-to-sepsis in days.
    """
    developed = outcomes[outcomes["developed_sepsis"]]["time_to_sepsis_days"].dropna()

    if len(developed) == 0:
        return {
            "n_events": 0,
            "median_days": np.nan,
            "mean_days": np.nan,
        }

    return {
        "n_events": len(developed),
        "median_days": float(developed.median()),
        "mean_days": float(developed.mean()),
        "p25_days": float(developed.quantile(0.25)),
        "p75_days": float(developed.quantile(0.75)),
        "min_days": float(developed.min()),
        "max_days": float(developed.max()),
    }


def run_longitudinal_analysis(
    fp_result: ExtractionResult,
    tn_result_or_df: pd.DataFrame | ExtractionResult,
    admissions: pd.DataFrame,
    labels: pd.DataFrame,
    output_dir: Path | None = None,
) -> dict:
    """Run full longitudinal analysis pipeline.

    Args:
        fp_result: ExtractionResult containing false positives.
        tn_result_or_df: True negatives (ExtractionResult or DataFrame).
        admissions: MIMIC-IV admissions table.
        labels: Sepsis labels table.
        output_dir: Directory to save outputs.

    Returns:
        Dict with outcomes, hazard ratio, and time-to-event statistics.
    """
    save_dir = output_dir or OUTPUT_DIR
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build patient timeline
    timeline = build_patient_timeline(admissions, labels)

    # Track FP outcomes
    fp_outcomes = track_fp_outcomes(fp_result, timeline)

    # Track TN outcomes for comparison
    if isinstance(tn_result_or_df, ExtractionResult):
        tn_data = tn_result_or_df.true_negatives
    else:
        tn_data = tn_result_or_df

    # Create a synthetic ExtractionResult for TNs if needed
    from bioagentics.diagnostics.fp_mining.extract import ExtractionResult as ER, OperatingPoint
    tn_er = ER(
        domain=fp_result.domain,
        operating_point=fp_result.operating_point,
        false_positives=tn_data,  # Trick: put TNs in FP slot for tracking
        true_negatives=pd.DataFrame(),
        true_positives=pd.DataFrame(),
        false_negatives=pd.DataFrame(),
    )
    tn_outcomes = track_fp_outcomes(tn_er, timeline)

    # Compute hazard ratio
    hazard = compute_hazard_ratio(fp_outcomes, tn_outcomes)

    # Time-to-event stats
    fp_tte = compute_time_to_event_stats(fp_outcomes)
    tn_tte = compute_time_to_event_stats(tn_outcomes)

    # Save outputs
    domain = fp_result.domain
    op_name = fp_result.operating_point.name

    fp_outcomes.to_parquet(
        save_dir / f"{domain}_{op_name}_fp_outcomes.parquet",
        index=False,
    )
    tn_outcomes.to_parquet(
        save_dir / f"{domain}_{op_name}_tn_outcomes.parquet",
        index=False,
    )

    summary = {
        "domain": domain,
        "operating_point": op_name,
        "hazard_ratio": hazard,
        "fp_time_to_event": fp_tte,
        "tn_time_to_event": tn_tte,
    }

    logger.info(
        "%s @ %s: IRR=%.2f, FP events=%d/%d, TN events=%d/%d",
        domain,
        op_name,
        hazard["incidence_rate_ratio"],
        hazard["n_fp_developed"],
        hazard["n_fp"],
        hazard["n_tn_developed"],
        hazard["n_tn"],
    )

    return {
        "fp_outcomes": fp_outcomes,
        "tn_outcomes": tn_outcomes,
        "hazard_ratio": hazard,
        "fp_time_to_event": fp_tte,
        "tn_time_to_event": tn_tte,
        "summary": summary,
    }
