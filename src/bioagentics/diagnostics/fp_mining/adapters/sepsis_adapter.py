"""Data adapter for ehr-sepsis-early-warning prediction outputs.

Loads sepsis model predictions and maps them to the FP extraction framework.
The ehr-sepsis project is currently in development — this adapter defines the
interface and provides mock data for testing until real outputs are available.

MimicDemoSepsisAdapter loads directly from MIMIC-IV Clinical Database Demo v2.2,
applying Sepsis-3 labeling and a heuristic severity risk score to generate
per-stay predictions for the FP mining pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

SEPSIS_OUTPUT_DIR = Path("output/diagnostics/ehr-sepsis-early-warning")
SEPSIS_DATASETS_DIR = SEPSIS_OUTPUT_DIR / "datasets"

MIMIC_DEMO_DIR = Path(
    "data/diagnostics/false-positive-biomarker-mining/mimic_demo/flat"
)
MIMIC_DEMO_OUTPUT_DIR = Path(
    "output/diagnostics/false-positive-biomarker-mining/mimic_demo"
)


class SepsisAdapter:
    """Adapter to load ehr-sepsis-early-warning predictions for FP mining.

    Implements the PredictionSource protocol from extract.py.
    """

    domain = "sepsis"

    def __init__(
        self,
        output_dir: Path | None = None,
        lookahead_hours: int = 6,
    ) -> None:
        self.output_dir = output_dir or SEPSIS_OUTPUT_DIR
        self.lookahead_hours = lookahead_hours

    def load_predictions(self) -> pd.DataFrame:
        """Load sepsis model predictions with features.

        Combines model prediction scores from the ensemble classifier with
        the engineered feature matrix and sepsis labels. Returns a DataFrame
        conforming to the PredictionSource protocol.

        Raises:
            FileNotFoundError: If required output files are not yet available.
        """
        features_path = self.output_dir / "engineered_features.parquet"
        labels_path = self.output_dir / "sepsis_labels.parquet"
        results_path = self.output_dir / f"results/ensemble_{self.lookahead_hours}h.json"

        for path in [features_path, labels_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"Sepsis output not found: {path}. "
                    f"The ehr-sepsis-early-warning pipeline must be run first."
                )

        # Load labels
        labels = pd.read_parquet(labels_path)

        # Load features — use chunked reading for large feature matrices
        features = pd.read_parquet(features_path)

        # Create composite sample ID from subject + admission
        features["sample_id"] = (
            features["subject_id"].astype(str) + "_" + features["hadm_id"].astype(str)
        )

        # Take the latest time-step per admission for cross-sectional analysis
        latest_per_admission = (
            features.sort_values("hours_in")
            .groupby(["subject_id", "hadm_id"])
            .last()
            .reset_index()
        )

        # Merge with labels
        merged = latest_per_admission.merge(
            labels[["subject_id", "hadm_id", "sepsis_label"]],
            on=["subject_id", "hadm_id"],
            how="inner",
        )
        merged = merged.rename(columns={"sepsis_label": "y_true"})

        # Load ensemble predictions if available
        if results_path.exists():
            with open(results_path) as f:
                results = json.load(f)
            # Use fold-level predictions if stored
            if "predictions" in results:
                preds = pd.DataFrame(results["predictions"])
                merged = merged.merge(preds[["sample_id", "y_score"]], on="sample_id", how="left")
            else:
                logger.warning(
                    "No per-sample predictions in %s; using placeholder scores",
                    results_path,
                )
                merged["y_score"] = 0.5
        else:
            logger.warning(
                "Ensemble results not found at %s; using placeholder scores",
                results_path,
            )
            merged["y_score"] = 0.5

        # Select feature columns (exclude IDs and metadata)
        exclude = {"subject_id", "hadm_id", "hours_in", "sample_id", "y_true", "y_score"}
        feature_cols = [c for c in merged.columns if c not in exclude]

        result = merged[["sample_id", "y_true", "y_score"] + feature_cols].copy()
        logger.info(
            "Loaded %d sepsis predictions (%d features, %d positive)",
            len(result),
            len(feature_cols),
            int(result["y_true"].sum()),
        )
        return result


def create_mock_sepsis_data(
    n_admissions: int = 500,
    sepsis_rate: float = 0.15,
    n_features: int = 10,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic sepsis prediction data for testing.

    Generates data that mimics the schema of real sepsis model outputs,
    with realistic class imbalance and feature distributions.

    Args:
        n_admissions: Number of admissions to simulate.
        sepsis_rate: Fraction of admissions with sepsis.
        n_features: Number of clinical features.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame conforming to PredictionSource protocol.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n_admissions * sepsis_rate)
    n_neg = n_admissions - n_pos

    # Simulated prediction scores (overlap between classes)
    neg_scores = rng.beta(2, 5, n_neg)  # Skewed low
    pos_scores = rng.beta(5, 2, n_pos)  # Skewed high

    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    scores = np.concatenate([neg_scores, pos_scores])

    # Simulated clinical features with class-dependent means
    feature_names = [
        "heart_rate", "sbp", "resp_rate", "temperature", "wbc",
        "lactate", "creatinine", "platelets", "spo2", "map",
    ][:n_features]

    features = {}
    for i, name in enumerate(feature_names):
        neg_mean = rng.uniform(0.3, 0.7)
        pos_shift = rng.uniform(0.1, 0.4) * rng.choice([-1, 1])
        neg_vals = rng.normal(neg_mean, 0.15, n_neg)
        pos_vals = rng.normal(neg_mean + pos_shift, 0.2, n_pos)
        features[name] = np.concatenate([neg_vals, pos_vals])

    df = pd.DataFrame({
        "sample_id": [f"sepsis_{i:04d}" for i in range(n_admissions)],
        "y_true": labels.astype(int),
        "y_score": scores,
        **features,
    })

    # Shuffle
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class MockSepsisAdapter:
    """Mock sepsis adapter for testing when real data is unavailable."""

    domain = "sepsis"

    def __init__(self, n_admissions: int = 500, seed: int = 42) -> None:
        self.n_admissions = n_admissions
        self.seed = seed

    def load_predictions(self) -> pd.DataFrame:
        return create_mock_sepsis_data(
            n_admissions=self.n_admissions,
            seed=self.seed,
        )


# ---------------------------------------------------------------------------
# MIMIC-IV Demo adapter
# ---------------------------------------------------------------------------


def _load_demo_table(data_dir: Path, name: str) -> pd.DataFrame:
    """Load a table from the MIMIC-IV demo flat directory."""
    gz = data_dir / f"{name}.csv.gz"
    if gz.exists():
        return pd.read_csv(gz)
    csv = data_dir / f"{name}.csv"
    if csv.exists():
        return pd.read_csv(csv)
    raise FileNotFoundError(f"Table {name} not found in {data_dir}")


def _extract_stay_features(
    chartevents: pd.DataFrame,
    labevents: pd.DataFrame,
    icustays: pd.DataFrame,
) -> pd.DataFrame:
    """Extract per-stay summary features from chartevents and labevents.

    Computes median values for key vitals and labs over each ICU stay,
    returning one row per stay_id.
    """
    from bioagentics.diagnostics.sepsis.config import LAB_ITEMIDS, VITAL_ITEMIDS

    # Build itemid -> feature name maps
    vital_map: dict[int, str] = {}
    for feat, ids in VITAL_ITEMIDS.items():
        for iid in ids:
            vital_map[iid] = feat

    lab_map: dict[int, str] = {}
    for feat, ids in LAB_ITEMIDS.items():
        for iid in ids:
            lab_map[iid] = feat

    # --- Vitals from chartevents ---
    all_vital_ids = list(vital_map.keys())
    cv = chartevents[chartevents["itemid"].isin(all_vital_ids)].copy()
    cv["feature"] = cv["itemid"].map(vital_map)
    cv["valuenum"] = pd.to_numeric(cv["valuenum"], errors="coerce")
    cv = cv.dropna(subset=["valuenum"])

    # Temperature F -> C
    from bioagentics.diagnostics.sepsis.config import TEMP_FAHRENHEIT_ITEMIDS

    f_mask = cv["itemid"].isin(TEMP_FAHRENHEIT_ITEMIDS)
    cv.loc[f_mask, "valuenum"] = (cv.loc[f_mask, "valuenum"] - 32) * 5 / 9

    # Aggregate per stay
    vital_agg = (
        cv.groupby(["stay_id", "feature"])["valuenum"]
        .median()
        .reset_index()
        .pivot(index="stay_id", columns="feature", values="valuenum")
    )
    vital_agg.columns.name = None

    # --- Labs from labevents ---
    all_lab_ids = list(lab_map.keys())
    lv = labevents[labevents["itemid"].isin(all_lab_ids)].copy()
    lv["feature"] = lv["itemid"].map(lab_map)
    lv["valuenum"] = pd.to_numeric(lv["valuenum"], errors="coerce")
    lv = lv.dropna(subset=["valuenum"])

    # Labs don't have stay_id directly; join via hadm_id + time window
    lv["charttime"] = pd.to_datetime(lv["charttime"])
    icu_times = icustays[["stay_id", "hadm_id", "intime", "outtime"]].copy()
    icu_times["intime"] = pd.to_datetime(icu_times["intime"])
    icu_times["outtime"] = pd.to_datetime(icu_times["outtime"])

    lv_merged = lv.merge(icu_times, on="hadm_id", how="inner")
    lv_merged = lv_merged[
        (lv_merged["charttime"] >= lv_merged["intime"])
        & (lv_merged["charttime"] <= lv_merged["outtime"])
    ]

    lab_agg = (
        lv_merged.groupby(["stay_id", "feature"])["valuenum"]
        .median()
        .reset_index()
        .pivot(index="stay_id", columns="feature", values="valuenum")
    )
    lab_agg.columns.name = None

    # Merge vitals + labs
    features = vital_agg.join(lab_agg, how="outer")

    # Join with icustays metadata
    icu_meta = icustays[["stay_id", "subject_id", "hadm_id"]].set_index("stay_id")
    features = icu_meta.join(features, how="left").reset_index()

    return features


def _compute_sepsis3_labels(
    data_dir: Path,
    icustays: pd.DataFrame,
    stay_features: pd.DataFrame,
) -> pd.DataFrame:
    """Apply Sepsis-3 criteria to label each ICU stay.

    Uses SOFA scoring from the labeling module and suspected infection times
    from prescriptions/microbiology. Falls back to ICD-coded sepsis diagnoses
    when clinical data is insufficient.

    Returns DataFrame with: stay_id, subject_id, hadm_id, sepsis_label, max_sofa.
    """
    from bioagentics.diagnostics.sepsis.labeling import (
        compute_sofa_coagulation,
        compute_sofa_cardiovascular,
        compute_sofa_liver,
        compute_sofa_renal,
        compute_sofa_respiratory,
    )

    df = stay_features.set_index("stay_id").copy()

    # Compute per-component SOFA from stay-level features
    sofa = pd.Series(0, index=df.index, dtype=int)
    if "pao2" in df.columns and "fio2" in df.columns:
        sofa = sofa + compute_sofa_respiratory(df["pao2"], df["fio2"])
    if "platelets" in df.columns:
        sofa = sofa + compute_sofa_coagulation(df["platelets"])
    if "bilirubin_total" in df.columns:
        sofa = sofa + compute_sofa_liver(df["bilirubin_total"])
    if "map" in df.columns:
        sofa = sofa + compute_sofa_cardiovascular(df["map"])
    if "creatinine" in df.columns:
        sofa = sofa + compute_sofa_renal(df["creatinine"])

    df["max_sofa"] = sofa

    # Suspected infection: check prescriptions and microbiology
    has_infection = set()
    try:
        prescriptions = _load_demo_table(data_dir, "prescriptions")
        abx_keywords = [
            "cillin", "mycin", "cycline", "floxacin", "metronidazole",
            "vancomycin", "meropenem", "imipenem", "azithromycin",
            "piperacillin", "ceftriaxone", "cefepime", "ampicillin",
            "trimethoprim", "linezolid", "daptomycin",
        ]
        drug_col = "drug" if "drug" in prescriptions.columns else "drug_name_generic"
        if drug_col in prescriptions.columns:
            pattern = "|".join(abx_keywords)
            abx = prescriptions[
                prescriptions[drug_col].str.lower().str.contains(pattern, na=False)
            ]
            has_infection |= set(abx["hadm_id"].unique())
    except FileNotFoundError:
        pass

    try:
        micro = _load_demo_table(data_dir, "microbiologyevents")
        has_infection |= set(micro["hadm_id"].unique())
    except FileNotFoundError:
        pass

    # Also check ICD-coded sepsis diagnoses as supplementary signal
    icd_sepsis_hadm = set()
    try:
        diag = _load_demo_table(data_dir, "diagnoses_icd")
        sepsis_icd9 = {"99591", "99592", "78552", "78559"}
        sepsis_icd10_prefix = ("A40", "A41", "R652")
        mask9 = (diag["icd_version"] == 9) & diag["icd_code"].astype(str).isin(sepsis_icd9)
        mask10 = (diag["icd_version"] == 10) & diag["icd_code"].astype(str).str.startswith(
            sepsis_icd10_prefix
        )
        icd_sepsis_hadm = set(diag[mask9 | mask10]["hadm_id"].unique())
    except FileNotFoundError:
        pass

    # Sepsis-3: SOFA >= 2 AND (suspected infection OR ICD sepsis)
    icu_hadm = icustays.set_index("stay_id")["hadm_id"]
    stay_hadm = icu_hadm.reindex(df.index)

    infected = stay_hadm.isin(has_infection) | stay_hadm.isin(icd_sepsis_hadm)
    sofa_met = df["max_sofa"] >= 2

    df["sepsis_label"] = (infected & sofa_met).astype(int)

    result = df[["sepsis_label", "max_sofa"]].reset_index()
    result = result.merge(
        icustays[["stay_id", "subject_id", "hadm_id"]], on="stay_id", how="left"
    )
    return result


def _compute_heuristic_risk_score(features: pd.DataFrame) -> pd.Series:
    """Compute a heuristic sepsis risk score from clinical features.

    Combines normalized SOFA-like components with vital sign and lab
    abnormality indicators through a logistic transform to produce a
    calibrated [0, 1] risk score.

    Handles missing data by computing the score from available features only,
    normalizing by the per-row count of non-NaN components.
    """
    components = []

    # Each component produces a 0-1 value (NaN where input is NaN)
    component_defs = [
        ("heart_rate", lambda s: (s.clip(60, 150) - 60) / 90),
        ("resp_rate", lambda s: (s.clip(10, 40) - 10) / 30),
        ("temperature", lambda s: (s - 37).abs().clip(0, 3) / 3),
        ("sbp", lambda s: (130 - s.clip(60, 130)) / 70),
        ("map", lambda s: (85 - s.clip(50, 85)) / 35),
        ("spo2", lambda s: (100 - s.clip(80, 100)) / 20),
        ("wbc", lambda s: ((s - 8).abs() - 4).clip(0, 15) / 15),
        ("lactate", lambda s: s.clip(0, 8) / 8),
        ("creatinine", lambda s: (s.clip(0.5, 5) - 0.5) / 4.5),
        ("platelets", lambda s: (300 - s.clip(20, 300)) / 280),
        ("bilirubin_total", lambda s: s.clip(0, 12) / 12),
    ]

    for col, fn in component_defs:
        if col in features.columns:
            components.append(fn(features[col]))

    if not components:
        return pd.Series(0.5, index=features.index)

    # Stack into a DataFrame, then nanmean across components
    comp_df = pd.concat(components, axis=1)
    score = comp_df.mean(axis=1)  # nanmean by default

    # Logistic transform for calibration: map [0, 1] raw → [0, 1] calibrated
    # Center at 0.35 (most patients are low-risk)
    logit = 8 * (score - 0.35)
    calibrated = 1 / (1 + np.exp(-logit))

    return calibrated.fillna(0.5)


class MimicDemoSepsisAdapter:
    """Adapter that loads MIMIC-IV demo data directly for FP mining.

    Extracts vitals/labs from chartevents and labevents, applies Sepsis-3
    labeling, and computes a heuristic severity-based risk score as a
    stand-in for a trained sepsis model.

    Implements the PredictionSource protocol.
    """

    domain = "mimic_demo_sepsis"

    def __init__(
        self,
        data_dir: Path | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.data_dir = data_dir or MIMIC_DEMO_DIR
        self.output_dir = output_dir or MIMIC_DEMO_OUTPUT_DIR

    def load_predictions(self) -> pd.DataFrame:
        """Load MIMIC-IV demo data, label with Sepsis-3, score with heuristic.

        Returns PredictionSource-compliant DataFrame with:
            sample_id, y_true, y_score, + clinical feature columns.

        Also exports per_stay_predictions.parquet with stay-level detail.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"MIMIC-IV demo data not found at {self.data_dir}. "
                f"Download from PhysioNet and place in the expected location."
            )

        logger.info("Loading MIMIC-IV demo tables from %s", self.data_dir)

        # Load required tables
        icustays = _load_demo_table(self.data_dir, "icustays")
        chartevents = _load_demo_table(self.data_dir, "chartevents")
        labevents = _load_demo_table(self.data_dir, "labevents")
        admissions = _load_demo_table(self.data_dir, "admissions")

        logger.info(
            "Loaded %d ICU stays, %d chart rows, %d lab rows",
            len(icustays),
            len(chartevents),
            len(labevents),
        )

        # Extract per-stay features
        stay_features = _extract_stay_features(chartevents, labevents, icustays)
        logger.info("Extracted features for %d stays", len(stay_features))

        # Compute Sepsis-3 labels
        labels = _compute_sepsis3_labels(
            self.data_dir, icustays, stay_features
        )

        # Merge labels into features
        merged = stay_features.merge(
            labels[["stay_id", "sepsis_label", "max_sofa"]],
            on="stay_id",
            how="left",
        )
        merged["sepsis_label"] = merged["sepsis_label"].fillna(0).astype(int)

        # Compute heuristic risk score
        merged["y_score"] = _compute_heuristic_risk_score(merged)

        # Build per-stay predictions export
        merged["y_true"] = merged["sepsis_label"]
        merged["sample_id"] = (
            merged["subject_id"].astype(str)
            + "_"
            + merged["hadm_id"].astype(str)
        )

        # Export per_stay_predictions.parquet (full detail with stay_id)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        export_cols = ["subject_id", "hadm_id", "stay_id", "y_true", "y_score"]
        feature_cols = [
            c
            for c in merged.columns
            if c
            not in {
                "subject_id", "hadm_id", "stay_id", "sample_id",
                "y_true", "y_score", "sepsis_label", "max_sofa",
            }
        ]
        export = merged[export_cols + feature_cols].copy()
        export_path = self.output_dir / "per_stay_predictions.parquet"
        export.to_parquet(export_path, index=False)
        logger.info("Exported %d stay predictions to %s", len(export), export_path)

        # For the FP mining pipeline: one row per admission (max risk across stays)
        admission_preds = (
            merged.sort_values("y_score", ascending=False)
            .groupby(["subject_id", "hadm_id"])
            .first()
            .reset_index()
        )

        # Select columns for PredictionSource output
        result_cols = ["sample_id", "y_true", "y_score"] + feature_cols
        result = admission_preds[result_cols].copy()

        n_pos = int(result["y_true"].sum())
        logger.info(
            "MIMIC demo: %d admissions, %d features, %d sepsis-positive (%.1f%%)",
            len(result),
            len(feature_cols),
            n_pos,
            100 * n_pos / max(len(result), 1),
        )

        return result
