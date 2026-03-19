"""Data adapter for ehr-sepsis-early-warning prediction outputs.

Loads sepsis model predictions and maps them to the FP extraction framework.
The ehr-sepsis project is currently in development — this adapter defines the
interface and provides mock data for testing until real outputs are available.
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
