"""False positive extraction framework.

Loads model prediction outputs and extracts false positive cases (AI predicted
disease, ground truth healthy) at configurable operating points. Works generically
across disease domains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/diagnostics/false-positive-biomarker-mining")


class PredictionSource(Protocol):
    """Protocol for domain-specific prediction adapters."""

    @property
    def domain(self) -> str:
        """Disease domain name (e.g., 'sepsis', 'crc')."""
        ...

    def load_predictions(self) -> pd.DataFrame:
        """Load predictions with columns: sample_id, y_true, y_score, + feature columns.

        Returns:
            DataFrame with at minimum:
                - sample_id: unique identifier for each sample/patient
                - y_true: binary ground truth label (1=disease, 0=healthy)
                - y_score: model prediction score (higher = more likely disease)
                - Additional columns treated as feature vectors
        """
        ...


@dataclass
class OperatingPoint:
    """A threshold operating point for binary classification."""

    name: str
    threshold: float
    specificity: float
    sensitivity: float


@dataclass
class ExtractionResult:
    """Result of false positive extraction at a given operating point."""

    domain: str
    operating_point: OperatingPoint
    false_positives: pd.DataFrame
    true_negatives: pd.DataFrame
    true_positives: pd.DataFrame
    false_negatives: pd.DataFrame
    summary: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_total = (
            len(self.false_positives)
            + len(self.true_negatives)
            + len(self.true_positives)
            + len(self.false_negatives)
        )
        self.summary = {
            "domain": self.domain,
            "operating_point": self.operating_point.name,
            "threshold": self.operating_point.threshold,
            "specificity": self.operating_point.specificity,
            "sensitivity": self.operating_point.sensitivity,
            "n_total": n_total,
            "n_fp": len(self.false_positives),
            "n_tn": len(self.true_negatives),
            "n_tp": len(self.true_positives),
            "n_fn": len(self.false_negatives),
            "fp_rate": len(self.false_positives) / max(1, len(self.false_positives) + len(self.true_negatives)),
        }


def find_threshold_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float,
) -> tuple[float, float, float]:
    """Find the decision threshold that achieves the target specificity.

    Args:
        y_true: Binary ground truth labels.
        y_score: Prediction scores.
        target_specificity: Desired specificity (e.g., 0.95).

    Returns:
        Tuple of (threshold, actual_specificity, sensitivity).
    """
    negatives = y_score[y_true == 0]
    positives = y_score[y_true == 1]

    if len(negatives) == 0 or len(positives) == 0:
        raise ValueError("Need both positive and negative samples to find threshold")

    # Threshold = percentile of negative scores such that (1 - target_specificity)
    # fraction of negatives fall above it
    threshold = float(np.percentile(negatives, target_specificity * 100))

    actual_spec = float(np.mean(negatives <= threshold))
    sensitivity = float(np.mean(positives > threshold))

    return threshold, actual_spec, sensitivity


def extract_false_positives(
    predictions: pd.DataFrame,
    threshold: float,
    score_col: str = "y_score",
    label_col: str = "y_true",
    id_col: str = "sample_id",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split predictions into TP, TN, FP, FN at a given threshold.

    Args:
        predictions: DataFrame with id, score, label, and feature columns.
        threshold: Decision threshold (predict positive if score > threshold).
        score_col: Column name for prediction scores.
        label_col: Column name for ground truth labels.
        id_col: Column name for sample identifiers.

    Returns:
        Tuple of (false_positives, true_negatives, true_positives, false_negatives).
    """
    y_pred = (predictions[score_col] > threshold).astype(int)
    y_true = predictions[label_col].astype(int)

    fp_mask = (y_pred == 1) & (y_true == 0)
    tn_mask = (y_pred == 0) & (y_true == 0)
    tp_mask = (y_pred == 1) & (y_true == 1)
    fn_mask = (y_pred == 0) & (y_true == 1)

    return (
        predictions[fp_mask].copy(),
        predictions[tn_mask].copy(),
        predictions[tp_mask].copy(),
        predictions[fn_mask].copy(),
    )


def get_feature_columns(df: pd.DataFrame, exclude: set[str] | None = None) -> list[str]:
    """Get feature columns from a predictions DataFrame.

    Excludes metadata columns (sample_id, y_true, y_score) and any additional
    columns specified in the exclude set.
    """
    default_exclude = {"sample_id", "y_true", "y_score"}
    if exclude:
        default_exclude |= exclude
    return [c for c in df.columns if c not in default_exclude]


def extract_at_operating_points(
    source: PredictionSource,
    specificities: list[float] | None = None,
) -> list[ExtractionResult]:
    """Extract false positives at multiple operating points from a prediction source.

    Args:
        source: A PredictionSource adapter for a specific disease domain.
        specificities: Target specificities (default: [0.90, 0.95, 0.99]).

    Returns:
        List of ExtractionResult, one per operating point.
    """
    if specificities is None:
        specificities = [0.90, 0.95, 0.99]

    predictions = source.load_predictions()

    required_cols = {"sample_id", "y_true", "y_score"}
    missing = required_cols - set(predictions.columns)
    if missing:
        raise ValueError(f"Predictions missing required columns: {missing}")

    logger.info(
        "Loaded %d predictions from %s (pos=%d, neg=%d)",
        len(predictions),
        source.domain,
        int(predictions["y_true"].sum()),
        int((predictions["y_true"] == 0).sum()),
    )

    results = []
    for spec in specificities:
        threshold, actual_spec, sensitivity = find_threshold_at_specificity(
            predictions["y_true"].values,
            predictions["y_score"].values,
            spec,
        )

        op = OperatingPoint(
            name=f"spec_{spec:.0%}".replace("%", "pct"),
            threshold=threshold,
            specificity=actual_spec,
            sensitivity=sensitivity,
        )

        fp, tn, tp, fn = extract_false_positives(predictions, threshold)

        result = ExtractionResult(
            domain=source.domain,
            operating_point=op,
            false_positives=fp,
            true_negatives=tn,
            true_positives=tp,
            false_negatives=fn,
        )

        logger.info(
            "%s @ %s: threshold=%.4f, FP=%d, TN=%d, TP=%d, FN=%d",
            source.domain,
            op.name,
            threshold,
            len(fp),
            len(tn),
            len(tp),
            len(fn),
        )

        results.append(result)

    return results


def save_extraction(result: ExtractionResult, output_dir: Path | None = None) -> Path:
    """Save extraction result to disk.

    Args:
        result: ExtractionResult to save.
        output_dir: Base output directory. Defaults to OUTPUT_DIR.

    Returns:
        Path to the saved directory.
    """
    base = output_dir or OUTPUT_DIR
    save_dir = base / result.domain / result.operating_point.name
    save_dir.mkdir(parents=True, exist_ok=True)

    result.false_positives.to_parquet(save_dir / "false_positives.parquet", index=False)
    result.true_negatives.to_parquet(save_dir / "true_negatives.parquet", index=False)

    summary_df = pd.DataFrame([result.summary])
    summary_df.to_csv(save_dir / "summary.csv", index=False)

    logger.info("Saved extraction to %s", save_dir)
    return save_dir
