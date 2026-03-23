"""Data adapter for crc-liquid-biopsy-panel prediction outputs.

Loads CRC classifier predictions and methylation features for FP mining.
Also provides a TCGA-COAD/READ stage adapter for pre-clinical gradient analysis.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CRC_OUTPUT_DIR = Path("output/diagnostics/crc-liquid-biopsy-panel")


class CrcAdapter:
    """Adapter to load crc-liquid-biopsy-panel predictions for FP mining.

    Implements the PredictionSource protocol from extract.py.

    The CRC project stores fold-level aggregate results rather than per-sample
    predictions. This adapter reconstructs sample-level data from the available
    outputs (stage-stratified results + feature matrices).
    """

    domain = "crc"

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or CRC_OUTPUT_DIR

    def load_predictions(self) -> pd.DataFrame:
        """Load CRC classifier predictions with features.

        Prefers real per-sample predictions from the ensemble classifier
        (per_sample_predictions.parquet). Falls back to reconstructing
        synthetic sample-level data from stage-stratified confusion matrices.

        Raises:
            FileNotFoundError: If required output files are missing.
        """
        predictions_path = self.output_dir / "per_sample_predictions.parquet"
        if predictions_path.exists():
            return self._load_real_predictions(predictions_path)
        return self._load_reconstructed_predictions()

    def _load_real_predictions(self, path: Path) -> pd.DataFrame:
        """Load real per-sample predictions exported by the ensemble classifier."""
        df = pd.read_parquet(path)
        logger.info(
            "Loaded %d real per-sample CRC predictions (%d cols) from %s",
            len(df), len(df.columns), path,
        )
        return df

    def _load_reconstructed_predictions(self) -> pd.DataFrame:
        """Reconstruct sample-level predictions from stage-stratified results.

        Fallback when per_sample_predictions.parquet is not yet available.
        """
        stage_path = self.output_dir / "stage_stratified_results.parquet"
        classifier_path = self.output_dir / "classifier_results.json"

        for path in [stage_path, classifier_path]:
            if not path.exists():
                raise FileNotFoundError(
                    f"CRC output not found: {path}. "
                    f"The crc-liquid-biopsy-panel pipeline must be run first."
                )

        stage_results = pd.read_parquet(stage_path)

        samples = []
        for _, stage_row in stage_results.iterrows():
            stage = stage_row.get("stage", "Unknown")
            stage_num = int(stage_row.get("stage_numeric", 0))

            if stage_num < 0:
                continue

            n_tp = int(stage_row.get("tp", 0))
            n_fn = int(stage_row.get("fn", 0))
            n_tn = int(stage_row.get("tn", 0))
            n_fp = int(stage_row.get("fp", 0))

            rng = np.random.default_rng(stage_num)

            for i in range(n_tp):
                samples.append({
                    "sample_id": f"crc_{stage}_{i:04d}_tp",
                    "y_true": 1,
                    "y_score": float(rng.beta(5, 2)),
                    "stage": stage,
                    "stage_numeric": stage_num,
                })

            for i in range(n_fn):
                samples.append({
                    "sample_id": f"crc_{stage}_{i:04d}_fn",
                    "y_true": 1,
                    "y_score": float(rng.beta(2, 5)),
                    "stage": stage,
                    "stage_numeric": stage_num,
                })

            for i in range(n_tn):
                samples.append({
                    "sample_id": f"crc_{stage}_{i:04d}_tn",
                    "y_true": 0,
                    "y_score": float(rng.beta(2, 5)),
                    "stage": stage,
                    "stage_numeric": stage_num,
                })

            for i in range(n_fp):
                samples.append({
                    "sample_id": f"crc_{stage}_{i:04d}_fp",
                    "y_true": 0,
                    "y_score": float(rng.beta(5, 2)),
                    "stage": stage,
                    "stage_numeric": stage_num,
                })

        if not samples:
            raise ValueError("No samples reconstructed from stage-stratified results")

        df = pd.DataFrame(samples)
        logger.info(
            "Reconstructed %d CRC samples from stage-stratified results "
            "(pos=%d, neg=%d) — run ensemble_classifier with --force to "
            "generate real per-sample predictions",
            len(df),
            int(df["y_true"].sum()),
            int((df["y_true"] == 0).sum()),
        )
        return df


def create_mock_crc_data(
    n_samples: int = 300,
    cancer_rate: float = 0.3,
    n_features: int = 7,
    seed: int = 42,
) -> pd.DataFrame:
    """Create synthetic CRC prediction data for testing.

    Args:
        n_samples: Total samples.
        cancer_rate: Fraction with CRC.
        n_features: Number of biomarker features.
        seed: Random seed.

    Returns:
        DataFrame conforming to PredictionSource protocol.
    """
    rng = np.random.default_rng(seed)
    n_pos = int(n_samples * cancer_rate)
    n_neg = n_samples - n_pos

    neg_scores = rng.beta(2, 4, n_neg)
    pos_scores = rng.beta(4, 2, n_pos)

    labels = np.concatenate([np.zeros(n_neg), np.ones(n_pos)])
    scores = np.concatenate([neg_scores, pos_scores])

    # Simulated protein biomarker features
    feature_names = [
        "prot_MMP9", "prot_CXCL8", "prot_S100A4", "prot_CEA",
        "prot_CA19_9", "prot_TIMP1", "prot_IL6",
    ][:n_features]

    features = {}
    for name in feature_names:
        neg_vals = rng.normal(0, 1, n_neg)
        pos_vals = rng.normal(rng.uniform(0.3, 1.0), 1.2, n_pos)
        features[name] = np.concatenate([neg_vals, pos_vals])

    # Stage assignments for positive samples
    stages = np.concatenate([
        np.zeros(n_neg, dtype=int),  # Normal
        rng.choice([1, 2, 3, 4], n_pos, p=[0.15, 0.35, 0.30, 0.20]),
    ])

    df = pd.DataFrame({
        "sample_id": [f"crc_{i:04d}" for i in range(n_samples)],
        "y_true": labels.astype(int),
        "y_score": scores,
        "stage_numeric": stages,
        **features,
    })

    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


class MockCrcAdapter:
    """Mock CRC adapter for testing when real data is unavailable."""

    domain = "crc"

    def __init__(self, n_samples: int = 300, seed: int = 42) -> None:
        self.n_samples = n_samples
        self.seed = seed

    def load_predictions(self) -> pd.DataFrame:
        return create_mock_crc_data(
            n_samples=self.n_samples,
            seed=self.seed,
        )
