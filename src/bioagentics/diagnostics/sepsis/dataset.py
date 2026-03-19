"""Prediction window and dataset generator for sepsis prediction.

Given labeled admissions and feature matrices, generates prediction datasets
at 4h, 6h, 8h, 12h lookahead before sepsis onset. For controls, samples
random time-steps. Implements stratified train/test splitting.

Output: ready-to-train X, y arrays per lookahead window.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from bioagentics.diagnostics.sepsis.config import (
    ALL_FEATURES,
    LOOKAHEAD_HOURS,
    OUTPUT_DIR,
    RANDOM_STATE,
)

logger = logging.getLogger(__name__)


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Get all numeric feature columns (excluding IDs and metadata)."""
    exclude = {"subject_id", "hadm_id", "hours_in", "age", "sex", "ethnicity", "sofa"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return [c for c in numeric_cols if c not in exclude]


def extract_prediction_samples(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookahead: int,
    min_hours_in: int = 6,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract prediction samples at a given lookahead before sepsis onset.

    For sepsis-positive admissions:
        - Take the time-step at (onset_hour - lookahead). If not available,
          take the closest earlier time-step.
    For sepsis-negative admissions:
        - Sample a random time-step after min_hours_in.

    Parameters
    ----------
    features : Engineered feature matrix with subject_id, hadm_id, hours_in.
    labels : Sepsis labels with subject_id, hadm_id, sepsis_label, sepsis_onset_hour.
    lookahead : Hours before onset to extract features.
    min_hours_in : Minimum hours into admission for control sampling.
    rng : Random number generator.

    Returns
    -------
    X : Feature DataFrame (one row per admission).
    y : Binary labels Series.
    """
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    feat_cols = get_feature_columns(features)
    samples_x = []
    samples_y = []

    for _, row in labels.iterrows():
        sid = row["subject_id"]
        hid = row["hadm_id"]
        label = int(row["sepsis_label"])

        adm_data = features[
            (features["subject_id"] == sid) & (features["hadm_id"] == hid)
        ]
        if adm_data.empty:
            continue

        if label == 1:
            onset = row["sepsis_onset_hour"]
            target_hour = onset - lookahead
            if target_hour < min_hours_in:
                continue  # Not enough lead time
            # Find closest hour <= target_hour
            eligible = adm_data[adm_data["hours_in"] <= target_hour]
            if eligible.empty:
                continue
            sample = eligible.iloc[-1]
        else:
            # Control: sample random time-step after min_hours_in
            eligible = adm_data[adm_data["hours_in"] >= min_hours_in]
            if eligible.empty:
                continue
            idx = rng.integers(0, len(eligible))
            sample = eligible.iloc[idx]

        samples_x.append(sample[feat_cols])
        samples_y.append(label)

    if not samples_x:
        return pd.DataFrame(columns=feat_cols), pd.Series(dtype=int)

    X = pd.DataFrame(samples_x).reset_index(drop=True)
    y = pd.Series(samples_y, name="sepsis_label")
    return X, y


def generate_datasets(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    lookaheads: list[int] | None = None,
    test_size: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict[int, dict[str, np.ndarray]]:
    """Generate train/test datasets for each lookahead window.

    Parameters
    ----------
    features : Engineered feature matrix.
    labels : Sepsis labels.
    lookaheads : List of lookahead hours (default: [4, 6, 8, 12]).
    test_size : Fraction for test set (default: 0.2).
    rng : Random number generator.

    Returns
    -------
    Dictionary keyed by lookahead, each containing:
    {"X_train", "X_test", "y_train", "y_test", "feature_names"}.
    """
    if lookaheads is None:
        lookaheads = LOOKAHEAD_HOURS
    if rng is None:
        rng = np.random.default_rng(RANDOM_STATE)

    datasets = {}
    for lh in lookaheads:
        logger.info("Generating dataset for %dh lookahead", lh)
        X, y = extract_prediction_samples(features, labels, lh, rng=rng)

        if len(X) == 0 or y.sum() < 2 or (y == 0).sum() < 2:
            logger.warning(
                "Insufficient samples for %dh lookahead: %d total, %d positive",
                lh,
                len(y),
                int(y.sum()),
            )
            continue

        # Stratified split
        splitter = StratifiedShuffleSplit(
            n_splits=1, test_size=test_size, random_state=RANDOM_STATE
        )
        train_idx, test_idx = next(splitter.split(X, y))

        X_arr = X.to_numpy(dtype=np.float64, na_value=np.nan)
        y_arr = y.to_numpy(dtype=np.int64)

        datasets[lh] = {
            "X_train": X_arr[train_idx],
            "X_test": X_arr[test_idx],
            "y_train": y_arr[train_idx],
            "y_test": y_arr[test_idx],
            "feature_names": X.columns.tolist(),
        }
        logger.info(
            "  %dh: %d train (%d pos), %d test (%d pos)",
            lh,
            len(train_idx),
            y_arr[train_idx].sum(),
            len(test_idx),
            y_arr[test_idx].sum(),
        )

    return datasets


def run_dataset_generation(
    output_dir: Path = OUTPUT_DIR,
    force: bool = False,
) -> dict[int, dict[str, np.ndarray]]:
    """Run dataset generation pipeline.

    Reads engineered_features.parquet and sepsis_labels.parquet.
    Saves per-lookahead .npz files.
    """
    datasets_dir = output_dir / "datasets"

    # Check if already generated
    if not force and datasets_dir.exists():
        existing = list(datasets_dir.glob("dataset_*h.npz"))
        if len(existing) == len(LOOKAHEAD_HOURS):
            logger.info("Loading cached datasets from %s", datasets_dir)
            datasets = {}
            for lh in LOOKAHEAD_HOURS:
                path = datasets_dir / f"dataset_{lh}h.npz"
                if path.exists():
                    data = np.load(path, allow_pickle=True)
                    datasets[lh] = {k: data[k] for k in data.files}
            return datasets

    # Load inputs
    features_path = output_dir / "engineered_features.parquet"
    labels_path = output_dir / "sepsis_labels.parquet"
    for p in [features_path, labels_path]:
        if not p.exists():
            raise FileNotFoundError(f"Run previous pipeline steps first: {p}")

    features = pd.read_parquet(features_path)
    labels = pd.read_parquet(labels_path)

    datasets = generate_datasets(features, labels)

    # Save
    datasets_dir.mkdir(parents=True, exist_ok=True)
    for lh, data in datasets.items():
        path = datasets_dir / f"dataset_{lh}h.npz"
        np.savez(path, **data)
        logger.info("Saved %dh dataset to %s", lh, path)

    return datasets


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate prediction datasets")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    datasets = run_dataset_generation(args.output_dir, force=args.force)

    print(f"\n=== Dataset Generation Complete ===")
    for lh, data in sorted(datasets.items()):
        n_train = len(data["X_train"])
        n_test = len(data["X_test"])
        pos_train = data["y_train"].sum()
        pos_test = data["y_test"].sum()
        print(
            f"  {lh}h: train={n_train} ({pos_train} pos), "
            f"test={n_test} ({pos_test} pos), "
            f"features={data['X_train'].shape[1]}"
        )


if __name__ == "__main__":
    main()
