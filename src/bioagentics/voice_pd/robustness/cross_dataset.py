"""Cross-dataset generalization evaluation.

Trains a classifier on one dataset and evaluates on held-out datasets
to measure how well voice biomarker models generalize across recording
conditions, languages, and populations.
"""

import json
import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import EVAL_DIR

log = logging.getLogger(__name__)


def cross_dataset_evaluate(
    train_X: np.ndarray,
    train_y: np.ndarray,
    test_datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path | None = None,
) -> dict:
    """Train on one dataset and evaluate on multiple held-out datasets.

    Uses a gradient boosting classifier (matching the classical pipeline)
    trained on ``train_X``/``train_y``, then computes AUC on each test set.

    Args:
        train_X: Training features (n_train, n_features).
        train_y: Training labels (n_train,).
        test_datasets: Mapping of dataset name to (X, y) tuples.
        output_dir: Directory to save results JSON.

    Returns:
        Dict with per-dataset AUC, accuracy, and sample counts.
    """
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score

    if output_dir is None:
        output_dir = EVAL_DIR / "cross_dataset"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Train classifier
    clf = GradientBoostingClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
    )
    clf.fit(train_X, train_y)

    dataset_results: dict[str, dict] = {}

    for name, (test_X, test_y) in test_datasets.items():
        y_prob = clf.predict_proba(test_X)[:, 1]
        y_pred = clf.predict(test_X)

        n_classes = len(np.unique(test_y))
        auc = float(roc_auc_score(test_y, y_prob)) if n_classes >= 2 else None
        acc = float(accuracy_score(test_y, y_pred))

        dataset_results[name] = {
            "auc": auc,
            "accuracy": acc,
            "n_samples": len(test_y),
            "n_pd": int(np.sum(test_y == 1)),
            "n_healthy": int(np.sum(test_y == 0)),
        }
        log.info("  %s: AUC=%.4f  acc=%.4f (n=%d)", name, auc or 0, acc, len(test_y))

    results = {
        "datasets": dataset_results,
        "n_train": len(train_y),
    }

    results_path = output_dir / "cross_dataset_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Cross-dataset results saved to %s", results_path)

    return results
