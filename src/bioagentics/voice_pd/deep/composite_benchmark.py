"""Benchmark: composite spectrogram vs single-vowel and feature-fusion baselines.

Compares the composite spectrogram CNN against:
  1. Single-vowel CNN (existing SpectrogramCNN)
  2. Classical gradient boosting on acoustic features
  3. Late-fusion ensemble

Reports AUROC with 95% bootstrap confidence intervals on the same test splits.
"""

import json
import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import EVAL_DIR

log = logging.getLogger(__name__)


def bootstrap_auroc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42,
) -> tuple[float, float, float]:
    """Compute AUROC with bootstrap confidence interval.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    from sklearn.metrics import roc_auc_score

    rng = np.random.RandomState(random_state)
    point = float(roc_auc_score(y_true, y_prob))

    aucs = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        if len(np.unique(y_true[idx])) < 2:
            continue
        aucs.append(roc_auc_score(y_true[idx], y_prob[idx]))

    alpha = 1 - ci
    lower = float(np.percentile(aucs, 100 * alpha / 2))
    upper = float(np.percentile(aucs, 100 * (1 - alpha / 2)))
    return point, lower, upper


def run_benchmark(
    models: dict[str, tuple[np.ndarray, np.ndarray]],
    output_dir: str | Path | None = None,
) -> dict:
    """Run head-to-head AUROC comparison across models.

    Args:
        models: Dict mapping model name to (y_true, y_prob) tuples.
            All models must be evaluated on the same test split.
        output_dir: Directory to save results JSON.

    Returns:
        Dict with per-model AUROC, 95% CI, and ranking.
    """
    if output_dir is None:
        output_dir = EVAL_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for name, (y_true, y_prob) in models.items():
        auc, ci_lo, ci_hi = bootstrap_auroc_ci(y_true, y_prob)
        results[name] = {
            "auroc": auc,
            "ci_95_lower": ci_lo,
            "ci_95_upper": ci_hi,
            "n_samples": len(y_true),
            "n_pd": int(y_true.sum()),
            "n_healthy": int((1 - y_true).sum()),
        }
        log.info("%s: AUROC=%.4f [%.4f, %.4f]", name, auc, ci_lo, ci_hi)

    # Rank by AUROC descending
    ranked = sorted(results.items(), key=lambda x: x[1]["auroc"], reverse=True)
    for rank, (name, _) in enumerate(ranked, 1):
        results[name]["rank"] = rank

    # Compute composite vs best-baseline delta
    composite_auc = results.get("composite_cnn", {}).get("auroc")
    baseline_aucs = {
        k: v["auroc"] for k, v in results.items() if k != "composite_cnn"
    }
    if composite_auc is not None and baseline_aucs:
        best_baseline = max(baseline_aucs.values())
        best_baseline_name = max(baseline_aucs, key=lambda k: baseline_aucs[k])
        results["composite_improvement"] = {
            "delta": composite_auc - best_baseline,
            "best_baseline": best_baseline_name,
            "best_baseline_auroc": best_baseline,
        }

    output = {
        "benchmark": "composite_spectrogram_comparison",
        "models": results,
    }

    results_path = output_dir / "composite_benchmark.json"
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    log.info("Benchmark results saved to %s", results_path)

    return output
