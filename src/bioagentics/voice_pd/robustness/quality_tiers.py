"""Recording quality tier assignment and per-tier evaluation.

Assigns audio recordings to quality tiers (high / medium / low) based on
signal characteristics, then evaluates classifier performance within each
tier to identify where the model degrades.
"""

import json
import logging
from pathlib import Path

import numpy as np

from bioagentics.voice_pd.config import EVAL_DIR, QUALITY_TIERS, SAMPLE_RATE

log = logging.getLogger(__name__)


def _spectral_flatness(y: np.ndarray, n_fft: int = 2048) -> float:
    """Estimate spectral flatness (Wiener entropy) of the signal.

    Flatness = geometric_mean(power_spectrum) / arithmetic_mean(power_spectrum).
    Clean tonal signals -> low flatness (near 0).
    White noise -> high flatness (near 1).

    Returns a value in [0, 1].
    """
    n_fft = min(n_fft, len(y))
    if n_fft < 64:
        return 1.0

    hop = n_fft // 2
    n_frames = max(1, (len(y) - n_fft) // hop)
    flatness_values = []

    for i in range(n_frames):
        frame = y[i * hop : i * hop + n_fft]
        power = np.abs(np.fft.rfft(frame)) ** 2
        power = power + 1e-12  # avoid log(0)
        geo_mean = np.exp(np.mean(np.log(power)))
        arith_mean = np.mean(power)
        if arith_mean < 1e-12:
            continue
        flatness_values.append(geo_mean / arith_mean)

    if not flatness_values:
        return 1.0
    return float(np.mean(flatness_values))


def _clipping_ratio(y: np.ndarray, threshold: float = 0.99) -> float:
    """Fraction of samples at or beyond clipping threshold."""
    if len(y) == 0:
        return 0.0
    return float(np.mean(np.abs(y) >= threshold))


def _spectral_bandwidth_mean(y: np.ndarray, sr: int = SAMPLE_RATE) -> float:
    """Mean spectral bandwidth in Hz (simplified, no librosa dependency)."""
    n_fft = min(2048, len(y))
    if n_fft < 64:
        return 0.0

    hop = n_fft // 2
    n_frames = max(1, (len(y) - n_fft) // hop)
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)

    bw_sum = 0.0
    for i in range(n_frames):
        frame = y[i * hop : i * hop + n_fft]
        mag = np.abs(np.fft.rfft(frame))
        mag_sum = mag.sum()
        if mag_sum < 1e-12:
            continue
        centroid = np.sum(freqs * mag) / mag_sum
        bw_sum += np.sqrt(np.sum(mag * (freqs - centroid) ** 2) / mag_sum)

    return float(bw_sum / max(1, n_frames))


def assign_quality_tier(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> str:
    """Classify a recording into a quality tier based on signal properties.

    Uses three metrics:
    - Spectral flatness (0=tonal/clean, 1=noise-like)
    - Clipping ratio (fraction of clipped samples)
    - Spectral bandwidth (Hz)

    Criteria (applied in order):
    - **high**: flatness < 0.15, clipping < 1%, bandwidth >= 1200 Hz
    - **low**: flatness >= 0.4, or clipping >= 5%, or bandwidth < 800 Hz
    - **medium**: everything else

    Args:
        y: Audio signal array (mono).
        sr: Sample rate.

    Returns:
        One of ``"high"``, ``"medium"``, ``"low"``.
    """
    flatness = _spectral_flatness(y)
    clip = _clipping_ratio(y)
    bw = _spectral_bandwidth_mean(y, sr=sr)

    if flatness < 0.15 and clip < 0.01 and bw >= 1200.0:
        return "high"
    if flatness >= 0.4 or clip >= 0.05 or bw < 800.0:
        return "low"
    return "medium"


def evaluate_by_quality_tier(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    tiers: list[str],
    output_dir: str | Path | None = None,
) -> dict:
    """Compute per-tier classification metrics.

    Args:
        y_true: Binary ground-truth labels (n_samples,).
        y_prob: Predicted probabilities for PD class (n_samples,).
        tiers: Quality tier string for each sample (n_samples,).
        output_dir: Directory to save results JSON.

    Returns:
        Dict mapping tier name to metrics (AUC, n_samples, etc.)
        plus an ``"overall"`` key.
    """
    from sklearn.metrics import roc_auc_score

    if output_dir is None:
        output_dir = EVAL_DIR / "quality_tiers"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tier_results: dict[str, dict] = {}

    for tier in QUALITY_TIERS:
        mask = np.array([t == tier for t in tiers])
        n = int(mask.sum())
        if n == 0:
            tier_results[tier] = {"auc": None, "n_samples": 0, "note": "no samples"}
            continue

        y_t = y_true[mask]
        y_p = y_prob[mask]
        n_classes = len(np.unique(y_t))

        if n_classes < 2:
            tier_results[tier] = {
                "auc": None,
                "n_samples": n,
                "n_pd": int(y_t.sum()),
                "n_healthy": n - int(y_t.sum()),
                "note": "single class — cannot compute AUC",
            }
            continue

        auc = float(roc_auc_score(y_t, y_p))
        tier_results[tier] = {
            "auc": auc,
            "n_samples": n,
            "n_pd": int(y_t.sum()),
            "n_healthy": n - int(y_t.sum()),
        }
        log.info("  %s tier: AUC=%.4f (n=%d)", tier, auc, n)

    # Overall
    n_classes_all = len(np.unique(y_true))
    overall_auc = (
        float(roc_auc_score(y_true, y_prob)) if n_classes_all >= 2 else None
    )

    results = {
        "tiers": tier_results,
        "overall_auc": overall_auc,
        "n_total": len(y_true),
    }

    results_path = output_dir / "quality_tier_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info("Quality tier results saved to %s", results_path)

    return results
