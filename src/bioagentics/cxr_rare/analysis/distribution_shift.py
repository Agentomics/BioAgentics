"""Inter-institutional distribution shift analysis (MIMIC vs MIDRC).

Compares label distributions, demographic distributions, and image
characteristics between institutions. Computes statistical divergence
measures.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.cxr_rare.config import LABEL_NAMES, OUTPUT_DIR, ensure_dirs

logger = logging.getLogger(__name__)


def compare_label_prevalence(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    name_a: str = "MIMIC",
    name_b: str = "MIDRC",
    label_names: list[str] | None = None,
) -> pd.DataFrame:
    """Compare per-class prevalence between two datasets.

    Returns DataFrame with prevalence rates and absolute differences.
    """
    labels = label_names or LABEL_NAMES
    total_a = max(sum(counts_a.get(l, 0) for l in labels), 1)
    total_b = max(sum(counts_b.get(l, 0) for l in labels), 1)

    rows = []
    for lbl in labels:
        ca = counts_a.get(lbl, 0)
        cb = counts_b.get(lbl, 0)
        prev_a = ca / total_a
        prev_b = cb / total_b
        rows.append({
            "label": lbl,
            f"{name_a}_count": ca,
            f"{name_b}_count": cb,
            f"{name_a}_prevalence": prev_a,
            f"{name_b}_prevalence": prev_b,
            "prevalence_diff": abs(prev_a - prev_b),
        })
    return pd.DataFrame(rows).sort_values("prevalence_diff", ascending=False)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Compute KL divergence D(P || Q) with smoothing."""
    eps = 1e-10
    p = np.array(p, dtype=np.float64) + eps
    q = np.array(q, dtype=np.float64) + eps
    p = p / p.sum()
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def chi_squared_test(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    label_names: list[str] | None = None,
) -> tuple[float, float]:
    """Chi-squared test for distribution difference.

    Returns (chi2_statistic, p_value).
    """
    labels = label_names or LABEL_NAMES
    observed_a = np.array([counts_a.get(l, 0) for l in labels])
    observed_b = np.array([counts_b.get(l, 0) for l in labels])

    # Contingency table
    table = np.array([observed_a, observed_b])
    # Filter out columns with all zeros
    nonzero_cols = table.sum(axis=0) > 0
    table = table[:, nonzero_cols]

    if table.shape[1] < 2:
        return 0.0, 1.0

    chi2, p, _, _ = stats.chi2_contingency(table)
    return float(chi2), float(p)


def analyze_distribution_shift(
    counts_a: dict[str, int],
    counts_b: dict[str, int],
    name_a: str = "MIMIC",
    name_b: str = "MIDRC",
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    """Full distribution shift analysis between two datasets.

    Computes prevalence comparison, KL divergence, chi-squared test,
    and generates comparison plots.

    Returns
    -------
    dict with keys: comparison_df, kl_ab, kl_ba, chi2, chi2_p
    """
    ensure_dirs()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Label prevalence comparison
    comparison_df = compare_label_prevalence(counts_a, counts_b, name_a, name_b)
    csv_path = output_dir / f"prevalence_comparison_{name_a}_vs_{name_b}.csv"
    comparison_df.to_csv(csv_path, index=False)
    logger.info("Saved prevalence comparison: %s", csv_path)

    # KL divergence
    labels = LABEL_NAMES
    p = np.array([counts_a.get(l, 0) for l in labels], dtype=float)
    q = np.array([counts_b.get(l, 0) for l in labels], dtype=float)
    kl_ab = kl_divergence(p, q)
    kl_ba = kl_divergence(q, p)

    # Chi-squared test
    chi2, chi2_p = chi_squared_test(counts_a, counts_b)

    logger.info(
        "Distribution shift: KL(%s||%s)=%.4f, KL(%s||%s)=%.4f, chi2=%.2f (p=%.2e)",
        name_a, name_b, kl_ab, name_b, name_a, kl_ba, chi2, chi2_p,
    )

    # Plot
    fig_path = output_dir / "figures" / f"prevalence_{name_a}_vs_{name_b}.png"
    _plot_prevalence_comparison(comparison_df, name_a, name_b, fig_path)

    return {
        "comparison_df": comparison_df,
        "kl_ab": kl_ab,
        "kl_ba": kl_ba,
        "chi2": chi2,
        "chi2_p": chi2_p,
    }


def _plot_prevalence_comparison(
    df: pd.DataFrame,
    name_a: str,
    name_b: str,
    output_path: Path,
) -> None:
    """Side-by-side bar chart comparing label prevalence."""
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(df))
    width = 0.35

    ax.bar(x - width / 2, df[f"{name_a}_prevalence"], width, label=name_a, color="#2196F3")
    ax.bar(x + width / 2, df[f"{name_b}_prevalence"], width, label=name_b, color="#FF9800")
    ax.set_xticks(x)
    ax.set_xticklabels(df["label"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Prevalence")
    ax.set_title(f"Label Prevalence: {name_a} vs {name_b}")
    ax.legend()

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved prevalence plot: %s", output_path)
