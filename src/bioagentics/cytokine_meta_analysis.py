"""Random-effects meta-analysis engine for PANDAS/PANS cytokine data.

Computes pooled effect sizes (Hedges' g) using the DerSimonian-Laird
random-effects model, assesses heterogeneity (I², Cochran's Q, τ²),
generates forest plots, and supports sensitivity analysis by sample type
and measurement method.

Usage::

    from bioagentics.cytokine_extraction import CytokineDataset
    from bioagentics.cytokine_meta_analysis import run_meta_analysis

    ds = CytokineDataset.from_csv("data/.../extracted_cytokines.csv")
    results = run_meta_analysis(ds)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.cytokine_extraction import CytokineDataset

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Effect size computation
# ---------------------------------------------------------------------------


def hedges_g(n1: int, m1: float, s1: float, n2: int, m2: float, s2: float) -> tuple[float, float]:
    """Compute Hedges' g (bias-corrected SMD) and its variance.

    Parameters
    ----------
    n1, m1, s1 : group 1 (e.g. flare) sample size, mean, SD
    n2, m2, s2 : group 2 (e.g. remission) sample size, mean, SD

    Returns
    -------
    g : Hedges' g effect size
    v : variance of g
    """
    n1, n2 = int(n1), int(n2)
    df = n1 + n2 - 2
    if df <= 0 or s1 <= 0 or s2 <= 0:
        return np.nan, np.nan

    # Pooled SD
    sp = np.sqrt(((n1 - 1) * s1**2 + (n2 - 1) * s2**2) / df)
    if sp == 0:
        return np.nan, np.nan

    # Cohen's d
    d = (m1 - m2) / sp

    # Bias correction factor (Hedges)
    j = 1 - 3 / (4 * df - 1)
    g = d * j

    # Variance of g
    v = (n1 + n2) / (n1 * n2) + g**2 / (2 * (n1 + n2))
    return float(g), float(v)


# ---------------------------------------------------------------------------
# DerSimonian-Laird random-effects model
# ---------------------------------------------------------------------------


@dataclass
class MetaAnalysisResult:
    """Result of a random-effects meta-analysis for one analyte."""

    analyte: str
    k: int  # number of studies
    effects: list[float] = field(default_factory=list)  # per-study g
    variances: list[float] = field(default_factory=list)  # per-study v
    study_ids: list[str] = field(default_factory=list)
    pooled_effect: float = 0.0
    pooled_se: float = 0.0
    ci_lower: float = 0.0
    ci_upper: float = 0.0
    z_value: float = 0.0
    p_value: float = 0.0
    tau_sq: float = 0.0  # between-study variance
    q_stat: float = 0.0  # Cochran's Q
    q_p_value: float = 0.0
    i_sq: float = 0.0  # I² heterogeneity
    weights: list[float] = field(default_factory=list)


def dersimonian_laird(effects: np.ndarray, variances: np.ndarray) -> MetaAnalysisResult:
    """Run DerSimonian-Laird random-effects meta-analysis.

    Parameters
    ----------
    effects : array of study-level effect sizes (Hedges' g)
    variances : array of study-level variances

    Returns
    -------
    MetaAnalysisResult with pooled estimate and heterogeneity statistics.
    """
    from scipy import stats

    k = len(effects)
    result = MetaAnalysisResult(analyte="", k=k)

    if k == 0:
        return result

    w_fe = 1.0 / variances  # fixed-effect weights

    # Fixed-effect pooled estimate (needed for Q)
    theta_fe = np.sum(w_fe * effects) / np.sum(w_fe)

    # Cochran's Q
    q = float(np.sum(w_fe * (effects - theta_fe) ** 2))
    q_df = k - 1
    q_p = float(1 - stats.chi2.cdf(q, q_df)) if q_df > 0 else 1.0

    # τ² (DerSimonian-Laird estimator)
    c = np.sum(w_fe) - np.sum(w_fe**2) / np.sum(w_fe)
    tau_sq = max(0.0, (q - q_df) / c) if c > 0 else 0.0

    # Random-effects weights
    w_re = 1.0 / (variances + tau_sq)
    total_w = np.sum(w_re)

    # Pooled effect
    theta_re = float(np.sum(w_re * effects) / total_w)
    se_re = float(np.sqrt(1.0 / total_w))

    # 95% CI and z-test
    ci_lo = theta_re - 1.96 * se_re
    ci_hi = theta_re + 1.96 * se_re
    z_val = theta_re / se_re if se_re > 0 else 0.0
    p_val = float(2 * (1 - stats.norm.cdf(abs(z_val))))

    # I²
    i_sq = max(0.0, (q - q_df) / q * 100) if q > 0 else 0.0

    result.pooled_effect = theta_re
    result.pooled_se = se_re
    result.ci_lower = float(ci_lo)
    result.ci_upper = float(ci_hi)
    result.z_value = float(z_val)
    result.p_value = p_val
    result.tau_sq = float(tau_sq)
    result.q_stat = q
    result.q_p_value = q_p
    result.i_sq = float(i_sq)
    result.weights = (w_re / total_w).tolist()
    return result


# ---------------------------------------------------------------------------
# Forest plot generation
# ---------------------------------------------------------------------------


def forest_plot(result: MetaAnalysisResult, output_path: Path | None = None) -> Path | None:
    """Generate a forest plot for a meta-analysis result.

    Returns the path to the saved PNG, or None if fewer than 2 studies.
    """
    if result.k < 2:
        return None

    fig, ax = plt.subplots(figsize=(8, max(3, 0.6 * result.k + 2)))

    effects = np.array(result.effects)
    variances = np.array(result.variances)
    ses = np.sqrt(variances)
    y_positions = list(range(result.k, 0, -1))

    # Individual study estimates
    for i, (y, g, se, sid) in enumerate(zip(y_positions, effects, ses, result.study_ids)):
        ci_lo = g - 1.96 * se
        ci_hi = g + 1.96 * se
        ax.plot([ci_lo, ci_hi], [y, y], color="black", linewidth=1)
        marker_size = max(4, result.weights[i] * 100)
        ax.plot(g, y, "s", color="steelblue", markersize=marker_size, zorder=3)
        ax.text(-0.05, y, sid, ha="right", va="center", fontsize=8, transform=ax.get_yaxis_transform())

    # Pooled diamond
    diamond_y = 0.3
    hw = 0.3  # half-width of diamond vertically
    diamond_x = [result.ci_lower, result.pooled_effect, result.ci_upper, result.pooled_effect]
    diamond_y_pts = [diamond_y, diamond_y + hw, diamond_y, diamond_y - hw]
    ax.fill(diamond_x, diamond_y_pts, color="firebrick", alpha=0.7, zorder=3)

    # Zero line
    ax.axvline(0, color="grey", linestyle="--", linewidth=0.5)

    ax.set_yticks([])
    ax.set_xlabel("Hedges' g (flare vs remission)")
    ax.set_title(f"{result.analyte} — RE Meta-Analysis (k={result.k}, I²={result.i_sq:.0f}%)")

    # Annotation
    sig = "p < 0.001" if result.p_value < 0.001 else f"p = {result.p_value:.3f}"
    ax.text(
        0.02, 0.02,
        f"Pooled g = {result.pooled_effect:.2f} [{result.ci_lower:.2f}, {result.ci_upper:.2f}], {sig}",
        transform=ax.transAxes, fontsize=8, verticalalignment="bottom",
    )

    plt.tight_layout()

    if output_path is None:
        output_path = OUTPUT_DIR / f"forest_{result.analyte.replace(' ', '_')}.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved forest plot: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Main analysis runner
# ---------------------------------------------------------------------------


def run_meta_analysis(
    dataset: CytokineDataset,
    min_studies: int = 3,
    condition_a: str = "flare",
    condition_b: str = "remission",
    output_dir: Path | None = None,
) -> list[MetaAnalysisResult]:
    """Run meta-analysis for all analytes with sufficient study coverage.

    Parameters
    ----------
    dataset : Validated cytokine dataset.
    min_studies : Minimum number of studies with paired data to include an analyte.
    condition_a : Treatment/exposed condition.
    condition_b : Control condition.
    output_dir : Directory for forest plot PNGs (default: standard output dir).

    Returns
    -------
    List of MetaAnalysisResult, one per eligible analyte.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    results: list[MetaAnalysisResult] = []

    for analyte in dataset.analytes():
        pairs = dataset.paired_effects(analyte, condition_a, condition_b)
        if len(pairs) < min_studies:
            logger.debug("Skipping %s: only %d paired studies (need %d)", analyte, len(pairs), min_studies)
            continue

        # Drop rows missing SD (can't compute effect size)
        pairs = pairs.dropna(subset=["sd_a", "sd_b"])
        if len(pairs) < min_studies:
            continue

        effects_list = []
        variances_list = []
        study_ids = []

        for _, row in pairs.iterrows():
            g, v = hedges_g(
                row["n_a"], row["mean_a"], row["sd_a"],
                row["n_b"], row["mean_b"], row["sd_b"],
            )
            if np.isfinite(g) and np.isfinite(v) and v > 0:
                effects_list.append(g)
                variances_list.append(v)
                study_ids.append(row["study_id"])

        if len(effects_list) < min_studies:
            continue

        effects_arr = np.array(effects_list)
        variances_arr = np.array(variances_list)

        ma_result = dersimonian_laird(effects_arr, variances_arr)
        ma_result.analyte = analyte
        ma_result.effects = effects_list
        ma_result.variances = variances_list
        ma_result.study_ids = study_ids

        # Generate forest plot
        forest_plot(ma_result, output_dir / f"forest_{analyte.replace(' ', '_')}.png")

        results.append(ma_result)
        logger.info(
            "Meta-analysis for %s: g=%.2f [%.2f, %.2f], p=%.4f, I²=%.0f%%, k=%d",
            analyte, ma_result.pooled_effect, ma_result.ci_lower,
            ma_result.ci_upper, ma_result.p_value, ma_result.i_sq, ma_result.k,
        )

    return results


def sensitivity_analysis(
    dataset: CytokineDataset,
    analyte: str,
    group_by: str = "sample_type",
    condition_a: str = "flare",
    condition_b: str = "remission",
) -> dict[str, MetaAnalysisResult]:
    """Run meta-analysis stratified by sample_type or measurement_method.

    Returns a dict mapping each group value to its MetaAnalysisResult.
    """
    df = dataset.to_dataframe()
    df = df[df["analyte_name"] == analyte]

    if group_by not in df.columns:
        raise ValueError(f"Cannot group by '{group_by}' — column not found")

    results: dict[str, MetaAnalysisResult] = {}
    for group_val, group_df in df.groupby(group_by):
        sub_records = dataset.__class__._from_dataframe(group_df)
        pairs = sub_records.paired_effects(analyte, condition_a, condition_b)
        if len(pairs) < 2:
            continue
        pairs = pairs.dropna(subset=["sd_a", "sd_b"])
        if len(pairs) < 2:
            continue

        eff, var, sids = [], [], []
        for _, row in pairs.iterrows():
            g, v = hedges_g(row["n_a"], row["mean_a"], row["sd_a"], row["n_b"], row["mean_b"], row["sd_b"])
            if np.isfinite(g) and np.isfinite(v) and v > 0:
                eff.append(g)
                var.append(v)
                sids.append(row["study_id"])
        if len(eff) < 2:
            continue

        ma = dersimonian_laird(np.array(eff), np.array(var))
        ma.analyte = f"{analyte} [{group_val}]"
        ma.effects = eff
        ma.variances = var
        ma.study_ids = sids
        results[str(group_val)] = ma

    return results


def results_to_dataframe(results: list[MetaAnalysisResult]) -> pd.DataFrame:
    """Convert meta-analysis results to a summary DataFrame."""
    rows = []
    for r in results:
        rows.append({
            "analyte": r.analyte,
            "k": r.k,
            "pooled_g": r.pooled_effect,
            "se": r.pooled_se,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
            "p_value": r.p_value,
            "i_sq": r.i_sq,
            "tau_sq": r.tau_sq,
            "q_stat": r.q_stat,
            "q_p_value": r.q_p_value,
            "significant": r.p_value < 0.05,
            "direction": "up" if r.pooled_effect > 0 else "down",
        })
    return pd.DataFrame(rows).sort_values("p_value")
