"""Immune axis classifier for PANDAS/PANS flare cytokine signatures.

Categorizes the flare signature by immune axis (Th1, Th2, Th17, regulatory,
innate), computes composite scores from member cytokine effect sizes,
determines the dominant axis, and generates radar/spider plots.

Usage::

    from bioagentics.cytokine_axis_classifier import classify_axes, radar_plot

    scores = classify_axes(meta_results)
    radar_plot(scores, output_path="radar.png")
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bioagentics.config import REPO_ROOT
from bioagentics.cytokine_meta_analysis import MetaAnalysisResult

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Axis definitions
# ---------------------------------------------------------------------------

IMMUNE_AXES: dict[str, list[str]] = {
    "Th1": ["IFN-γ", "TNF-α"],
    "Th2": ["IL-4", "IL-13"],
    "Th17": ["IL-17A", "IL-22"],
    "Regulatory": ["IL-10", "TGF-β"],
    "Innate": ["IL-1β", "IL-6", "TNF-α"],
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class AxisScore:
    """Composite score for a single immune axis."""

    axis: str
    members: list[str]
    member_effects: dict[str, float] = field(default_factory=dict)
    member_p_values: dict[str, float] = field(default_factory=dict)
    composite_score: float = 0.0
    n_measured: int = 0
    n_significant: int = 0


@dataclass
class AxisClassification:
    """Full classification result across all axes."""

    scores: dict[str, AxisScore] = field(default_factory=dict)
    dominant_axis: str = ""
    dominant_score: float = 0.0


# ---------------------------------------------------------------------------
# Classification logic
# ---------------------------------------------------------------------------


def classify_axes(
    results: list[MetaAnalysisResult],
    alpha: float = 0.05,
) -> AxisClassification:
    """Classify the flare signature by immune axis.

    For each axis, the composite score is the mean absolute effect size
    of its member cytokines (using only those measured in the meta-analysis).
    Direction-aware: positive means net upregulation, negative means net
    downregulation of that axis in flares.

    Parameters
    ----------
    results : List of meta-analysis results (one per analyte).
    alpha : Significance threshold.

    Returns
    -------
    AxisClassification with per-axis scores and dominant axis.
    """
    result_map = {r.analyte: r for r in results}
    classification = AxisClassification()

    for axis_name, members in IMMUNE_AXES.items():
        score = AxisScore(axis=axis_name, members=members)
        effects = []

        for cytokine in members:
            if cytokine in result_map:
                r = result_map[cytokine]
                score.member_effects[cytokine] = r.pooled_effect
                score.member_p_values[cytokine] = r.p_value
                score.n_measured += 1
                effects.append(r.pooled_effect)
                if r.p_value < alpha:
                    score.n_significant += 1

        if effects:
            score.composite_score = float(np.mean(effects))

        classification.scores[axis_name] = score

    # Determine dominant axis (highest absolute composite score)
    if classification.scores:
        dominant = max(classification.scores.values(), key=lambda s: abs(s.composite_score))
        classification.dominant_axis = dominant.axis
        classification.dominant_score = dominant.composite_score

    logger.info(
        "Axis classification: dominant=%s (score=%.2f), measured axes=%d",
        classification.dominant_axis, classification.dominant_score,
        sum(1 for s in classification.scores.values() if s.n_measured > 0),
    )

    return classification


# ---------------------------------------------------------------------------
# Radar / spider plot
# ---------------------------------------------------------------------------


def radar_plot(
    classification: AxisClassification,
    output_path: Path | str | None = None,
    title: str = "PANDAS/PANS Flare — Immune Axis Profile",
) -> Path:
    """Generate a radar/spider plot of immune axis composite scores.

    Each axis is a spoke; the radius encodes the absolute composite score,
    and color encodes direction (red=up, blue=down).
    """
    if output_path is None:
        output_path = OUTPUT_DIR / "immune_axis_radar.png"
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    axes = list(IMMUNE_AXES.keys())
    n = len(axes)
    scores_raw = [classification.scores[a].composite_score for a in axes]
    scores_abs = [abs(s) for s in scores_raw]
    colors = ["red" if s > 0 else "blue" if s < 0 else "grey" for s in scores_raw]

    # Radar plot angles
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    scores_abs += scores_abs[:1]  # close the polygon
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={"polar": True})

    # Fill polygon
    ax.fill(angles, scores_abs, alpha=0.15, color="steelblue")
    ax.plot(angles, scores_abs, "o-", linewidth=2, color="steelblue")

    # Color individual points by direction
    for i, (angle, score, color) in enumerate(zip(angles[:-1], scores_abs[:-1], colors)):
        ax.plot(angle, score, "o", markersize=10, color=color, zorder=3)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axes, fontsize=10, fontweight="bold")

    # Annotate scores
    for i, (angle, score, raw) in enumerate(zip(angles[:-1], scores_abs[:-1], scores_raw)):
        direction = "↑" if raw > 0 else "↓" if raw < 0 else "—"
        ax.text(
            angle, score + 0.05, f"{raw:+.2f} {direction}",
            ha="center", va="bottom", fontsize=8,
        )

    ax.set_title(title, fontsize=12, fontweight="bold", pad=20)

    # Add legend for dominant axis
    if classification.dominant_axis:
        fig.text(
            0.5, 0.02,
            f"Dominant axis: {classification.dominant_axis} "
            f"(score = {classification.dominant_score:+.2f})",
            ha="center", fontsize=10, fontstyle="italic",
        )

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved radar plot: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------


def classification_to_dict(classification: AxisClassification) -> dict:
    """Convert classification to a serializable dict."""
    return {
        "dominant_axis": classification.dominant_axis,
        "dominant_score": classification.dominant_score,
        "axes": {
            name: {
                "composite_score": s.composite_score,
                "n_measured": s.n_measured,
                "n_significant": s.n_significant,
                "member_effects": s.member_effects,
                "member_p_values": s.member_p_values,
            }
            for name, s in classification.scores.items()
        },
    }
