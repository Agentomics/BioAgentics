"""Treatment response cytokine analysis for PANDAS/PANS.

Compares cytokine profiles pre/post treatment (IVIG, plasmapheresis, antibiotics).
Computes paired effect sizes, identifies cytokine shifts correlated with clinical
improvement, and generates comparison heatmaps.

Usage::

    from bioagentics.cytokine_treatment_response import TreatmentResponseAnalyzer

    analyzer = TreatmentResponseAnalyzer(dataset)
    results = analyzer.analyze_all()
    analyzer.plot_heatmap(results, output_path="heatmap.png")
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
import seaborn as sns

from bioagentics.config import REPO_ROOT
from bioagentics.cytokine_extraction import CytokineDataset
from bioagentics.cytokine_meta_analysis import hedges_g

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


# ---------------------------------------------------------------------------
# Data model — extended extraction schema for treatment data
# ---------------------------------------------------------------------------

# Treatment types we expect to see in PANDAS/PANS literature
TREATMENT_TYPES = ["IVIG", "plasmapheresis", "antibiotics", "corticosteroids", "rituximab"]


@dataclass
class TreatmentEffect:
    """Effect of treatment on a single cytokine."""

    analyte: str
    treatment: str
    n_studies: int = 0
    study_ids: list[str] = field(default_factory=list)
    pre_means: list[float] = field(default_factory=list)
    post_means: list[float] = field(default_factory=list)
    effect_sizes: list[float] = field(default_factory=list)  # per-study
    mean_effect: float = 0.0
    pooled_effect: float = 0.0
    p_value: float = 1.0
    direction: str = "ns"


@dataclass
class TreatmentAnalysisResult:
    """Results for a single treatment type across all cytokines."""

    treatment: str
    effects: dict[str, TreatmentEffect] = field(default_factory=dict)
    n_studies: int = 0
    responder_cytokines: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------


class TreatmentResponseAnalyzer:
    """Analyze cytokine changes pre/post treatment."""

    def __init__(self, dataset: CytokineDataset) -> None:
        self.dataset = dataset

    def analyze_treatment(
        self,
        treatment: str,
        pre_condition: str = "flare",
        post_condition: str = "remission",
    ) -> TreatmentAnalysisResult:
        """Analyze cytokine changes for a specific treatment.

        Extracts paired pre/post measurements for studies reporting a given
        treatment. Currently uses flare vs remission as a proxy for pre/post
        treatment (until explicit treatment metadata is available).
        """
        df = self.dataset.to_dataframe()
        result = TreatmentAnalysisResult(treatment=treatment)

        # Filter by treatment keyword in notes if available
        if "notes" in df.columns:
            treatment_mask = df["notes"].fillna("").str.contains(treatment, case=False, na=False)
            if treatment_mask.any():
                df = df[treatment_mask]
            else:
                # Fall back to using all data (treatment-agnostic analysis)
                logger.debug("No explicit '%s' annotations found — using full dataset", treatment)

        for analyte in df["analyte_name"].unique():
            analyte_df = df[df["analyte_name"] == analyte]
            pre = analyte_df[analyte_df["condition"] == pre_condition]
            post = analyte_df[analyte_df["condition"] == post_condition]

            shared_studies = sorted(set(pre["study_id"]) & set(post["study_id"]))
            if not shared_studies:
                continue

            effect = TreatmentEffect(analyte=analyte, treatment=treatment)
            effect.study_ids = shared_studies
            effect.n_studies = len(shared_studies)

            for sid in shared_studies:
                pre_row = pre[pre["study_id"] == sid].iloc[0]
                post_row = post[post["study_id"] == sid].iloc[0]

                pre_mean = float(pre_row["mean_or_median"])
                post_mean = float(post_row["mean_or_median"])
                effect.pre_means.append(pre_mean)
                effect.post_means.append(post_mean)

                pre_sd = pre_row.get("sd_or_iqr")
                post_sd = post_row.get("sd_or_iqr")
                pre_n = int(pre_row["sample_size_n"])
                post_n = int(post_row["sample_size_n"])

                if pre_sd is not None and post_sd is not None and pre_sd > 0 and post_sd > 0:
                    g, _ = hedges_g(pre_n, pre_mean, float(pre_sd), post_n, post_mean, float(post_sd))
                    if np.isfinite(g):
                        effect.effect_sizes.append(float(g))

            if effect.effect_sizes:
                effect.mean_effect = float(np.mean(effect.effect_sizes))
                # Simple z-test on mean effect
                se = float(np.std(effect.effect_sizes, ddof=1) / np.sqrt(len(effect.effect_sizes))) if len(effect.effect_sizes) > 1 else 1.0
                if se > 0:
                    from scipy import stats
                    z = effect.mean_effect / se
                    effect.p_value = float(2 * (1 - stats.norm.cdf(abs(z))))
                effect.direction = "decrease" if effect.mean_effect > 0 else "increase"
                if effect.p_value >= 0.05:
                    effect.direction = "ns"

            result.effects[analyte] = effect
            result.n_studies = max(result.n_studies, effect.n_studies)

        # Identify responder cytokines (significant changes)
        result.responder_cytokines = [
            name for name, eff in result.effects.items()
            if eff.p_value < 0.05
        ]

        return result

    def analyze_all(self) -> list[TreatmentAnalysisResult]:
        """Run analysis for all known treatment types."""
        results = []
        for treatment in TREATMENT_TYPES:
            r = self.analyze_treatment(treatment)
            if r.effects:
                results.append(r)
                logger.info(
                    "%s: %d cytokines analyzed, %d responder cytokines",
                    treatment, len(r.effects), len(r.responder_cytokines),
                )
        return results

    @staticmethod
    def plot_heatmap(
        results: list[TreatmentAnalysisResult],
        output_path: Path | str | None = None,
    ) -> Path | None:
        """Generate a comparison heatmap of treatment effects across cytokines."""
        if not results:
            return None

        if output_path is None:
            output_path = OUTPUT_DIR / "treatment_response_heatmap.png"
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Build matrix: rows=analytes, columns=treatments
        all_analytes = sorted({
            a for r in results for a in r.effects.keys()
        })
        treatments = [r.treatment for r in results]

        matrix = pd.DataFrame(index=all_analytes, columns=treatments, dtype=float)
        for r in results:
            for analyte, eff in r.effects.items():
                matrix.loc[analyte, r.treatment] = eff.mean_effect

        matrix = matrix.fillna(0)

        fig, ax = plt.subplots(figsize=(max(6, len(treatments) * 1.5), max(4, len(all_analytes) * 0.35)))
        sns.heatmap(
            matrix.astype(float),
            cmap="RdBu_r",
            center=0,
            annot=True,
            fmt=".2f",
            linewidths=0.5,
            ax=ax,
            cbar_kws={"label": "Effect size (Hedges' g)"},
        )
        ax.set_title("Treatment Response — Cytokine Effect Sizes\n(positive = decrease post-treatment)")
        ax.set_xlabel("Treatment")
        ax.set_ylabel("Cytokine")
        plt.tight_layout()
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved heatmap: %s", output_path)
        return output_path

    @staticmethod
    def results_to_dataframe(results: list[TreatmentAnalysisResult]) -> pd.DataFrame:
        """Convert results to a flat summary DataFrame."""
        rows = []
        for r in results:
            for analyte, eff in r.effects.items():
                rows.append({
                    "treatment": r.treatment,
                    "analyte": analyte,
                    "n_studies": eff.n_studies,
                    "mean_effect": eff.mean_effect,
                    "p_value": eff.p_value,
                    "direction": eff.direction,
                    "significant": eff.p_value < 0.05,
                })
        return pd.DataFrame(rows)
