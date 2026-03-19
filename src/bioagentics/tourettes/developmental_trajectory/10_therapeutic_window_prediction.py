"""Step 10 — Therapeutic window prediction (Phase 5).

Uses the developmental trajectory model (Phases 1-4) to identify optimal
intervention windows for Tourette syndrome:

  1. Therapeutic window scoring: composite score per developmental stage
     indicating amenability to intervention
  2. DBS efficacy modelling: map molecular dynamics to known DBS outcome
     data (pediatric 70% vs adult 56% improvement at 60 months)
  3. Druggable target identification: TS genes encoding druggable protein
     classes mapped to peak expression windows
  4. Early intervention analysis: could onset-window targeting prevent
     tic escalation?
  5. Testable predictions for clinical validation (P5.1-P5.x)

Task: #787 (Phase 5: Therapeutic Window Prediction)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.10_therapeutic_window_prediction
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.analysis.tourettes.brainspan_trajectories import DEV_STAGES
from bioagentics.data.tourettes.gene_sets import get_gene_set

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

# ── Clinical stage mapping ─────────────────────────────────────────────────

POSTNATAL_STAGES = [
    "infancy", "early_childhood", "late_childhood", "adolescence", "adulthood",
]

# Clinical windows aligned with developmental stages
CLINICAL_WINDOWS: dict[str, dict] = {
    "pre_onset": {
        "stages": ["infancy"],
        "age_range": "0-4 years",
        "description": "Before tic onset — preventive window",
    },
    "onset": {
        "stages": ["early_childhood"],
        "age_range": "4-7 years",
        "description": "Tic onset window — early intervention",
    },
    "peak_severity": {
        "stages": ["late_childhood"],
        "age_range": "8-12 years",
        "description": "Peak tic severity — symptom management",
    },
    "remission_window": {
        "stages": ["adolescence"],
        "age_range": "12-18 years",
        "description": "Spontaneous remission window — circuit maturation",
    },
    "adult": {
        "stages": ["adulthood"],
        "age_range": "18+ years",
        "description": "Adult persistence — compensatory strategies",
    },
}

# ── Druggable gene classification ──────────────────────────────────────────
# TS-relevant genes classified by druggable protein class.
# Sources: DGIdb, ChEMBL, OpenTargets, curated from gene set descriptions.

DRUGGABLE_CLASSES: dict[str, dict[str, str]] = {
    "receptor": {
        "DRD1": "D1 dopamine receptor — agonists/antagonists available",
        "DRD2": "D2 dopamine receptor — antipsychotic target",
        "ADORA2A": "adenosine A2a receptor — istradefylline approved",
        "OPRD1": "delta opioid receptor — naltrindole, SNC80",
        "AR": "androgen receptor — anti-androgens available",
        "ESR1": "estrogen receptor alpha — SERMs available",
        "ESR2": "estrogen receptor beta — selective agonists available",
        "GPR6": "orphan GPCR — tool compounds in development",
        "GRM5": "metabotropic glutamate receptor 5 — NAMs available",
    },
    "ion_channel": {
        "CACNA1A": "P/Q-type Ca2+ channel — existing modulators",
        "KCND2": "Kv4.2 — potassium channel modulators",
        "KCNC1": "Kv3.1 — PV interneuron fast-spiking channel",
        "KCNC2": "Kv3.2 — PV interneuron fast-spiking channel",
        "GABRA1": "GABA-A alpha1 subunit — benzodiazepine target",
        "GABRG2": "GABA-A gamma2 subunit — allosteric modulators",
        "GRIN1": "NMDA receptor NR1 — memantine target",
        "GRIN2A": "NMDA receptor NR2A — subunit-selective modulators",
        "GRIN2B": "NMDA receptor NR2B — ifenprodil-site modulators",
    },
    "enzyme": {
        "HDC": "histidine decarboxylase — enzyme replacement concept",
        "MAOA": "monoamine oxidase A — MAOIs available",
        "TH": "tyrosine hydroxylase — rate-limiting DA synthesis",
        "GAD1": "glutamate decarboxylase 67 — GABA synthesis",
        "GAD2": "glutamate decarboxylase 65 — GABA synthesis",
    },
    "transporter": {
        "SLC6A3": "dopamine transporter — methylphenidate target",
        "SLC6A1": "GABA transporter — tiagabine target",
        "SLC32A1": "vesicular GABA transporter — vigabatrin related",
    },
    "kinase": {
        "FLT3": "FLT3 receptor tyrosine kinase — inhibitors in oncology",
        "MET": "MET receptor kinase — capmatinib, tepotinib",
        "LATS1": "Hippo pathway kinase — emerging targets",
        "LATS2": "Hippo pathway kinase — emerging targets",
    },
}

# Flatten for quick lookup
_DRUGGABLE_LOOKUP: dict[str, str] = {}
for _cls, _genes in DRUGGABLE_CLASSES.items():
    for _sym in _genes:
        _DRUGGABLE_LOOKUP[_sym] = _cls


# ── Therapeutic window scoring ─────────────────────────────────────────────


def compute_therapeutic_window_scores(
    remission_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Compute a per-stage therapeutic amenability score.

    Combines three signals:
    1. **Dynamic change rate**: stages with rapid change in remission/
       persistence scores represent windows where the system is most
       malleable to intervention.
    2. **E/I imbalance**: stages where excitatory drive exceeds inhibitory
       tone offer the most room for rebalancing.
    3. **Druggable target density**: number of druggable TS genes with
       elevated expression at each stage.

    Returns
    -------
    DataFrame with columns: dev_stage, change_rate, ei_imbalance,
    druggable_density, therapeutic_score, stage_order, clinical_window.
    """
    stage_order = list(DEV_STAGES.keys())

    if remission_df.empty and persistence_df.empty:
        return pd.DataFrame()

    # 1. Change rate of remission score (finite differences)
    rem_by_stage: dict[str, float] = {}
    if not remission_df.empty:
        for _, row in remission_df.iterrows():
            rem_by_stage[row["dev_stage"]] = row["remission_score"]

    change_rates: dict[str, float] = {}
    for i, stage in enumerate(stage_order):
        if i == 0 or stage_order[i - 1] not in rem_by_stage or stage not in rem_by_stage:
            change_rates[stage] = 0.0
        else:
            change_rates[stage] = abs(
                rem_by_stage[stage] - rem_by_stage[stage_order[i - 1]]
            )

    # 2. E/I imbalance from persistence scores
    ei_imbalance: dict[str, float] = {}
    if not persistence_df.empty:
        for _, row in persistence_df.iterrows():
            ei_imbalance[row["dev_stage"]] = row["persistence_score"]
    else:
        ei_imbalance = {s: 0.0 for s in stage_order}

    # 3. Druggable target density
    druggable_density = _compute_druggable_density(trajectory_df, region)

    # Build composite score
    records = []
    for stage in stage_order:
        cr = change_rates.get(stage, 0.0)
        ei = ei_imbalance.get(stage, 0.0)
        dd = druggable_density.get(stage, 0.0)

        # Therapeutic score: high change rate + high E/I imbalance +
        # high druggable density = more amenable to intervention
        score = (cr + max(ei, 0.0) + dd) / 3.0

        # Map to clinical window
        cw = _stage_to_clinical_window(stage)

        records.append({
            "dev_stage": stage,
            "change_rate": float(cr),
            "ei_imbalance": float(ei),
            "druggable_density": float(dd),
            "therapeutic_score": float(score),
            "stage_order": stage_order.index(stage),
            "clinical_window": cw,
        })

    return pd.DataFrame(records).sort_values("stage_order").reset_index(drop=True)


def _stage_to_clinical_window(stage: str) -> str:
    """Map a developmental stage to its clinical window name."""
    for window_name, window_info in CLINICAL_WINDOWS.items():
        if stage in window_info["stages"]:
            return window_name
    return "unknown"


def _compute_druggable_density(
    trajectory_df: pd.DataFrame,
    region: str,
) -> dict[str, float]:
    """Compute the fraction of druggable genes with above-median expression
    at each stage (normalized 0-1)."""
    if trajectory_df.empty:
        return {}

    region_df = trajectory_df[trajectory_df["cstc_region"] == region]
    if region_df.empty:
        return {}

    druggable_genes = set(_DRUGGABLE_LOOKUP.keys())
    available = druggable_genes & set(region_df["gene_symbol"].unique())
    if not available:
        return {}

    sub = region_df[region_df["gene_symbol"].isin(available)]

    # Per gene, median across all stages
    gene_medians = sub.groupby("gene_symbol")["mean_log2_rpkm"].median()

    result: dict[str, float] = {}
    for stage in sub["dev_stage"].unique():
        stage_data = sub[sub["dev_stage"] == stage]
        n_above = 0
        for _, row in stage_data.iterrows():
            if row["mean_log2_rpkm"] >= gene_medians.get(row["gene_symbol"], 0):
                n_above += 1
        result[stage] = n_above / max(len(available), 1)

    return result


# ── DBS efficacy window modelling ──────────────────────────────────────────

# Clinical DBS outcome data (Baldermann et al. 2019 meta-analysis;
# Deeb et al. 2016 long-term follow-up)
DBS_CLINICAL_DATA = {
    "pediatric": {"age_range": "12-17", "improvement_pct": 70.0, "followup_months": 60},
    "adult": {"age_range": "25+", "improvement_pct": 56.0, "followup_months": 60},
}


def model_dbs_efficacy_window(
    remission_df: pd.DataFrame,
    persistence_df: pd.DataFrame,
) -> dict:
    """Model DBS efficacy as a function of developmental stage.

    Hypothesis: DBS is more effective when applied during windows of active
    inhibitory circuit maturation (high remission score rate of change).
    The model predicts adolescence as the optimal DBS window because it
    coincides with PV interneuron maturation — DBS may amplify this natural
    compensatory process.

    Returns
    -------
    Dict with efficacy predictions per clinical window and comparison
    with known DBS outcomes.
    """
    if remission_df.empty and persistence_df.empty:
        return {"hypothesis": "DBS efficacy window", "error": "no data"}

    stage_order = list(DEV_STAGES.keys())

    # Compute circuit malleability: rate of change of remission score
    rem_trajectory: dict[str, float] = {}
    if not remission_df.empty:
        for _, row in remission_df.iterrows():
            rem_trajectory[row["dev_stage"]] = row["remission_score"]

    # Malleability = absolute change from previous stage
    malleability: dict[str, float] = {}
    for i, stage in enumerate(stage_order):
        if stage not in rem_trajectory:
            malleability[stage] = 0.0
            continue
        if i == 0 or stage_order[i - 1] not in rem_trajectory:
            malleability[stage] = 0.0
        else:
            malleability[stage] = abs(
                rem_trajectory[stage] - rem_trajectory[stage_order[i - 1]]
            )

    # E/I gap: larger gap = more room for DBS to help
    ei_gap: dict[str, float] = {}
    if not persistence_df.empty:
        for _, row in persistence_df.iterrows():
            ei_gap[row["dev_stage"]] = max(row["persistence_score"], 0.0)

    # Composite DBS amenability per postnatal stage
    window_scores: dict[str, dict] = {}
    for window_name, window_info in CLINICAL_WINDOWS.items():
        stages = window_info["stages"]
        m_vals = [malleability.get(s, 0.0) for s in stages]
        e_vals = [ei_gap.get(s, 0.0) for s in stages]
        avg_m = float(np.mean(m_vals)) if m_vals else 0.0
        avg_e = float(np.mean(e_vals)) if e_vals else 0.0
        dbs_score = (avg_m + avg_e) / 2.0
        window_scores[window_name] = {
            "malleability": avg_m,
            "ei_gap": avg_e,
            "dbs_amenability_score": float(dbs_score),
            "age_range": window_info["age_range"],
        }

    # Identify optimal window
    best_window = max(
        window_scores,
        key=lambda w: window_scores[w]["dbs_amenability_score"],
    )

    # Compare with clinical data
    adolescent_score = window_scores.get("remission_window", {}).get(
        "dbs_amenability_score", 0.0
    )
    adult_score = window_scores.get("adult", {}).get(
        "dbs_amenability_score", 0.0
    )
    model_predicts_pediatric_advantage = adolescent_score > adult_score

    return {
        "hypothesis": "DBS efficacy window",
        "window_scores": window_scores,
        "optimal_window": best_window,
        "model_predicts_pediatric_advantage": bool(model_predicts_pediatric_advantage),
        "clinical_comparison": {
            "pediatric_dbs_improvement": DBS_CLINICAL_DATA["pediatric"]["improvement_pct"],
            "adult_dbs_improvement": DBS_CLINICAL_DATA["adult"]["improvement_pct"],
            "model_consistent_with_clinical": bool(model_predicts_pediatric_advantage),
        },
        "remission_trajectory": rem_trajectory,
    }


# ── Druggable target identification ───────────────────────────────────────


def identify_druggable_targets(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Identify TS genes encoding druggable proteins and map their peak
    expression to developmental windows.

    For each druggable gene found in the trajectory data, determines:
    - Peak expression stage and clinical window
    - Druggable class (receptor, channel, enzyme, transporter, kinase)
    - Whether it's a TS risk gene
    - Expression fold change at therapeutic windows vs baseline

    Returns
    -------
    DataFrame sorted by peak_expression descending, with columns:
    gene_symbol, druggable_class, peak_stage, peak_clinical_window,
    peak_expression, is_ts_gene, onset_fold, remission_fold.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region]
    if region_df.empty:
        return pd.DataFrame()

    druggable_genes = set(_DRUGGABLE_LOOKUP.keys())
    available = druggable_genes & set(region_df["gene_symbol"].unique())
    if not available:
        return pd.DataFrame()

    ts_combined = set(get_gene_set("ts_combined").keys())

    records = []
    for gene in sorted(available):
        gene_data = region_df[region_df["gene_symbol"] == gene]
        stage_expr = gene_data.set_index("dev_stage")["mean_log2_rpkm"]

        if stage_expr.empty:
            continue

        # Peak stage
        peak_stage = stage_expr.idxmax()
        peak_expr = float(stage_expr.max())

        # Baseline = mean across all stages
        baseline = float(stage_expr.mean())

        # Fold change at onset window vs baseline
        onset_expr = stage_expr.get("early_childhood", baseline)
        onset_fold = float(onset_expr / max(baseline, 1e-6))

        # Fold change at remission window vs baseline
        rem_expr = stage_expr.get("adolescence", baseline)
        remission_fold = float(rem_expr / max(baseline, 1e-6))

        records.append({
            "gene_symbol": gene,
            "druggable_class": _DRUGGABLE_LOOKUP[gene],
            "peak_stage": peak_stage,
            "peak_clinical_window": _stage_to_clinical_window(peak_stage),
            "peak_expression": peak_expr,
            "baseline_expression": baseline,
            "is_ts_gene": gene in ts_combined,
            "onset_fold": onset_fold,
            "remission_fold": remission_fold,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values(
        "peak_expression", ascending=False
    ).reset_index(drop=True)


# ── Early intervention analysis ───────────────────────────────────────────


def analyze_early_intervention(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> dict:
    """Analyze whether early intervention during the onset window could
    prevent tic escalation.

    Examines genes that:
    1. Peak during the onset window (early_childhood / late_childhood)
    2. Are druggable
    3. Are TS risk genes or part of pathogenic pathways

    A high density of druggable, TS-relevant genes peaking during onset
    supports the hypothesis that early pharmacological intervention could
    modulate disease trajectory.

    Returns
    -------
    Dict with onset-window druggable target analysis and early
    intervention feasibility assessment.
    """
    if trajectory_df.empty:
        return {"hypothesis": "Early intervention feasibility", "error": "no data"}

    region_df = trajectory_df[trajectory_df["cstc_region"] == region]
    if region_df.empty:
        return {"hypothesis": "Early intervention feasibility",
                "error": "no region data"}

    onset_stages = {"early_childhood", "late_childhood"}
    druggable_genes = set(_DRUGGABLE_LOOKUP.keys())
    ts_combined = set(get_gene_set("ts_combined").keys())
    available_genes = set(region_df["gene_symbol"].unique())

    # Find genes peaking in onset window
    onset_peak_genes: list[str] = []
    for gene in available_genes:
        gene_data = region_df[region_df["gene_symbol"] == gene]
        stage_expr = gene_data.set_index("dev_stage")["mean_log2_rpkm"]
        if stage_expr.empty:
            continue
        peak_stage = stage_expr.idxmax()
        if peak_stage in onset_stages:
            onset_peak_genes.append(gene)

    # Classify onset-peak genes
    onset_druggable = [g for g in onset_peak_genes if g in druggable_genes]
    onset_ts = [g for g in onset_peak_genes if g in ts_combined]
    onset_druggable_ts = [g for g in onset_peak_genes
                          if g in druggable_genes and g in ts_combined]

    # Druggable class breakdown
    class_counts: dict[str, int] = {}
    for gene in onset_druggable:
        cls = _DRUGGABLE_LOOKUP[gene]
        class_counts[cls] = class_counts.get(cls, 0) + 1

    # Compare onset druggable density to other windows
    other_stages = set(POSTNATAL_STAGES) - onset_stages
    other_peak_druggable = []
    for gene in available_genes:
        gene_data = region_df[region_df["gene_symbol"] == gene]
        stage_expr = gene_data.set_index("dev_stage")["mean_log2_rpkm"]
        if stage_expr.empty:
            continue
        peak_stage = stage_expr.idxmax()
        if peak_stage in other_stages and gene in druggable_genes:
            other_peak_druggable.append(gene)

    onset_density = len(onset_druggable) / max(len(onset_peak_genes), 1)
    other_density = len(other_peak_druggable) / max(
        len(available_genes) - len(onset_peak_genes), 1
    )
    enriched = onset_density > other_density

    return {
        "hypothesis": "Early intervention feasibility",
        "region": region,
        "n_onset_peak_genes": len(onset_peak_genes),
        "n_onset_druggable": len(onset_druggable),
        "n_onset_ts_risk": len(onset_ts),
        "n_onset_druggable_ts": len(onset_druggable_ts),
        "onset_druggable_genes": sorted(onset_druggable),
        "onset_druggable_ts_genes": sorted(onset_druggable_ts),
        "druggable_class_breakdown": class_counts,
        "onset_druggable_density": float(onset_density),
        "other_window_druggable_density": float(other_density),
        "onset_enriched_for_druggable": bool(enriched),
        "feasibility": _assess_feasibility(
            len(onset_druggable_ts), len(onset_druggable), class_counts,
        ),
    }


def _assess_feasibility(
    n_druggable_ts: int,
    n_druggable: int,
    class_counts: dict[str, int],
) -> str:
    """Qualitative feasibility assessment for early intervention."""
    if n_druggable_ts >= 3 and "receptor" in class_counts:
        return "high"
    if n_druggable >= 3:
        return "moderate"
    if n_druggable >= 1:
        return "low"
    return "not_feasible"


# ── Testable predictions ──────────────────────────────────────────────────


def generate_phase5_predictions(
    therapeutic_scores: pd.DataFrame,
    dbs_analysis: dict,
    druggable_targets: pd.DataFrame,
    early_intervention: dict,
) -> list[dict]:
    """Generate Phase 5 testable predictions for therapeutic windows.

    Each prediction includes hypothesis, supporting evidence, validation
    approach, and confidence level.
    """
    predictions: list[dict] = []

    # P5.1: Optimal DBS timing
    if dbs_analysis.get("model_predicts_pediatric_advantage"):
        optimal = dbs_analysis.get("optimal_window", "unknown")
        predictions.append({
            "id": "P5.1",
            "prediction": (
                "DBS for Tourette syndrome is most effective when applied "
                "during the remission window (12-18 years), when PV "
                "interneuron maturation and inhibitory circuit strengthening "
                "are actively occurring. DBS may amplify this natural "
                "compensatory process."
            ),
            "evidence": {
                "optimal_window": optimal,
                "model_consistent_with_clinical": dbs_analysis.get(
                    "clinical_comparison", {}
                ).get("model_consistent_with_clinical"),
                "window_scores": {
                    k: v.get("dbs_amenability_score")
                    for k, v in dbs_analysis.get("window_scores", {}).items()
                },
            },
            "validation": (
                "Prospective DBS trial stratified by age at implantation, "
                "with pre-operative PV interneuron proxy measures (e.g. "
                "gamma oscillation power on EEG) as predictive biomarker."
            ),
            "confidence": "medium-high",
        })

    # P5.2: Druggable target window
    if not druggable_targets.empty:
        remission_targets = druggable_targets[
            druggable_targets["peak_clinical_window"] == "remission_window"
        ]
        if not remission_targets.empty:
            top_df = remission_targets.head(5)
            top_names = top_df["gene_symbol"].tolist()
            predictions.append({
                "id": "P5.2",
                "prediction": (
                    f"The druggable targets {', '.join(top_names)} peak "
                    "in expression during the remission window and represent "
                    "candidate pharmacological targets to augment spontaneous "
                    "remission in patients at risk for tic persistence."
                ),
                "evidence": {
                    "n_remission_druggable": len(remission_targets),
                    "top_targets": [
                        {
                            "gene": row["gene_symbol"],
                            "class": row["druggable_class"],
                            "peak_expression": row["peak_expression"],
                        }
                        for _, row in top_df.iterrows()
                    ],
                },
                "validation": (
                    "Screen remission-window targets in TS animal models "
                    "(e.g. Hdc-KO mice) during the equivalent developmental "
                    "period; measure tic-like behavior and PV interneuron "
                    "density."
                ),
                "confidence": "medium",
            })

    # P5.3: Early intervention
    if early_intervention.get("n_onset_druggable_ts", 0) > 0:
        feasibility = early_intervention.get("feasibility", "unknown")
        genes = early_intervention.get("onset_druggable_ts_genes", [])
        predictions.append({
            "id": "P5.3",
            "prediction": (
                f"Early pharmacological intervention targeting "
                f"{', '.join(genes[:3])} during the onset window (4-7 years) "
                "could prevent tic escalation by modulating pathogenic "
                "pathways before compensatory circuit failure occurs."
            ),
            "evidence": {
                "n_druggable_ts_at_onset": early_intervention.get(
                    "n_onset_druggable_ts"
                ),
                "feasibility": feasibility,
                "druggable_classes": early_intervention.get(
                    "druggable_class_breakdown"
                ),
            },
            "validation": (
                "Retrospective analysis of children treated with onset-window "
                "target medications for comorbid conditions — do they show "
                "different tic trajectories than untreated TS patients?"
            ),
            "confidence": "low-medium" if feasibility == "low" else "medium",
        })

    # P5.4: Therapeutic score trajectory
    if not therapeutic_scores.empty:
        postnatal = therapeutic_scores[
            therapeutic_scores["dev_stage"].isin(POSTNATAL_STAGES)
        ]
        if not postnatal.empty:
            best_stage = postnatal.loc[
                postnatal["therapeutic_score"].idxmax(), "dev_stage"
            ]
            best_window = _stage_to_clinical_window(best_stage)
            predictions.append({
                "id": "P5.4",
                "prediction": (
                    f"The developmental stage with highest therapeutic "
                    f"amenability is {best_stage} ({best_window}), where "
                    "circuit malleability, E/I imbalance, and druggable "
                    "target density converge to create the optimal "
                    "intervention window."
                ),
                "evidence": {
                    "best_stage": best_stage,
                    "best_window": best_window,
                    "score_trajectory": {
                        row["dev_stage"]: row["therapeutic_score"]
                        for _, row in postnatal.iterrows()
                    },
                },
                "validation": (
                    "Meta-analysis of age-stratified treatment response data "
                    "across TS pharmacological trials to test whether "
                    f"{best_stage}-age patients show superior treatment "
                    "response."
                ),
                "confidence": "medium",
            })

    # P5.5: GABAergic augmentation timing
    if not druggable_targets.empty:
        gaba_targets = druggable_targets[
            druggable_targets["gene_symbol"].isin(
                {"GABRA1", "GABRG2", "GAD1", "GAD2", "SLC6A1", "SLC32A1"}
            )
        ]
        if not gaba_targets.empty:
            predictions.append({
                "id": "P5.5",
                "prediction": (
                    "GABAergic augmentation therapy (targeting GABA-A "
                    "receptors or GABA transporters) should be initiated "
                    "during late childhood or early adolescence, before the "
                    "natural remission window, to support compensatory "
                    "inhibitory circuit maturation in at-risk patients."
                ),
                "evidence": {
                    "gaba_targets_found": gaba_targets["gene_symbol"].tolist(),
                    "gaba_peak_stages": gaba_targets.set_index(
                        "gene_symbol"
                    )["peak_stage"].to_dict(),
                },
                "validation": (
                    "Randomized trial of low-dose GABAergic medication in "
                    "late-childhood TS patients at high persistence risk, "
                    "measuring tic trajectory through adolescence vs placebo."
                ),
                "confidence": "medium",
            })

    return predictions


# ── Pipeline runner ────────────────────────────────────────────────────────


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Run Phase 5 therapeutic window prediction analysis."""
    import importlib
    step09_mod = importlib.import_module(
        "bioagentics.tourettes.developmental_trajectory.09_persistence_remission_model"
    )

    trajectory_path = output_dir / "expression_trajectories.csv"
    phase4_dir = output_dir / "phase4_persistence_remission"
    phase5_dir = output_dir / "phase5_therapeutic_windows"
    phase5_dir.mkdir(parents=True, exist_ok=True)

    if not trajectory_path.exists():
        msg = (
            f"Trajectory data not found at {trajectory_path}. "
            "Run steps 01-03 first."
        )
        logger.error(msg)
        return {"error": msg}

    logger.info("Loading trajectory data from %s", trajectory_path)
    trajectories = pd.read_csv(trajectory_path)

    # Load Phase 4 scores (or recompute)
    remission_path = phase4_dir / "remission_scores.csv"
    persistence_path = phase4_dir / "persistence_scores.csv"

    if remission_path.exists():
        remission_df = pd.read_csv(remission_path)
    else:
        logger.info("Recomputing remission scores")
        remission_df = step09_mod.compute_remission_score(trajectories)

    if persistence_path.exists():
        persistence_df = pd.read_csv(persistence_path)
    else:
        logger.info("Recomputing persistence scores")
        persistence_df = step09_mod.compute_persistence_score(trajectories)

    # ── Therapeutic window scores ──────────────────────────────────────
    logger.info("Computing therapeutic window scores")
    therapeutic_scores = compute_therapeutic_window_scores(
        remission_df, persistence_df, trajectories,
    )
    if not therapeutic_scores.empty:
        therapeutic_scores.to_csv(
            phase5_dir / "therapeutic_window_scores.csv", index=False,
        )

    # ── DBS efficacy modelling ─────────────────────────────────────────
    logger.info("Modelling DBS efficacy window")
    dbs_analysis = model_dbs_efficacy_window(remission_df, persistence_df)
    with open(phase5_dir / "dbs_efficacy_analysis.json", "w") as f:
        json.dump(dbs_analysis, f, indent=2)

    # ── Druggable targets ──────────────────────────────────────────────
    logger.info("Identifying druggable targets")
    druggable_targets = identify_druggable_targets(trajectories)
    if not druggable_targets.empty:
        druggable_targets.to_csv(
            phase5_dir / "druggable_targets.csv", index=False,
        )

    # ── Early intervention ─────────────────────────────────────────────
    logger.info("Analysing early intervention feasibility")
    early_intervention = analyze_early_intervention(trajectories)
    with open(phase5_dir / "early_intervention.json", "w") as f:
        json.dump(early_intervention, f, indent=2)

    # ── Predictions ────────────────────────────────────────────────────
    logger.info("Generating Phase 5 predictions")
    predictions = generate_phase5_predictions(
        therapeutic_scores, dbs_analysis, druggable_targets,
        early_intervention,
    )
    with open(phase5_dir / "testable_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # ── Summary ────────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 5: Therapeutic Window Prediction",
        "n_stages_scored": len(therapeutic_scores),
        "optimal_dbs_window": dbs_analysis.get("optimal_window"),
        "n_druggable_targets": len(druggable_targets),
        "early_intervention_feasibility": early_intervention.get("feasibility"),
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    with open(phase5_dir / "phase5_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 5 complete: %d predictions, %d druggable targets identified",
        len(predictions), len(druggable_targets),
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 5: Therapeutic window prediction"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(output_dir=args.output)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\nPhase 5: Therapeutic Window Prediction")
    print(f"  Stages scored: {summary['n_stages_scored']}")
    print(f"  Optimal DBS window: {summary['optimal_dbs_window']}")
    print(f"  Druggable targets: {summary['n_druggable_targets']}")
    print(f"  Early intervention: {summary['early_intervention_feasibility']}")
    print(f"  Predictions: {summary['n_predictions']}")

    for p in summary["predictions"]:
        print(f"\n  [{p['id']}] ({p['confidence']})")
        print(f"    {p['prediction'][:100]}...")


if __name__ == "__main__":
    main()
