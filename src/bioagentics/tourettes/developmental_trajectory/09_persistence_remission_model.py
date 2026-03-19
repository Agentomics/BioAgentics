"""Step 09 — Persistence vs. remission molecular model (Phase 4).

Integrates Phases 1-3 to build a computational model predicting molecular
features of persistent vs. remitting Tourette syndrome:

  1. Remission signature: PV interneuron maturation + cortical GABA
     strengthening → increased inhibitory control over CSTC circuit
  2. Persistence signature: Disrupted PV interneuron development →
     failure of compensatory inhibition
  3. Striatal zonation attenuation: Dorsal-zone gene spatial convergence
     during adolescence may reduce aberrant CSTC activity
  4. Hormone modulation: AR/ESR1/ESR2 trajectories vs. cell-type dynamics
  5. Predictive gene scoring: rank genes by persistence/remission
     discriminative power
  6. Testable predictions for clinical validation

Task: #786 (Phase 4: Persistence vs. Remission Model)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.09_persistence_remission_model
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
from bioagentics.data.tourettes.gene_sets import (
    get_gene_set,
    get_celltype_markers,
    list_celltype_markers,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

# ── Clinical trajectory stage mapping ──────────────────────────────────────

POSTNATAL_STAGES = [
    "infancy", "early_childhood", "late_childhood", "adolescence", "adulthood",
]

# Stages before/during/after the remission boundary (~15-18y)
PRE_REMISSION_STAGES = ["infancy", "early_childhood", "late_childhood"]
REMISSION_STAGES = ["adolescence"]
POST_REMISSION_STAGES = ["adulthood"]

# ── Dorsal striatal zone markers ───────────────────────────────────────────
# Curated from literature on dorsal (sensorimotor) vs ventral (limbic)
# striatal compartments. Dorsal caudate/putamen receives CSTC motor circuit
# input — the zone most relevant to tic generation.
# These genes show dorsal-zone-enriched expression that attenuates with age
# (consistent with mesoscale striatal atlas findings).

DORSAL_ZONE_MARKERS: dict[str, str] = {
    "FOXP2": "dorsal striatal enrichment; corticostriatal circuit identity",
    "EPHA4": "EphA4 receptor; dorsal striatal topography",
    "MET": "MET receptor tyrosine kinase; dorsal striatal circuit formation",
    "SEMA3A": "semaphorin 3A; dorsal striatal axon guidance",
    "CACNA1A": "P/Q-type Ca2+ channel; dorsal striatal synaptic transmission",
    "KCND2": "Kv4.2; dorsal striatal MSN excitability",
    "GRIA3": "GluA3; dorsal striatal glutamatergic input",
    "GRID2": "GluD2; dorsal striatal cerebellar-like signaling",
}


# ── Remission signature computation ───────────────────────────────────────


def compute_remission_score(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Compute a per-stage remission-associated score from inhibitory maturation.

    The remission signature captures the developmental increase of inhibitory
    circuit components: PV interneuron markers + GABA signaling genes. A high
    score in adolescence supports the compensatory inhibition hypothesis.

    Parameters
    ----------
    trajectory_df
        DataFrame with columns: gene_symbol, dev_stage, cstc_region,
        mean_log2_rpkm.
    region
        CSTC region to analyze.

    Returns
    -------
    DataFrame with columns: dev_stage, pv_score, gaba_score,
    remission_score, stage_order.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region].copy()
    if region_df.empty:
        return pd.DataFrame()

    # PV interneuron markers
    pv_markers = set(get_celltype_markers("pv_interneuron").keys())
    # GABA signaling genes (from step 07 pathway)
    gaba_genes = {"GAD1", "GAD2", "SLC32A1", "GABRA1", "GABRA2",
                  "GABRB2", "GABRG2", "SLC6A1", "ABAT"}

    available = set(region_df["gene_symbol"].unique())

    def _stage_zscore(gene_set: set[str]) -> dict[str, float]:
        found = gene_set & available
        if not found:
            return {}
        sub = region_df[region_df["gene_symbol"].isin(found)].copy()
        # Z-score per gene
        g_mean = sub.groupby("gene_symbol")["mean_log2_rpkm"].transform("mean")
        g_std = sub.groupby("gene_symbol")["mean_log2_rpkm"].transform("std").replace(0, 1)
        sub["z"] = (sub["mean_log2_rpkm"] - g_mean) / g_std
        return sub.groupby("dev_stage")["z"].mean().to_dict()

    pv_scores = _stage_zscore(pv_markers)
    gaba_scores = _stage_zscore(gaba_genes)

    if not pv_scores and not gaba_scores:
        return pd.DataFrame()

    records = []
    for stage in stage_order:
        pv = pv_scores.get(stage, 0.0)
        gaba = gaba_scores.get(stage, 0.0)
        records.append({
            "dev_stage": stage,
            "pv_score": float(pv),
            "gaba_score": float(gaba),
            "remission_score": float((pv + gaba) / 2),
            "stage_order": stage_order.index(stage),
        })

    return pd.DataFrame(records).sort_values("stage_order").reset_index(drop=True)


def compute_persistence_score(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Compute a per-stage persistence-associated score.

    The persistence signature reflects failure of compensatory inhibition:
    low PV maturation combined with sustained excitatory drive. A flat or
    declining score in adolescence suggests persistence risk.

    Returns
    -------
    DataFrame with columns: dev_stage, excitatory_score, inhibitory_deficit,
    persistence_score, stage_order.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region].copy()
    if region_df.empty:
        return pd.DataFrame()

    # Excitatory drive: glutamate signaling + dopamine markers
    excitatory_genes = {"GRIN1", "GRIN2A", "GRIN2B", "GRIA1", "GRIA2",
                        "GRM5", "SLC17A7", "SLC17A6",
                        "DRD1", "DRD2", "TH", "SLC6A3"}
    # Inhibitory maturation markers (inverted — low = persistence)
    inhibitory_genes = set(get_celltype_markers("pv_interneuron").keys())
    inhibitory_genes.update({"GAD1", "GAD2", "SLC32A1"})

    available = set(region_df["gene_symbol"].unique())

    def _stage_zscore(gene_set: set[str]) -> dict[str, float]:
        found = gene_set & available
        if not found:
            return {}
        sub = region_df[region_df["gene_symbol"].isin(found)].copy()
        g_mean = sub.groupby("gene_symbol")["mean_log2_rpkm"].transform("mean")
        g_std = sub.groupby("gene_symbol")["mean_log2_rpkm"].transform("std").replace(0, 1)
        sub["z"] = (sub["mean_log2_rpkm"] - g_mean) / g_std
        return sub.groupby("dev_stage")["z"].mean().to_dict()

    exc_scores = _stage_zscore(excitatory_genes)
    inh_scores = _stage_zscore(inhibitory_genes)

    if not exc_scores and not inh_scores:
        return pd.DataFrame()

    records = []
    for stage in stage_order:
        exc = exc_scores.get(stage, 0.0)
        inh = inh_scores.get(stage, 0.0)
        # Persistence = high excitation + low inhibition (inverted)
        records.append({
            "dev_stage": stage,
            "excitatory_score": float(exc),
            "inhibitory_deficit": float(-inh),
            "persistence_score": float((exc + (-inh)) / 2),
            "stage_order": stage_order.index(stage),
        })

    return pd.DataFrame(records).sort_values("stage_order").reset_index(drop=True)


# ── Remission vs persistence boundary analysis ────────────────────────────


def test_remission_adolescent_increase(
    remission_df: pd.DataFrame,
) -> dict:
    """Test whether remission score increases significantly in adolescence.

    Compares remission score in adolescence vs. earlier postnatal stages.
    """
    if remission_df.empty:
        return {"hypothesis": "Remission score adolescent increase", "error": "no data"}

    postnatal = remission_df[remission_df["dev_stage"].isin(POSTNATAL_STAGES)]
    if postnatal.empty:
        return {"hypothesis": "Remission score adolescent increase",
                "error": "no postnatal data"}

    adol = postnatal[postnatal["dev_stage"].isin(REMISSION_STAGES)]["remission_score"].values
    pre = postnatal[postnatal["dev_stage"].isin(PRE_REMISSION_STAGES)]["remission_score"].values

    if len(adol) == 0 or len(pre) == 0:
        return {"hypothesis": "Remission score adolescent increase",
                "error": "insufficient data"}

    adol_mean = float(np.mean(adol))
    pre_mean = float(np.mean(pre))
    effect = adol_mean - pre_mean

    # Spearman trend across postnatal stages
    orders = postnatal["stage_order"].values
    scores = postnatal["remission_score"].values
    if len(orders) >= 3:
        rho, p_val = stats.spearmanr(orders, scores)
    else:
        rho, p_val = 0.0, 1.0

    return {
        "hypothesis": "Remission score adolescent increase",
        "adolescent_mean": adol_mean,
        "pre_remission_mean": pre_mean,
        "effect_size": float(effect),
        "increases_in_adolescence": bool(effect > 0),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "significant_trend": bool(p_val < 0.05),
        "trajectory": {
            row["dev_stage"]: float(row["remission_score"])
            for _, row in postnatal.iterrows()
        },
    }


def test_persistence_plateau(
    persistence_df: pd.DataFrame,
) -> dict:
    """Test whether persistence score plateaus or remains elevated in adolescence.

    A flat/increasing persistence score across adolescence suggests failure
    of compensatory inhibition.
    """
    if persistence_df.empty:
        return {"hypothesis": "Persistence score plateau", "error": "no data"}

    postnatal = persistence_df[persistence_df["dev_stage"].isin(POSTNATAL_STAGES)]
    if postnatal.empty:
        return {"hypothesis": "Persistence score plateau",
                "error": "no postnatal data"}

    adol = postnatal[postnatal["dev_stage"].isin(REMISSION_STAGES)]["persistence_score"].values
    pre = postnatal[postnatal["dev_stage"].isin(PRE_REMISSION_STAGES)]["persistence_score"].values

    if len(adol) == 0 or len(pre) == 0:
        return {"hypothesis": "Persistence score plateau",
                "error": "insufficient data"}

    adol_mean = float(np.mean(adol))
    pre_mean = float(np.mean(pre))

    # Persistence persists if score does NOT decline
    does_not_decline = adol_mean >= pre_mean - 0.1  # tolerance for noise

    # Trend test
    orders = postnatal["stage_order"].values
    scores = postnatal["persistence_score"].values
    if len(orders) >= 3:
        rho, p_val = stats.spearmanr(orders, scores)
    else:
        rho, p_val = 0.0, 1.0

    return {
        "hypothesis": "Persistence score plateau",
        "adolescent_mean": adol_mean,
        "pre_remission_mean": pre_mean,
        "does_not_decline": bool(does_not_decline),
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "trajectory": {
            row["dev_stage"]: float(row["persistence_score"])
            for _, row in postnatal.iterrows()
        },
    }


# ── Striatal zonation attenuation ─────────────────────────────────────────


def compute_zonation_attenuation(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
    dorsal_markers: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Compute dorsal-zone gene expression variability across development.

    Tests whether dorsal-zone-specific genes lose expression differentiation
    (spatial attenuation) during adolescence, which could provide a molecular
    substrate for tic remission.

    The attenuation score is the coefficient of variation (CV) of dorsal-zone
    marker expression per stage. Decreasing CV suggests transcriptomic
    convergence (spatial attenuation).

    Returns
    -------
    DataFrame with columns: dev_stage, mean_expression, cv, n_markers,
    stage_order.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    if dorsal_markers is None:
        dorsal_markers = DORSAL_ZONE_MARKERS

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region].copy()
    if region_df.empty:
        return pd.DataFrame()

    marker_genes = set(dorsal_markers.keys())
    available = marker_genes & set(region_df["gene_symbol"].unique())
    if not available:
        return pd.DataFrame()

    sub = region_df[region_df["gene_symbol"].isin(available)]

    records = []
    for stage in sub["dev_stage"].unique():
        stage_data = sub[sub["dev_stage"] == stage]["mean_log2_rpkm"]
        if len(stage_data) < 2:
            continue
        mean_expr = float(stage_data.mean())
        std_expr = float(stage_data.std())
        cv = std_expr / max(abs(mean_expr), 1e-6)
        records.append({
            "dev_stage": stage,
            "mean_expression": mean_expr,
            "cv": cv,
            "n_markers": len(stage_data),
            "stage_order": stage_order.index(stage) if stage in stage_order else -1,
        })

    if not records:
        return pd.DataFrame()

    return pd.DataFrame(records).sort_values("stage_order").reset_index(drop=True)


def test_zonation_attenuation_trend(
    zonation_df: pd.DataFrame,
) -> dict:
    """Test whether dorsal-zone CV decreases across postnatal development.

    A negative Spearman correlation of CV with stage order indicates
    spatial attenuation (gene expression convergence) over time.
    """
    if zonation_df.empty:
        return {"hypothesis": "Striatal zonation attenuation", "error": "no data"}

    postnatal = zonation_df[zonation_df["dev_stage"].isin(POSTNATAL_STAGES)]
    if len(postnatal) < 3:
        return {"hypothesis": "Striatal zonation attenuation",
                "error": "insufficient postnatal stages"}

    orders = postnatal["stage_order"].values
    cvs = postnatal["cv"].values
    rho, p_val = stats.spearmanr(orders, cvs)

    # Check if adolescence has lower CV than childhood
    adol_cv = postnatal[postnatal["dev_stage"] == "adolescence"]["cv"].values
    child_cv = postnatal[
        postnatal["dev_stage"].isin(["early_childhood", "late_childhood"])
    ]["cv"].values

    attenuation_in_adolescence = False
    if len(adol_cv) > 0 and len(child_cv) > 0:
        attenuation_in_adolescence = bool(np.mean(adol_cv) < np.mean(child_cv))

    return {
        "hypothesis": "Striatal zonation attenuation",
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "cv_decreases": bool(rho < 0),
        "significant_trend": bool(p_val < 0.05),
        "attenuation_in_adolescence": attenuation_in_adolescence,
        "cv_trajectory": {
            row["dev_stage"]: float(row["cv"])
            for _, row in postnatal.iterrows()
        },
    }


# ── Hormone modulation analysis ───────────────────────────────────────────


def analyze_hormone_modulation(
    trajectory_df: pd.DataFrame,
    celltype_scores_df: pd.DataFrame | None = None,
    region: str = "striatum",
) -> dict:
    """Analyze hormone receptor expression trajectory and correlation with
    cell-type dynamics.

    Tests whether AR/ESR1/ESR2 expression peaks near puberty and correlates
    with PV interneuron or D2:D1 ratio changes.
    """
    if trajectory_df.empty:
        return {"hypothesis": "Hormone modulation", "error": "no data"}

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region]
    if region_df.empty:
        return {"hypothesis": "Hormone modulation", "error": "no region data"}

    hormone_genes = set(get_gene_set("hormone_receptors").keys())
    available = hormone_genes & set(region_df["gene_symbol"].unique())

    if not available:
        return {"hypothesis": "Hormone modulation",
                "error": "no hormone receptor genes in data"}

    # Expression trajectory per hormone receptor
    trajectories: dict[str, dict[str, float]] = {}
    for gene in sorted(available):
        gene_data = region_df[region_df["gene_symbol"] == gene]
        trajectories[gene] = {
            row["dev_stage"]: float(row["mean_log2_rpkm"])
            for _, row in gene_data.iterrows()
        }

    # Test pubertal peak (late_childhood / adolescence vs others)
    pubertal_stages = {"late_childhood", "adolescence"}
    postnatal_data = region_df[
        (region_df["gene_symbol"].isin(available))
        & (region_df["dev_stage"].isin(POSTNATAL_STAGES))
    ]

    pub_expr = postnatal_data[
        postnatal_data["dev_stage"].isin(pubertal_stages)
    ]["mean_log2_rpkm"].values
    nonpub_expr = postnatal_data[
        ~postnatal_data["dev_stage"].isin(pubertal_stages)
    ]["mean_log2_rpkm"].values

    pubertal_elevation = False
    pub_effect = 0.0
    pub_p = 1.0
    if len(pub_expr) >= 1 and len(nonpub_expr) >= 1:
        pub_effect = float(np.mean(pub_expr) - np.mean(nonpub_expr))
        pubertal_elevation = pub_effect > 0
        if len(pub_expr) >= 2 and len(nonpub_expr) >= 2:
            _, pub_p = stats.mannwhitneyu(
                pub_expr, nonpub_expr, alternative="greater"
            )
            pub_p = float(pub_p)

    # Correlation with PV scores if available
    pv_correlation = None
    if celltype_scores_df is not None and not celltype_scores_df.empty:
        pv_data = celltype_scores_df[
            (celltype_scores_df["celltype"] == "pv_interneuron")
            & (celltype_scores_df["cstc_region"] == region)
        ]
        if not pv_data.empty:
            # Mean hormone expression per stage
            hr_stage = postnatal_data.groupby("dev_stage")["mean_log2_rpkm"].mean()
            pv_stage = pv_data.set_index("dev_stage")["score"]
            common = sorted(set(hr_stage.index) & set(pv_stage.index))
            if len(common) >= 3:
                hr_vals = [hr_stage[s] for s in common]
                pv_vals = [pv_stage[s] for s in common]
                rho, p = stats.spearmanr(hr_vals, pv_vals)
                pv_correlation = {
                    "spearman_rho": float(rho),
                    "spearman_p": float(p),
                    "stages_compared": common,
                }

    return {
        "hypothesis": "Hormone modulation",
        "region": region,
        "hormone_genes_found": sorted(available),
        "pubertal_elevation": bool(pubertal_elevation),
        "pubertal_effect_size": float(pub_effect),
        "pubertal_p": float(pub_p),
        "pv_correlation": pv_correlation,
        "gene_trajectories": trajectories,
    }


# ── Predictive gene scoring ──────────────────────────────────────────────


def score_genes_persistence_remission(
    trajectory_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Score each gene by how well its trajectory discriminates remission
    vs. persistence patterns.

    For each gene, computes:
    - adolescent_change: expression change from late childhood → adolescence
    - remission_direction: positive = increases (remission-like),
      negative = decreases/flat (persistence-like)
    - discrimination_score: |adolescent_change| * consistency across regions

    Returns sorted DataFrame with gene rankings.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    stage_order = list(DEV_STAGES.keys())
    region_df = trajectory_df[trajectory_df["cstc_region"] == region]
    if region_df.empty:
        return pd.DataFrame()

    records = []
    for gene in region_df["gene_symbol"].unique():
        gene_data = region_df[region_df["gene_symbol"] == gene]
        stage_expr = gene_data.set_index("dev_stage")["mean_log2_rpkm"]

        late_child = stage_expr.get("late_childhood")
        adolescence = stage_expr.get("adolescence")
        adulthood = stage_expr.get("adulthood")

        if late_child is None or adolescence is None:
            continue

        adol_change = float(adolescence - late_child)
        adult_change = float(adulthood - late_child) if adulthood is not None else adol_change

        # Postnatal trend
        postnatal = gene_data[gene_data["dev_stage"].isin(POSTNATAL_STAGES)]
        if len(postnatal) >= 3:
            rho, _ = stats.spearmanr(
                [stage_order.index(s) for s in postnatal["dev_stage"]],
                postnatal["mean_log2_rpkm"].values,
            )
        else:
            rho = 0.0

        # TS gene membership
        ts_combined = set(get_gene_set("ts_combined").keys())
        is_ts_gene = gene in ts_combined

        records.append({
            "gene_symbol": gene,
            "late_childhood_expr": float(late_child),
            "adolescence_expr": float(adolescence),
            "adolescent_change": adol_change,
            "adult_change": adult_change,
            "postnatal_trend_rho": float(rho),
            "remission_direction": "remission" if adol_change > 0 else "persistence",
            "discrimination_score": abs(adol_change),
            "is_ts_gene": is_ts_gene,
        })

    if not records:
        return pd.DataFrame()

    result = pd.DataFrame(records)
    result = result.sort_values("discrimination_score", ascending=False)
    return result.reset_index(drop=True)


# ── Testable predictions ─────────────────────────────────────────────────


def generate_predictions(
    remission_test: dict,
    persistence_test: dict,
    zonation_test: dict,
    hormone_test: dict,
    gene_scores: pd.DataFrame,
) -> list[dict]:
    """Generate testable predictions from the persistence/remission model.

    Each prediction includes a hypothesis statement, supporting evidence
    from this analysis, and suggested validation approach.
    """
    predictions: list[dict] = []

    # Prediction 1: PV interneuron maturation timing
    if remission_test.get("increases_in_adolescence"):
        predictions.append({
            "id": "P4.1",
            "prediction": (
                "PV interneuron markers and GABAergic genes increase expression "
                "in striatum during adolescence, providing the molecular basis "
                "for spontaneous tic remission via enhanced inhibitory control."
            ),
            "evidence": {
                "remission_score_trend": remission_test.get("spearman_rho"),
                "adolescent_increase": remission_test.get("effect_size"),
            },
            "validation": (
                "Compare PV+ interneuron density in postmortem striatum of "
                "remitted vs. persistent TS patients using immunohistochemistry "
                "or snRNA-seq."
            ),
            "confidence": "medium-high",
        })

    # Prediction 2: Persistence from failed compensatory inhibition
    if persistence_test.get("does_not_decline"):
        predictions.append({
            "id": "P4.2",
            "prediction": (
                "Persistent TS reflects a failure of compensatory inhibitory "
                "circuit maturation: excitatory/inhibitory balance does not "
                "shift toward inhibition during adolescence."
            ),
            "evidence": {
                "persistence_plateau": persistence_test.get("does_not_decline"),
                "trajectory": persistence_test.get("trajectory"),
            },
            "validation": (
                "Longitudinal MRS study measuring GABA/glutamate ratio in "
                "striatum of TS patients from childhood through adulthood, "
                "comparing remitters vs persisters."
            ),
            "confidence": "medium",
        })

    # Prediction 3: Striatal zonation attenuation
    if zonation_test.get("attenuation_in_adolescence"):
        predictions.append({
            "id": "P4.3",
            "prediction": (
                "Dorsal striatal zone gene expression converges during "
                "adolescence (spatial attenuation), reducing aberrant "
                "CSTC circuit activity. Delayed attenuation predicts tic "
                "persistence."
            ),
            "evidence": {
                "cv_trend": zonation_test.get("spearman_rho"),
                "attenuation_in_adolescence": True,
            },
            "validation": (
                "Spatial transcriptomics (MERFISH/Visium) of striatum "
                "across developmental ages to confirm dorsal-zone gene "
                "expression convergence timing."
            ),
            "confidence": "medium",
        })

    # Prediction 4: Hormone modulation window
    if hormone_test.get("pubertal_elevation"):
        predictions.append({
            "id": "P4.4",
            "prediction": (
                "Gonadal steroid hormone receptors show elevated expression "
                "during the pubertal window, modulating dopaminergic and "
                "GABAergic signaling in striatum. This hormonal surge "
                "contributes to both peak tic severity and subsequent "
                "remission via promotion of inhibitory circuit maturation."
            ),
            "evidence": {
                "pubertal_effect": hormone_test.get("pubertal_effect_size"),
                "pv_correlation": hormone_test.get("pv_correlation"),
            },
            "validation": (
                "Test whether anti-androgen therapy timing affects tic "
                "trajectory in adolescent TS patients; examine AR/ESR "
                "expression in TS vs control striatal tissue."
            ),
            "confidence": "low-medium",
        })

    # Prediction 5: Top predictive genes
    if not gene_scores.empty:
        ts_predictive = gene_scores[gene_scores["is_ts_gene"]].head(5)
        if not ts_predictive.empty:
            top_genes = ts_predictive["gene_symbol"].tolist()
            predictions.append({
                "id": "P4.5",
                "prediction": (
                    f"The TS risk genes {', '.join(top_genes)} show the "
                    "largest developmental expression changes at the "
                    "persistence/remission boundary and are candidate "
                    "biomarkers for predicting tic outcome."
                ),
                "evidence": {
                    "top_genes": [
                        {
                            "gene": row["gene_symbol"],
                            "adolescent_change": row["adolescent_change"],
                            "direction": row["remission_direction"],
                        }
                        for _, row in ts_predictive.iterrows()
                    ],
                },
                "validation": (
                    "Measure expression of these genes in accessible tissue "
                    "(blood, CSF) of TS patients during adolescence and "
                    "correlate with clinical outcome at follow-up."
                ),
                "confidence": "medium",
            })

    return predictions


# ── Pipeline runner ────────────────────────────────────────────────────────


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Run Phase 4 persistence vs remission model analysis."""
    trajectory_path = output_dir / "expression_trajectories.csv"
    phase3_dir = output_dir / "phase3_celltype_dynamics"
    phase4_dir = output_dir / "phase4_persistence_remission"
    phase4_dir.mkdir(parents=True, exist_ok=True)

    if not trajectory_path.exists():
        msg = (
            f"Trajectory data not found at {trajectory_path}. "
            "Run steps 01-03 first."
        )
        logger.error(msg)
        return {"error": msg}

    logger.info("Loading trajectory data from %s", trajectory_path)
    trajectories = pd.read_csv(trajectory_path)

    # Load Phase 3 cell-type scores if available
    ct_scores_path = phase3_dir / "celltype_scores.csv"
    celltype_scores = None
    if ct_scores_path.exists():
        logger.info("Loading Phase 3 cell-type scores")
        celltype_scores = pd.read_csv(ct_scores_path)

    # ── Remission signature ─────────────────────────────────────────
    logger.info("Computing remission signature")
    remission_df = compute_remission_score(trajectories)
    if not remission_df.empty:
        remission_df.to_csv(phase4_dir / "remission_scores.csv", index=False)

    # ── Persistence signature ───────────────────────────────────────
    logger.info("Computing persistence signature")
    persistence_df = compute_persistence_score(trajectories)
    if not persistence_df.empty:
        persistence_df.to_csv(phase4_dir / "persistence_scores.csv", index=False)

    # ── Zonation attenuation ────────────────────────────────────────
    logger.info("Computing striatal zonation attenuation")
    zonation_df = compute_zonation_attenuation(trajectories)
    if not zonation_df.empty:
        zonation_df.to_csv(phase4_dir / "zonation_attenuation.csv", index=False)

    # ── Hypothesis tests ────────────────────────────────────────────
    logger.info("Running persistence/remission hypothesis tests")
    remission_test = test_remission_adolescent_increase(remission_df)
    persistence_test = test_persistence_plateau(persistence_df)
    zonation_test = test_zonation_attenuation_trend(zonation_df)
    hormone_test = analyze_hormone_modulation(
        trajectories, celltype_scores,
    )

    hypothesis_results = {
        "remission_increase": remission_test,
        "persistence_plateau": persistence_test,
        "zonation_attenuation": zonation_test,
        "hormone_modulation": hormone_test,
    }

    with open(phase4_dir / "hypothesis_tests.json", "w") as f:
        json.dump(hypothesis_results, f, indent=2)

    # ── Gene scoring ────────────────────────────────────────────────
    logger.info("Scoring genes for persistence/remission discrimination")
    gene_scores = score_genes_persistence_remission(trajectories)
    if not gene_scores.empty:
        gene_scores.to_csv(phase4_dir / "gene_scores.csv", index=False)

    # ── Predictions ─────────────────────────────────────────────────
    logger.info("Generating testable predictions")
    predictions = generate_predictions(
        remission_test, persistence_test, zonation_test,
        hormone_test, gene_scores,
    )

    with open(phase4_dir / "testable_predictions.json", "w") as f:
        json.dump(predictions, f, indent=2)

    # ── Summary ─────────────────────────────────────────────────────
    summary = {
        "phase": "Phase 4: Persistence vs. Remission Model",
        "hypothesis_results": hypothesis_results,
        "n_genes_scored": len(gene_scores) if not gene_scores.empty else 0,
        "n_ts_genes_scored": int(gene_scores["is_ts_gene"].sum())
        if not gene_scores.empty else 0,
        "n_predictions": len(predictions),
        "predictions": predictions,
    }

    with open(phase4_dir / "phase4_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(
        "Phase 4 complete: %d predictions generated, %d genes scored",
        len(predictions), len(gene_scores),
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 4: Persistence vs. remission molecular model"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(output_dir=args.output)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\nPhase 4: Persistence vs. Remission Molecular Model")
    print(f"  Genes scored: {summary['n_genes_scored']}")
    print(f"  TS genes scored: {summary['n_ts_genes_scored']}")
    print(f"  Predictions generated: {summary['n_predictions']}")

    for p in summary["predictions"]:
        print(f"\n  [{p['id']}] ({p['confidence']})")
        print(f"    {p['prediction'][:100]}...")


if __name__ == "__main__":
    main()
