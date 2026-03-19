"""Step 08 — Cell-type developmental dynamics (Phase 3).

Estimates cell-type proportion trajectories across BrainSpan developmental
stages using marker gene scoring, then tests hypotheses about striatal
cell-type dynamics in relation to TS clinical trajectory:

  1. Cholinergic interneuron signature peaks during tic-onset window (5-7y)
  2. PV interneuron signature increases during remission window (15-18y)
  3. D2-MSN:D1-MSN ratio shifts across development
  4. Microglia activation has a developmental component

Uses canonical cell-type markers from literature. Can be supplemented with
Wang et al. 2025 snRNA-seq markers when data_curator task #237 is resolved.

Task: #785 (Phase 3: Cell-Type Developmental Dynamics)
Project: ts-developmental-trajectory-modeling

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.08_celltype_deconvolution
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
    get_celltype_markers,
    list_celltype_markers,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"

# ── Clinical trajectory stage mapping ──────────────────────────────────────
# Maps TS clinical milestones to BrainSpan developmental stages for testing.

TS_TRAJECTORY_STAGES = {
    "pre_onset": ["infancy", "early_childhood"],
    "onset": ["early_childhood", "late_childhood"],
    "peak_severity": ["late_childhood"],
    "remission": ["adolescence"],
    "post_remission": ["adulthood"],
}


# ── Marker gene scoring ───────────────────────────────────────────────────


def compute_celltype_scores(
    trajectory_df: pd.DataFrame,
    celltype_markers: dict[str, dict[str, str]] | None = None,
) -> pd.DataFrame:
    """Compute cell-type enrichment scores per developmental stage and region.

    For each cell type, computes the mean z-scored expression of its marker
    genes across samples in each stage/region combination. This gives a proxy
    for relative cell-type abundance trajectory.

    Parameters
    ----------
    trajectory_df
        DataFrame with columns: gene_symbol, dev_stage, cstc_region,
        mean_log2_rpkm (output of step 03).
    celltype_markers
        Dict mapping cell-type name to {gene_symbol: description}.
        If None, uses all canonical markers from gene_sets module.

    Returns
    -------
    DataFrame with columns: celltype, dev_stage, cstc_region, score,
    n_markers_found, n_markers_total, stage_order.
    """
    if trajectory_df.empty:
        return pd.DataFrame()

    if celltype_markers is None:
        celltype_markers = {
            name: get_celltype_markers(name)
            for name in list_celltype_markers()
        }

    stage_order = list(DEV_STAGES.keys())
    available_genes = set(trajectory_df["gene_symbol"].unique())

    records: list[dict] = []
    for ct_name, markers in celltype_markers.items():
        marker_genes = set(markers.keys())
        found = marker_genes & available_genes
        if not found:
            continue

        ct_data = trajectory_df[
            trajectory_df["gene_symbol"].isin(found)
        ].copy()

        # Z-score each gene across all stages/regions
        gene_means = ct_data.groupby("gene_symbol")["mean_log2_rpkm"].transform("mean")
        gene_stds = ct_data.groupby("gene_symbol")["mean_log2_rpkm"].transform("std")
        gene_stds = gene_stds.replace(0, 1)  # avoid div-by-zero for constant genes
        ct_data["zscore"] = (ct_data["mean_log2_rpkm"] - gene_means) / gene_stds

        for region in ct_data["cstc_region"].unique():
            region_data = ct_data[ct_data["cstc_region"] == region]
            for stage in region_data["dev_stage"].unique():
                stage_data = region_data[region_data["dev_stage"] == stage]
                score = stage_data["zscore"].mean()
                records.append({
                    "celltype": ct_name,
                    "dev_stage": stage,
                    "cstc_region": region,
                    "score": float(score),
                    "n_markers_found": len(found),
                    "n_markers_total": len(marker_genes),
                    "stage_order": stage_order.index(stage)
                    if stage in stage_order else -1,
                })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values(["celltype", "cstc_region", "stage_order"])
        result = result.reset_index(drop=True)
    return result


def compute_msn_ratio(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> pd.DataFrame:
    """Compute D2-MSN:D1-MSN score ratio across developmental stages.

    Parameters
    ----------
    scores_df
        Output of compute_celltype_scores.
    region
        CSTC region to analyze (default: striatum).

    Returns
    -------
    DataFrame with columns: dev_stage, d1_score, d2_score, d2_d1_ratio,
    stage_order.
    """
    if scores_df.empty:
        return pd.DataFrame()

    region_scores = scores_df[scores_df["cstc_region"] == region]

    d1 = region_scores[region_scores["celltype"] == "d1_msn"][
        ["dev_stage", "score", "stage_order"]
    ].rename(columns={"score": "d1_score"})

    d2 = region_scores[region_scores["celltype"] == "d2_msn"][
        ["dev_stage", "score"]
    ].rename(columns={"score": "d2_score"})

    if d1.empty or d2.empty:
        return pd.DataFrame()

    merged = d1.merge(d2, on="dev_stage", how="inner")

    # Shift scores to positive range before ratio (z-scores can be negative)
    shift = max(abs(merged["d1_score"].min()), abs(merged["d2_score"].min())) + 1
    merged["d2_d1_ratio"] = (merged["d2_score"] + shift) / (merged["d1_score"] + shift)

    return merged.sort_values("stage_order").reset_index(drop=True)


# ── Developmental trajectory hypothesis tests ─────────────────────────────


def test_onset_cholinergic_peak(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> dict:
    """Test whether cholinergic interneuron score peaks during onset window.

    Compares cholinergic scores in onset stages (early_childhood,
    late_childhood) vs. all other stages using a one-sided t-test.

    Returns dict with test statistic, p-value, and peak stage info.
    """
    return _test_celltype_window_peak(
        scores_df, "cholinergic_interneuron", region,
        TS_TRAJECTORY_STAGES["onset"],
        "Cholinergic interneuron peak at onset",
    )


def test_remission_pv_increase(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> dict:
    """Test whether PV interneuron score increases during remission window.

    Compares PV scores in adolescence vs. earlier postnatal stages using
    a one-sided t-test (testing for higher scores in remission window).

    Returns dict with test statistic, p-value, and effect info.
    """
    return _test_celltype_window_peak(
        scores_df, "pv_interneuron", region,
        TS_TRAJECTORY_STAGES["remission"],
        "PV interneuron increase during remission",
    )


def test_microglia_developmental_component(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> dict:
    """Test whether microglia scores vary significantly across development.

    Uses one-way ANOVA across postnatal developmental stages.

    Returns dict with F-statistic, p-value, and stage-wise scores.
    """
    if scores_df.empty:
        return {"hypothesis": "Microglia developmental dynamics", "error": "no data"}

    ct_data = scores_df[
        (scores_df["celltype"] == "microglia")
        & (scores_df["cstc_region"] == region)
    ].copy()

    # Only postnatal stages
    postnatal = ["infancy", "early_childhood", "late_childhood",
                 "adolescence", "adulthood"]
    ct_data = ct_data[ct_data["dev_stage"].isin(postnatal)]

    if ct_data.empty or ct_data["dev_stage"].nunique() < 2:
        return {"hypothesis": "Microglia developmental dynamics",
                "error": "insufficient data"}

    groups = [
        group["score"].values
        for _, group in ct_data.groupby("dev_stage")
        if len(group) > 0
    ]

    if len(groups) < 2:
        return {"hypothesis": "Microglia developmental dynamics",
                "error": "insufficient groups"}

    # With single observations per group, test trend instead
    ct_data = ct_data.sort_values("stage_order")
    stage_scores = ct_data.set_index("dev_stage")["score"].to_dict()

    # Spearman correlation of score with stage order
    orders = ct_data["stage_order"].values
    scores = ct_data["score"].values
    if len(orders) >= 3:
        rho, p_val = stats.spearmanr(orders, scores)
    else:
        rho, p_val = 0.0, 1.0

    peak_stage = ct_data.loc[ct_data["score"].idxmax(), "dev_stage"]
    trough_stage = ct_data.loc[ct_data["score"].idxmin(), "dev_stage"]

    return {
        "hypothesis": "Microglia developmental dynamics",
        "region": region,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "significant_trend": bool(p_val < 0.05),
        "peak_stage": peak_stage,
        "trough_stage": trough_stage,
        "stage_scores": stage_scores,
    }


def test_msn_ratio_trajectory(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> dict:
    """Test whether D2:D1 MSN ratio changes across development.

    Computes Spearman correlation of D2:D1 ratio with developmental stage
    order to test for a monotonic trend.

    Returns dict with correlation, p-value, and ratio trajectory.
    """
    ratio_df = compute_msn_ratio(scores_df, region)

    if ratio_df.empty or len(ratio_df) < 3:
        return {"hypothesis": "D2:D1 MSN ratio trajectory",
                "error": "insufficient data"}

    orders = ratio_df["stage_order"].values
    ratios = ratio_df["d2_d1_ratio"].values
    rho, p_val = stats.spearmanr(orders, ratios)

    peak_idx = ratio_df["d2_d1_ratio"].idxmax()
    peak_stage = ratio_df.loc[peak_idx, "dev_stage"]

    return {
        "hypothesis": "D2:D1 MSN ratio trajectory",
        "region": region,
        "spearman_rho": float(rho),
        "spearman_p": float(p_val),
        "significant_trend": bool(p_val < 0.05),
        "peak_ratio_stage": peak_stage,
        "trajectory": {
            row["dev_stage"]: float(row["d2_d1_ratio"])
            for _, row in ratio_df.iterrows()
        },
    }


def _test_celltype_window_peak(
    scores_df: pd.DataFrame,
    celltype: str,
    region: str,
    window_stages: list[str],
    hypothesis_name: str,
) -> dict:
    """Test whether a cell-type score is elevated during specified window stages.

    Compares mean score in window_stages vs. all other postnatal stages using
    a one-sided Mann-Whitney U test (or falls back to effect size if n is too
    small for meaningful p-values).
    """
    if scores_df.empty:
        return {"hypothesis": hypothesis_name, "error": "no data"}

    ct_data = scores_df[
        (scores_df["celltype"] == celltype)
        & (scores_df["cstc_region"] == region)
    ]

    if ct_data.empty:
        return {"hypothesis": hypothesis_name, "error": f"no {celltype} data"}

    # Only postnatal stages for comparison
    postnatal = ["infancy", "early_childhood", "late_childhood",
                 "adolescence", "adulthood"]
    ct_data = ct_data[ct_data["dev_stage"].isin(postnatal)]

    window_scores = ct_data[ct_data["dev_stage"].isin(window_stages)]["score"].values
    other_scores = ct_data[~ct_data["dev_stage"].isin(window_stages)]["score"].values

    if len(window_scores) == 0 or len(other_scores) == 0:
        return {"hypothesis": hypothesis_name, "error": "insufficient data"}

    window_mean = float(np.mean(window_scores))
    other_mean = float(np.mean(other_scores))
    effect_size = window_mean - other_mean

    # Mann-Whitney U test (one-sided: window > other)
    if len(window_scores) >= 2 and len(other_scores) >= 2:
        u_stat, p_two = stats.mannwhitneyu(
            window_scores, other_scores, alternative="greater"
        )
        p_val = float(p_two)
    else:
        u_stat = 0.0
        p_val = 1.0

    ct_sorted = ct_data.sort_values("stage_order")
    peak_stage = ct_sorted.loc[ct_sorted["score"].idxmax(), "dev_stage"]

    return {
        "hypothesis": hypothesis_name,
        "celltype": celltype,
        "region": region,
        "window_stages": window_stages,
        "window_mean": window_mean,
        "other_mean": other_mean,
        "effect_size": effect_size,
        "elevated_in_window": bool(effect_size > 0),
        "u_statistic": float(u_stat),
        "p_value": p_val,
        "significant": bool(p_val < 0.05),
        "peak_stage": peak_stage,
        "stage_scores": {
            row["dev_stage"]: float(row["score"])
            for _, row in ct_sorted.iterrows()
        },
    }


# ── Aggregate analysis ─────────────────────────────────────────────────────


def run_all_hypotheses(
    scores_df: pd.DataFrame,
    region: str = "striatum",
) -> list[dict]:
    """Run all four TS-relevant cell-type hypotheses.

    Returns list of hypothesis test result dicts.
    """
    return [
        test_onset_cholinergic_peak(scores_df, region),
        test_remission_pv_increase(scores_df, region),
        test_microglia_developmental_component(scores_df, region),
        test_msn_ratio_trajectory(scores_df, region),
    ]


# ── Pipeline runner ────────────────────────────────────────────────────────


def run(output_dir: Path = OUTPUT_DIR) -> dict:
    """Run Phase 3 cell-type deconvolution analysis.

    Loads Phase 1 trajectory data, computes cell-type scores, runs
    hypothesis tests, and writes results.
    """
    trajectory_path = output_dir / "expression_trajectories.csv"
    phase3_dir = output_dir / "phase3_celltype_dynamics"
    phase3_dir.mkdir(parents=True, exist_ok=True)

    if not trajectory_path.exists():
        msg = (
            f"Trajectory data not found at {trajectory_path}. "
            "Run steps 01-03 first."
        )
        logger.error(msg)
        return {"error": msg}

    logger.info("Loading trajectory data from %s", trajectory_path)
    trajectories = pd.read_csv(trajectory_path)

    logger.info("Computing cell-type scores for %d markers sets",
                len(list_celltype_markers()))
    scores = compute_celltype_scores(trajectories)

    if scores.empty:
        return {"error": "No cell-type scores computed — check marker gene overlap"}

    scores.to_csv(phase3_dir / "celltype_scores.csv", index=False)

    # D2:D1 ratio
    ratio_df = compute_msn_ratio(scores)
    if not ratio_df.empty:
        ratio_df.to_csv(phase3_dir / "d2_d1_ratio.csv", index=False)

    # Hypothesis tests
    logger.info("Running cell-type trajectory hypothesis tests")
    results = run_all_hypotheses(scores)

    with open(phase3_dir / "hypothesis_tests.json", "w") as f:
        json.dump(results, f, indent=2)

    # Summary
    n_significant = sum(1 for r in results if r.get("significant", False)
                        or r.get("significant_trend", False))

    summary = {
        "phase": "Phase 3: Cell-Type Developmental Dynamics",
        "celltype_markers_used": list_celltype_markers(),
        "n_celltypes": len(list_celltype_markers()),
        "n_stages": scores["dev_stage"].nunique(),
        "n_regions": scores["cstc_region"].nunique(),
        "hypotheses_tested": len(results),
        "hypotheses_significant": n_significant,
        "results": results,
    }

    with open(phase3_dir / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Phase 3 complete: %d/%d hypotheses significant",
                n_significant, len(results))
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Phase 3: Cell-type developmental dynamics"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    summary = run(output_dir=args.output)

    if "error" in summary:
        print(f"ERROR: {summary['error']}")
        return

    print("\nPhase 3: Cell-Type Developmental Dynamics")
    print(f"  Cell types analyzed: {summary['n_celltypes']}")
    print(f"  Hypotheses: {summary['hypotheses_significant']}/{summary['hypotheses_tested']} significant")

    for r in summary["results"]:
        status = "ERROR" if "error" in r else ("*" if r.get("significant", False) or r.get("significant_trend", False) else " ")
        hyp = r.get("hypothesis", "unknown")
        if "error" in r:
            print(f"  [{status}] {hyp}: {r['error']}")
        elif "p_value" in r:
            print(f"  [{status}] {hyp}: effect={r['effect_size']:.3f}, p={r['p_value']:.4f}")
        elif "spearman_p" in r:
            print(f"  [{status}] {hyp}: rho={r['spearman_rho']:.3f}, p={r['spearman_p']:.4f}")


if __name__ == "__main__":
    main()
