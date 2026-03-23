"""Phase 4 — Predict autoantibody targets from prolonged vs acute GAS exposure.

Task 110: Model how innate immune deficiency (prolonged pathogen exposure) leads
to broader autoantibody target diversification via molecular mimicry.

Hypothesis: Under normal innate immunity, acute GAS exposure only generates
cross-reactive antibodies against the strongest molecular mimicry matches
(high sequence similarity). Under lectin complement deficiency, prolonged
exposure allows weaker mimicry matches to also become autoantibody targets,
expanding the autoantibody repertoire.

Approach:
  1. Load per-serotype mimicry hits with their sequence similarity metrics
  2. Rank human targets by mimicry strength → predict targeting tiers
  3. Cross-reference with autoantibody network to validate predictions
  4. Integrate with epitope spreading model from Task 109

Usage::

    uv run python -m pandas_pans.innate_immunity_deficiency_model.mimicry_exposure_prediction
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from bioagentics.config import DATA_DIR, REPO_ROOT

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "innate-immunity-deficiency-model"
MIMICRY_DIR = DATA_DIR / "pandas_pans" / "gas-molecular-mimicry-mapping" / "mimicry_screen"
AUTOAB_DIR = DATA_DIR / "pandas_pans" / "autoantibody_network"

# BLAST column names (standard tabular format)
BLAST_COLS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    "qlen", "slen", "qcovhsp",
]

# Serotypes to analyze (per-serotype hit files)
SEROTYPES = ["m1", "m3", "m5", "m12", "m18", "m3.93", "m49"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _parse_human_target(sseqid: str) -> tuple[str, str]:
    """Extract human accession and gene from sseqid like 'P09104|ENO2|caudate'."""
    parts = str(sseqid).split("|")
    accession = parts[0] if len(parts) > 0 else ""
    gene = parts[1] if len(parts) > 1 else ""
    return accession, gene


def load_mimicry_hits() -> pd.DataFrame:
    """Load all per-serotype mimicry hits into a single DataFrame."""
    frames: list[pd.DataFrame] = []

    for serotype in SEROTYPES:
        path = MIMICRY_DIR / f"hits_{serotype}.tsv"
        if not path.exists():
            continue

        df = pd.read_csv(path, sep="\t", header=None, names=BLAST_COLS)
        df["serotype"] = serotype.upper()
        frames.append(df)

    if not frames:
        logger.warning("No per-serotype hit files found")
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)

    # Parse human target info
    parsed = combined["sseqid"].apply(lambda x: pd.Series(_parse_human_target(x)))
    combined["human_accession"] = parsed[0]
    combined["human_gene"] = parsed[1]

    # Extract GAS protein name from qseqid
    combined["gas_protein"] = combined["qseqid"].apply(
        lambda x: str(x).split("|")[-1].split("_")[0] if "|" in str(x) else str(x)
    )

    return combined


def load_filtered_hits() -> pd.DataFrame:
    """Load the filtered (high-confidence) mimicry hits."""
    path = MIMICRY_DIR / "hits_filtered.tsv"
    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    return df


def load_autoantibody_seeds() -> set[str]:
    """Load autoantibody seed target gene symbols."""
    comm_path = AUTOAB_DIR / "community_assignments.tsv"
    if not comm_path.exists():
        return set()
    communities = pd.read_csv(comm_path, sep="\t")
    return set(communities[communities["is_seed"] == True]["gene_symbol"].values)  # noqa: E712


# ---------------------------------------------------------------------------
# 1. Mimicry strength ranking
# ---------------------------------------------------------------------------


def rank_by_mimicry_strength(hits: pd.DataFrame) -> pd.DataFrame:
    """Rank human targets by molecular mimicry strength.

    Higher bitscore, lower evalue, higher pident → stronger mimicry.
    Aggregate across all serotypes to get overall mimicry strength per target.
    """
    if hits.empty:
        return pd.DataFrame()

    # Per-target best hit across all serotypes
    agg = hits.groupby("human_accession").agg(
        human_gene=("human_gene", "first"),
        best_pident=("pident", "max"),
        best_bitscore=("bitscore", "max"),
        best_evalue=("evalue", "min"),
        mean_pident=("pident", "mean"),
        n_serotypes_hitting=("serotype", "nunique"),
        n_gas_proteins_hitting=("gas_protein", "nunique"),
        n_total_hits=("qseqid", "count"),
        best_qcovhsp=("qcovhsp", "max"),
        serotypes=("serotype", lambda x: "; ".join(sorted(set(x)))),
    ).reset_index()

    # Composite mimicry strength score
    if agg["best_bitscore"].max() > 0:
        agg["norm_bitscore"] = agg["best_bitscore"] / agg["best_bitscore"].max()
    else:
        agg["norm_bitscore"] = 0.0

    agg["norm_serotype_breadth"] = agg["n_serotypes_hitting"] / len(SEROTYPES)
    agg["norm_pident"] = agg["best_pident"] / 100.0

    # Weighted score: similarity strength + serotype breadth
    agg["mimicry_strength"] = (
        0.40 * agg["norm_bitscore"]
        + 0.30 * agg["norm_pident"]
        + 0.20 * agg["norm_serotype_breadth"]
        + 0.10 * (agg["best_qcovhsp"] / 100.0)
    )

    return agg.sort_values("mimicry_strength", ascending=False)


# ---------------------------------------------------------------------------
# 2. Exposure duration tiers
# ---------------------------------------------------------------------------


def assign_exposure_tiers(ranked: pd.DataFrame) -> pd.DataFrame:
    """Assign targets to exposure-duration tiers.

    Tier 1 (acute): Strongest mimicry — targeted even during brief GAS exposure.
      These are the highest-similarity matches that generate cross-reactive
      antibodies quickly.

    Tier 2 (subacute): Moderate mimicry — targeted during prolonged exposure
      (days to weeks, as would occur with impaired lectin complement clearance).

    Tier 3 (chronic): Weaker mimicry — targeted only under chronic/recurrent
      exposure (months, as in recurrent strep with innate deficiency).

    Thresholds based on natural breaks in mimicry strength distribution.
    """
    if ranked.empty:
        return ranked

    df = ranked.copy()

    # Use percentile-based tiers
    q67 = df["mimicry_strength"].quantile(0.67)
    q33 = df["mimicry_strength"].quantile(0.33)

    def _tier(strength: float) -> str:
        if strength >= q67:
            return "tier_1_acute"
        elif strength >= q33:
            return "tier_2_subacute"
        else:
            return "tier_3_chronic"

    df["exposure_tier"] = df["mimicry_strength"].apply(_tier)
    df["tier_order"] = df["exposure_tier"].map({
        "tier_1_acute": 1,
        "tier_2_subacute": 2,
        "tier_3_chronic": 3,
    })

    return df.sort_values(["tier_order", "mimicry_strength"], ascending=[True, False])


# ---------------------------------------------------------------------------
# 3. Cross-reference with autoantibody network
# ---------------------------------------------------------------------------


def cross_reference_autoantibody_network(
    tiered: pd.DataFrame,
) -> pd.DataFrame:
    """Cross-reference mimicry targets with autoantibody network seeds."""
    seeds = load_autoantibody_seeds()
    if tiered.empty:
        return tiered

    df = tiered.copy()
    df["is_autoantibody_seed"] = df["human_gene"].apply(
        lambda g: g in seeds if pd.notna(g) and g else False
    )

    # Also load epitope spreading predictions if available
    spreading_path = OUTPUT_DIR / "epitope_spreading_predictions.csv"
    if spreading_path.exists():
        spreading = pd.read_csv(spreading_path)
        spread_map = dict(zip(
            spreading["autoantibody_target"],
            spreading["predicted_targeting_score"],
        ))
        df["network_targeting_score"] = df["human_gene"].map(spread_map)
    else:
        df["network_targeting_score"] = np.nan

    return df


# ---------------------------------------------------------------------------
# 4. Integrated prediction
# ---------------------------------------------------------------------------


def compute_integrated_prediction(df: pd.DataFrame) -> pd.DataFrame:
    """Combine mimicry strength with network topology for final predictions.

    Final prediction = mimicry_strength * network_accessibility
    Targets that are both strong mimics AND network-accessible from innate
    immune genes are the most likely autoantibody targets under innate deficiency.
    """
    if df.empty:
        return df

    result = df.copy()

    # For targets with network scores, combine signals
    has_network = result["network_targeting_score"].notna()
    if has_network.any():
        result.loc[has_network, "integrated_score"] = (
            0.5 * result.loc[has_network, "mimicry_strength"]
            + 0.5 * result.loc[has_network, "network_targeting_score"]
        )
    # For others, use mimicry strength alone
    result.loc[~has_network, "integrated_score"] = result.loc[~has_network, "mimicry_strength"]

    result["integrated_rank"] = (
        result["integrated_score"].rank(ascending=False, method="min").astype(int)
    )

    return result.sort_values("integrated_rank")


# ---------------------------------------------------------------------------
# 5. Summary
# ---------------------------------------------------------------------------


def compute_summary(
    hits: pd.DataFrame,
    predictions: pd.DataFrame,
) -> dict:
    """Generate summary statistics."""
    summary: dict = {
        "total_mimicry_hits": len(hits),
        "unique_human_targets": int(hits["human_accession"].nunique()) if not hits.empty else 0,
        "unique_gas_proteins": int(hits["gas_protein"].nunique()) if not hits.empty else 0,
    }

    if not predictions.empty:
        for tier in ["tier_1_acute", "tier_2_subacute", "tier_3_chronic"]:
            tier_df = predictions[predictions["exposure_tier"] == tier]
            summary[f"n_{tier}"] = len(tier_df)
            if not tier_df.empty:
                genes = tier_df["human_gene"].dropna().tolist()
                summary[f"{tier}_targets"] = [g for g in genes if g]

        seeds_hit = predictions[predictions["is_autoantibody_seed"] == True]  # noqa: E712
        summary["n_autoantibody_seeds_with_mimicry"] = len(seeds_hit)
        if not seeds_hit.empty:
            summary["autoantibody_seeds_with_mimicry"] = list(
                seeds_hit["human_gene"].dropna().values
            )

        # Key prediction: innate deficiency expands targeting
        summary["prediction"] = (
            "Innate deficiency (lectin complement failure) extends GAS exposure duration, "
            "allowing progression from tier 1 (acute) to tier 2-3 (subacute/chronic) "
            "mimicry targets. This predicts broader autoantibody diversification in "
            "patients with MBL2/MASP variants compared to those with intact innate immunity."
        )

    return summary


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_mimicry_exposure_prediction() -> dict[str, Path]:
    """Run the mimicry-based autoantibody target prediction pipeline."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Path] = {}

    # Load data
    logger.info("Loading per-serotype mimicry hits...")
    hits = load_mimicry_hits()
    if hits.empty:
        logger.warning("No mimicry hits loaded — cannot proceed")
        return outputs
    logger.info("Loaded %d hits across %d serotypes", len(hits), hits["serotype"].nunique())

    # 1. Rank by mimicry strength
    logger.info("\n=== Mimicry strength ranking ===")
    ranked = rank_by_mimicry_strength(hits)
    if not ranked.empty:
        path = OUTPUT_DIR / "mimicry_target_ranking.csv"
        ranked.to_csv(path, index=False)
        outputs["ranking"] = path
        logger.info("Ranked %d unique human targets:", len(ranked))
        logger.info("\n%s", ranked[
            ["human_accession", "human_gene", "best_pident", "best_bitscore",
             "n_serotypes_hitting", "mimicry_strength"]
        ].to_string(index=False))

    # 2. Assign exposure tiers
    logger.info("\n=== Exposure duration tiers ===")
    tiered = assign_exposure_tiers(ranked)
    if not tiered.empty:
        for tier in ["tier_1_acute", "tier_2_subacute", "tier_3_chronic"]:
            tier_df = tiered[tiered["exposure_tier"] == tier]
            genes = [g for g in tier_df["human_gene"].dropna() if g]
            logger.info("  %s: %d targets %s", tier, len(tier_df), genes if genes else "")

    # 3. Cross-reference with autoantibody network
    logger.info("\n=== Autoantibody network cross-reference ===")
    cross_ref = cross_reference_autoantibody_network(tiered)
    seeds = cross_ref[cross_ref["is_autoantibody_seed"] == True]  # noqa: E712
    if not seeds.empty:
        logger.info(
            "Autoantibody seeds also hit by mimicry: %s",
            list(seeds["human_gene"].dropna().values),
        )

    # 4. Integrated prediction
    logger.info("\n=== Integrated predictions ===")
    predictions = compute_integrated_prediction(cross_ref)
    if not predictions.empty:
        path = OUTPUT_DIR / "mimicry_exposure_predictions.csv"
        predictions.to_csv(path, index=False)
        outputs["predictions"] = path
        logger.info(
            "Top predicted targets:\n%s",
            predictions[
                ["human_gene", "exposure_tier", "mimicry_strength",
                 "is_autoantibody_seed", "network_targeting_score", "integrated_rank"]
            ].head(15).to_string(index=False),
        )

    # 5. Summary
    logger.info("\n=== Summary ===")
    summary = compute_summary(hits, predictions)
    path = OUTPUT_DIR / "mimicry_exposure_summary.json"
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    outputs["summary"] = path
    for k, v in summary.items():
        if k != "prediction":
            logger.info("  %s: %s", k, v)
    logger.info("\n  PREDICTION: %s", summary.get("prediction", ""))

    return outputs


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_mimicry_exposure_prediction()
