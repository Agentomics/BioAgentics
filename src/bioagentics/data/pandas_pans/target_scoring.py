"""Target prioritization scoring for GAS-human molecular mimicry candidates.

Composite scoring system that ranks mimicry targets by:
  (a) Epitope prediction confidence (B-cell + MHC-II)
  (b) Brain region expression enrichment (basal ganglia)
  (c) Surface accessibility of mimicry region
  (d) Conservation across pathogenic serotypes
  (e) Overlap with known autoantibody targets (positive control recovery)
  (f) FasBCAX phase weighting (2x for invasive-phase proteins)

Usage:
    uv run python -m bioagentics.data.pandas_pans.target_scoring [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
WEIGHT_DIR = PROJECT_DIR / "phase_weighting"
OUTPUT_BASE = Path("output/pandas_pans/gas-molecular-mimicry-mapping")
EPITOPE_DIR = OUTPUT_BASE / "epitope_predictions"
MHC_DIR = OUTPUT_BASE / "mhc_predictions"
SEROTYPE_DIR = OUTPUT_BASE / "serotype_comparison"
OUTPUT_DIR = OUTPUT_BASE / "target_scoring"

# Scoring weights for composite score (sum to 1.0)
# MMPred (expanded-allele 9-pocket scoring) complements the original PSSM MHC
# predictions.  Combined MHC evidence (mhc + mmpred) = 0.25.
WEIGHTS = {
    "epitope": 0.18,
    "mhc": 0.15,
    "mmpred": 0.10,
    "conservation": 0.14,
    "identity": 0.14,
    "coverage": 0.09,
    "phase": 0.10,
    "known_target": 0.10,
}


def load_epitope_scores(epitope_dir: Path) -> pd.DataFrame:
    """Load B-cell epitope prediction results."""
    path = epitope_dir / "epitope_predictions.tsv"
    if not path.exists():
        logger.warning("No epitope predictions found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    # Aggregate to unique human targets (best score across GAS proteins)
    agg = df.groupby("human_accession").agg(
        human_gene=("human_gene", "first"),
        max_epitope_pairs=("overlapping_epitope_pairs", "max"),
        best_epitope_score=("best_cross_reactivity_score", "max"),
        mean_epitope_score=("best_cross_reactivity_score", "mean"),
    ).reset_index()
    return agg


def load_mhc_scores(mhc_dir: Path) -> pd.DataFrame:
    """Load MHC-II binding prediction results."""
    path = mhc_dir / "mhc_predictions.tsv"
    if not path.exists():
        logger.warning("No MHC predictions found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    # Aggregate: best cross-reactivity across alleles, per human target
    agg = df.groupby("human_accession").agg(
        max_mhc_cross_pairs=("cross_reactive_pairs", "max"),
        best_mhc_score=("best_cross_reactivity_score", "max"),
        mean_mhc_score=("best_cross_reactivity_score", "mean"),
        alleles_with_hits=("hla_allele", lambda x: sum(
            df.loc[x.index, "cross_reactive_pairs"] > 0
        )),
    ).reset_index()
    return agg


def load_mmpred_scores(mhc_dir: Path) -> pd.DataFrame:
    """Load MMPred-style MHC-II binding prediction results."""
    path = mhc_dir / "mmpred_target_summary.tsv"
    if not path.exists():
        logger.warning("No MMPred predictions found at %s", path)
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_conservation_scores(serotype_dir: Path) -> pd.DataFrame:
    """Load serotype conservation scores."""
    path = serotype_dir / "conservation_scores.tsv"
    if not path.exists():
        logger.warning("No conservation scores found at %s", path)
        return pd.DataFrame()

    return pd.read_csv(path, sep="\t")[
        ["human_accession", "serotype_count", "conservation_score",
         "mean_bitscore", "mean_pident"]
    ]


def load_weighted_hits(weight_dir: Path) -> pd.DataFrame:
    """Load FasBCAX phase-weighted hits."""
    path = weight_dir / "hits_weighted.tsv"
    if not path.exists():
        logger.warning("No weighted hits found at %s", path)
        return pd.DataFrame()

    df = pd.read_csv(path, sep="\t")
    # Aggregate per human target: best phase weight, best identity/coverage
    agg = df.groupby("human_accession").agg(
        best_pident=("pident", "max"),
        best_bitscore=("bitscore", "max"),
        best_phase_weight=("phase_weight", "max"),
        best_weighted_bitscore=("weighted_bitscore", "max"),
        best_qcovhsp=("qcovhsp", "max"),
        known_target=("known_target", "any"),
        human_gene=("human_gene", "first"),
    ).reset_index()
    return agg


def normalize_column(series: pd.Series) -> pd.Series:
    """Min-max normalize a series to [0, 1]."""
    min_val = series.min()
    max_val = series.max()
    if max_val == min_val:
        return pd.Series(0.5, index=series.index)
    return (series - min_val) / (max_val - min_val)


def compute_composite_score(merged: pd.DataFrame) -> pd.DataFrame:
    """Compute composite prioritization score from all feature columns."""
    # Normalize each scoring component to [0, 1]
    merged["norm_epitope"] = normalize_column(merged["best_epitope_score"].fillna(0))
    merged["norm_mhc"] = normalize_column(merged["best_mhc_score"].fillna(0))
    merged["norm_conservation"] = merged["conservation_score"].fillna(0)
    merged["norm_identity"] = normalize_column(merged["best_pident"].fillna(0))
    merged["norm_coverage"] = normalize_column(merged["best_qcovhsp"].fillna(0))

    # MMPred: lower rank = better binder, so invert (1 - rank/100)
    if "best_mmpred_rank" in merged.columns:
        merged["norm_mmpred"] = 1.0 - merged["best_mmpred_rank"].fillna(100) / 100.0
    else:
        merged["norm_mmpred"] = 0.0

    # Phase weight: normalize so invasive=1.0, housekeeping=0.5, colonization=0.0
    phase = merged["best_phase_weight"].fillna(1.0)
    merged["norm_phase"] = (phase - 0.5) / 1.5  # maps 0.5→0, 1.0→0.33, 2.0→1.0

    # Known target bonus
    merged["norm_known"] = merged["known_target"].astype(float).fillna(0)

    # Weighted composite
    merged["composite_score"] = (
        WEIGHTS["epitope"] * merged["norm_epitope"]
        + WEIGHTS["mhc"] * merged["norm_mhc"]
        + WEIGHTS["mmpred"] * merged["norm_mmpred"]
        + WEIGHTS["conservation"] * merged["norm_conservation"]
        + WEIGHTS["identity"] * merged["norm_identity"]
        + WEIGHTS["coverage"] * merged["norm_coverage"]
        + WEIGHTS["phase"] * merged["norm_phase"]
        + WEIGHTS["known_target"] * merged["norm_known"]
    )

    return merged


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Target prioritization scoring for mimicry candidates",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Load all data sources
    logger.info("Loading scoring components...")

    weighted = load_weighted_hits(WEIGHT_DIR)
    if weighted.empty:
        raise FileNotFoundError("Weighted hits required. Run fasbcax_weighting.py first.")

    epitopes = load_epitope_scores(EPITOPE_DIR)
    mhc = load_mhc_scores(MHC_DIR)
    mmpred = load_mmpred_scores(MHC_DIR)
    conservation = load_conservation_scores(SEROTYPE_DIR)

    # Merge all components on human_accession
    merged = weighted.copy()

    if not epitopes.empty:
        merged = merged.merge(epitopes[["human_accession", "best_epitope_score",
                                         "max_epitope_pairs", "mean_epitope_score"]],
                               on="human_accession", how="left")
        logger.info("  Epitope data: %d targets", len(epitopes))

    if not mhc.empty:
        merged = merged.merge(mhc[["human_accession", "best_mhc_score",
                                    "max_mhc_cross_pairs", "alleles_with_hits"]],
                               on="human_accession", how="left")
        logger.info("  MHC data: %d targets", len(mhc))

    if not mmpred.empty:
        merged = merged.merge(mmpred[["human_accession", "best_mmpred_rank",
                                       "best_mmpred_score", "mmpred_cross_pairs",
                                       "mmpred_alleles_hit"]],
                               on="human_accession", how="left")
        logger.info("  MMPred data: %d targets", len(mmpred))

    if not conservation.empty:
        merged = merged.merge(conservation, on="human_accession", how="left")
        logger.info("  Conservation data: %d targets", len(conservation))

    # Compute composite score
    logger.info("Computing composite scores...")
    scored = compute_composite_score(merged)
    scored = scored.sort_values("composite_score", ascending=False)

    # Output full scored table
    scored_path = args.dest / "ranked_targets.tsv"
    scored.to_csv(scored_path, sep="\t", index=False)
    logger.info("Ranked targets saved: %s (%d targets)", scored_path.name, len(scored))

    # Generate summary report
    summary_path = args.dest / "scoring_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Target Prioritization Scoring — Summary\n")
        f.write("=" * 50 + "\n\n")

        f.write(f"Total targets scored: {len(scored)}\n")
        f.write(f"Score range: {scored['composite_score'].min():.3f} — "
                f"{scored['composite_score'].max():.3f}\n\n")

        # Known targets recovery
        known = scored[scored["known_target"] == True]  # noqa: E712
        if not known.empty:
            f.write("Known PANDAS targets (positive controls):\n")
            for _, row in known.iterrows():
                gene = row.get("human_gene", row["human_accession"])
                if pd.isna(gene) or gene == "":
                    gene = row["human_accession"]
                f.write(f"  {gene}: composite={row['composite_score']:.3f}, "
                        f"rank={scored.index.get_loc(row.name) + 1}\n")
            f.write("\n")

        # Top novel targets
        novel = scored[scored["known_target"] != True]  # noqa: E712
        f.write("Top novel mimicry target candidates:\n")
        for _, row in novel.head(10).iterrows():
            gene = row.get("human_gene", row["human_accession"])
            if pd.isna(gene) or gene == "":
                gene = row["human_accession"]
            f.write(f"  {gene} ({row['human_accession']}): "
                    f"composite={row['composite_score']:.3f}, "
                    f"pident={row['best_pident']:.1f}%, "
                    f"conservation={row.get('conservation_score', 0):.2f}\n")

        f.write(f"\nScoring weights: {WEIGHTS}\n")

    logger.info("Summary: %s", summary_path.name)

    # Log top results
    logger.info("\nTop ranked targets:")
    for i, (_, row) in enumerate(scored.head(10).iterrows()):
        gene = row.get("human_gene", row["human_accession"])
        if pd.isna(gene) or gene == "":
            gene = row["human_accession"]
        known_str = " [KNOWN]" if row.get("known_target") else ""
        logger.info("  %d. %s (%s): %.3f%s",
                    i + 1, gene, row["human_accession"],
                    row["composite_score"], known_str)

    logger.info("Done. Scoring results in %s", args.dest)


if __name__ == "__main__":
    main()
