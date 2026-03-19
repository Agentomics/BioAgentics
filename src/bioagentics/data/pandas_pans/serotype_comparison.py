"""Serotype comparison of GAS molecular mimicry profiles.

Compares mimicry hit profiles across GAS serotypes (M1, M3, M5, M12, M18, M49).
Identifies conserved vs serotype-specific mimicry targets and generates
heatmaps of mimicry scores per serotype x human target.

Usage:
    uv run python -m bioagentics.data.pandas_pans.serotype_comparison [--dest DIR]
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

PROJECT_DIR = Path("data/pandas_pans/gas-molecular-mimicry-mapping")
SCREEN_DIR = PROJECT_DIR / "mimicry_screen"
OUTPUT_DIR = Path("output/pandas_pans/gas-molecular-mimicry-mapping/serotype_comparison")

# Same filtering thresholds as mimicry_screen.py
MAX_EVALUE = 1e-3
MIN_ALIGNMENT_LENGTH = 8
MIN_PIDENT = 40.0

DIAMOND_COLUMNS = [
    "qseqid", "sseqid", "pident", "length", "mismatch", "gapopen",
    "qstart", "qend", "sstart", "send", "evalue", "bitscore",
    "qlen", "slen", "qcovhsp",
]

# PANDAS-associated serotypes (rheumatic fever / neuropsychiatric)
PANDAS_ASSOCIATED = {"M1", "M3", "M5", "M12", "M18"}


def load_per_serotype_hits(screen_dir: Path) -> dict[str, pd.DataFrame]:
    """Load and filter per-serotype DIAMOND results."""
    results = {}
    for f in sorted(screen_dir.glob("hits_m*.tsv")):
        serotype = f.stem.replace("hits_", "").upper()
        df = pd.read_csv(f, sep="\t", header=None, names=DIAMOND_COLUMNS)

        # Apply mimicry filters
        filtered = df[
            (df["evalue"] <= MAX_EVALUE)
            & (df["length"] >= MIN_ALIGNMENT_LENGTH)
            & (df["pident"] >= MIN_PIDENT)
        ].copy()

        # Parse human target info
        filtered["human_accession"] = filtered["sseqid"].str.split("|").str[0]
        filtered["human_gene"] = filtered["sseqid"].str.split("|").str[1]
        filtered["serotype"] = serotype

        results[serotype] = filtered
        logger.info("  %s: %d filtered hits, %d human targets",
                    serotype, len(filtered), filtered["human_accession"].nunique())

    return results


def build_comparison_matrix(per_serotype: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Build serotype x human target matrix with best bitscore as values."""
    rows = []
    for serotype, df in per_serotype.items():
        # Best hit per human target
        best = df.groupby("human_accession").agg(
            best_bitscore=("bitscore", "max"),
            best_pident=("pident", "max"),
            hit_count=("bitscore", "count"),
            human_gene=("human_gene", "first"),
        ).reset_index()

        for _, r in best.iterrows():
            rows.append({
                "serotype": serotype,
                "human_accession": r["human_accession"],
                "human_gene": r["human_gene"] if pd.notna(r["human_gene"]) else r["human_accession"],
                "best_bitscore": r["best_bitscore"],
                "best_pident": r["best_pident"],
                "hit_count": int(r["hit_count"]),
            })

    return pd.DataFrame(rows)


def compute_conservation(matrix_df: pd.DataFrame, serotypes: list[str]) -> pd.DataFrame:
    """Compute conservation scores for each human target across serotypes."""
    targets = matrix_df.groupby("human_accession").agg(
        human_gene=("human_gene", "first"),
        serotype_count=("serotype", "nunique"),
        serotypes_list=("serotype", lambda x: ",".join(sorted(x))),
        mean_bitscore=("best_bitscore", "mean"),
        max_bitscore=("best_bitscore", "max"),
        mean_pident=("best_pident", "mean"),
        total_hits=("hit_count", "sum"),
    ).reset_index()

    n_serotypes = len(serotypes)
    targets["conservation_score"] = targets["serotype_count"] / n_serotypes
    targets["conserved"] = targets["conservation_score"] == 1.0

    # Check PANDAS-association enrichment
    for _, row in targets.iterrows():
        seros = set(row["serotypes_list"].split(","))
        pandas_count = len(seros & PANDAS_ASSOCIATED)
        non_pandas_count = len(seros - PANDAS_ASSOCIATED)
        targets.loc[targets["human_accession"] == row["human_accession"], "pandas_serotype_count"] = pandas_count
        targets.loc[targets["human_accession"] == row["human_accession"], "non_pandas_serotype_count"] = non_pandas_count

    targets = targets.sort_values("conservation_score", ascending=False)
    return targets


def differential_mimicry(per_serotype: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Identify mimicry differences between PANDAS-associated and non-associated serotypes."""
    pandas_hits = set()
    non_pandas_hits = set()

    for serotype, df in per_serotype.items():
        targets = set(df["human_accession"].unique())
        if serotype in PANDAS_ASSOCIATED:
            pandas_hits.update(targets)
        else:
            non_pandas_hits.update(targets)

    all_targets = pandas_hits | non_pandas_hits
    rows = []
    for target in sorted(all_targets):
        in_pandas = target in pandas_hits
        in_non_pandas = target in non_pandas_hits

        # Get gene name from any serotype that has this target
        gene = target
        for df in per_serotype.values():
            match = df[df["human_accession"] == target]
            if not match.empty:
                g = match.iloc[0].get("human_gene", "")
                if pd.notna(g) and g:
                    gene = g
                break

        rows.append({
            "human_accession": target,
            "human_gene": gene,
            "in_pandas_serotypes": in_pandas,
            "in_non_pandas_serotypes": in_non_pandas,
            "category": (
                "pandas_specific" if in_pandas and not in_non_pandas
                else "non_pandas_specific" if not in_pandas and in_non_pandas
                else "shared"
            ),
        })

    return pd.DataFrame(rows)


def generate_heatmap(matrix_df: pd.DataFrame, dest: Path) -> None:
    """Generate heatmap of mimicry scores per serotype x human target."""
    # Pivot to matrix form
    pivot = matrix_df.pivot_table(
        index="human_gene", columns="serotype", values="best_bitscore", fill_value=0,
    )

    # Order serotypes
    serotype_order = [s for s in ["M1", "M3", "M5", "M12", "M18", "M49"] if s in pivot.columns]
    pivot = pivot[serotype_order]

    fig, ax = plt.subplots(figsize=(10, max(6, len(pivot) * 0.5)))
    sns.heatmap(
        pivot, annot=True, fmt=".0f", cmap="YlOrRd",
        linewidths=0.5, ax=ax, cbar_kws={"label": "Best DIAMOND bitscore"},
    )
    ax.set_title("GAS Molecular Mimicry: Serotype × Human Target", fontsize=14)
    ax.set_xlabel("GAS Serotype")
    ax.set_ylabel("Human Target")

    plt.tight_layout()
    fig.savefig(dest / "serotype_heatmap.png", dpi=150)
    plt.close(fig)
    logger.info("Heatmap saved: serotype_heatmap.png")

    # Also generate percent identity heatmap
    pivot_pid = matrix_df.pivot_table(
        index="human_gene", columns="serotype", values="best_pident", fill_value=0,
    )
    pivot_pid = pivot_pid[[s for s in serotype_order if s in pivot_pid.columns]]

    fig2, ax2 = plt.subplots(figsize=(10, max(6, len(pivot_pid) * 0.5)))
    sns.heatmap(
        pivot_pid, annot=True, fmt=".1f", cmap="YlGnBu",
        linewidths=0.5, ax=ax2, cbar_kws={"label": "Best % identity"},
    )
    ax2.set_title("GAS Molecular Mimicry: Serotype × Human Target (% Identity)", fontsize=14)
    ax2.set_xlabel("GAS Serotype")
    ax2.set_ylabel("Human Target")

    plt.tight_layout()
    fig2.savefig(dest / "serotype_heatmap_pident.png", dpi=150)
    plt.close(fig2)
    logger.info("Heatmap saved: serotype_heatmap_pident.png")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Compare mimicry profiles across GAS serotypes",
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args.dest.mkdir(parents=True, exist_ok=True)

    # Load per-serotype hits
    logger.info("Loading per-serotype mimicry hits...")
    per_serotype = load_per_serotype_hits(SCREEN_DIR)

    if not per_serotype:
        raise FileNotFoundError("No per-serotype hit files found. Run mimicry_screen.py first.")

    serotypes = sorted(per_serotype.keys())
    logger.info("Serotypes: %s", ", ".join(serotypes))

    # Build comparison matrix
    logger.info("Building comparison matrix...")
    matrix_df = build_comparison_matrix(per_serotype)
    matrix_path = args.dest / "serotype_comparison_matrix.tsv"
    matrix_df.to_csv(matrix_path, sep="\t", index=False)
    logger.info("Comparison matrix: %s (%d entries)", matrix_path.name, len(matrix_df))

    # Conservation analysis
    logger.info("Computing conservation scores...")
    conservation = compute_conservation(matrix_df, serotypes)
    conservation_path = args.dest / "conservation_scores.tsv"
    conservation.to_csv(conservation_path, sep="\t", index=False)

    conserved = conservation[conservation["conserved"]]
    logger.info("Conserved targets (all %d serotypes): %d / %d",
                len(serotypes), len(conserved), len(conservation))
    for _, row in conserved.iterrows():
        logger.info("  %s (%s): mean bitscore=%.0f, mean pident=%.1f%%",
                    row["human_gene"], row["human_accession"],
                    row["mean_bitscore"], row["mean_pident"])

    # Differential mimicry
    logger.info("Analyzing differential mimicry (PANDAS-associated vs other)...")
    diff = differential_mimicry(per_serotype)
    diff_path = args.dest / "differential_mimicry.tsv"
    diff.to_csv(diff_path, sep="\t", index=False)

    for cat in ["pandas_specific", "non_pandas_specific", "shared"]:
        n = len(diff[diff["category"] == cat])
        logger.info("  %s: %d targets", cat, n)

    # Heatmaps
    logger.info("Generating heatmaps...")
    generate_heatmap(matrix_df, args.dest)

    logger.info("Done. Serotype comparison in %s", args.dest)


if __name__ == "__main__":
    main()
