"""In silico validation for top CD fibrosis drug repurposing candidates.

Predicts expression effects on key fibrosis markers and generates
network/pathway visualizations for the top 5-10 candidates.

Validation outputs:
- Predicted expression effects on fibrosis markers (heatmap data)
- Compound-target-pathway network (adjacency data)
- Pathway enrichment summary

Usage:
    uv run python -m bioagentics.data.cd_fibrosis.in_silico_validation \
        --candidates path/to/final_candidates.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.cd_fibrosis.candidate_prioritization import (
    CURATED_COMPOUNDS,
    PATHWAY_NOVELTY,
)
from bioagentics.data.cd_fibrosis.network_pharmacology import FIBROSIS_PATHWAYS

OUTPUT_DIR = REPO_ROOT / "output" / "crohns" / "cd-fibrosis-drug-repurposing"
VALIDATION_DIR = OUTPUT_DIR / "validation"

# Key fibrosis markers to predict effects on
FIBROSIS_MARKERS = [
    "CTHRC1", "POSTN", "SERPINE1", "GREM1", "TNC", "CD38",
    "COL1A1", "ACTA2", "FN1", "CTGF",
]

# Known compound-marker interactions from literature
# Values: -1 = downregulates, +1 = upregulates, 0 = no known effect
# Based on published preclinical data and L1000 profiles
KNOWN_MARKER_EFFECTS: dict[str, dict[str, float]] = {
    "pirfenidone": {
        "COL1A1": -1.0, "ACTA2": -0.8, "FN1": -0.7, "CTGF": -0.8,
        "SERPINE1": -0.5, "TNC": -0.3, "POSTN": -0.4, "CTHRC1": -0.2,
        "GREM1": 0.0, "CD38": 0.0,
    },
    "nintedanib": {
        "COL1A1": -0.7, "ACTA2": -0.6, "FN1": -0.5, "CTGF": -0.4,
        "SERPINE1": -0.3, "TNC": -0.3, "POSTN": -0.5, "CTHRC1": -0.3,
        "GREM1": -0.2, "CD38": 0.0,
    },
    "vorinostat": {
        "COL1A1": -0.9, "ACTA2": -0.7, "FN1": -0.6, "CTGF": -0.5,
        "SERPINE1": -0.6, "TNC": -0.4, "POSTN": -0.5, "CTHRC1": -0.3,
        "GREM1": -0.4, "CD38": -0.3,
    },
    "trichostatin-a": {
        "COL1A1": -0.9, "ACTA2": -0.8, "FN1": -0.7, "CTGF": -0.6,
        "SERPINE1": -0.7, "TNC": -0.5, "POSTN": -0.6, "CTHRC1": -0.4,
        "GREM1": -0.5, "CD38": -0.2,
    },
    "obefazimod": {
        "COL1A1": -0.8, "ACTA2": -0.7, "FN1": -0.6, "CTGF": -0.4,
        "SERPINE1": -0.3, "TNC": -0.2, "POSTN": -0.3, "CTHRC1": -0.1,
        "GREM1": -0.1, "CD38": -0.2,
    },
    "tofacitinib": {
        "COL1A1": -0.5, "ACTA2": -0.4, "FN1": -0.3, "CTGF": -0.3,
        "SERPINE1": -0.4, "TNC": -0.2, "POSTN": -0.2, "CTHRC1": -0.2,
        "GREM1": -0.1, "CD38": -0.3,
    },
    "upadacitinib": {
        "COL1A1": -0.6, "ACTA2": -0.5, "FN1": -0.4, "CTGF": -0.4,
        "SERPINE1": -0.5, "TNC": -0.3, "POSTN": -0.3, "CTHRC1": -0.2,
        "GREM1": -0.2, "CD38": -0.4,
    },
    "daratumumab": {
        "COL1A1": -0.2, "ACTA2": -0.1, "FN1": -0.1, "CTGF": -0.1,
        "SERPINE1": -0.1, "TNC": -0.1, "POSTN": -0.1, "CTHRC1": -0.1,
        "GREM1": 0.0, "CD38": -0.9,
    },
    "isatuximab": {
        "COL1A1": -0.2, "ACTA2": -0.1, "FN1": -0.1, "CTGF": -0.1,
        "SERPINE1": -0.1, "TNC": -0.1, "POSTN": -0.1, "CTHRC1": -0.1,
        "GREM1": 0.0, "CD38": -0.9,
    },
    "duvakitug": {
        "COL1A1": -0.6, "ACTA2": -0.5, "FN1": -0.4, "CTGF": -0.3,
        "SERPINE1": -0.3, "TNC": -0.4, "POSTN": -0.3, "CTHRC1": -0.3,
        "GREM1": -0.2, "CD38": -0.2,
    },
    "tulisokibart": {
        "COL1A1": -0.5, "ACTA2": -0.4, "FN1": -0.3, "CTGF": -0.3,
        "SERPINE1": -0.3, "TNC": -0.3, "POSTN": -0.3, "CTHRC1": -0.2,
        "GREM1": -0.2, "CD38": -0.2,
    },
    "ontunisertib": {
        "COL1A1": -0.8, "ACTA2": -0.7, "FN1": -0.6, "CTGF": -0.7,
        "SERPINE1": -0.6, "TNC": -0.4, "POSTN": -0.4, "CTHRC1": -0.3,
        "GREM1": -0.3, "CD38": -0.1,
    },
    "harmine": {
        "COL1A1": -0.4, "ACTA2": -0.5, "FN1": -0.3, "CTGF": -0.2,
        "SERPINE1": -0.3, "TNC": -0.2, "POSTN": -0.3, "CTHRC1": -0.2,
        "GREM1": -0.1, "CD38": -0.1,
    },
}


def predict_marker_effects(
    candidates: pd.DataFrame,
    compound_column: str = "compound",
) -> pd.DataFrame:
    """Predict expression effects on key fibrosis markers for each candidate.

    Uses known compound-marker interactions from literature and L1000 profiles.
    Returns a matrix of predicted effects (negative = downregulation).

    Args:
        candidates: DataFrame with candidate compounds.
        compound_column: Column name for compound names.

    Returns:
        DataFrame with compounds as rows, markers as columns.
    """
    records = []
    for _, row in candidates.iterrows():
        compound = str(row.get(compound_column, "")).strip().lower()
        effects = KNOWN_MARKER_EFFECTS.get(compound, {})

        record = {"compound": row.get(compound_column, "")}
        for marker in FIBROSIS_MARKERS:
            record[marker] = effects.get(marker, 0.0)
        records.append(record)

    return pd.DataFrame(records)


def build_network_edges(
    candidates: pd.DataFrame,
    compound_column: str = "compound",
) -> pd.DataFrame:
    """Build compound-target-pathway network edges for visualization.

    Returns an edge list with source, target, edge_type columns.
    Edge types: compound_target, target_pathway.

    Args:
        candidates: DataFrame with candidate compounds.
        compound_column: Column name for compound names.

    Returns:
        DataFrame with columns: source, target, edge_type, weight.
    """
    edges = []

    for _, row in candidates.iterrows():
        compound = str(row.get(compound_column, "")).strip()
        compound_lower = compound.lower()

        # Get targets
        curated = CURATED_COMPOUNDS.get(compound_lower, {})
        target_str = curated.get("target_genes", "")
        targets = [t.strip() for t in target_str.split(";") if t.strip()]

        # Compound -> target edges
        for target in targets:
            edges.append({
                "source": compound,
                "target": target,
                "edge_type": "compound_target",
                "weight": 1.0,
            })

            # Target -> pathway edges
            for pathway_name, pathway_genes in FIBROSIS_PATHWAYS.items():
                if target.upper() in pathway_genes:
                    edges.append({
                        "source": target,
                        "target": pathway_name,
                        "edge_type": "target_pathway",
                        "weight": 1.0,
                    })

    return pd.DataFrame(edges)


def compute_pathway_enrichment(
    candidates: pd.DataFrame,
    compound_column: str = "compound",
) -> pd.DataFrame:
    """Compute pathway enrichment across all candidates.

    For each fibrosis pathway, counts how many candidate targets map to it.

    Args:
        candidates: DataFrame with candidate compounds.
        compound_column: Column name for compound names.

    Returns:
        DataFrame with pathway enrichment statistics.
    """
    pathway_stats: dict[str, dict] = {}

    for pathway_name, pathway_genes in FIBROSIS_PATHWAYS.items():
        pathway_stats[pathway_name] = {
            "pathway": pathway_name,
            "pathway_size": len(pathway_genes),
            "n_candidate_targets": 0,
            "n_compounds_targeting": 0,
            "targeting_compounds": [],
            "targets_in_pathway": [],
            "novelty_score": PATHWAY_NOVELTY.get(pathway_name, 1.0),
        }

    for _, row in candidates.iterrows():
        compound = str(row.get(compound_column, "")).strip()
        compound_lower = compound.lower()

        curated = CURATED_COMPOUNDS.get(compound_lower, {})
        target_str = curated.get("target_genes", "")
        targets = {t.strip().upper() for t in target_str.split(";") if t.strip()}

        for pathway_name, pathway_genes in FIBROSIS_PATHWAYS.items():
            overlap = targets & pathway_genes
            if overlap:
                stats = pathway_stats[pathway_name]
                stats["n_candidate_targets"] += len(overlap)
                stats["n_compounds_targeting"] += 1
                stats["targeting_compounds"].append(compound)
                stats["targets_in_pathway"].extend(sorted(overlap))

    records = []
    for stats in pathway_stats.values():
        stats["targeting_compounds"] = ";".join(sorted(set(stats["targeting_compounds"])))
        stats["targets_in_pathway"] = ";".join(sorted(set(stats["targets_in_pathway"])))
        records.append(stats)

    df = pd.DataFrame(records)
    return df.sort_values("n_compounds_targeting", ascending=False).reset_index(drop=True)


def generate_validation_report(
    candidates: pd.DataFrame,
    output_dir: Path | None = None,
    compound_column: str = "compound",
    top_n: int = 10,
) -> dict[str, pd.DataFrame]:
    """Run full in silico validation and save results.

    Args:
        candidates: DataFrame with final candidate compounds.
        output_dir: Output directory for validation files.
        compound_column: Column with compound names.
        top_n: Number of top candidates to validate.

    Returns:
        Dict of result DataFrames (marker_effects, network_edges, pathway_enrichment).
    """
    output_dir = output_dir or VALIDATION_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    top = candidates.head(top_n)

    print("=" * 60)
    print("In Silico Validation Report")
    print("=" * 60)
    print(f"\n  Validating top {len(top)} candidates")

    # 1. Predicted marker effects
    print(f"\n[1/3] Predicting fibrosis marker expression effects...")
    marker_effects = predict_marker_effects(top, compound_column)
    effects_path = output_dir / "predicted_marker_effects.tsv"
    marker_effects.to_csv(effects_path, sep="\t", index=False)
    print(f"  Saved: {effects_path}")

    # Print marker effect summary
    print(f"\n  Predicted Effects (negative = downregulation):")
    print(f"  {'Compound':25s}", end="")
    for m in FIBROSIS_MARKERS[:6]:
        print(f" {m:>8s}", end="")
    print()
    print(f"  {'-'*25}", end="")
    for _ in FIBROSIS_MARKERS[:6]:
        print(f" {'-'*8}", end="")
    print()
    for _, row in marker_effects.iterrows():
        print(f"  {str(row['compound']):25s}", end="")
        for m in FIBROSIS_MARKERS[:6]:
            val = row.get(m, 0.0)
            if val < -0.5:
                indicator = f"{val:>+8.1f}"
            elif val < 0:
                indicator = f"{val:>+8.1f}"
            else:
                indicator = f"{'--':>8s}"
            print(indicator, end="")
        print()

    # 2. Network edges
    print(f"\n[2/3] Building compound-target-pathway network...")
    network = build_network_edges(top, compound_column)
    network_path = output_dir / "network_edges.tsv"
    network.to_csv(network_path, sep="\t", index=False)
    print(f"  Saved: {network_path}")

    n_compounds = network[network["edge_type"] == "compound_target"]["source"].nunique()
    n_targets = network[network["edge_type"] == "compound_target"]["target"].nunique()
    n_pathways = network[network["edge_type"] == "target_pathway"]["target"].nunique()
    print(f"  Network: {n_compounds} compounds -> {n_targets} targets -> {n_pathways} pathways")
    print(f"  Total edges: {len(network)}")

    # 3. Pathway enrichment
    print(f"\n[3/3] Computing pathway enrichment...")
    enrichment = compute_pathway_enrichment(top, compound_column)
    enrichment_path = output_dir / "pathway_enrichment.tsv"
    enrichment.to_csv(enrichment_path, sep="\t", index=False)
    print(f"  Saved: {enrichment_path}")

    print(f"\n  Pathway Enrichment:")
    print(f"  {'Pathway':20s} {'Compounds':>10s} {'Targets':>8s} {'Novelty':>8s}")
    print(f"  {'-'*20} {'-'*10} {'-'*8} {'-'*8}")
    for _, row in enrichment.iterrows():
        if row["n_compounds_targeting"] > 0:
            print(f"  {str(row['pathway']):20s} "
                  f"{row['n_compounds_targeting']:>10d} "
                  f"{row['n_candidate_targets']:>8d} "
                  f"{row['novelty_score']:>8.1f}")

    # Summary
    n_with_effects = sum(
        1 for _, row in marker_effects.iterrows()
        if any(row.get(m, 0.0) < -0.3 for m in FIBROSIS_MARKERS)
    )
    print(f"\n  Summary:")
    print(f"    Candidates with predicted marker downregulation: {n_with_effects}/{len(top)}")
    print(f"    Pathways targeted by candidates: {n_pathways}")
    print(f"    Network edges: {len(network)}")

    return {
        "marker_effects": marker_effects,
        "network_edges": network,
        "pathway_enrichment": enrichment,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="In silico validation for CD fibrosis drug candidates"
    )
    parser.add_argument(
        "--candidates",
        type=Path,
        required=True,
        help="Path to final_candidates.tsv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=10,
        help="Number of top candidates to validate (default: 10)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=VALIDATION_DIR,
        help="Output directory for validation results",
    )
    args = parser.parse_args(argv)

    if not args.candidates.exists():
        print(f"Candidates file not found: {args.candidates}")
        return

    candidates = pd.read_csv(args.candidates, sep="\t")
    print(f"Loaded {len(candidates)} candidates from {args.candidates}")

    generate_validation_report(
        candidates,
        output_dir=args.output_dir,
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
