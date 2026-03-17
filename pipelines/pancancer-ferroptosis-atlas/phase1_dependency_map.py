"""Phase 1: Pan-cancer ferroptosis dependency extraction and cancer-type ranking.

Extracts CRISPR dependency scores for a 20-gene ferroptosis panel across all
DepMap 25Q3 cell lines, computes per-cancer-type statistics, ranks cancer types
by composite ferroptosis vulnerability, and produces a hierarchical clustering
heatmap.

Usage:
    uv run python -m pipelines.pancancer-ferroptosis-atlas.phase1_dependency_map
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import pdist

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix, load_depmap_model_metadata

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
RESULTS_DIR = REPO_ROOT / "data" / "results" / "pancancer-ferroptosis-atlas" / "phase1"

# --- Ferroptosis gene panel (20 genes) ---
# Keys = DepMap HUGO symbol, values = (category, evidence_tier)
# Tier A: in vivo-validated; Tier B: in vitro-only; Tier C: non-ferroptotic; None: untiered
FERROPTOSIS_GENES = {
    # Defense (pro-survival)
    "AIFM2":  ("defense", "A"),   # FSP1 — in vivo validated (icFSP1, Wu et al.)
    "GPX4":   ("defense", "B"),   # in vitro only (bioRxiv 10.64898/2026.03.11.711115)
    "SLC7A11": ("defense", "B"),  # in vitro only
    "GLS":    ("defense", None),  # GLS1 in literature, GLS in DepMap
    "GCLC":   ("defense", "B"),   # in vitro only
    "GCLM":   ("defense", None),
    "TXNRD1": ("defense", "C"),   # separate (non-ferroptotic) cell death mechanism
    "NQO1":   ("defense", None),
    "FTH1":   ("defense", None),
    "HMOX1":  ("defense", None),
    # Pro-ferroptotic
    "ACSL4":  ("pro-ferroptotic", None),
    "LPCAT3": ("pro-ferroptotic", None),
    "SAT1":   ("pro-ferroptotic", None),
    "NCOA4":  ("pro-ferroptotic", None),
    "TFRC":   ("pro-ferroptotic", None),
    "ALOX15": ("pro-ferroptotic", None),
    # Metabolic modulators
    "SHMT1":  ("metabolic", None),
    "SHMT2":  ("metabolic", None),
    "MTHFD2": ("metabolic", None),
    "CBS":    ("metabolic", None),
}

DEFENSE_GENES = [g for g, (cat, _) in FERROPTOSIS_GENES.items() if cat == "defense"]

# Literature-flagged cancer types for validation
PRIORITY_VALIDATION = {"Kidney", "Ovary/Fallopian Tube", "Skin", "Breast"}


def load_ferroptosis_dependencies(depmap_dir: Path) -> pd.DataFrame:
    """Load CRISPRGeneEffect and extract ferroptosis gene panel."""
    print("Loading CRISPRGeneEffect.csv...")
    crispr = load_depmap_matrix(depmap_dir / "CRISPRGeneEffect.csv")

    gene_list = list(FERROPTOSIS_GENES.keys())
    missing = [g for g in gene_list if g not in crispr.columns]
    if missing:
        print(f"  WARNING: genes not found in CRISPR data: {missing}")
        gene_list = [g for g in gene_list if g in crispr.columns]

    deps = crispr[gene_list].copy()
    print(f"  Extracted {len(gene_list)} genes for {len(deps)} cell lines")
    return deps


def annotate_cancer_types(deps: pd.DataFrame, depmap_dir: Path) -> pd.DataFrame:
    """Join dependency matrix with Model.csv to add OncotreeLineage."""
    print("Loading Model.csv...")
    meta = load_depmap_model_metadata(depmap_dir / "Model.csv")

    # Join on ModelID index
    common = deps.index.intersection(meta.index)
    print(f"  Matched {len(common)} / {len(deps)} cell lines to metadata")

    deps = deps.loc[common].copy()
    deps["OncotreeLineage"] = meta.loc[common, "OncotreeLineage"]
    deps["CellLineName"] = meta.loc[common, "CellLineName"]

    # Drop lines with no lineage annotation
    n_before = len(deps)
    deps = deps.dropna(subset=["OncotreeLineage"])
    if len(deps) < n_before:
        print(f"  Dropped {n_before - len(deps)} lines with no lineage annotation")

    return deps


def build_dependency_matrix(deps: pd.DataFrame) -> pd.DataFrame:
    """Build the full output matrix with gene tier annotations in column names."""
    gene_cols = [c for c in deps.columns if c in FERROPTOSIS_GENES]
    meta_cols = ["CellLineName", "OncotreeLineage"]

    out = deps[meta_cols + gene_cols].copy()
    out.index.name = "ModelID"
    return out


def compute_cancer_type_stats(deps: pd.DataFrame) -> pd.DataFrame:
    """Compute per-cancer-type statistics for each ferroptosis gene."""
    gene_cols = [c for c in deps.columns if c in FERROPTOSIS_GENES]
    rows = []

    for lineage, grp in deps.groupby("OncotreeLineage"):
        n = len(grp)
        for gene in gene_cols:
            vals = grp[gene].dropna()
            if len(vals) == 0:
                continue
            cat, tier = FERROPTOSIS_GENES[gene]
            rows.append({
                "cancer_type": lineage,
                "n_lines": n,
                "gene": gene,
                "category": cat,
                "evidence_tier": tier if tier else "untiered",
                "mean_dependency": vals.mean(),
                "median_dependency": vals.median(),
                "fraction_dependent": (vals < -0.5).mean(),
                "iqr": vals.quantile(0.75) - vals.quantile(0.25),
                "q25": vals.quantile(0.25),
                "q75": vals.quantile(0.75),
            })

    stats = pd.DataFrame(rows)
    return stats


def compute_cancer_type_ranking(stats: pd.DataFrame) -> pd.DataFrame:
    """Rank cancer types by composite ferroptosis vulnerability score.

    Composite = mean dependency across defense genes (more negative = more vulnerable).
    """
    defense_stats = stats[stats["gene"].isin(DEFENSE_GENES)].copy()

    ranking = defense_stats.groupby("cancer_type").agg(
        composite_vulnerability=("mean_dependency", "mean"),
        n_lines=("n_lines", "first"),
        n_defense_genes_dependent=("fraction_dependent", lambda x: (x > 0.2).sum()),
    ).reset_index()

    ranking = ranking.sort_values("composite_vulnerability", ascending=True)
    ranking["rank"] = range(1, len(ranking) + 1)
    ranking["priority_validation"] = ranking["cancer_type"].isin(PRIORITY_VALIDATION)

    return ranking


def plot_clustering_heatmap(deps: pd.DataFrame, out_path: Path) -> None:
    """Hierarchical clustering heatmap of cancer types by ferroptosis gene profile."""
    gene_cols = [c for c in deps.columns if c in FERROPTOSIS_GENES]

    # Mean dependency per cancer type
    mean_deps = deps.groupby("OncotreeLineage")[gene_cols].mean()

    # Drop cancer types with too few lines
    line_counts = deps.groupby("OncotreeLineage").size()
    keep = line_counts[line_counts >= 3].index
    mean_deps = mean_deps.loc[mean_deps.index.isin(keep)]

    if len(mean_deps) < 3:
        print("  Too few cancer types for clustering — skipping heatmap")
        return

    # Cluster rows (cancer types) and columns (genes)
    row_dist = pdist(mean_deps.values, metric="euclidean")
    col_dist = pdist(mean_deps.values.T, metric="euclidean")
    row_link = sch.linkage(row_dist, method="ward")
    col_link = sch.linkage(col_dist, method="ward")

    row_order = sch.leaves_list(row_link)
    col_order = sch.leaves_list(col_link)

    data = mean_deps.values[row_order][:, col_order]
    row_labels = mean_deps.index[row_order]
    col_labels = mean_deps.columns[col_order]

    # Annotate column labels with tier
    col_annot = []
    for g in col_labels:
        _, tier = FERROPTOSIS_GENES.get(g, (None, None))
        suffix = f" [{tier}]" if tier else ""
        col_annot.append(f"{g}{suffix}")

    fig, ax = plt.subplots(figsize=(14, max(8, len(row_labels) * 0.35)))
    vmax = max(abs(np.nanmin(data)), abs(np.nanmax(data)), 0.5)
    im = ax.imshow(data, aspect="auto", cmap="RdBu", vmin=-vmax, vmax=vmax)

    ax.set_xticks(range(len(col_annot)))
    ax.set_xticklabels(col_annot, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels, fontsize=8)

    cbar = fig.colorbar(im, ax=ax, shrink=0.6, label="Mean CRISPR dependency score")
    ax.set_title("Pan-Cancer Ferroptosis Gene Dependencies (DepMap 25Q3)", fontsize=11)
    ax.set_xlabel("Ferroptosis genes [evidence tier]")
    ax.set_ylabel("Cancer type (OncotreeLineage)")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved heatmap: {out_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Phase 1: Pan-cancer ferroptosis dependency map")
    parser.add_argument("--depmap-dir", type=Path, default=DEPMAP_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args(argv)

    args.results_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract ferroptosis dependencies
    deps = load_ferroptosis_dependencies(args.depmap_dir)

    # Step 2: Annotate with cancer types
    deps = annotate_cancer_types(deps, args.depmap_dir)

    # Step 3: Save full dependency matrix
    matrix = build_dependency_matrix(deps)
    matrix_path = args.results_dir / "ferroptosis_dependency_matrix.csv"
    matrix.to_csv(matrix_path)
    print(f"Saved dependency matrix: {matrix_path} ({len(matrix)} lines x {len(FERROPTOSIS_GENES)} genes)")

    # Step 4: Per-cancer-type stats
    stats = compute_cancer_type_stats(deps)
    stats_path = args.results_dir / "cancer_type_stats.csv"
    stats.to_csv(stats_path, index=False)
    print(f"Saved cancer type stats: {stats_path} ({len(stats)} rows)")

    # Step 5: Cancer type ranking
    ranking = compute_cancer_type_ranking(stats)
    ranking_path = args.results_dir / "cancer_type_ranking.csv"
    ranking.to_csv(ranking_path, index=False)
    print(f"\nCancer type ferroptosis vulnerability ranking:")
    print(ranking[["rank", "cancer_type", "composite_vulnerability", "n_lines",
                    "n_defense_genes_dependent", "priority_validation"]].to_string(index=False))

    # Step 6: Hierarchical clustering heatmap
    heatmap_path = args.results_dir / "ferroptosis_clustering_heatmap.png"
    plot_clustering_heatmap(deps, heatmap_path)

    print("\nPhase 1 complete.")


if __name__ == "__main__":
    main()
