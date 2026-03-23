"""DBS fiber tract expression profiling for TS risk genes.

Maps TS risk gene expression along the three optimal DBS fiber tract corridors
identified in the multi-center DBS study (n=115, 12 centers, medRxiv Feb 2026):
  1. Ansa lenticularis (AL) — GPi → thalamus, ventral route
  2. Fasciculus lenticularis (FL) — GPi → thalamus, dorsal route through H2 field
  3. Posterior intralaminar-lentiform projections (PIL) — CM/Pf → putamen

These three tract corridors explained 19% of tic improvement variance. OCD
response maps diverge from tic maps in the thalamus.

Uses AHBA spatial data to approximate tract corridors via anatomical regions
traversed by each tract.

Dependency: gene_sets module + ahba_spatial module.

Usage:
    uv run python -m bioagentics.analysis.tourettes.dbs_tracts [--gene-set ts_combined]
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
from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets
from bioagentics.data.tourettes.hmba_reference import (
    get_cstc_region_cell_types,
    CellType,
)
from bioagentics.analysis.tourettes.ahba_spatial import (
    CSTC_STRUCTURES,
    fetch_gene_expression,
)

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"

# ── DBS fiber tract corridor definitions ──────────────────────────────────
# Each tract is approximated by the CSTC regions it traverses.
# Allen Brain Atlas structure IDs are used (same as ahba_spatial module).
#
# Ansa lenticularis: GPi → ventral around IC → VA/VL thalamus
# Fasciculus lenticularis: GPi → dorsal through H2 field of Forel → VA/VL thalamus
# Posterior intralaminar-lentiform: CM/Pf thalamic nuclei → putamen

DBS_TRACTS: dict[str, dict] = {
    "ansa_lenticularis": {
        "abbreviation": "AL",
        "description": "GPi to thalamus via ventral route around internal capsule",
        "regions": ["GPi", "thalamus"],
        "origin": "GPi",
        "target": "thalamus",
    },
    "fasciculus_lenticularis": {
        "abbreviation": "FL",
        "description": "GPi to thalamus via dorsal route through H2 field of Forel",
        "regions": ["GPi", "STN", "thalamus"],
        "origin": "GPi",
        "target": "thalamus",
    },
    "posterior_intralaminar_lentiform": {
        "abbreviation": "PIL",
        "description": "CM/Pf thalamic nuclei to putamen (intralaminar projections)",
        "regions": ["thalamus", "putamen"],
        "origin": "thalamus",
        "target": "putamen",
    },
}

# ── Thalamic response subdivisions ───────────────────────────────────────
# Based on the multi-center DBS study findings: tic response mapped to
# anterior/ventral thalamic targets (VA, VL), while OCD response mapped
# to more medial/posterior targets (MD, CM/Pf).
# Since AHBA uses a single thalamus structure ID, we classify thalamic
# sub-regions by their functional relevance.

THALAMIC_RESPONSE_MAP: dict[str, dict] = {
    "tic_responsive": {
        "description": "Ventral anterior (VA) and ventrolateral (VL) thalamus — primary tic improvement targets",
        "nuclei": ["VA", "VL"],
    },
    "ocd_responsive": {
        "description": "Mediodorsal (MD) and centromedian-parafascicular (CM/Pf) — OCD improvement targets",
        "nuclei": ["MD", "CM", "Pf"],
    },
}

# Non-tract basal ganglia regions for comparison
NON_TRACT_BG_REGIONS = ["caudate", "GPe"]


def get_tract_cell_types(tract_name: str) -> list[CellType]:
    """Return HMBA cell types affected by a given DBS tract corridor.

    Aggregates cell types from all CSTC regions traversed by the tract.
    """
    regions = get_tract_regions(tract_name)
    cell_types: list[CellType] = []
    seen: set[str] = set()
    for region in regions:
        for ct in get_cstc_region_cell_types(region):
            if ct.label not in seen:
                cell_types.append(ct)
                seen.add(ct.label)
    return cell_types


def get_all_tract_cell_type_map() -> dict[str, list[str]]:
    """Return mapping of tract name -> list of HMBA cell-type labels."""
    return {
        name: [ct.label for ct in get_tract_cell_types(name)]
        for name in DBS_TRACTS
    }


def get_tract_regions(tract_name: str) -> list[str]:
    """Return CSTC region names traversed by a given DBS tract."""
    if tract_name not in DBS_TRACTS:
        raise KeyError(
            f"Unknown tract {tract_name!r}. "
            f"Available: {', '.join(DBS_TRACTS.keys())}"
        )
    return list(DBS_TRACTS[tract_name]["regions"])


def extract_tract_expression(
    expr_df: pd.DataFrame,
    tract_name: str,
) -> pd.DataFrame:
    """Extract expression data for regions along a DBS tract corridor.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression DataFrame from ahba_spatial.fetch_gene_expression.
    tract_name : str
        Name of the DBS tract.

    Returns
    -------
    pd.DataFrame
        Filtered expression data for tract regions, with 'tract' column added.
    """
    regions = get_tract_regions(tract_name)
    df = expr_df[expr_df["cstc_region"].isin(regions)].copy()
    df["tract"] = tract_name
    return df


def profile_all_tracts(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Profile gene expression along all three DBS tract corridors.

    Returns a DataFrame with tract-level expression summaries per gene.
    """
    records: list[dict] = []

    for tract_name, tract_info in DBS_TRACTS.items():
        tract_df = extract_tract_expression(expr_df, tract_name)
        if tract_df.empty:
            continue

        for gene, gene_group in tract_df.groupby("gene_symbol"):
            mean_z = gene_group["mean_zscore"].mean()
            std_z = gene_group["mean_zscore"].std()
            n = len(gene_group)
            tract_cts = get_tract_cell_types(tract_name)
            records.append({
                "gene_symbol": gene,
                "tract": tract_name,
                "tract_abbrev": tract_info["abbreviation"],
                "mean_zscore": mean_z,
                "std_zscore": std_z if not np.isnan(std_z) else 0.0,
                "n_observations": n,
                "regions": ",".join(tract_info["regions"]),
                "hmba_cell_types": ",".join(ct.label for ct in tract_cts),
            })

    return pd.DataFrame(records)


def compare_tract_vs_nontract(expr_df: pd.DataFrame) -> pd.DataFrame:
    """Compare expression between tract corridor regions and non-tract BG regions.

    For each gene, computes the mean expression in tract regions vs non-tract
    BG regions and tests for significant differences (Mann-Whitney U).
    """
    # Collect all tract regions (unique)
    tract_regions: list[str] = []
    for info in DBS_TRACTS.values():
        for r in info["regions"]:
            if r not in tract_regions:
                tract_regions.append(r)

    nontract = expr_df[expr_df["cstc_region"].isin(NON_TRACT_BG_REGIONS)]
    tract = expr_df[expr_df["cstc_region"].isin(tract_regions)]

    records: list[dict] = []
    for gene in expr_df["gene_symbol"].unique():
        t_vals = np.asarray(tract.loc[tract["gene_symbol"] == gene, "mean_zscore"])
        nt_vals = np.asarray(nontract.loc[nontract["gene_symbol"] == gene, "mean_zscore"])

        if len(t_vals) < 2 or len(nt_vals) < 2:
            continue

        u_stat, p_val = stats.mannwhitneyu(
            t_vals, nt_vals, alternative="two-sided",
        )
        records.append({
            "gene_symbol": gene,
            "tract_mean": float(np.mean(t_vals)),
            "nontract_mean": float(np.mean(nt_vals)),
            "diff": float(np.mean(t_vals) - np.mean(nt_vals)),
            "u_statistic": float(u_stat),
            "p_value": float(p_val),
            "significant": bool(p_val < 0.05),
            "n_tract": len(t_vals),
            "n_nontract": len(nt_vals),
        })

    result = pd.DataFrame(records)
    if not result.empty:
        result = result.sort_values("p_value").reset_index(drop=True)
    return result


def analyze_thalamic_gene_subsets(
    expr_df: pd.DataFrame,
    gene_set_names: list[str] | None = None,
) -> dict:
    """Examine whether different TS gene subsets show differential expression
    in thalamic regions.

    Since AHBA uses a single thalamus structure ID, we compare gene set
    expression levels within thalamus to identify which gene subsets are
    most enriched in the DBS target region.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression data from fetch_gene_expression.
    gene_set_names : list of str, optional
        Gene set names to compare. Defaults to all non-combined sets.
    """
    if gene_set_names is None:
        gene_set_names = [
            "tsaicg_gwas", "rare_variant", "iron_homeostasis", "hippo_signaling",
        ]

    thalamic = expr_df.loc[expr_df["cstc_region"] == "thalamus"]
    if len(thalamic) == 0:
        return {"error": "No thalamic expression data available"}

    subset_stats: dict[str, dict] = {}
    subset_values: dict[str, np.ndarray] = {}

    for gs_name in gene_set_names:
        try:
            gs_genes = sorted(get_gene_set(gs_name).keys())
        except KeyError:
            continue

        gs_thalamic = thalamic.loc[thalamic["gene_symbol"].isin(gs_genes)]
        if len(gs_thalamic) == 0:
            continue

        vals = np.asarray(gs_thalamic["mean_zscore"], dtype=float)
        subset_values[gs_name] = vals
        subset_stats[gs_name] = {
            "mean_zscore": float(np.mean(vals)),
            "std_zscore": float(np.std(vals)),
            "n_genes": int(gs_thalamic["gene_symbol"].nunique()),
            "n_observations": len(vals),
        }

    # Pairwise comparisons between gene subsets in thalamus
    pairwise: list[dict] = []
    names = list(subset_values.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = subset_values[names[i]], subset_values[names[j]]
            if len(a) >= 2 and len(b) >= 2:
                u_stat, p_val = stats.mannwhitneyu(a, b, alternative="two-sided")
                pairwise.append({
                    "set_a": names[i],
                    "set_b": names[j],
                    "mean_a": float(np.mean(a)),
                    "mean_b": float(np.mean(b)),
                    "u_statistic": float(u_stat),
                    "p_value": float(p_val),
                })

    return {
        "subset_stats": subset_stats,
        "pairwise_comparisons": pairwise,
        "interpretation": (
            "Differential thalamic expression between gene subsets may indicate "
            "which genetic pathways are most relevant to DBS target regions. "
            "Higher expression in tic-relevant gene sets (GWAS, rare variants) "
            "supports the thalamic convergence hypothesis."
        ),
    }


def generate_tract_heatmap(
    tract_profiles: pd.DataFrame,
    output_path: Path,
    title: str = "TS Gene Expression Along DBS Fiber Tract Corridors",
) -> Path:
    """Generate a heatmap of gene expression across DBS tract corridors."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    if tract_profiles.empty:
        logger.warning("No tract profile data for heatmap")
        return output_path

    pivot = tract_profiles.pivot_table(
        index="gene_symbol",
        columns="tract_abbrev",
        values="mean_zscore",
        aggfunc="mean",
    )

    fig, ax = plt.subplots(figsize=(8, max(6, len(pivot) * 0.35)))
    sns.heatmap(
        pivot,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title(title, fontsize=13, pad=15)
    ax.set_ylabel("Gene")
    ax.set_xlabel("DBS Tract Corridor")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved tract heatmap to %s", output_path)
    return output_path


def run_analysis(
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
    cache: bool = True,
) -> dict:
    """Run the full DBS fiber tract expression profiling analysis.

    Returns dict with tract profiles, comparison stats, and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading gene set: %s", gene_set_name)
    genes = get_gene_set(gene_set_name)
    gene_list = sorted(genes.keys())
    logger.info("  %d genes in set", len(gene_list))

    # Fetch expression data (reuses ahba_spatial cache)
    logger.info("Fetching AHBA expression data...")
    expr_df = fetch_gene_expression(gene_list, cache=cache)

    if expr_df.empty:
        logger.error("No expression data retrieved.")
        return {"error": "No expression data"}

    # Profile all tracts
    logger.info("Profiling expression along DBS tract corridors...")
    tract_profiles = profile_all_tracts(expr_df)

    # Compare tract vs non-tract
    logger.info("Comparing tract vs non-tract expression...")
    comparison = compare_tract_vs_nontract(expr_df)

    # Thalamic gene subset analysis
    logger.info("Analyzing thalamic gene subsets...")
    thalamic_analysis = analyze_thalamic_gene_subsets(expr_df)

    # Save outputs
    profiles_path = output_dir / f"dbs_tract_profiles_{gene_set_name}.csv"
    tract_profiles.to_csv(profiles_path, index=False)
    logger.info("Saved tract profiles: %s", profiles_path)

    comparison_path = output_dir / f"dbs_tract_vs_nontract_{gene_set_name}.csv"
    comparison.to_csv(comparison_path, index=False)

    thalamic_path = output_dir / f"dbs_thalamic_subsets_{gene_set_name}.json"
    with open(thalamic_path, "w") as f:
        json.dump(thalamic_analysis, f, indent=2)

    heatmap_path = output_dir / f"dbs_tract_heatmap_{gene_set_name}.png"
    generate_tract_heatmap(tract_profiles, heatmap_path)

    # Summary stats
    n_sig_genes = int(comparison["significant"].sum()) if not comparison.empty else 0
    top_tract_genes = (
        comparison[comparison["significant"]]["gene_symbol"].tolist()[:10]
        if not comparison.empty else []
    )

    return {
        "gene_set": gene_set_name,
        "n_genes": len(gene_list),
        "n_tracts": len(DBS_TRACTS),
        "tract_profile_rows": len(tract_profiles),
        "n_significant_tract_genes": n_sig_genes,
        "top_tract_differential_genes": top_tract_genes,
        "thalamic_analysis": thalamic_analysis,
        "outputs": {
            "tract_profiles_csv": str(profiles_path),
            "tract_vs_nontract_csv": str(comparison_path),
            "thalamic_subsets_json": str(thalamic_path),
            "tract_heatmap_png": str(heatmap_path),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="DBS fiber tract expression profiling for TS risk genes"
    )
    parser.add_argument(
        "--gene-set",
        default="ts_combined",
        choices=list_gene_sets(),
        help="Gene set to analyze (default: ts_combined)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Force fresh data download (ignore cache)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_DIR,
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    results = run_analysis(
        gene_set_name=args.gene_set,
        output_dir=args.output,
        cache=not args.no_cache,
    )

    if "error" in results:
        print(f"ERROR: {results['error']}")
        return

    print(f"\nDBS Tract Analysis complete for gene set: {results['gene_set']}")
    print(f"  Genes: {results['n_genes']}")
    print(f"  Tracts profiled: {results['n_tracts']}")
    print(f"  Genes with significant tract vs non-tract diff: {results['n_significant_tract_genes']}")
    if results["top_tract_differential_genes"]:
        print(f"  Top differential genes: {', '.join(results['top_tract_differential_genes'])}")
    print("\nOutputs:")
    for name, path in results["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
