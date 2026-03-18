"""AHBA spatial mapping of TS risk genes across CSTC circuit regions.

Queries Allen Human Brain Atlas (AHBA) microarray data via the Allen Institute
API to profile expression of Tourette syndrome risk genes across the
cortico-striato-thalamo-cortical (CSTC) circuit.

Pipeline:
1. Resolve gene symbols → AHBA probe IDs
2. Query microarray expression per donor across CSTC structures
3. Compute z-scored regional enrichment (vs whole-brain baseline)
4. Generate CSTC heatmap and statistical tests

Data source: Allen Human Brain Atlas — ~3,700 samples from 6 donors.
API docs: https://help.brain-map.org/display/api/Allen+Human+Brain+Atlas

Usage:
    uv run python -m bioagentics.analysis.tourettes.ahba_spatial [--gene-set ts_combined]
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set, list_gene_sets

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "cstc-circuit-expression-atlas"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "ahba"

AHBA_API = "https://api.brain-map.org/api/v2"

# CSTC circuit structures — Allen ontology structure IDs
# Mapped from the Allen Human Brain Atlas structure ontology.
CSTC_STRUCTURES: dict[str, list[int]] = {
    "prefrontal_cortex": [4251],  # orbital gyrus / prefrontal
    "motor_cortex": [4260],  # precentral gyrus (M1)
    "caudate": [4229],  # caudate nucleus
    "putamen": [4278],  # putamen
    "GPe": [4249],  # globus pallidus, external segment
    "GPi": [4250],  # globus pallidus, internal segment
    "STN": [10657],  # subthalamic nucleus
    "thalamus": [4394],  # thalamus (parent)
}

# Flatten for API queries
ALL_STRUCTURE_IDS = [sid for sids in CSTC_STRUCTURES.values() for sid in sids]

# AHBA donor IDs (all 6 donors in the human microarray dataset)
DONOR_IDS = [9861, 10021, 12876, 14380, 15496, 15697]


def _api_get(path: str, params: dict | None = None, retries: int = 3) -> dict:
    """GET request to Allen Brain Atlas API with retry."""
    url = f"{AHBA_API}/{path}"
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            if attempt == retries - 1:
                raise
            logger.warning("AHBA API attempt %d failed: %s", attempt + 1, e)
            time.sleep(2 ** attempt)
    return {}  # unreachable


def resolve_probes(gene_symbol: str) -> list[dict]:
    """Find AHBA microarray probe IDs for a gene symbol.

    Returns list of dicts with keys: id, name, gene_id, gene_symbol.
    """
    data = _api_get(
        "data/query.json",
        params={
            "criteria": (
                f"model::Probe,"
                f"rma::criteria,gene[acronym$eq'{gene_symbol}'],"
                f"products[abbreviation$eq'HumanMA']"
            ),
            "num_rows": "all",
            "only": "probes.id,name,gene_id,genes.acronym",
        },
    )
    rows = data.get("msg", [])
    return [
        {
            "id": r["id"],
            "name": r.get("name", ""),
            "gene_id": r.get("gene_id"),
            "gene_symbol": gene_symbol,
        }
        for r in rows
    ]


def query_expression(
    probe_ids: list[int],
    donor_id: int,
) -> pd.DataFrame:
    """Query microarray expression for probes in one donor across all structures.

    Returns DataFrame: rows = samples, columns include structure_id,
    structure_name, mri_voxel coordinates, and expression z-score per probe.
    """
    if not probe_ids:
        return pd.DataFrame()

    probes_str = ",".join(str(p) for p in probe_ids)
    data = _api_get(
        "data/query.json",
        params={
            "criteria": (
                f"service::human_microarray_expression"
                f"[probes$in{probes_str}]"
                f"[donors$eq{donor_id}]"
            ),
        },
    )

    msg = data.get("msg", {})
    if not msg:
        return pd.DataFrame()

    probes_info = msg.get("probes", [])
    samples = msg.get("samples", [])

    if not samples or not probes_info:
        return pd.DataFrame()

    # Build sample metadata
    sample_records = []
    for s in samples:
        sample_records.append({
            "sample_id": s.get("sample", {}).get("well", 0),
            "structure_id": s.get("structure", {}).get("id", 0),
            "structure_name": s.get("structure", {}).get("name", ""),
            "structure_acronym": s.get("structure", {}).get("acronym", ""),
            "donor_id": donor_id,
            "mni_x": s.get("sample", {}).get("mni", [0, 0, 0])[0] if s.get("sample", {}).get("mni") else 0,
            "mni_y": s.get("sample", {}).get("mni", [0, 0, 0])[1] if s.get("sample", {}).get("mni") else 0,
            "mni_z": s.get("sample", {}).get("mni", [0, 0, 0])[2] if s.get("sample", {}).get("mni") else 0,
        })

    df = pd.DataFrame(sample_records)

    # Add probe expression values
    for pi in probes_info:
        probe_id = pi.get("id", 0)
        z_scores = pi.get("z-score", [])
        if len(z_scores) == len(df):
            df[f"probe_{probe_id}"] = z_scores

    return df


def map_samples_to_cstc(df: pd.DataFrame) -> pd.DataFrame:
    """Map AHBA samples to CSTC regions using structure IDs.

    Adds a 'cstc_region' column. Samples not in CSTC get 'other'.
    """
    # Build reverse lookup: structure_id → region name
    sid_to_region: dict[int, str] = {}
    for region, sids in CSTC_STRUCTURES.items():
        for sid in sids:
            sid_to_region[sid] = region

    df = df.copy()
    df["cstc_region"] = df["structure_id"].map(sid_to_region).fillna("other")
    return df


def fetch_gene_expression(
    gene_symbols: list[str],
    donors: list[int] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """Fetch AHBA expression for multiple genes across all donors.

    Returns a DataFrame with columns: gene_symbol, cstc_region, donor_id,
    mean_zscore (averaged across probes per gene).
    """
    donors = donors or DONOR_IDS
    cache_path = CACHE_DIR / "ahba_expression_cache.parquet"

    if cache and cache_path.exists():
        logger.info("Loading cached AHBA expression data from %s", cache_path)
        return pd.read_parquet(cache_path)

    all_records: list[dict] = []
    failed_genes: list[str] = []

    for gene in gene_symbols:
        logger.info("Resolving probes for %s...", gene)
        probes = resolve_probes(gene)
        if not probes:
            logger.warning("No AHBA probes found for %s", gene)
            failed_genes.append(gene)
            continue

        probe_ids = [p["id"] for p in probes]
        logger.info("  %s: %d probes found", gene, len(probe_ids))

        for donor_id in donors:
            df = query_expression(probe_ids, donor_id)
            if df.empty:
                continue

            df = map_samples_to_cstc(df)

            # Average expression across probes for this gene
            probe_cols = [c for c in df.columns if c.startswith("probe_")]
            if not probe_cols:
                continue

            df["mean_zscore"] = df[probe_cols].mean(axis=1)

            # Aggregate by CSTC region
            for region, group in df.groupby("cstc_region"):
                all_records.append({
                    "gene_symbol": gene,
                    "cstc_region": region,
                    "donor_id": donor_id,
                    "mean_zscore": group["mean_zscore"].mean(),
                    "n_samples": len(group),
                    "n_probes": len(probe_cols),
                })

        # Rate-limit to be kind to the API
        time.sleep(0.5)

    if failed_genes:
        logger.warning("Genes with no AHBA probes: %s", ", ".join(failed_genes))

    result = pd.DataFrame(all_records)

    if cache and not result.empty:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        result.to_parquet(cache_path, index=False)
        logger.info("Cached expression data to %s", cache_path)

    return result


def compute_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    """Compute regional enrichment scores for each gene.

    For each gene, z-score its CSTC regional expression relative to
    the whole-brain baseline (all regions including 'other').
    Returns a gene x region enrichment matrix.
    """
    if df.empty:
        return pd.DataFrame()

    # Average across donors
    avg = (
        df.groupby(["gene_symbol", "cstc_region"])["mean_zscore"]
        .mean()
        .reset_index()
    )

    # Pivot to gene x region matrix
    pivot = avg.pivot(index="gene_symbol", columns="cstc_region", values="mean_zscore")

    # Z-score each gene across regions (row-wise)
    enrichment = pivot.apply(
        lambda row: (row - row.mean()) / row.std() if row.std() > 0 else row * 0,
        axis=1,
    )

    # Reorder columns to match CSTC circuit flow
    cstc_order = [
        "prefrontal_cortex", "motor_cortex", "caudate", "putamen",
        "GPe", "GPi", "STN", "thalamus",
    ]
    available = [c for c in cstc_order if c in enrichment.columns]
    return enrichment[available]


def check_regional_specificity(enrichment: pd.DataFrame) -> dict:
    """Test whether TS genes show non-uniform expression across CSTC nodes.

    Uses one-way ANOVA across regions (genes as observations).
    Returns dict with F-statistic, p-value, and top-enriched regions.
    """
    from scipy import stats

    if enrichment.empty or enrichment.shape[1] < 2:
        return {"error": "Insufficient data for statistical test"}

    # Each column is a region, each row is a gene
    groups = [enrichment[col].dropna().values for col in enrichment.columns]
    groups = [g for g in groups if len(g) > 0]

    if len(groups) < 2:
        return {"error": "Need at least 2 regions with data"}

    f_stat, p_value = stats.f_oneway(*groups)

    # Identify top-enriched regions (by mean enrichment score)
    region_means = enrichment.mean().sort_values(ascending=False)

    return {
        "f_statistic": float(f_stat),
        "p_value": float(p_value),
        "significant": bool(p_value < 0.05),
        "top_regions": {k: float(v) for k, v in region_means.head(3).items()},
        "n_genes": enrichment.shape[0],
        "n_regions": enrichment.shape[1],
    }


def generate_heatmap(
    enrichment: pd.DataFrame,
    output_path: Path,
    title: str = "TS Risk Gene Expression Across CSTC Circuit",
) -> Path:
    """Generate and save a heatmap of gene x region enrichment scores."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(12, max(6, len(enrichment) * 0.35)))
    sns.heatmap(
        enrichment,
        cmap="RdBu_r",
        center=0,
        annot=True,
        fmt=".2f",
        linewidths=0.5,
        ax=ax,
        xticklabels=[r.replace("_", " ").title() for r in enrichment.columns],
    )
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_ylabel("Gene")
    ax.set_xlabel("CSTC Region")
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved heatmap to %s", output_path)
    return output_path


def run_analysis(
    gene_set_name: str = "ts_combined",
    output_dir: Path = OUTPUT_DIR,
    cache: bool = True,
) -> dict:
    """Run the full AHBA spatial mapping analysis.

    Returns dict with enrichment matrix, stats, and output paths.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading gene set: %s", gene_set_name)
    genes = get_gene_set(gene_set_name)
    gene_list = sorted(genes.keys())
    logger.info("  %d genes in set", len(gene_list))

    # Fetch expression data
    logger.info("Fetching AHBA expression data...")
    expr_df = fetch_gene_expression(gene_list, cache=cache)

    if expr_df.empty:
        logger.error("No expression data retrieved. Check API connectivity.")
        return {"error": "No expression data"}

    # Compute enrichment
    logger.info("Computing regional enrichment scores...")
    enrichment = compute_enrichment(expr_df)

    # Statistical test
    stats_result = check_regional_specificity(enrichment)
    logger.info("Regional specificity test: F=%.2f, p=%.4f",
                stats_result.get("f_statistic", 0),
                stats_result.get("p_value", 1))

    # Save outputs
    enrichment_path = output_dir / f"cstc_enrichment_{gene_set_name}.csv"
    enrichment.to_csv(enrichment_path)
    logger.info("Saved enrichment matrix: %s", enrichment_path)

    stats_path = output_dir / f"cstc_stats_{gene_set_name}.json"
    with open(stats_path, "w") as f:
        json.dump(stats_result, f, indent=2)

    heatmap_path = output_dir / f"cstc_heatmap_{gene_set_name}.png"
    generate_heatmap(enrichment, heatmap_path,
                     title=f"TS Risk Gene CSTC Expression ({gene_set_name})")

    expr_path = output_dir / f"cstc_expression_{gene_set_name}.parquet"
    expr_df.to_parquet(expr_path, index=False)

    return {
        "gene_set": gene_set_name,
        "n_genes_queried": len(gene_list),
        "n_genes_found": enrichment.shape[0],
        "n_regions": enrichment.shape[1],
        "stats": stats_result,
        "outputs": {
            "enrichment_csv": str(enrichment_path),
            "stats_json": str(stats_path),
            "heatmap_png": str(heatmap_path),
            "expression_parquet": str(expr_path),
        },
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="AHBA spatial mapping of TS risk genes across CSTC circuit"
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

    print(f"\nAnalysis complete for gene set: {results['gene_set']}")
    print(f"  Genes queried: {results['n_genes_queried']}")
    print(f"  Genes found in AHBA: {results['n_genes_found']}")
    print(f"  CSTC regions: {results['n_regions']}")
    stats = results["stats"]
    print(f"  Regional specificity F={stats.get('f_statistic', 0):.2f}, "
          f"p={stats.get('p_value', 1):.4f}")
    if stats.get("significant"):
        print("  ** Significant non-uniform expression across CSTC regions **")
    print("\nOutputs:")
    for name, path in results["outputs"].items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
