"""SL benchmark integration: load and harmonize 3 synthetic lethality datasets.

Datasets:
1. Genome Biology 2025 combinatorial CRISPR (experimental, 117+ validated pairs)
2. Vermeulen et al. Boruta/PRISM (computational, high-confidence from DepMap screens)
3. Desjardins et al. isogenic CRISPR (isogenic validation, 15 drivers)

Usage:
    from bioagentics.models.sl_benchmarks import load_sl_benchmarks
    df = load_sl_benchmarks("data/benchmarks")
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

CONFIDENCE_TIERS = {
    "genome_biology": "experimental",
    "desjardins": "isogenic",
    "vermeulen": "computational",
}


def _load_genome_biology(sl_dir: Path, min_hits: int = 1) -> pd.DataFrame:
    """Load Genome Biology 2025 combinatorial CRISPR SL pairs.

    Pairs with hit_count >= min_hits are considered validated.
    """
    path = sl_dir / "sl_pairs" / "sl_pairs_summary.tsv"
    df = pd.read_csv(path, sep="\t")
    hits = df[df["hit_count"] >= min_hits][["gene_1", "gene_2"]].copy()
    hits.columns = ["gene_a", "gene_b"]
    hits["source"] = "genome_biology"
    hits["confidence_tier"] = "experimental"
    hits["tissue"] = ""
    hits["driver_context"] = ""
    logger.info("Genome Biology: %d validated SL pairs (hit_count >= %d)", len(hits), min_hits)
    return hits


def _load_vermeulen(sl_dir: Path) -> pd.DataFrame:
    """Load Vermeulen et al. high-confidence SL pairs (included == 'included')."""
    path = sl_dir / "vermeulen_sl" / "vermeulen_sl_interactions.tsv"
    df = pd.read_csv(path, sep="\t")
    incl = df[df["included"] == "included"][["source", "target"]].copy()
    incl.columns = ["gene_a", "gene_b"]
    incl["source"] = "vermeulen"
    incl["confidence_tier"] = "computational"
    incl["tissue"] = "pan-cancer"
    incl["driver_context"] = incl["gene_a"]
    logger.info("Vermeulen: %d high-confidence SL pairs", len(incl))
    return incl


def _load_desjardins(sl_dir: Path, min_cca: float = 1.0) -> pd.DataFrame:
    """Load Desjardins et al. isogenic CRISPR screen SL hits.

    Reads all 15 driver sheets from Table1, filtering by CCA >= min_cca.
    """
    path = sl_dir / "desjardins_isogenic_sl" / "Table1_isogenic_depmap_SL_screen_data.xlsx"
    xl = pd.ExcelFile(path)

    frames = []
    for sheet in xl.sheet_names:
        sdf = pd.read_excel(xl, sheet_name=sheet)
        hits = sdf[sdf["CCA"] >= min_cca][["Gene", "Lesion"]].copy()
        hits.columns = ["gene_b", "driver"]
        hits["gene_a"] = sheet  # driver is the sheet name
        frames.append(hits)

    if not frames:
        return pd.DataFrame(columns=["gene_a", "gene_b", "source", "confidence_tier", "tissue", "driver_context"])

    combined = pd.concat(frames, ignore_index=True)
    combined["source"] = "desjardins"
    combined["confidence_tier"] = "isogenic"
    combined["tissue"] = ""
    combined["driver_context"] = combined["gene_a"]
    combined = combined[["gene_a", "gene_b", "source", "confidence_tier", "tissue", "driver_context"]]

    logger.info("Desjardins: %d SL hits across %d drivers (CCA >= %.1f)",
                len(combined), len(xl.sheet_names), min_cca)
    return combined


def load_sl_benchmarks(
    benchmark_dir: str | Path,
    *,
    min_gb_hits: int = 1,
    min_cca: float = 1.0,
) -> pd.DataFrame:
    """Load and combine all 3 SL benchmark datasets into a unified reference.

    Returns DataFrame with columns:
        gene_a, gene_b, source, confidence_tier, tissue, driver_context, n_sources

    Pairs appearing in multiple sources get all source annotations preserved
    and a higher n_sources count.
    """
    benchmark_dir = Path(benchmark_dir)

    gb = _load_genome_biology(benchmark_dir, min_hits=min_gb_hits)
    vm = _load_vermeulen(benchmark_dir)
    dj = _load_desjardins(benchmark_dir, min_cca=min_cca)

    combined = pd.concat([gb, vm, dj], ignore_index=True)

    # Normalize gene pair ordering for deduplication (alphabetical)
    combined["pair_key"] = combined.apply(
        lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1
    )

    # Count sources per pair
    source_counts = combined.groupby("pair_key")["source"].apply(
        lambda x: ",".join(sorted(set(x)))
    ).rename("sources")
    n_sources = combined.groupby("pair_key")["source"].nunique().rename("n_sources")

    # Aggregate driver context and tissue per pair
    driver_ctx = combined.groupby("pair_key")["driver_context"].apply(
        lambda x: ",".join(sorted(set(s for s in x if s)))
    ).rename("driver_context_all")

    pair_info = pd.concat([source_counts, n_sources, driver_ctx], axis=1).reset_index()

    # Keep one row per unique pair with the highest confidence tier
    tier_rank = {"experimental": 3, "isogenic": 2, "computational": 1}
    combined["tier_rank"] = combined["confidence_tier"].map(tier_rank)
    best = combined.sort_values("tier_rank", ascending=False).drop_duplicates(
        subset="pair_key", keep="first"
    )

    result = best.merge(pair_info, on="pair_key")
    result = result[["gene_a", "gene_b", "sources", "confidence_tier",
                      "tissue", "driver_context_all", "n_sources"]]
    result = result.rename(columns={"driver_context_all": "driver_context"})
    result = result.sort_values(["n_sources", "confidence_tier"],
                                ascending=[False, True]).reset_index(drop=True)

    logger.info("Combined: %d unique SL pairs (%d multi-source)",
                len(result), (result["n_sources"] > 1).sum())
    return result
