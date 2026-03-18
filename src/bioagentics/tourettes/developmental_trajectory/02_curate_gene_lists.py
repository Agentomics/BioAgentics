"""Step 02 — Curate and validate TS developmental gene lists.

Assembles the complete gene lists needed for developmental trajectory modeling,
validates gene symbol coverage against BrainSpan, and outputs curated gene
lists with metadata for downstream analysis.

Task: #779 (TS gene list curation)
Project: ts-developmental-trajectory-modeling

Gene categories for this project:
  - TSAICG GWAS risk genes (including 2024 additions: BCL11B, NDFIP2, RBM26)
  - Rare variant genes (SLITRK1, HDC, NRXN1, CNTN6, WWC1)
  - De novo variant genes (PPP5C, EXOC1, GXYLT1)
  - Hormone receptors (AR, ESR1, ESR2) for Phase 4 puberty analysis
  - Iron homeostasis genes (for CSTC iron depletion context)
  - Hippo signaling genes (WWC1 pathway)

Usage:
    uv run python -m bioagentics.tourettes.developmental_trajectory.02_curate_gene_lists
    uv run python -m bioagentics.tourettes.developmental_trajectory.02_curate_gene_lists --validate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.tourettes.gene_sets import get_gene_set

logger = logging.getLogger(__name__)

OUTPUT_DIR = REPO_ROOT / "output" / "tourettes" / "ts-developmental-trajectory-modeling"
CACHE_DIR = REPO_ROOT / "data" / "cache" / "brainspan"


def assemble_developmental_gene_lists() -> dict[str, dict[str, str]]:
    """Assemble all gene lists needed for developmental trajectory modeling.

    Returns dict of {set_name: {gene_symbol: description}}.
    """
    gene_lists: dict[str, dict[str, str]] = {}

    # Core TS risk gene sets
    for name in ["tsaicg_gwas", "rare_variant", "de_novo_variant"]:
        gene_lists[name] = get_gene_set(name)

    # Combined TS risk (all genetic evidence)
    ts_risk_combined: dict[str, str] = {}
    for name in ["tsaicg_gwas", "rare_variant", "de_novo_variant"]:
        ts_risk_combined.update(gene_lists[name])
    gene_lists["ts_risk_combined"] = ts_risk_combined

    # Pathway sets
    for name in ["iron_homeostasis", "hippo_signaling"]:
        gene_lists[name] = get_gene_set(name)

    # Hormone receptors for Phase 4
    gene_lists["hormone_receptors"] = get_gene_set("hormone_receptors")

    # Full union for BrainSpan query
    all_genes: dict[str, str] = {}
    for gs in gene_lists.values():
        all_genes.update(gs)
    gene_lists["ts_developmental_all"] = all_genes

    return gene_lists


def validate_against_brainspan(
    gene_lists: dict[str, dict[str, str]],
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Validate gene symbols against BrainSpan rows metadata.

    Returns validation report with coverage stats per gene set.
    """
    rows_path = cache_dir / "rows_metadata.parquet"
    if not rows_path.exists():
        logger.warning("BrainSpan data not cached yet — run 01_download_brainspan first")
        return {"error": "BrainSpan data not cached. Run step 01 first."}

    rows_meta = pd.read_parquet(rows_path)

    # Find gene symbol column
    gene_col = next(
        (c for c in rows_meta.columns if "gene_symbol" in c.lower() or "symbol" in c.lower()),
        None,
    )
    if gene_col is None:
        return {"error": f"No gene symbol column found in rows_metadata: {rows_meta.columns.tolist()}"}

    brainspan_symbols = set(rows_meta[gene_col].dropna().str.upper().unique())
    logger.info("BrainSpan contains %d unique gene symbols", len(brainspan_symbols))

    report: dict = {
        "brainspan_total_genes": len(brainspan_symbols),
        "gene_set_coverage": {},
    }

    for set_name, genes in gene_lists.items():
        query_symbols = set(g.upper() for g in genes.keys())
        found = query_symbols & brainspan_symbols
        missing = query_symbols - brainspan_symbols

        report["gene_set_coverage"][set_name] = {
            "total": len(query_symbols),
            "found": len(found),
            "missing": len(missing),
            "coverage_pct": round(len(found) / max(len(query_symbols), 1) * 100, 1),
            "found_genes": sorted(found),
            "missing_genes": sorted(missing),
        }

    return report


def export_gene_lists(
    gene_lists: dict[str, dict[str, str]],
    output_dir: Path = OUTPUT_DIR,
) -> list[Path]:
    """Export gene lists as CSV files for downstream analysis.

    Each CSV has columns: gene_symbol, description, gene_set.
    Also exports a combined master list.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Individual gene sets
    for set_name, genes in gene_lists.items():
        records = [
            {"gene_symbol": sym, "description": desc, "gene_set": set_name}
            for sym, desc in sorted(genes.items())
        ]
        df = pd.DataFrame(records)
        path = output_dir / f"gene_list_{set_name}.csv"
        df.to_csv(path, index=False)
        paths.append(path)

    # Summary table: gene x set membership matrix
    all_symbols = sorted(gene_lists.get("ts_developmental_all", {}).keys())
    membership: list[dict] = []
    for sym in all_symbols:
        row: dict[str, object] = {"gene_symbol": sym}
        for set_name, genes in gene_lists.items():
            if set_name == "ts_developmental_all":
                continue
            row[set_name] = sym in genes
        membership.append(row)

    membership_df = pd.DataFrame(membership)
    membership_path = output_dir / "gene_set_membership.csv"
    membership_df.to_csv(membership_path, index=False)
    paths.append(membership_path)

    return paths


def run(
    output_dir: Path = OUTPUT_DIR,
    validate: bool = True,
    cache_dir: Path = CACHE_DIR,
) -> dict:
    """Assemble, validate, and export developmental gene lists."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Assemble
    gene_lists = assemble_developmental_gene_lists()
    logger.info("Assembled %d gene sets:", len(gene_lists))
    for name, genes in gene_lists.items():
        logger.info("  %s: %d genes", name, len(genes))

    # Export
    paths = export_gene_lists(gene_lists, output_dir)
    logger.info("Exported %d files to %s", len(paths), output_dir)

    result: dict = {
        "gene_sets": {name: len(genes) for name, genes in gene_lists.items()},
        "exported_files": [str(p) for p in paths],
    }

    # Validate against BrainSpan if requested
    if validate:
        validation = validate_against_brainspan(gene_lists, cache_dir)
        result["brainspan_validation"] = validation

        validation_path = output_dir / "gene_list_validation.json"
        with open(validation_path, "w") as f:
            json.dump(validation, f, indent=2)
        logger.info("Validation report saved to %s", validation_path)

    # Save summary
    summary_path = output_dir / "gene_curation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Curate and validate TS developmental gene lists"
    )
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=Path, default=CACHE_DIR)
    parser.add_argument(
        "--validate", action="store_true", default=False,
        help="Validate gene symbols against BrainSpan (requires step 01 first)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    result = run(output_dir=args.output, validate=args.validate, cache_dir=args.cache)

    print(f"\nTS Developmental Gene List Curation")
    print(f"  Gene sets assembled:")
    for name, count in result["gene_sets"].items():
        print(f"    {name}: {count} genes")

    if "brainspan_validation" in result:
        validation = result["brainspan_validation"]
        if "error" in validation:
            print(f"\n  BrainSpan validation: {validation['error']}")
        else:
            print(f"\n  BrainSpan validation:")
            for set_name, cov in validation.get("gene_set_coverage", {}).items():
                status = "OK" if cov["coverage_pct"] >= 80 else "LOW"
                print(f"    {set_name}: {cov['found']}/{cov['total']} ({cov['coverage_pct']}%) [{status}]")
                if cov["missing_genes"]:
                    print(f"      Missing: {', '.join(cov['missing_genes'])}")


if __name__ == "__main__":
    main()
