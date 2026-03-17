"""Orchestrate the full PANS pathway analysis pipeline end-to-end.

Runs all analysis modules in sequence:
  1. Load curated gene data
  2. Pathway enrichment (ORA via Enrichr)
  3. GEO GSE102482 download and DE analysis
  4. Neuroinflammation cross-reference
  5. PPI network analysis (STRING)
  6. mTOR convergence analysis
  7. Collate summary

Usage:
    uv run python -m bioagentics.models.run_pans_pathway_analysis [--dest DIR]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("output/pandas_pans/pans-genetic-variant-pathway-analysis")
DATA_DIR = Path("data/pandas_pans/pans-genetic-variant-pathway-analysis")


def run_pipeline(output_dir: Path | None = None,
                 data_dir: Path | None = None,
                 skip_download: bool = False) -> dict:
    """Run the full PANS pathway analysis pipeline.

    Args:
        output_dir: Directory for analysis outputs.
        data_dir: Directory for downloaded/processed data.
        skip_download: Skip GEO download (use cached data).

    Returns:
        Summary dict with results from all pipeline steps.
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR
    if data_dir is None:
        data_dir = DATA_DIR

    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    summary: dict = {
        "pipeline": "pans-genetic-variant-pathway-analysis",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "steps": {},
    }

    # Step 1: Load gene data
    logger.info("=" * 60)
    logger.info("Step 1: Loading PANS variant gene data")
    logger.info("=" * 60)
    from bioagentics.data.pans_variants import get_pans_gene_symbols, get_pans_variant_genes

    gene_df = get_pans_variant_genes()
    gene_symbols = get_pans_gene_symbols()
    summary["steps"]["gene_data"] = {
        "status": "done",
        "gene_count": len(gene_symbols),
        "pathway_axes": gene_df["pathway_axis"].nunique(),
        "genes": gene_symbols,
    }
    logger.info("  Loaded %d genes across %d pathway axes",
                len(gene_symbols), gene_df["pathway_axis"].nunique())

    # Step 2: Pathway enrichment
    logger.info("=" * 60)
    logger.info("Step 2: Running pathway enrichment analysis")
    logger.info("=" * 60)
    try:
        from bioagentics.models.pans_pathway_enrichment import run_pans_pathway_enrichment

        enrichment_df = run_pans_pathway_enrichment(dest_dir=output_dir)
        n_sig = 0
        if not enrichment_df.empty and "adj_p_value" in enrichment_df.columns:
            n_sig = int((enrichment_df["adj_p_value"] < 0.05).sum())
        summary["steps"]["pathway_enrichment"] = {
            "status": "done",
            "total_terms": len(enrichment_df),
            "significant_terms": n_sig,
        }
        logger.info("  %d terms, %d significant (FDR<0.05)", len(enrichment_df), n_sig)
    except Exception as e:
        logger.error("  Pathway enrichment failed: %s", e)
        summary["steps"]["pathway_enrichment"] = {"status": "error", "error": str(e)}

    # Step 3: GEO GSE102482 download and DE
    logger.info("=" * 60)
    logger.info("Step 3: GEO GSE102482 differential expression")
    logger.info("=" * 60)
    de_df = None
    try:
        from bioagentics.data.pans_geo_expression import get_neuroinflammation_de

        de_df = get_neuroinflammation_de(dest_dir=data_dir, force=not skip_download)
        n_de_sig = int((de_df["padj"] < 0.05).sum()) if not de_df.empty else 0
        summary["steps"]["geo_expression"] = {
            "status": "done",
            "total_genes": len(de_df),
            "significant_de": n_de_sig,
        }
        logger.info("  %d genes, %d significant DE", len(de_df), n_de_sig)
    except Exception as e:
        logger.error("  GEO expression failed: %s", e)
        summary["steps"]["geo_expression"] = {"status": "error", "error": str(e)}

    # Step 4: Neuroinflammation cross-reference
    logger.info("=" * 60)
    logger.info("Step 4: Neuroinflammation cross-reference")
    logger.info("=" * 60)
    try:
        from bioagentics.models.pans_neuroinflammation_xref import run_neuroinflammation_xref

        xref_df = run_neuroinflammation_xref(dest_dir=output_dir)
        status_counts = xref_df["de_status"].value_counts().to_dict()
        summary["steps"]["neuroinflammation_xref"] = {
            "status": "done",
            "gene_count": len(xref_df),
            "de_status_counts": status_counts,
        }
        logger.info("  Cross-reference: %s", status_counts)
    except Exception as e:
        logger.error("  Neuroinflammation xref failed: %s", e)
        summary["steps"]["neuroinflammation_xref"] = {"status": "error", "error": str(e)}

    # Step 5: PPI network analysis
    logger.info("=" * 60)
    logger.info("Step 5: PPI network analysis (STRING)")
    logger.info("=" * 60)
    try:
        from bioagentics.models.pans_ppi_network import run_pans_ppi_analysis

        G, metrics_df = run_pans_ppi_analysis(dest_dir=output_dir)
        hub_genes = []
        if not metrics_df.empty:
            hub_genes = metrics_df.head(5)["gene_symbol"].tolist()
        summary["steps"]["ppi_network"] = {
            "status": "done",
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
            "hub_genes": hub_genes,
        }
        logger.info("  Network: %d nodes, %d edges, hubs: %s",
                    G.number_of_nodes(), G.number_of_edges(), hub_genes)
    except Exception as e:
        logger.error("  PPI network failed: %s", e)
        summary["steps"]["ppi_network"] = {"status": "error", "error": str(e)}

    # Step 6: mTOR convergence analysis
    logger.info("=" * 60)
    logger.info("Step 6: mTOR convergence analysis")
    logger.info("=" * 60)
    try:
        from bioagentics.models.pans_mtor_convergence import run_mtor_convergence

        mtor_results = run_mtor_convergence(dest_dir=output_dir)
        direct_count = mtor_results["direct_overlap"]["direct_overlap_count"]
        ppi_count = len(mtor_results["ppi_connections"].get("ppi_mtor_connections", []))
        summary["steps"]["mtor_convergence"] = {
            "status": "done",
            "direct_overlap": direct_count,
            "ppi_connections": ppi_count,
        }
        logger.info("  Direct mTOR overlap: %d, PPI connections: %d",
                    direct_count, ppi_count)
    except Exception as e:
        logger.error("  mTOR convergence failed: %s", e)
        summary["steps"]["mtor_convergence"] = {"status": "error", "error": str(e)}

    # Step 7: Save summary
    logger.info("=" * 60)
    logger.info("Step 7: Saving analysis summary")
    logger.info("=" * 60)

    completed = sum(1 for s in summary["steps"].values() if s.get("status") == "done")
    errored = sum(1 for s in summary["steps"].values() if s.get("status") == "error")
    summary["completed_steps"] = completed
    summary["errored_steps"] = errored
    summary["total_steps"] = len(summary["steps"])

    summary_path = output_dir / "analysis_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Saved summary: %s", summary_path)

    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run full PANS pathway analysis pipeline"
    )
    parser.add_argument("--dest", type=Path, default=OUTPUT_DIR,
                        help="Output directory (default: %(default)s)")
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR,
                        help="Data directory (default: %(default)s)")
    parser.add_argument("--skip-download", action="store_true",
                        help="Skip GEO download, use cached data")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")

    summary = run_pipeline(output_dir=args.dest,
                           data_dir=args.data_dir,
                           skip_download=args.skip_download)

    print(f"\n{'=' * 60}")
    print("PANS Pathway Analysis — Pipeline Complete")
    print(f"{'=' * 60}")
    print(f"Steps completed: {summary['completed_steps']}/{summary['total_steps']}")
    if summary["errored_steps"] > 0:
        print(f"Steps with errors: {summary['errored_steps']}")
        for name, step in summary["steps"].items():
            if step.get("status") == "error":
                print(f"  {name}: {step.get('error', 'unknown')}")
        sys.exit(1)
    else:
        print("All steps completed successfully.")


if __name__ == "__main__":
    main()
