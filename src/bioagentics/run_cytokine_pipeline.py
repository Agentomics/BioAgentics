"""Orchestrator for the cytokine network flare prediction pipeline.

Runs the full pipeline end-to-end:
1. Load extracted cytokine data
2. Run random-effects meta-analysis
3. Build cytokine interaction network with meta-analysis overlay
4. Classify immune axes
5. Train flare prediction models
6. Analyze treatment responses
7. Generate all output artifacts

Usage::

    uv run python -m bioagentics.run_cytokine_pipeline
    uv run python -m bioagentics.run_cytokine_pipeline --data path/to/data.csv
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bioagentics.config import REPO_ROOT

logger = logging.getLogger("bioagentics.cytokine_pipeline")

DATA_DIR = REPO_ROOT / "data" / "pandas_pans" / "cytokine-network-flare-prediction"
OUTPUT_DIR = REPO_ROOT / "output" / "pandas_pans" / "cytokine-network-flare-prediction"


def run_pipeline(data_path: Path | None = None, output_dir: Path | None = None) -> dict:
    """Run the full cytokine network flare prediction pipeline.

    Parameters
    ----------
    data_path : Path to extracted cytokine CSV. If None, uses default location.
    output_dir : Directory for output artifacts. If None, uses default.

    Returns
    -------
    dict with summary of all pipeline outputs.
    """
    from bioagentics.cytokine_axis_classifier import (
        classify_axes,
        classification_to_dict,
        radar_plot,
    )
    from bioagentics.cytokine_extraction import CytokineDataset
    from bioagentics.cytokine_flare_predictor import FlarePredictor, results_summary
    from bioagentics.cytokine_meta_analysis import (
        results_to_dataframe,
        run_meta_analysis,
    )
    from bioagentics.cytokine_network import (
        build_network,
        export_network,
        overlay_meta_results,
        visualize_network,
    )
    from bioagentics.cytokine_treatment_response import TreatmentResponseAnalyzer

    if output_dir is None:
        output_dir = OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    if data_path is None:
        data_path = DATA_DIR / "extracted_cytokines.csv"

    summary: dict = {"pipeline": "cytokine-network-flare-prediction", "outputs": {}}

    # -----------------------------------------------------------------------
    # Step 1: Load data
    # -----------------------------------------------------------------------
    logger.info("Step 1: Loading extracted cytokine data from %s", data_path)
    dataset = CytokineDataset.from_csv(data_path)
    ds_summary = dataset.summary()
    logger.info("Loaded: %d records, %d studies, %d analytes", ds_summary["n_records"], ds_summary["n_studies"], ds_summary["n_analytes"])
    summary["dataset"] = ds_summary

    # -----------------------------------------------------------------------
    # Step 2: Meta-analysis
    # -----------------------------------------------------------------------
    logger.info("Step 2: Running random-effects meta-analysis")
    meta_results = run_meta_analysis(dataset, min_studies=3, output_dir=output_dir)
    meta_df = results_to_dataframe(meta_results)
    meta_csv = output_dir / "meta_analysis_summary.csv"
    meta_df.to_csv(meta_csv, index=False)
    logger.info("Meta-analysis: %d analytes with sufficient data", len(meta_results))
    summary["outputs"]["meta_analysis_csv"] = str(meta_csv)
    summary["meta_analysis"] = {
        "n_analytes_analyzed": len(meta_results),
        "significant_analytes": meta_df[meta_df["significant"]]["analyte"].tolist() if len(meta_df) > 0 else [],
    }

    # -----------------------------------------------------------------------
    # Step 3: Network construction + overlay
    # -----------------------------------------------------------------------
    logger.info("Step 3: Building cytokine interaction network")
    G = build_network()
    overlay_meta_results(G, meta_results)
    net_img = visualize_network(G, output_path=output_dir / "cytokine_network.png")
    net_json = export_network(G, output_path=output_dir / "cytokine_network.json")
    summary["outputs"]["network_image"] = str(net_img)
    summary["outputs"]["network_json"] = str(net_json)
    summary["network"] = {"n_nodes": G.number_of_nodes(), "n_edges": G.number_of_edges()}

    # -----------------------------------------------------------------------
    # Step 4: Immune axis classification
    # -----------------------------------------------------------------------
    logger.info("Step 4: Classifying immune axes")
    classification = classify_axes(meta_results)
    radar_path = radar_plot(classification, output_path=output_dir / "immune_axis_radar.png")
    axis_dict = classification_to_dict(classification)
    axis_json = output_dir / "axis_classification.json"
    with open(axis_json, "w") as f:
        json.dump(axis_dict, f, indent=2)
    summary["outputs"]["axis_radar"] = str(radar_path)
    summary["outputs"]["axis_json"] = str(axis_json)
    summary["axis_classification"] = {
        "dominant_axis": classification.dominant_axis,
        "dominant_score": classification.dominant_score,
    }

    # -----------------------------------------------------------------------
    # Step 5: Flare prediction
    # -----------------------------------------------------------------------
    logger.info("Step 5: Training flare prediction models")
    predictor = FlarePredictor(dataset)
    pred_results = predictor.run()
    if pred_results:
        roc_path = predictor.plot_roc(pred_results, output_path=output_dir / "flare_roc_curves.png")
        pred_df = results_summary(pred_results)
        pred_csv = output_dir / "prediction_summary.csv"
        pred_df.to_csv(pred_csv, index=False)
        summary["outputs"]["roc_curves"] = str(roc_path)
        summary["outputs"]["prediction_csv"] = str(pred_csv)

        # Feature importance for best model
        best = max(pred_results, key=lambda r: r.auc_score)
        fi_path = predictor.plot_feature_importance(best, output_path=output_dir / f"feature_importance_{best.model_name}.png")
        if fi_path:
            summary["outputs"]["feature_importance"] = str(fi_path)

        summary["prediction"] = {
            "models": [{
                "name": r.model_name, "auc": r.auc_score,
                "sensitivity": r.sensitivity, "specificity": r.specificity,
            } for r in pred_results],
        }
    else:
        logger.warning("Insufficient data for flare prediction")
        summary["prediction"] = {"models": [], "note": "insufficient data"}

    # -----------------------------------------------------------------------
    # Step 6: Treatment response
    # -----------------------------------------------------------------------
    logger.info("Step 6: Analyzing treatment responses")
    analyzer = TreatmentResponseAnalyzer(dataset)
    tx_results = analyzer.analyze_all()
    if tx_results:
        heatmap_path = analyzer.plot_heatmap(tx_results, output_path=output_dir / "treatment_response_heatmap.png")
        tx_df = analyzer.results_to_dataframe(tx_results)
        tx_csv = output_dir / "treatment_response_summary.csv"
        tx_df.to_csv(tx_csv, index=False)
        summary["outputs"]["treatment_heatmap"] = str(heatmap_path)
        summary["outputs"]["treatment_csv"] = str(tx_csv)
        summary["treatment_response"] = {
            "treatments_analyzed": [r.treatment for r in tx_results],
            "responder_cytokines": {r.treatment: r.responder_cytokines for r in tx_results},
        }
    else:
        logger.warning("No treatment response data available")
        summary["treatment_response"] = {"note": "no treatment data"}

    # -----------------------------------------------------------------------
    # Step 7: Save pipeline summary
    # -----------------------------------------------------------------------
    summary_path = output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info("Pipeline complete. Summary saved to %s", summary_path)

    return summary


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="Run cytokine network flare prediction pipeline")
    parser.add_argument("--data", type=Path, default=None, help="Path to extracted cytokine CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    try:
        summary = run_pipeline(data_path=args.data, output_dir=args.output_dir)
        print(f"\nPipeline complete: {summary['dataset']['n_records']} records processed")
        print(f"Outputs in: {args.output_dir or OUTPUT_DIR}")
    except FileNotFoundError as e:
        logger.error("Data file not found: %s", e)
        logger.info("Create extracted data CSV first, or specify --data path")
        sys.exit(1)


if __name__ == "__main__":
    main()
