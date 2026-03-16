"""Phase 2 end-to-end pipeline: dependency prediction for NSCLC subtypes.

Runs the full Phase 2 pipeline:
1. Feature matrix preparation (expression + CRISPR alignment)
2. ElasticNet model training and selection
3. TCGA patient dependency prediction
4. Subtype-specific dependency analysis
5. Positive control validation

Usage:
    uv run python -m bioagentics.models.run_phase2
    uv run python -m bioagentics.models.run_phase2 --skip-training --results-dir data/results
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

from bioagentics.data.gene_ids import load_tcga_expression_matrix
from bioagentics.data.nsclc_tcga import classify_patients

from bioagentics.models.dependency_model import (
    load_models,
    save_results,
    train_all_models,
)
from bioagentics.models.feature_prep import prepare_features
from bioagentics.models.subtype_analysis import (
    analyze_subtype_dependencies,
    save_subtype_results,
)
from bioagentics.models.tcga_prediction import (
    predict_tcga_dependencies,
    save_predictions,
)
from bioagentics.models.validation import (
    save_validation_report,
    validate_positive_controls,
)

REPO_ROOT = Path(__file__).resolve().parents[3]

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Phase 2 pipeline: NSCLC dependency prediction and analysis",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=REPO_ROOT / "data",
        help="Root data directory (default: data/)",
    )
    parser.add_argument(
        "--results-dir", type=Path, default=REPO_ROOT / "data" / "results",
        help="Results output directory (default: data/results/)",
    )
    parser.add_argument("--n-features", type=int, default=6000, help="Number of expression features")
    parser.add_argument("--min-r", type=float, default=0.3, help="Min CV r for model selection")
    parser.add_argument("--n-folds", type=int, default=5, help="CV folds for model training")
    parser.add_argument("--grouping", default="kl", choices=["kl", "kp", "separate"],
                        help="KPL grouping strategy")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training, load existing models")
    parser.add_argument("--skip-prediction", action="store_true",
                        help="Skip TCGA prediction, load existing predictions")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip subtype analysis")
    parser.add_argument("--skip-validation", action="store_true",
                        help="Skip positive control validation")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    depmap_dir = args.data_dir / "depmap" / "25q3"
    tcga_dir = args.data_dir / "tcga"
    results_dir = args.results_dir
    results_dir.mkdir(parents=True, exist_ok=True)

    summary = {}

    # Step 1: Feature matrix preparation
    logger.info("=== Step 1: Feature matrix preparation ===")
    feat = prepare_features(
        depmap_dir=depmap_dir,
        n_features=args.n_features,
        grouping=args.grouping,
    )
    logger.info("  %d cell lines, %d features, %d target genes",
                feat.n_lines, feat.n_features, feat.n_targets)
    summary["n_cell_lines"] = feat.n_lines
    summary["n_features"] = feat.n_features
    summary["n_target_genes"] = feat.n_targets

    # Steps 2-3: Model training and selection
    if args.skip_training:
        logger.info("=== Steps 2-3: Loading existing models ===")
        models = load_models(results_dir)
        # Load metrics if available
        metrics_path = results_dir / "gene_metrics.csv"
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path, index_col=0)
            predictable = metrics[metrics["cv_r"] > args.min_r].index.tolist()
        else:
            predictable = list(models.keys())
        logger.info("  Loaded %d models (%d predictable)", len(models), len(predictable))
    else:
        logger.info("=== Steps 2-3: Training elastic-net models ===")
        model_results = train_all_models(
            feat.X, feat.Y,
            n_folds=args.n_folds,
            min_r=args.min_r,
        )
        save_results(model_results, results_dir)
        models = model_results.models
        predictable = model_results.predictable_genes
        logger.info("  %d / %d genes predictable (CV r > %.2f)",
                    model_results.n_predictable, model_results.n_total, args.min_r)

    summary["n_predictable_genes"] = len(predictable)

    # Step 4: TCGA prediction
    if args.skip_prediction:
        logger.info("=== Step 4: Loading existing TCGA predictions ===")
        pred_path = results_dir / "tcga_predicted_dependencies.csv"
        dep_matrix = pd.read_csv(pred_path, index_col=0)
        logger.info("  Loaded predictions: %d patients x %d genes", *dep_matrix.shape)
    else:
        logger.info("=== Step 4: Predicting TCGA patient dependencies ===")
        # Load TCGA expression
        tcga_expr_dirs = []
        for ct in ["luad", "lusc"]:
            expr_dir = tcga_dir / ct / "expression"
            if expr_dir.exists():
                tcga_expr_dirs.append(expr_dir)

        if not tcga_expr_dirs:
            logger.error("No TCGA expression directories found in %s", tcga_dir)
            sys.exit(1)

        # Load and concatenate TCGA expression
        tcga_frames = []
        for d in tcga_expr_dirs:
            logger.info("  Loading TCGA expression from %s", d)
            tcga_frames.append(load_tcga_expression_matrix(d))
        tcga_expr = pd.concat(tcga_frames)

        # Log-transform if needed (DepMap is log2(TPM+1), TCGA TPM needs same)
        tcga_expr_log = tcga_expr.apply(lambda x: x.clip(lower=0) + 1).apply(lambda x: x.map(
            lambda v: __import__("math").log2(v)))

        dep_matrix = predict_tcga_dependencies(models, feat.feature_genes, tcga_expr_log)

        # Get patient subtypes
        patients = classify_patients(tcga_dir)
        save_predictions(dep_matrix, results_dir, patient_meta=patients)
        logger.info("  Predicted: %d patients x %d genes", *dep_matrix.shape)

    summary["n_patients_predicted"] = dep_matrix.shape[0]

    # Step 5: Subtype-specific analysis
    if args.skip_analysis:
        logger.info("=== Step 5: Skipped (--skip-analysis) ===")
    else:
        logger.info("=== Step 5: Subtype-specific dependency analysis ===")
        patients = classify_patients(tcga_dir)
        patient_subtypes = pd.Series(
            patients.set_index("patient_id")["molecular_subtype"],
        )

        kras_alleles = pd.Series(
            patients.set_index("patient_id")["KRAS_allele"],
        ) if "KRAS_allele" in patients.columns else None

        subtype_results = analyze_subtype_dependencies(
            dep_matrix, patient_subtypes, kras_alleles=kras_alleles,
        )
        save_subtype_results(subtype_results, results_dir / "subtype_dependencies")
        summary["n_significant_subtype_genes"] = subtype_results.n_significant
        logger.info("  %d significant genes (FDR < 0.05)", subtype_results.n_significant)

    # Step 6: Positive control validation
    if args.skip_validation:
        logger.info("=== Step 6: Skipped (--skip-validation) ===")
    else:
        logger.info("=== Step 6: Positive control validation ===")
        patients = classify_patients(tcga_dir)
        patients_indexed = patients.set_index("patient_id")
        patient_subtypes = patients_indexed["molecular_subtype"]

        # Build mutations DataFrame
        mut_cols = [c for c in patients_indexed.columns if c.endswith("_mutated")]
        mutations = patients_indexed[mut_cols]

        val_report = validate_positive_controls(dep_matrix, patient_subtypes, mutations)
        save_validation_report(val_report, results_dir)
        summary["validation_aucs"] = val_report.control_aucs
        summary["n_controls_passed"] = val_report.n_controls_passed
        logger.info("  %d / %d controls passed",
                    val_report.n_controls_passed, val_report.n_controls_tested)

    # Save summary
    summary_path = results_dir / "phase2_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("=== Pipeline complete. Summary saved to %s ===", summary_path)


if __name__ == "__main__":
    main()
