#!/usr/bin/env python3
"""Ensemble classifier model (gradient boosting + logistic regression).

Trains 3-class GOF/neutral/LOF ensemble with nested cross-validation.
Computes per-class AUC and feature importance analysis.

Usage:
    uv run python -m crohns.nod2_variant_functional_impact.07_model
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from bioagentics.data.nod2.classifier import train_and_save

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("data/crohns/nod2-variant-functional-impact")
OUTPUT_DIR = Path("output/crohns/nod2-variant-functional-impact")


def main():
    """Run ensemble classifier training pipeline."""
    logger.info("=== Ensemble Classifier Pipeline ===")

    results = train_and_save(data_dir=DATA_DIR, output_dir=OUTPUT_DIR)

    # Report
    cv = results["cv_results"]
    logger.info("Macro AUC (OvR): %.3f", cv["macro_auc"])
    logger.info("Classification report:")
    report = cv["classification_report"]
    for cls in ["GOF", "neutral", "LOF"]:
        if cls in report:
            r = report[cls]
            logger.info("  %s: precision=%.3f recall=%.3f f1=%.3f",
                        cls, r["precision"], r["recall"], r["f1-score"])

    logger.info("Top 5 features:")
    for _, row in results["feature_importance"].head(5).iterrows():
        logger.info("  %s: %.4f", row["feature"], row["importance"])

    # Check AUC target
    if cv["macro_auc"] < 0.90:
        logger.warning(
            "AUC %.3f below 0.90 target. Likely due to class imbalance "
            "(small GOF/LOF sets). Consider augmenting training data.",
            cv["macro_auc"],
        )

    return results


if __name__ == "__main__":
    try:
        main()
    except Exception:
        logger.exception("Pipeline failed")
        sys.exit(1)
