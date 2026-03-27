#!/usr/bin/env python3
"""MIMIC-IV demo → FP mining pipeline.

Runs the MimicDemoSepsisAdapter end-to-end:
1. Load MIMIC-IV Clinical Database Demo v2.2
2. Extract per-stay vitals and lab features
3. Label stays with Sepsis-3 criteria
4. Score stays with a heuristic severity risk score
5. Export per_stay_predictions.parquet for downstream FP mining

Output: output/diagnostics/false-positive-biomarker-mining/mimic_demo/
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from bioagentics.diagnostics.fp_mining.adapters.sepsis_adapter import (
    MIMIC_DEMO_DIR,
    MIMIC_DEMO_OUTPUT_DIR,
    MimicDemoSepsisAdapter,
)
from bioagentics.diagnostics.fp_mining.extract import (
    extract_at_operating_points,
    save_extraction,
)

logger = logging.getLogger(__name__)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="MIMIC-IV demo FP mining pipeline")
    parser.add_argument(
        "--data-dir", type=Path, default=MIMIC_DEMO_DIR,
        help="Path to MIMIC-IV demo flat directory",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=MIMIC_DEMO_OUTPUT_DIR,
        help="Output directory for predictions and extractions",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    # Step 1: Load and score
    adapter = MimicDemoSepsisAdapter(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
    )
    predictions = adapter.load_predictions()

    print(f"\n=== MIMIC-IV Demo Predictions ===")
    print(f"Admissions: {len(predictions)}")
    print(f"Features: {[c for c in predictions.columns if c not in ('sample_id', 'y_true', 'y_score')]}")
    print(f"Sepsis positive: {int(predictions['y_true'].sum())}")
    print(f"Sepsis negative: {int((predictions['y_true'] == 0).sum())}")
    print(f"Score range: [{predictions['y_score'].min():.3f}, {predictions['y_score'].max():.3f}]")
    print(f"Score mean: {predictions['y_score'].mean():.3f}")

    # Step 2: Extract FPs at operating points
    results = extract_at_operating_points(adapter, specificities=[0.90, 0.95])
    for res in results:
        save_extraction(res, output_dir=args.output_dir)
        print(
            f"\n{res.operating_point.name}: "
            f"FP={res.summary['n_fp']}, TN={res.summary['n_tn']}, "
            f"TP={res.summary['n_tp']}, FN={res.summary['n_fn']}, "
            f"specificity={res.summary['specificity']:.3f}"
        )

    # Step 3: Verify longitudinal linkage
    per_stay = pd.read_parquet(args.output_dir / "per_stay_predictions.parquet")
    n_multi_adm = (
        per_stay.groupby("subject_id")["hadm_id"]
        .nunique()
        .gt(1)
        .sum()
    )
    print(f"\n=== Longitudinal Linkage ===")
    print(f"Unique subjects: {per_stay['subject_id'].nunique()}")
    print(f"Unique admissions: {per_stay['hadm_id'].nunique()}")
    print(f"Subjects with >1 admission: {n_multi_adm}")

    # Save summary metrics
    summary = {
        "n_stays": len(per_stay),
        "n_admissions": len(predictions),
        "n_subjects": int(per_stay["subject_id"].nunique()),
        "n_multi_admission_subjects": int(n_multi_adm),
        "n_sepsis": int(predictions["y_true"].sum()),
        "sepsis_rate": float(predictions["y_true"].mean()),
        "score_mean": float(predictions["y_score"].mean()),
        "score_std": float(predictions["y_score"].std()),
    }
    summary_path = args.output_dir / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == "__main__":
    main()
