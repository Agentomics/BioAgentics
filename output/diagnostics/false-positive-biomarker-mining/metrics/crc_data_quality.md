# CRC False Positive Extraction — Data Quality Assessment

**Date:** 2026-03-22
**Source:** output/diagnostics/crc-liquid-biopsy-panel/stage_stratified_results.parquet

## Data Summary

- **Total samples:** 353 (308 positive, 45 negative)
- **Source data type:** Stage-stratified confusion matrix counts (not per-sample predictions)
- **Stages:** Normal (n=45), Stage I (51), Stage II (121), Stage III (100), Stage IV (36)
- **All negatives are "Normal" stage** — no stage-mixed negatives

## Critical Limitation: Synthetic Score Reconstruction

The crc-liquid-biopsy-panel pipeline stores **aggregate confusion matrix counts per stage**, not per-sample prediction scores. The CrcAdapter reconstructs individual sample scores using:

- True positives: `Beta(5, 2)` — high scores
- False negatives: `Beta(2, 5)` — low scores
- True negatives: `Beta(2, 5)` — low scores
- False positives: `Beta(5, 2)` — high scores

**The original data shows 0 FPs and 0 FNs** (100% sensitivity at 100% specificity per stage). The FPs identified in this extraction are **artifacts of the synthetic score generation** — they arise because some Beta(2,5)-drawn TN scores happen to fall above the computed threshold, not because the original model actually made these errors.

## Extraction Results

| Operating Point | Threshold | Specificity | Sensitivity | FPs | TNs |
|----------------|-----------|-------------|-------------|-----|-----|
| 90% spec | 0.4824 | 88.9% | 88.0% | 5 | 40 |
| 95% spec | 0.4952 | 93.3% | 87.3% | 3 | 42 |
| 99% spec | 0.5441 | 97.8% | 81.5% | 1 | 44 |

All FPs originate from the "Normal" stage group (the only negatives in the dataset).

## Impact on Analysis

1. **FP characterization (Phase 1):** Can proceed with profiling/clustering using reconstructed data to validate pipeline mechanics, but findings are about synthetic score distributions, not real biological signal.
2. **Feature analysis:** The CRC adapter currently lacks methylation/protein feature columns in reconstructed samples — only `stage` and `stage_numeric` are present as non-score features.
3. **Biomarker discovery:** Not possible with this data. Real per-sample predictions with feature vectors are required.

## Recommendations

- **For pipeline validation:** These synthetic results are adequate to exercise the extract → profile → cluster pipeline end-to-end.
- **For real analysis:** The crc-liquid-biopsy-panel pipeline must be modified to output per-sample prediction scores and feature vectors (not just stage-level aggregates).
- **Create developer task:** Request per-sample prediction export from crc-liquid-biopsy-panel classifier.
