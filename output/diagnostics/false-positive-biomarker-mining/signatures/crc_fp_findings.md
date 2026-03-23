# CRC False Positive Findings — Phase 1 Pipeline Validation

**Date:** 2026-03-22
**Project:** false-positive-biomarker-mining
**Data source:** Reconstructed from crc-liquid-biopsy-panel stage-stratified results

## Summary

Pipeline ran end-to-end on CRC data reconstructed from aggregate confusion matrices. Extraction succeeded at all three operating points, but profiling and clustering are severely limited by the absence of per-sample biomarker features. **No meaningful biomarker candidates can be identified from this data** — the reconstructed predictions contain only `stage_numeric` as a feature, and all negatives (including FPs) are Normal stage, making feature comparison trivial (no variance).

## Extraction Results

| Operating Point | Threshold | Specificity | Sensitivity | FPs | TNs | FP Rate |
|----------------|-----------|-------------|-------------|-----|-----|---------|
| 90% target | 0.4824 | 88.9% | 88.0% | 5 | 40 | 11.1% |
| 95% target | 0.4952 | 93.3% | 87.3% | 3 | 42 | 6.7% |
| 99% target | 0.5441 | 97.8% | 81.5% | 1 | 44 | 2.2% |

## Confidence Score Analysis

FP confidence scores cluster narrowly above the threshold (mean 0.51 at 90% spec, 0.56 at 99% spec), while TNs have a much wider and lower distribution (mean 0.24–0.27). This pattern is expected from synthetic Beta(2,5) scores — FPs are the tail of the negative distribution that crossed the threshold.

## Feature Comparison

- **Features available:** 1 (stage_numeric)
- **Features with |Cohen's d| > 0.5:** 0
- **Features with p < 0.05:** 0
- **Clustering:** Skipped — all FPs have identical feature values

## Profiling/Clustering Blocked By

1. **No biomarker feature columns** — CrcAdapter reconstructs scores from stage-level counts but does not propagate cfDNA methylation or protein biomarker values to individual samples.
2. **All negatives are Normal stage** — no stage heterogeneity among FPs or TNs.
3. **Synthetic scores** — predictions drawn from Beta distributions, not from a real classifier.

## Recommendations

1. **Developer task (blocking):** Modify crc-liquid-biopsy-panel to export per-sample prediction scores and feature vectors (methylation deltas, protein levels) rather than stage-level aggregates. This is required for any real FP biomarker analysis.
2. **UK Biobank radiomics:** Feature extraction guidance (Task 1185) should prioritize defining what intermediate representations the CRC pipeline must save.
3. **Proceed with COMPOSER demo:** The sepsis demo data may offer more feature richness for pipeline validation while CRC data gaps are addressed.
