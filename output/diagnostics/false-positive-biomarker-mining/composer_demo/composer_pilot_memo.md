# COMPOSER Demo — Sepsis FP Pilot Analysis Memo

**Date:** 2026-03-22
**Project:** false-positive-biomarker-mining
**Data:** NematiLab COMPOSER sepsis prediction demo (500 ICU stays)

## Data Summary

- **Patients:** 500 ICU stays (62 septic, 438 non-septic)
- **Prevalence:** 12.4% sepsis
- **Feature extraction:** Per-patient aggregation of hourly COMPOSER risk scores
  - max score (used as patient-level y_score), mean, std, early/late phase means, score trend, peak timing, conformal alert count, stay length

## Extraction Results (95% Specificity Target)

| Metric | Value |
|--------|-------|
| Threshold | 0.8935 |
| Actual specificity | 95.0% |
| Sensitivity | 46.8% |
| False positives | 22 |
| True negatives | 416 |
| True positives | 29 |
| False negatives | 33 |

## Profiling: FP vs TN Feature Comparison

**6 of 8 features significantly different (p < 0.05), all with |Cohen's d| > 0.5:**

| Feature | FP Mean | TN Mean | Cohen's d | p-value | Interpretation |
|---------|---------|---------|-----------|---------|----------------|
| mean_score | 0.639 | 0.287 | **2.30** | <0.001 | FPs have persistently elevated risk throughout stay |
| late_score | 0.655 | 0.275 | **1.95** | <0.001 | FPs maintain high risk in late ICU phase |
| early_score | 0.585 | 0.301 | **1.37** | <0.001 | FPs already elevated early in stay |
| std_score | 0.198 | 0.114 | **1.21** | <0.001 | FPs have more volatile risk trajectories |
| n_alerts | 5.9 | 9.5 | **-0.52** | 0.0005 | FPs have *fewer* conformal rejections (model more confident) |
| n_timepoints | 48.1 | 34.9 | **0.57** | 0.023 | FPs have longer ICU stays |
| score_trend | — | — | 0.36 | 0.20 | Not significant |
| peak_fraction | — | — | 0.20 | 0.36 | Not significant |

### Key Observations

1. **FPs are not random noise.** All score-based features show massive effect sizes (d > 1.0). These patients have consistently elevated sepsis risk scores throughout their entire ICU stay — not just a single spike.

2. **Model confidence is *higher* for FPs than TNs.** Fewer conformal rejections mean the model is more confident these patients are at risk. This directly supports the pre-clinical signal hypothesis: these may be patients with genuine infection/inflammation that doesn't meet clinical sepsis criteria.

3. **Longer stays.** FP patients spend ~38% longer in the ICU (48 vs 35 timepoints), suggesting these are sicker patients even if they don't receive a sepsis diagnosis.

4. **Early and persistent signal.** Both early_score and late_score are elevated, meaning the model detects the risk signal from admission, not just before a late deterioration.

## Clustering

- **K-Means (k=2):** Silhouette = 0.30 (moderate structure). FPs split into two subgroups.
- **DBSCAN:** 1 cluster, no noise points — all FPs form a cohesive group.

Interpretation: FPs show some internal structure (K-Means finds 2 subgroups) but are fundamentally similar (DBSCAN treats them as one cluster). This suggests FPs are a distinct population from TNs rather than random scatter.

## Longitudinal Analysis: Blocked

The COMPOSER demo contains only **single ICU stays** — no multi-admission longitudinal data is available. The longitudinal module (`longitudinal.py`) requires:

- `admissions` DataFrame with `subject_id, hadm_id, admittime`
- `labels` DataFrame with `subject_id, hadm_id, sepsis_label`
- Multiple admissions per patient to track whether FP patients later develop sepsis

**This is the critical gap.** Without longitudinal follow-up, we can characterize FPs but cannot validate the core hypothesis (FPs develop sepsis at >= 2x baseline rate).

## Requirements from ehr-sepsis-early-warning Project

To proceed to MIMIC-IV-based longitudinal analysis, we need:

1. **`engineered_features.parquet`** — Per-patient/per-admission feature vectors from the sepsis prediction pipeline
2. **`sepsis_labels.parquet`** — Binary sepsis labels per patient and admission, with onset timing
3. **Ensemble predictions** — Per-patient prediction scores from the trained model
4. **MIMIC-IV admissions linkage** — `subject_id` and `hadm_id` mapping so we can look up multiple admissions per patient in the MIMIC-IV `admissions` table
5. **MIMIC-IV `admissions` table extract** — At minimum: subject_id, hadm_id, admittime for all patients in the cohort (not just those with predictions)

## Conclusion

The COMPOSER demo pilot successfully validates the FP mining pipeline on real sepsis prediction data. The finding that FPs show **large, consistent, and statistically significant** differences from TNs across multiple features is promising. The pre-clinical signal hypothesis is plausible for sepsis — these FP patients show a distinctive pattern of persistently elevated risk, higher model confidence, and longer ICU stays.

**Next steps:**
1. Obtain MIMIC-IV multi-admission data for longitudinal validation
2. Run hazard ratio analysis on FP vs TN patients' future sepsis incidence
3. Extract discriminative feature signatures for the FP subgroups (Phase 3)
