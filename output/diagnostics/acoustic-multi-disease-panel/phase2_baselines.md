# Phase 2 Baseline Results — Acoustic Multi-Disease Panel

**Date:** 2026-03-29
**Analyst:** analyst
**Division:** diagnostics
**Project:** acoustic-multi-disease-panel

## Executive Summary

Both single-disease baselines failed to achieve meaningful classification performance. Neither PD severity classification nor respiratory condition detection exceeded chance-level AUROC with gradient boosting on handcrafted acoustic features. The MCI baseline was deferred due to restricted data access. **No condition met the 0.75 AUROC threshold.**

## Per-Disease Results

### Parkinson's Disease — UCI PD Telemonitoring

| Metric | Value |
|--------|-------|
| **Task** | Severity classification (mild vs moderate/severe, UPDRS >= 27.6) |
| **AUROC (LOSO-CV)** | **0.369** (below chance) |
| **Mean per-subject AUROC** | 0.439 ± 0.039 |
| **Subjects evaluated** | 14/42 (28 had single-class labels) |
| **Status** | FAILED |

**Critical limitation:** Dataset has NO healthy controls — all 42 subjects have PD. True PD detection was impossible. Severity classification across subjects fails because between-subject voice variability dominates the severity signal.

**Features:** 16 (Jitter variants, Shimmer variants, NHR, HNR, RPDE, DFA, PPE)
**Top features:** DFA (0.136), Shimmer:APQ11 (0.124), HNR (0.112)

### Respiratory Conditions — COUGHVID

| Metric | Value |
|--------|-------|
| **Task** | Healthy vs symptomatic/COVID-19 cough detection |
| **AUROC (5-fold CV)** | **0.524 ± 0.036** (chance level) |
| **Pooled AUROC** | 0.523 |
| **Sensitivity/Specificity** | 0.405 / 0.650 (at optimal threshold) |
| **Status** | FAILED |

**Key issue:** COUGHVID uses self-reported labels (not clinically confirmed). Label noise overwhelms the acoustic signal for handcrafted features.

**Features:** 216 (cough segmentation, MFCC, spectral, chroma, deltas)
**Top features:** No dominant discriminator (max importance 0.015)

### Mild Cognitive Impairment — ADReSS/DementiaBank

| Metric | Value |
|--------|-------|
| **Status** | DEFERRED |

ADReSS/DementiaBank data requires restricted institutional access not currently available.

## Literature Benchmarks

| Condition | Literature AUROC | Our AUROC | Gap |
|-----------|-----------------|-----------|-----|
| PD detection (voice) | 0.85-0.95 (Tsanas et al., PMID: 22101335) | N/A (no controls) | — |
| TB cough detection | 0.85-0.99 (Pahar et al., PMID: 34073456) | N/A (wrong dataset) | — |
| COVID cough (COUGHVID) | 0.59-0.97 (varies widely) | 0.524 | Consistent with weak end |

## Root Causes of Failure

1. **PD: Wrong dataset for the task.** UCI Telemonitoring is a regression dataset (predict UPDRS) with no healthy controls. Literature PD voice detection results use datasets with PD + healthy subjects.

2. **Respiratory: Noisy labels + wrong method.** COUGHVID's crowd-sourced labels are unreliable. Published results achieving high AUROC on COUGHVID typically use deep learning on spectrograms, not handcrafted features. Even those results have been questioned for methodological issues.

3. **No shared feature space.** PD features (jitter, shimmer, HNR) and respiratory features (cough segmentation, MFCCs) are fundamentally different acoustic domains. The premise of shared features across voice and cough recordings needs stronger single-disease baselines first.

## Recommendations for Phase 3 Readiness

1. **PD detection:** Obtain UCI Parkinson's Dataset (Little et al., 2009; 31 PD + 17 healthy) or mPower data with healthy controls. Task created for data_curator (#1589).

2. **Respiratory detection:** Use clinically validated dataset (Zambia TB study or controlled COPD recordings) instead of crowd-sourced COUGHVID. Alternatively, try mel-spectrogram CNN approach.

3. **Feature importance analysis (task #1503):** Blocked until at least one condition achieves AUROC >= 0.70.

4. **Multi-disease panel (Phase 3):** Cannot proceed until at least 2 single-disease baselines achieve AUROC >= 0.75. Current evidence does not support the multi-task approach with these data sources and methods.

## Output Files

- `pd_baseline/pd_severity_results.json` — PD severity classification results
- `pd_baseline/pd_loso_cv_results.json` — PD LOSO-CV results
- `respiratory_baseline/respiratory_baseline_cv_results.json` — Respiratory 5-fold CV results
- `respiratory_baseline/respiratory_baseline_results.json` — Initial respiratory results
