# UK Biobank Radiomics Methods → FP Mining Feature Extraction Guide

**Date:** 2026-03-22
**Task:** 1185
**Reference:** Seletkov et al. 2025, "AI-driven preclinical disease risk assessment using imaging in UK Biobank," npj Digital Medicine 8:480. [PMC12297167](https://pmc.ncbi.nlm.nih.gov/articles/PMC12297167/)
**Code:** [github.com/yayapa/ukbb_risk](https://github.com/yayapa/ukbb_risk) (paths/configs removed for privacy)

## 1. Core Radiomics Feature Families (Seletkov et al.)

The study extracted PyRadiomics features from whole-body MRI (neck-to-knee T1-weighted Dixon, 69 organ segmentations via TotalVibeSegmentator) across 7 families:

| Family | What It Captures | PyRadiomics Module |
|--------|-----------------|-------------------|
| First-order statistics | Intensity distribution (mean, std, skewness, kurtosis, entropy) | `firstorder` |
| Gray Level Co-occurrence Matrix (GLCM) | Spatial texture patterns (contrast, correlation, homogeneity) | `glcm` |
| Gray Level Run Length Matrix (GLRLM) | Consecutive voxel runs of same intensity | `glrlm` |
| Gray Level Size Zone Matrix (GLSZM) | Connected regions of same intensity | `glszm` |
| Neighboring Gray-Tone Difference Matrix (NGTDM) | Intensity differences with neighbors | `ngtdm` |
| Gray Level Dependence Matrix (GLDM) | Voxel dependencies at specified distance | `gldm` |
| Shape-based features | Volume, surface area, sphericity, elongation | `shape` |

**Feature selection:** mRMR (minimum Redundancy Maximum Relevance), 20 features per anatomical category.
**Normalization:** Shape features normalized by height²; all features standardized (fit on training set).
**Preclinical window:** 3-year risk prediction — events counted only if first occurrence is between 3 months and 3 years after imaging.

**Key result for FP mining:** Radiomics features from whole-body MRI outperformed raw imaging (ResNet18 3D) for preclinical risk prediction across cardiovascular, metabolic, inflammatory, and oncologic conditions. The interpretable tabular features were more useful than end-to-end deep learning.

## 2. Adaptation to FP Mining Feature Domains

### 2a. CRC Methylation/Protein Features

Radiomics captures spatial heterogeneity in images. For CRC liquid biopsy, the analogous concept is **molecular heterogeneity** across biomarker panels:

| Radiomics Concept | CRC Adaptation | Feature Name Pattern |
|-------------------|---------------|---------------------|
| First-order statistics | Distribution of cfDNA methylation deltas across CpG sites | `meth_mean`, `meth_std`, `meth_skew`, `meth_kurtosis`, `meth_entropy` |
| Texture (GLCM-like) | Co-occurrence of elevated markers — which biomarkers co-elevate | `coelev_CEA_CA199`, `marker_correlation_matrix` |
| Shape (volume/area) | Panel-level aggregates — how many markers are elevated, by how much | `n_elevated_markers`, `total_marker_burden`, `max_single_marker` |
| Zone/run features | Consecutive methylation changes along chromosomal regions | `meth_run_length_chr17`, `meth_zone_size` |
| First-order on proteins | Protein panel statistics | `prot_mean`, `prot_std`, `prot_range`, `prot_entropy` |

**Key features to extract from CRC classifier:**
1. Per-sample cfDNA methylation delta vector (top 50 CpG sites by combined_score)
2. Per-sample protein biomarker values (MMP9, CXCL8, S100A4, CEA, CA19-9, TIMP1, IL6)
3. Panel-level summary statistics (mean, std, skewness of the above)
4. Classifier internal representations (if using ML — hidden layer activations, attention weights)

### 2b. MIMIC Sepsis Signals

For temporal ICU data, the analogy is **temporal texture** — patterns in how vital signs and lab values evolve:

| Radiomics Concept | Sepsis Adaptation | Feature Name Pattern |
|-------------------|------------------|---------------------|
| First-order statistics | Distribution of vital sign values over ICU stay | `hr_mean`, `hr_std`, `hr_skew`, `temp_entropy` |
| GLCM (texture) | Temporal co-occurrence — which vitals deteriorate together | `codeteriorate_hr_bp`, `vital_correlation` |
| GLRLM (runs) | Duration of abnormal episodes (runs of elevated heart rate) | `hr_elevated_run_max`, `temp_fever_duration_hrs` |
| Shape (trajectory) | Trajectory shape features — slope, curvature, inflection points | `hr_trend_slope`, `bp_trajectory_curvature` |
| NGTDM (differences) | Rate of change, volatility between consecutive measurements | `hr_delta_mean`, `bp_volatility`, `score_acceleration` |

**Key features already validated (COMPOSER pilot):**
- `mean_score`, `std_score`: First-order statistics of risk scores (d=2.30, d=1.21)
- `early_score`, `late_score`: Temporal phase features (d=1.37, d=1.95)
- `score_trend`: Trajectory feature (d=0.36, not significant — may need refinement)
- `n_alerts`: Conformal prediction confidence proxy (d=-0.52)
- `n_timepoints`: Stay duration proxy (d=0.57)

**Additional features to extract from ehr-sepsis-early-warning:**
1. Vital sign distributions per patient (HR, BP, SpO2, temperature, respiratory rate)
2. Lab value distributions (WBC, lactate, creatinine, bilirubin)
3. Temporal texture: run lengths of abnormal values, volatility metrics
4. SOFA/qSOFA component trajectories
5. Model intermediate features (hidden states, attention weights if transformer-based)

## 3. Developer Checklist: Intermediate Tensors/Metadata to Log

### For CRC Liquid Biopsy Pipeline

```
Required per sample:
  ├── sample_id                     # Unique identifier
  ├── y_true                        # Ground truth label
  ├── y_score                       # Classifier prediction probability
  ├── fold_id                       # Cross-validation fold (if applicable)
  ├── stage                         # Clinical stage (if available)
  ├── meth_deltas[50]               # Top 50 CpG methylation deltas
  ├── protein_values[7]             # Protein biomarker panel values
  ├── panel_summary                 # {mean, std, skew, kurtosis, entropy, n_elevated}
  └── model_intermediates (if ML):
      ├── hidden_layer_activations  # Last hidden layer output vector
      ├── feature_importances       # Per-feature SHAP or permutation importances
      └── prediction_uncertainty    # If ensemble: prediction variance across models

Output format: Parquet
Output path: output/diagnostics/crc-liquid-biopsy-panel/per_sample_predictions.parquet
```

### For EHR Sepsis Pipeline (MIMIC-IV)

```
Required per patient-admission:
  ├── subject_id, hadm_id           # MIMIC-IV identifiers
  ├── y_true                        # Sepsis label (binary)
  ├── y_score                       # Peak/mean prediction score
  ├── sepsis_onset_hour             # Hours from admission to sepsis onset (if positive)
  ├── vital_summaries               # Per-vital: {mean, std, min, max, skew, entropy}
  ├── lab_summaries                 # Per-lab: {mean, std, min, max, trend}
  ├── temporal_features:
  │   ├── early_score               # Mean score in first third of stay
  │   ├── late_score                # Mean score in last third of stay
  │   ├── score_trend               # Linear regression slope of scores
  │   ├── score_volatility          # Std of score deltas between timepoints
  │   ├── max_score_timing          # Fraction of stay at peak score
  │   ├── n_alert_episodes          # Number of times score exceeded threshold
  │   └── longest_alert_run         # Longest consecutive alert period (hours)
  ├── sofa_trajectory               # SOFA scores at each assessment point
  └── model_intermediates (optional):
      ├── attention_weights         # If transformer: attention over time steps
      └── hidden_states             # Last layer hidden state vector

Output format: Parquet
Output path: output/diagnostics/ehr-sepsis-early-warning/per_patient_predictions.parquet
```

## 4. Recommendations for Phases 2-3 Outputs

### Metrics to Add

1. **Preclinical detection lead time:** Following Seletkov's 3-year window concept, define domain-specific windows:
   - Sepsis: 24h, 48h, 72h before clinical onset
   - CRC: 6-month, 12-month, 24-month before clinical diagnosis (if longitudinal data available)

2. **Feature importance ranking for FP subgroups:** After clustering FPs, run SHAP or permutation importance on each cluster to identify which features drive the FP prediction. Compare to TP feature importances — are FPs driven by the same features at lower intensity, or by qualitatively different features?

3. **Radiomics-inspired heterogeneity metrics:**
   - **Marker entropy:** Shannon entropy across the biomarker panel (high entropy = many markers slightly elevated; low entropy = one marker dominantly elevated)
   - **Co-elevation matrix:** Binary co-occurrence of elevated markers (analogous to GLCM)
   - **Panel burden score:** Sum of normalized marker elevations (analogous to radiomics total volume)

4. **Calibration metrics for FP subgroups:** After identifying FP clusters, assess whether re-calibrating model confidence within each cluster improves the "time-adjusted sensitivity" metric from Phase 5.

### Reusable Code

The `ukbb_risk` repository structure suggests three separable modules:
- `PrepareDataset`: Data ingestion — not directly reusable (UK Biobank-specific)
- `TwoDImage`: ResNet3D training — not applicable to tabular data
- `analysisNumericFeatures`: **Most relevant** — tabular feature processing, mRMR selection, model training with Optuna hyperparameter tuning. The mRMR feature selection and Optuna-based model search could be adapted for FP signature extraction.

**Practical limitation:** Config files and paths are removed from the public repo. Feature extraction logic would need to be reimplemented using PyRadiomics directly for any imaging extensions (e.g., MIMIC-IV-CXR).

### Priority Order for Developer Implementation

1. **[Blocking]** CRC per-sample prediction export (Task 1217) — enables real FP profiling
2. **[Blocking]** EHR sepsis per-patient feature export — enables MIMIC-IV longitudinal analysis
3. **[Enhancement]** Panel-level summary statistics (entropy, co-elevation matrix)
4. **[Enhancement]** Model intermediate logging (hidden states, attention, SHAP)
5. **[Future]** MIMIC-IV-CXR radiomics extraction using PyRadiomics + TotalSegmentator (Phase 4)
