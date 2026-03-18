# EHR-Based Sepsis Early Warning System

## Objective
Build a machine learning system for early sepsis detection (≥6 hours before clinical onset) using routinely available EHR vitals and labs, with rigorous calibration and demographic fairness analysis.

## Background
Sepsis affects ~1.7M adults annually in the US with ~270K deaths. Early recognition and treatment (each hour of delay increases mortality ~4%) is critical, but current clinical screening tools (SIRS, qSOFA, NEWS) have poor sensitivity for early detection. ML-based early warning systems have shown promise (e.g., InSight, EASL) but adoption is hampered by poor calibration, alarm fatigue from high false-positive rates, and limited validation of fairness across demographic subgroups. The FDA-cleared Epic Sepsis Model was found to perform poorly in external validation (AUROC 0.63 in Emory study), highlighting the need for transparent, well-calibrated, and rigorously validated approaches.

## Data Sources
- **MIMIC-IV v3.1** (PhysioNet): ~50K ICU admissions, Beth Israel Deaconess Medical Center. Hourly vitals, labs, medications, ICD codes. Credentialed access via PhysioNet.
- **eICU Collaborative Research Database v2.0** (PhysioNet): ~200K ICU admissions across 208 US hospitals. Multi-center validation set.
- **Sepsis labels**: Sepsis-3 criteria (suspected infection + SOFA ≥2). Time of onset defined as earlier of (antibiotics order, culture order) + SOFA criteria met.

## Methodology

### Phase 1 — Data Preparation
- Extract hourly feature vectors: 6 vitals (HR, MAP, SBP, RR, SpO2, temp), 20 labs (WBC, lactate, creatinine, bilirubin, platelets, BUN, glucose, etc.), demographics (age, sex, ethnicity)
- Compute derived features: delta values (1h, 6h, 12h), rolling statistics (mean, std, slope over 6h windows), missingness indicators
- Define prediction windows: predict sepsis onset at 4h, 6h, 8h, 12h lookahead
- Handle class imbalance: ~10-15% sepsis prevalence in ICU; use stratified sampling

### Phase 2 — Model Development
- **Baseline**: Logistic regression with L1 regularization on hand-crafted features
- **Gradient boosting**: XGBoost/LightGBM on tabular features
- **Temporal model**: LSTM/GRU on sequential hourly observations (24h lookback window)
- **Ensemble**: Stacking of temporal + tabular models
- Nested cross-validation (5-fold outer, 3-fold inner) on MIMIC-IV

### Phase 3 — Calibration & Fairness
- Platt scaling and isotonic regression for probability calibration
- Calibration curves and expected calibration error (ECE) by subgroup
- Fairness analysis: AUROC, sensitivity, specificity, PPV stratified by age group, sex, race/ethnicity
- Alarm burden analysis: false positive rate at clinically relevant sensitivity thresholds (80%, 90%, 95%)

### Phase 4 — External Validation & Deployment Readiness
- Temporal validation on MIMIC-IV holdout (most recent 20% by admission date)
- External validation on eICU (multi-center generalization)
- SHAP explanations for top features driving each prediction
- Feature importance stability analysis across centers
- Operating point recommendations for different clinical settings (ICU vs ward vs ED)

## Expected Outputs
- Calibrated sepsis prediction model with AUROC ≥0.85 at 6h lookahead
- Demographic fairness report (AUROC within 0.05 across subgroups)
- Feature importance analysis with SHAP
- Alarm burden analysis at multiple operating points
- External validation results on eICU

## Success Criteria
- AUROC ≥0.85 on MIMIC-IV temporal holdout at 6h lookahead
- AUROC ≥0.80 on eICU external validation
- ECE <0.05 after calibration
- No subgroup AUROC disparity >0.05
- False positive rate <2:1 at 80% sensitivity

## Labels
screening, ai-diagnostic, accessibility, cost-reduction, high-priority
