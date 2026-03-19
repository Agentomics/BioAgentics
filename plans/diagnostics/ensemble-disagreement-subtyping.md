# Ensemble Disagreement Subtyping — Mining Model Conflict for Clinical Subtypes

## Objective
Systematically analyze *patterns of disagreement* among diagnostic AI models trained on the same patient cohort to discover clinically meaningful patient subgroups (disease subtypes, atypical presentations, comorbidity signatures) that are invisible to any single model or to standard ensemble averaging.

**Falsifiable hypothesis:** In the ehr-sepsis-early-warning multi-model predictions (LR, XGBoost, LightGBM), patients where models produce contradictory predictions cluster into >= 2 subgroups with statistically different clinical outcomes (mortality, ICU length of stay, treatment response), and these disagreement-defined subgroups correspond to clinically recognizable phenotypes (e.g., hyperinflammatory vs. immunosuppressed sepsis).

## Rationale

### Cross-Project Basis
- **ehr-sepsis-early-warning** (diagnostics, development, Phase 4 complete): Already trains LR + XGBoost + LightGBM + ensemble on MIMIC-IV sepsis cohort with temporal validation, external validation on eICU, and SHAP explainability. The multi-model architecture produces per-patient prediction vectors that naturally encode disagreement. SHAP analysis already reveals which features drive predictions — we can extend this to ask which features drive *disagreement*.
- **acoustic-multi-disease-panel** (diagnostics, development, Phase 1 complete): Multi-task model with disease-specific classification heads will produce multi-disease risk vectors. Patients with "high PD + high MCI" vs. "high PD only" represent disagreement between disease hypotheses — potentially indicating neurological subtypes with different progression trajectories.
- **false-positive-biomarker-mining** (plan exists, project lost in DB reset): Proposed mining false positives for pre-clinical signals. Our initiative extends this: false positives where models *agree* may be random noise, but false positives where models *disagree* about why they're positive may encode genuine clinical heterogeneity.

### Unconventional Reasoning
1. **Inverted assumption about ensembles:** Standard ML practice averages ensemble predictions to cancel out noise. This treats model disagreement as pure error. But in clinical settings, disagreement between diagnostic approaches is informative — it's why multidisciplinary tumor boards exist. When a pathologist and a radiologist disagree about a diagnosis, the case goes to conference. When LR and XGBoost disagree, we just average.
2. **Disagreement as difficulty signal:** Ensemble diversity theory shows that models disagree most on "hard" cases. In diagnostics, "hard" cases are atypical presentations, borderline states, or patients with confounding comorbidities. These are exactly the patients who benefit most from subtype-aware management.
3. **The sepsis subtyping parallel:** Seymour et al. (2019) identified 4 sepsis phenotypes (alpha, beta, gamma, delta) with different mortality rates and treatment responses using unsupervised clustering on clinical features. We hypothesize that model disagreement patterns can recover these phenotypes — and potentially discover new ones — without requiring the manual feature engineering that Seymour used.
4. **Transferable method:** If disagreement-based subtyping works for sepsis, the method transfers directly to every other division project that uses ensemble models. This is a division-wide capability, not a single-disease tool.

## Data Sources

| Dataset | Models Available | Disease Domain | Key Feature |
|---------|-----------------|----------------|-------------|
| MIMIC-IV (ehr-sepsis) | LR, XGBoost, LightGBM (Phase 4) | Sepsis | Multi-model predictions + SHAP values + outcomes |
| eICU (ehr-sepsis) | External validation models | Sepsis | Cross-institution disagreement patterns |
| MIMIC-IV Clinical | Full EHR context | Multiple | Longitudinal outcomes for disagreement cohorts |
| Seymour 2019 phenotypes | Reference clustering | Sepsis | Benchmark comparison for subtype discovery |

## Methodology

### Phase 1: Disagreement Characterization
1. Extract per-patient prediction vectors from ehr-sepsis Phase 4 models: P(sepsis) from LR, XGBoost, LightGBM.
2. Compute disagreement metrics per patient: (a) prediction variance, (b) maximum pairwise prediction difference, (c) binary disagreement (models split on diagnosis).
3. Characterize the "disagreement cohort": patients where max pairwise difference > 0.3 (at least one model says >0.65, another says <0.35).
4. Profile: demographics, SOFA scores, infection source, comorbidity burden, feature distributions.

### Phase 2: Disagreement Clustering
1. For each patient in the disagreement cohort, construct a "disagreement fingerprint": the full vector of [LR_pred, XGB_pred, LGBM_pred, LR_SHAP_top10, XGB_SHAP_top10, LGBM_SHAP_top10].
2. Cluster using: (a) k-means with gap statistic for optimal k, (b) hierarchical clustering with Ward's method, (c) UMAP visualization.
3. Validate cluster stability via bootstrap resampling (Jaccard similarity > 0.7 across 100 bootstrap iterations).
4. Characterize each cluster: which models agree, which disagree, which SHAP features drive the split.

### Phase 3: Clinical Outcome Validation
1. For each disagreement cluster, compare clinical outcomes: in-hospital mortality, 30-day mortality, ICU LOS, vasopressor duration, ventilator days, antibiotic escalation.
2. Statistical testing: Kaplan-Meier survival curves + log-rank tests; Cox proportional hazards adjusting for SOFA, age, comorbidity.
3. Treatment response heterogeneity: do clusters respond differently to early vs. late antibiotics, fluid resuscitation volume, vasopressor choice?
4. Benchmark against Seymour 2019 phenotypes: do our disagreement-based clusters align with alpha/beta/gamma/delta phenotypes, or discover orthogonal subgroups?

### Phase 4: External Validation on eICU
1. Apply disagreement fingerprinting to eICU external validation set.
2. Test whether the same cluster structure reproduces across institutions.
3. If it does: the subtypes are robust. If not: disagreement patterns are model-specific artifacts.

### Phase 5: Disagreement-Aware Routing
1. Build a meta-classifier that routes patients to disease-subtype-appropriate models based on initial disagreement pattern.
2. Test whether routing improves net clinical benefit (net reclassification improvement, NRI) vs. standard ensemble averaging.
3. Decision curve analysis: at what treatment threshold does disagreement-aware routing outperform?

## Success Criteria
- Identify >= 2 clinically distinct patient subgroups defined by model disagreement patterns
- Disagreement-based subgroups have statistically different 30-day mortality (hazard ratio > 1.5, p < 0.05)
- At least 1 subgroup aligns with a known sepsis phenotype (Seymour alpha/beta/gamma/delta)
- Disagreement-aware routing improves NRI >= 0.05 compared to standard ensemble
- Clusters replicate on eICU external data (Jaccard stability > 0.6)

## Risk Assessment
- **Most likely failure mode:** Model disagreement in the sepsis models is random — driven by noise, not clinical heterogeneity. The LR/XGBoost/LightGBM models may be too similar in their decision boundaries to produce meaningful disagreement patterns. Mitigation: if within-architecture disagreement is too low, test with architecturally diverse models (e.g., add a neural network).
- **Small disagreement cohort:** If models agree on >90% of patients, the disagreement cohort may be too small for clustering. Mitigation: use continuous disagreement metrics rather than binary cutoffs.
- **Confounding:** Disagreement clusters may simply recapitulate SOFA severity. Mitigation: adjust for severity in outcome analysis; test whether disagreement adds information beyond SOFA.
- **What we'd learn even if it fails:** Whether diagnostic AI ensemble disagreement is clinically random or structured. If random, this validates simple ensemble averaging as optimal. If structured but not clinically meaningful, this identifies model architecture issues. Either way, it advances the field's understanding of when to trust ensembles vs. investigate further.

## Labels
catalyst, novel-finding, high-priority, sepsis, ensemble-methods, clinical-subtyping
