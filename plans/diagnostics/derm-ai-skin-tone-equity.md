# Equitable Dermatology AI: Skin Lesion Classification Across Diverse Skin Tones

## Objective
Develop and benchmark skin lesion classification models that perform equitably across Fitzpatrick skin types I-VI, closing the diagnostic accuracy gap between light and dark skin tones that plagues current dermatology AI systems.

## Background
Skin cancer is the most common cancer globally, and AI-assisted dermoscopy/clinical image classification has shown promise for early detection. However, existing models exhibit dramatic performance disparities:
- A 2022 study (Daneshjou et al., Science Advances) showed dermatology AI models dropped 20-30% in accuracy on dark skin vs. light skin when evaluated on the DDI dataset
- Most training datasets (ISIC, HAM10000) are 80-95% Fitzpatrick types I-III, severely underrepresenting darker skin tones
- The FDA has begun scrutinizing algorithmic bias in medical AI, making equitable performance a regulatory requirement

This gap matters because melanoma on dark skin is diagnosed at later stages (thicker Breslow depth), has worse prognosis, and is more often acral (palms, soles, nails) — a distribution poorly represented in standard dermoscopy datasets. An AI system that works well only on light skin could actively worsen disparities by giving false reassurance to clinicians screening dark-skinned patients.

## Data Sources

| Dataset | Size | Description | Skin Tone Diversity | Access |
|---------|------|-------------|-------------------|--------|
| DDI (Diverse Dermatology Images) | 656 images, 570 patients | Histopathologically confirmed, dermatologist-assigned Fitzpatrick types | Balanced across FST I-VI (by design) | Open access |
| Fitzpatrick17k | 16,577 images | 114 skin conditions, FST I-VI labels | Moderate diversity, crowd-labeled FST | Open access |
| ISIC Archive | 100,000+ dermoscopy images | Multi-year challenge data, diagnostic labels | Predominantly FST I-III | Open access |
| SLICE-3D (ISIC 2024) | 400,000 crop images | 3D TBP-derived, 7 global centers | Multi-center but skin tone unlabeled | CC-BY 4.0 |
| HAM10000 | 10,015 dermoscopy images | 7 diagnostic categories, histopathology-confirmed | Predominantly light skin | Open access |
| PAD-UFES-20 | 2,298 clinical images | 6 categories, smartphone-captured, Brazilian patients | Greater diversity (Brazilian population) | Open access |

**Key insight:** DDI is small but gold-standard for fairness evaluation (expertly labeled FST, pathology-confirmed). Fitzpatrick17k provides scale with FST labels. ISIC/SLICE-3D provide volume. The strategy is to train on large datasets and rigorously evaluate fairness on DDI.

## Methodology

### Phase 1: Data Preparation & Skin Tone Annotation
1. Download DDI, Fitzpatrick17k, ISIC Archive, HAM10000, PAD-UFES-20 datasets.
2. Harmonize diagnostic labels into unified taxonomy: malignant melanoma, basal cell carcinoma, squamous cell carcinoma, actinic keratosis, benign nevus, seborrheic keratosis, dermatofibroma, vascular lesion, other.
3. Profile skin tone distribution per dataset. Identify underrepresented FST-condition combinations.
4. Create stratified train/val/test splits ensuring DDI is reserved as a held-out fairness evaluation set.

### Phase 2: Baseline Models
1. Train standard classifiers: EfficientNet-B2, ConvNeXt-Tiny, ViT-Small on combined training data.
2. Report per-class accuracy stratified by Fitzpatrick skin type.
3. Quantify the "fairness gap": difference in AUC/sensitivity between FST I-II and FST V-VI.
4. Establish baseline metrics on DDI held-out set.

### Phase 3: Fairness-Aware Training
1. **Resampling:** Oversample underrepresented FST-condition combinations.
2. **Reweighting:** Class-balanced and skin-tone-balanced loss functions.
3. **Domain adaptation:** Treat skin tone as a domain; adversarial domain adaptation to learn tone-invariant features.
4. **Contrastive learning:** FairDisCo-style disentanglement contrastive learning to separate diagnostic features from skin tone features.
5. **Foundation model transfer:** Fine-tune BiomedCLIP and DINOv2 on combined data, evaluate fairness of transferred representations.

### Phase 4: Acral Lesion Focus
1. Filter datasets for acral lesions (palms, soles, nails) — a distribution disproportionately affecting dark-skinned patients.
2. Train specialized acral lesion classifier.
3. Evaluate whether general models vs. specialized acral models perform better for this critical subtype.

### Phase 5: Evaluation & Calibration
1. Per-FST performance on DDI: AUC, sensitivity, specificity, positive predictive value.
2. Equalized odds analysis: equal true positive and false positive rates across skin tones.
3. Calibration per FST: reliability diagrams showing whether confidence estimates are trustworthy across skin tones.
4. Error analysis: which conditions on which skin tones cause the most misclassifications and why.
5. Comparison to published dermatologist performance stratified by skin tone.

## Expected Outputs
- Systematic benchmark of fairness-aware training methods for dermatology AI
- Per-FST performance matrices for all model variants
- Fairness gap analysis: which methods reduce disparity most effectively
- Acral lesion specialist model with cross-skin-tone evaluation
- Calibration analysis across skin tones
- Practical guidelines for building equitable dermatology AI systems

## Success Criteria
- Fairness gap (AUC difference between FST I-II and FST V-VI) reduced to < 5% on DDI (from ~20-30% baseline)
- Overall malignant lesion sensitivity >= 85% across ALL skin types on DDI
- At least one fairness-aware method achieves equitable performance without degrading light-skin accuracy by > 2%
- Acral lesion AUC >= 0.80 across all skin tones
- Calibration ECE < 0.10 for all FST groups

## Labels
ai-diagnostic, imaging, accessibility, screening, novel-finding, high-priority
