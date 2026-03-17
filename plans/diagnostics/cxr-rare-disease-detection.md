# Chest X-Ray Long-Tail Rare Disease Detection

## Objective

Develop a classification system for detecting rare and underrepresented thoracic conditions in chest X-rays, addressing the diagnostic accuracy gap where infrequent conditions are systematically missed by both radiologists and standard AI models.

## Background

Rare thoracic conditions — unusual pneumonia patterns, rare interstitial lung diseases, uncommon masses, foreign bodies, rare congenital anomalies — are frequently missed on chest X-rays because radiologists encounter them infrequently and AI models trained on imbalanced data learn to ignore minority classes. This contributes to diagnostic delays and errors.

The CXR-LT 2026 challenge (ISBI 2026) provides 300,000+ multi-institutional chest X-rays from MIMIC and MIDRC with long-tail label distributions, noisy supervision, and cross-center domain shift — precisely the conditions that cause standard classifiers to fail on rare findings.

Standard deep learning classifiers achieve strong performance on common findings (atelectasis, cardiomegaly, pleural effusion) but degrade severely on tail classes. Recent work in long-tail recognition (decoupled training, class-balanced losses, prototype learning) has shown promise but has not been systematically evaluated on large-scale medical imaging benchmarks.

## Data Sources

| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| CXR-LT 2026 (MIMIC-CXR component) | ~230,000 CXRs | Multi-label, 26+ findings, long-tail distribution | PhysioNet (credentialed) |
| CXR-LT 2026 (MIDRC component) | ~70,000 CXRs | Multi-institutional, distribution shift source | MIDRC portal |
| MultiCaRe | 130,791 images, 93,816 cases | 140+ category taxonomy from PubMed case reports | Open access |
| NIH ChestX-ray14 | 112,120 CXRs | 14 findings, NLP-extracted labels | Open access |
| CheXpert | 224,316 CXRs | 14 findings with uncertainty labels | Stanford (application) |

**Key characteristic:** CXR-LT 2026 specifically provides rare finding annotations and multi-institutional data, creating a realistic benchmark for long-tail performance.

## Methodology

### Phase 1: Data Acquisition & Analysis
1. Obtain CXR-LT 2026 challenge data (MIMIC-CXR + MIDRC components)
2. Download MultiCaRe dataset for supplementary rare cases
3. Profile label distribution: identify head, body, and tail classes
4. Analyze inter-institutional distribution shift between MIMIC and MIDRC

### Phase 2: Baseline Models
1. Train standard DenseNet-121 and ResNet-50 with binary cross-entropy (standard approach)
2. Document per-class performance to quantify the head-tail performance gap
3. Establish baseline metrics: per-class AUROC, macro-averaged AUROC, tail-class mean AUROC

### Phase 3: Long-Tail Classification Methods
1. **Class-balanced losses:** Focal loss, class-balanced focal loss (effective number weighting), asymmetric loss
2. **Decoupled training:** Standard representation learning (stage 1) → re-balanced classifier fine-tuning with cRT or tau-normalization (stage 2)
3. **Prototype-based:** Class prototype learning with cosine classifier heads
4. **Data augmentation for tail classes:** CutMix, mosaic augmentation, synthetic oversampling of rare findings
5. **Foundation model features:** Extract features from BiomedCLIP or CXR-FM, train balanced classifiers on frozen features

### Phase 4: Cross-Institutional Robustness
1. Train on MIMIC, evaluate on MIDRC (and vice versa) to assess domain shift impact
2. Domain adaptation: adversarial training, batch normalization alignment
3. Calibration under distribution shift: temperature scaling per-institution

### Phase 5: Evaluation & Error Analysis
1. Per-class performance stratified by head/body/tail frequency bins
2. Improvement analysis: which methods help which tail classes most
3. Confusion analysis: which rare conditions are confused with common ones
4. Calibration analysis: reliability under class imbalance
5. Comparison to radiologist performance on rare findings (from published literature)

## Expected Outputs

- Systematic benchmark of long-tail classification methods on CXR-LT 2026
- Per-class AUROC curves with head/body/tail stratification
- Cross-institutional generalization analysis
- Method recommendation matrix (which techniques help which types of rare findings)
- Attention/saliency maps for rare finding detection
- Practical guidelines for deploying rare disease detection in clinical CXR AI

## Success Criteria

- Tail-class mean AUROC improvement >= 10 percentage points over standard training baseline
- No more than 2% degradation in head-class performance (no accuracy sacrifice for common findings)
- Cross-institutional performance within 5% of intra-institutional (MIMIC→MIDRC transfer)
- At least 3 individual rare conditions achieving AUROC >= 0.80

## Labels

ai-diagnostic, imaging, rare-disease, screening, novel-finding, accessibility
