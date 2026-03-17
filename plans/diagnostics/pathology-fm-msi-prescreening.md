# Pathology Foundation Model MSI Pre-screening

## Objective

Benchmark pathology foundation models for predicting microsatellite instability (MSI) status from routine H&E histopathology slides across multiple cancer types, enabling low-cost pre-screening where molecular testing is unavailable.

## Background

MSI status is a critical biomarker: it determines eligibility for immune checkpoint inhibitors and predicts response to WRN inhibitors (relevant to cancer division's wrn-msi-pancancer-atlas). Current gold-standard MSI testing (PCR, IHC, NGS) requires molecular pathology infrastructure costing $300-500 per test — unavailable in most low- and middle-income settings.

H&E-stained slides are universally produced for every surgical pathology case. Recent benchmarks show pathology foundation models can predict MSI from H&E alone:
- **CONCH** achieves 77.5-77.8% balanced accuracy for MSI in CRC using 2-cluster aggregation
- **UNI 2** (Mahmood Lab, 200M+ training images, open-access) consistently top-performs on biomarker prediction
- Most studies evaluate only CRC; cross-cancer MSI prediction is underexplored

A reliable H&E-based MSI classifier could serve as a low-cost triage tool: screen all cases computationally, confirm positives with molecular testing. This could dramatically expand MSI testing coverage globally.

## Data Sources

| Dataset | Cancer Types | Slides | MSI Labels | Access |
|---------|-------------|--------|------------|--------|
| TCGA-DIAGNOSTIC | Pan-cancer (33 types) | ~11,000 diagnostic WSIs | MSI status from MANTIS/MSIsensor via GDC | Open (GDC portal) |
| TCGA MSI-relevant types | COAD, READ, UCEC, STAD | ~2,500 WSIs with high MSI prevalence | Gold-standard PCR/IHC | Open |
| CPTAC | COAD, UCEC, LSCC, LUAD | ~1,200 WSIs | MSI via MSIsensor2 | Open (CPTAC portal) |

**Primary cancer types:** Colorectal (COAD/READ), endometrial (UCEC), gastric (STAD) — highest MSI prevalence. Secondary: ovarian, hepatocellular, urothelial.

## Methodology

### Phase 1: Data Preparation
1. Download TCGA diagnostic WSIs for MSI-relevant cancer types via GDC API
2. Curate MSI labels from TCGA molecular subtypes (MANTIS scores, MSI-PCR calls)
3. Tile WSIs at 20x magnification (256x256 patches), filter background/artifact tiles
4. Stratified train/validation/test splits (70/15/15), stratified by cancer type and MSI status

### Phase 2: Feature Extraction
1. Extract patch-level features using frozen UNI 2 (ViT-Large, open-access)
2. Extract features from CONCH (visual-language FM) for comparison
3. Baseline: ResNet-50 pretrained on ImageNet (non-pathology control)

### Phase 3: Slide-Level Classification
1. Multiple instance learning (MIL) aggregation: attention-based pooling (ABMIL), CLAM, TransMIL
2. Train per-cancer-type classifiers and pan-cancer classifier
3. Nested cross-validation (outer 5-fold, inner 3-fold) for unbiased performance estimation

### Phase 4: Cross-Cancer Generalization
1. Train on high-prevalence types (CRC, UCEC, STAD), evaluate on held-out cancer types
2. Assess whether MSI morphological signature transfers across tissue origins
3. Domain adaptation experiments if direct transfer underperforms

### Phase 5: Validation & Clinical Utility
1. External validation on CPTAC cohort (completely held-out)
2. Calibration analysis (ECE, reliability diagrams)
3. Operating point analysis: set specificity to minimize unnecessary molecular testing
4. Cost-effectiveness modeling: H&E pre-screening + confirmatory molecular testing vs. universal molecular testing

## Expected Outputs

- Per-cancer-type MSI classification AUROC, AUPRC, sensitivity/specificity
- Cross-cancer generalization matrix
- Feature importance and morphological pattern analysis (attention heatmaps)
- Cost-effectiveness comparison vs. universal molecular testing
- Recommended operating thresholds for clinical deployment scenarios

## Success Criteria

- AUROC >= 0.85 for MSI prediction in at least 3 cancer types on held-out test sets
- Meaningful cross-cancer generalization (AUROC >= 0.75 when training excludes the target cancer type)
- External validation on CPTAC within 5% of internal performance
- Demonstrated cost reduction in pre-screening scenario modeling

## Labels

ai-diagnostic, imaging, novel-finding, high-priority, screening, cost-reduction, accessibility
