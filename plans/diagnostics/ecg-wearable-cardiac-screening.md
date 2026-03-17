# ECG-Based Cardiac Abnormality Screening from Wearable Single-Lead Data

## Objective
Develop a lightweight model that detects clinically significant cardiac abnormalities from single-lead ECG recordings, enabling population-scale screening via consumer wearable devices.

## Background
Consumer wearables (Apple Watch, Samsung Galaxy Watch, KardiaMobile) now record single-lead ECGs used by hundreds of millions of people. These devices currently detect only atrial fibrillation. However, 12-lead ECGs can reveal structural heart disease (low ejection fraction, left ventricular hypertrophy, valvular disease) — conditions with high morbidity when diagnosed late but excellent prognosis when caught early.

Key gaps:
- Most deep learning ECG research uses 12-lead data from clinical settings — not directly transferable to single-lead wearable recordings
- 12-lead-to-single-lead translation models exist but haven't been systematically evaluated for structural disease screening
- No public benchmarks for wearable-grade ECG screening across diverse populations — the CODE dataset (Brazilian) and PTB-XL (German) offer geographic diversity but need harmonization
- Clinical-grade models are too large for on-device inference

A single-lead model that screens for structural cardiac abnormalities could turn billions of existing wearable ECGs into a population-scale early warning system, catching disease years before symptoms.

## Data Sources

| Dataset | Size | Description | Access |
|---------|------|-------------|--------|
| PTB-XL | 21,799 12-lead ECGs, 18,869 patients | 10-second recordings, 71 diagnostic labels, age/sex | PhysioNet (open) |
| PTB-XL+ | Same ECGs + derived features | 100+ features per ECG (intervals, morphology) | PhysioNet (open) |
| CODE-15% | 345,779 12-lead ECGs, 233,770 patients | Brazilian telehealth, 6 diagnostic classes | Zenodo (open) |
| CODE-II-open | 15,000 patients subset | 66 diagnostic classes, cardiologist-reviewed | Open access |
| CPSC2018 | 6,877 12-lead ECGs | Chinese hospital data, 9 rhythm classes | Open access |
| MIMIC-IV-ECG | ~800,000 ECGs | Linked to clinical outcomes (EF, echo, mortality) | PhysioNet (credentialed) |

**Critical advantage:** MIMIC-IV-ECG links to echocardiography reports, enabling ground-truth labels for structural disease (low EF, valvular pathology) that other ECG datasets lack.

## Methodology

### Phase 1: Data Preparation & Lead Simulation
1. Download and harmonize PTB-XL, CODE-15%, CPSC2018 datasets. Standardize sampling rate (500Hz), duration (10s), and label taxonomy.
2. Simulate single-lead (Lead I) from 12-lead recordings to create paired 12-lead/single-lead data. Also evaluate Lead II and augmented limb leads.
3. Add realistic wearable noise augmentation: motion artifact, baseline wander, electrode contact noise, muscle artifact at levels matching published wearable signal quality studies.
4. Curate structural disease labels from MIMIC-IV-ECG linked echocardiography: low EF (<40%), LVH, valvular disease, diastolic dysfunction.

### Phase 2: 12-Lead Baseline Models
1. Train standard multi-label classifiers (ResNet-1D, InceptionTime, XResNet) on 12-lead data across all datasets.
2. Establish per-condition performance ceilings: arrhythmias, conduction abnormalities, hypertrophy, ischemia.
3. Report dataset-specific and cross-dataset generalization.

### Phase 3: Single-Lead Models
1. **Direct training:** Train lightweight 1D-CNN and Transformer models on simulated single-lead data.
2. **Knowledge distillation:** Use 12-lead teacher to train single-lead student via soft label distillation.
3. **Lead reconstruction:** Train single-lead-to-12-lead reconstruction network, then apply 12-lead classifier.
4. Compare all three approaches across all diagnostic categories.

### Phase 4: Mobile Optimization
1. Quantize best model to INT8 via post-training quantization and quantization-aware training.
2. Convert to TFLite and CoreML formats.
3. Benchmark: inference latency, model size, memory footprint on mobile-class hardware.
4. Target: <100ms inference, <5MB model size, <50MB memory.

### Phase 5: Robustness & Equity
1. Evaluate performance across age groups, sex, and geographic origin (German PTB-XL vs. Brazilian CODE vs. Chinese CPSC).
2. Noise robustness: test across wearable-grade SNR levels (5-25 dB).
3. Subgroup fairness analysis: ensure no demographic group has disproportionately lower sensitivity.

## Expected Outputs
- Single-lead ECG classifier for 10+ cardiac conditions including structural abnormalities
- 12-lead vs. single-lead performance comparison showing what's preserved and what's lost
- Cross-population generalization analysis (Europe, South America, Asia)
- Mobile-optimized model in TFLite/CoreML format
- Noise robustness curves mapping SNR to diagnostic performance
- Clinical utility analysis: number needed to screen, false positive burden

## Success Criteria
- Single-lead AUROC within 5% of 12-lead for at least 5 major diagnostic categories
- Low EF detection (if MIMIC-IV-ECG labels available): AUROC >= 0.80 from single lead
- Cross-dataset AUROC drop < 8% (PTB-XL → CODE or reverse)
- Model size < 5MB, inference < 100ms on mobile hardware
- No significant performance disparity (>5% AUROC) across sex or age subgroups

## Labels
ai-diagnostic, screening, point-of-care, accessibility, cost-reduction, novel-finding
