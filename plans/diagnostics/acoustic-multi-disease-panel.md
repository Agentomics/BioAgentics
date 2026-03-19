# Acoustic Multi-Disease Screening Panel

## Objective
Test whether a single 3-minute smartphone voice/cough recording can simultaneously screen for Parkinson's disease, respiratory conditions (TB, COPD), and mild cognitive impairment (MCI) using shared acoustic feature extraction — an "acoustic blood panel" yielding multi-disease risk scores from one free test.

**Falsifiable hypothesis:** A multi-task acoustic model trained on combined voice datasets achieves AUROC >= 0.80 for at least 2 out of 3 target conditions (PD, respiratory disease, MCI) from a single recording, without degrading single-disease performance by more than 3% AUROC compared to disease-specific models.

## Rationale

### Cross-Project Basis
- **voice-biomarkers-parkinsons** (diagnostics, development): Extracts jitter, shimmer, HNR, MFCCs, spectrograms from sustained vowels. Literature shows 91% accuracy on PD (small samples).
- **acoustic-tb-screening** (diagnostics, RD-approved, queued): Cough acoustics via speech foundation models. AUROC 85.2% on 512 participants.
- Both use phone microphones, spectral features, and audio foundation model fine-tuning — but as separate, siloed projects.

### Unconventional Reasoning
1. **Shared feature spaces:** Jitter/shimmer abnormalities appear in PD (vocal fold rigidity), COPD (airway obstruction), and MCI (reduced motor speech control). MFCCs carry disease signal across all three. Speech rate, pause patterns, and vocabulary complexity are MCI markers that are free byproducts of any voice recording.
2. **Multi-task signal amplification:** Disease-specific features may actually help discriminate other diseases. A model that knows about PD tremor patterns can better isolate non-PD vocal abnormalities, potentially improving respiratory and cognitive disease detection.
3. **The "one visit, many tests" paradigm:** Blood panels test for 20+ conditions from a single draw. No one has built the acoustic equivalent. A 3-minute recording protocol (30s sustained vowel, 30s cough, 60s counting, 60s reading passage) covers all task types needed for PD, respiratory, and cognitive screening.
4. **Cost: $0 marginal.** Aligns with division's accessibility mandate. Especially impactful in low-resource settings where aging populations face PD, TB, and dementia simultaneously.

## Data Sources

| Dataset | Size | Condition | Task Type | Access |
|---------|------|-----------|-----------|--------|
| mPower | 16K participants | Parkinson's | Sustained vowel, walking | Synapse (open) |
| UCI PD Telemonitoring | 5,875 recordings | Parkinson's | Sustained vowels | UCI ML Repo (open) |
| PC-GITA | 100 speakers | Parkinson's | Vowels, sentences, reading | Request-based |
| Zambia TB cough study | 512 participants | Tuberculosis | Cough recordings | Publication data |
| COUGHVID | 25,000+ | Respiratory | Crowdsourced cough | Open access |
| ADReSS Challenge 2020 | 156 speakers | Alzheimer's/MCI | Picture description | DementiaBank (request) |
| ADReSSo 2021 | 237 speakers | Alzheimer's/MCI | Picture description | DementiaBank (request) |
| Pitt DementiaBank | 500+ sessions | Dementia/MCI | Interviews | TalkBank (request) |

## Methodology

### Phase 1: Unified Feature Extraction Pipeline
1. Build shared acoustic feature extractor compatible with all recording types: sustained vowels, cough, connected speech, reading passages.
2. Features: classical (jitter, shimmer, HNR, F0, formants, speech rate, pause ratio) + spectral (MFCCs, mel-spectrograms, chromagram) + temporal (speaking rate, hesitation frequency, mean utterance length).
3. Audio foundation model embeddings: fine-tune Wav2Vec 2.0 or HuBERT on combined dataset for shared representations.
4. Standardize recording format: 16kHz mono WAV, noise-normalized, silence-trimmed.

### Phase 2: Single-Disease Baselines
1. Train disease-specific classifiers (gradient boosting + fine-tuned audio FM) for each condition independently.
2. Establish per-disease AUROC baselines on held-out test sets.
3. Feature importance analysis: which features are disease-specific vs. shared across conditions.

### Phase 3: Multi-Task Model
1. Multi-task learning architecture: shared encoder (audio FM backbone) + disease-specific classification heads.
2. Training strategies: (a) joint training, (b) sequential training with frozen shared layers, (c) gradient-balancing (GradNorm or similar).
3. Compare multi-task vs. single-task performance per disease.
4. Feature attribution: which acoustic features are shared discriminators vs. disease-specific.

### Phase 4: Recording Protocol Optimization
1. Ablation: which recording segments (vowel, cough, counting, reading) contribute most to each disease?
2. Minimum viable recording: what's the shortest protocol that maintains >= 0.80 AUROC per disease?
3. Noise robustness: test at wearable-grade SNR levels (5-25 dB).

### Phase 5: Clinical Utility & Cost Analysis
1. Multi-disease screening cost-effectiveness vs. individual screening programs.
2. Number-needed-to-screen analysis for combined panel.
3. Referral pathway modeling: which follow-up tests should positive screens trigger?
4. Demographic fairness: performance across age, sex, language/accent.

## Success Criteria
- Multi-task model achieves AUROC >= 0.80 for at least 2 of 3 conditions (PD, respiratory, MCI)
- Single-disease performance degradation < 3% AUROC compared to disease-specific baselines
- Shared feature analysis reveals >= 5 acoustic features with significant discriminative power for 2+ diseases
- Minimum viable recording protocol <= 3 minutes
- No significant performance disparity (> 5% AUROC) across sex or age subgroups

## Risk Assessment
- **Most likely failure mode:** Multi-task interference — PD vocal tremor features may conflict with MCI hesitation features, causing mutual performance degradation. Mitigation: gradient-balancing techniques and disease-specific fine-tuning heads.
- **Data heterogeneity:** Datasets use different recording conditions, microphones, and protocols. Mitigation: aggressive normalization and domain adaptation.
- **MCI detection from voice may be too hard:** ADReSS datasets are small (156-237 speakers). MCI voice changes may be too subtle for phone-grade audio. If MCI fails, the 2-of-3 success criterion still allows PD + respiratory to validate the concept.
- **What we'd learn even if it fails:** Which acoustic features are truly disease-specific vs. shared. Whether multi-task learning helps or hurts in acoustic diagnostics. These negative results prevent future groups from wasting resources on the same approach.

## Labels
catalyst, novel-finding, ai-diagnostic, point-of-care, accessibility, cost-reduction, screening
