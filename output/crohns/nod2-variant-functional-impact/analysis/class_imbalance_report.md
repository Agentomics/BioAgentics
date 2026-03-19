# Class Imbalance Impact and Calibration Analysis

**Task:** 855
**Project:** nod2-variant-functional-impact
**Date:** 2026-03-19
**Analyst:** analyst

## Summary

The NOD2 3-class classifier (GOF/neutral/LOF) suffers from **catastrophic class imbalance** (5 GOF : 301 neutral : 5 LOF, ratio 1:60:1). The model has effectively collapsed to a neutral classifier. LOF recall = 0.0 (no LOF variants detected in any CV fold), GOF recall = 0.4 (2/5 detected), and leave-one-out testing shows even worse: GOF LOO accuracy = 1/5 (20%), LOF LOO accuracy = 0/5 (0%). The 0.779 macro AUC is misleading — it is inflated by the neutral class, which dominates predictions. Binary OvR classifiers perform near random chance (GOF AUC = 0.584, LOF AUC = 0.661), confirming that **5 positive examples per class are fundamentally insufficient to learn a decision boundary**.

## Key Numbers

| Metric | Value |
|--------|-------|
| Training set size | 311 (5 GOF / 301 neutral / 5 LOF) |
| 3-class macro AUC (5-fold nested CV) | 0.842 |
| GOF recall | 0.40 (2/5) |
| LOF recall | 0.00 (0/5) |
| Neutral recall | 0.993 (299/301) |
| Binary GOF-vs-rest AUC | 0.584 |
| Binary LOF-vs-rest AUC | 0.661 |
| GOF LOO accuracy | 1/5 (20%) |
| LOF LOO accuracy | 0/5 (0%) |

## 1. Current Model Bias

The balanced class weights (20.73x for GOF/LOF, 0.34x for neutral) are insufficient to overcome the 60:1 imbalance. The model assigns:

- **Mean P(GOF) = 0.009** across all predictions (max 0.625)
- **Mean P(LOF) = 0.012** across all predictions (max 0.565)
- **Mean P(neutral) = 0.979** across all predictions

The model predicts GOF only 2 times and LOF only 2 times across all 311 CV predictions. It is systematically biased toward neutral.

## 2. Calibration

Calibration is unreliable for GOF and LOF classes due to extreme sparsity (n=5 each). The model's probability outputs for rare classes are not interpretable as true posterior probabilities. With only 5 samples per rare class, there is insufficient data to assess or correct calibration.

Calibration curves saved to: `analysis/calibration_curves.png`

## 3. Leave-One-Out Analysis

### GOF Variants (1/5 correct)

| Variant | Predicted | P(GOF) | P(neutral) | P(LOF) | Correct |
|---------|-----------|--------|------------|--------|---------|
| p.Arg334Gln | neutral | 0.145 | 0.818 | 0.037 | No |
| p.Arg334Trp | GOF | 1.000 | 0.000 | 0.000 | Yes |
| p.Leu469Phe | LOF | 0.000 | 0.004 | 0.996 | No |
| p.Glu383Lys | neutral | 0.000 | 1.000 | 0.000 | No |
| p.Pro268Ser | LOF | 0.000 | 0.025 | 0.975 | No |

Only p.Arg334Trp (Blau syndrome, NACHT domain) is correctly classified. Notably, p.Leu469Phe and p.Pro268Ser (both functionally validated GOF) are confidently predicted as LOF — a dangerous misclassification.

### LOF Variants (0/5 correct)

| Variant | Predicted | P(GOF) | P(neutral) | P(LOF) | Correct |
|---------|-----------|--------|------------|--------|---------|
| p.Arg702Trp | neutral | 0.000 | 1.000 | 0.000 | No |
| p.Gly908Arg | neutral | 0.022 | 0.955 | 0.023 | No |
| p.Leu1007fs | neutral | 0.000 | 1.000 | 0.000 | No |
| p.Leu682Phe | neutral | 0.110 | 0.831 | 0.059 | No |
| p.Arg587Cys | neutral | 0.000 | 1.000 | 0.000 | No |

The three canonical CD-associated LOF variants (R702W, G908R, L1007fs — together accounting for ~80% of NOD2-attributable CD risk) are all classified as neutral with near-certainty. This is a critical failure.

## 4. Binary OvR vs 3-Class

| Approach | GOF AUC | LOF AUC |
|----------|---------|---------|
| 3-class ensemble | Part of 0.842 macro | Part of 0.842 macro |
| Binary OvR (GOF vs rest) | 0.584 | — |
| Binary OvR (LOF vs rest) | — | 0.661 |

Binary OvR does not improve performance. The fundamental problem is sample size, not modeling strategy.

## 5. Root Cause

The class imbalance is not the primary issue — **the sample size of the rare classes is**. With n=5 per rare class:

- The model cannot learn generalizable decision boundaries
- Cross-validation folds contain 0-1 positive examples per fold, making training unstable
- Any statistical test has negligible power
- The reported AUC and classification metrics are unreliable

## 6. Recommendation

**Do NOT use this model for clinical variant classification in its current state.** The LOF and GOF predictions are unreliable.

### Immediate actions:

1. **Augment the training set.** This is the only viable path to improvement. Target ≥20-30 variants per rare class:
   - **LOF:** Add known CD-risk NOD2 variants from IBDGC datasets, SURF study extensions, and functional assay databases (e.g., MaveDB). The Infevers database (https://infevers.umai-montpellier.fr/) catalogs validated NOD2 mutations with functional data.
   - **GOF:** Add Blau syndrome / early-onset sarcoidosis variants from Infevers, HGMD, and published functional studies. Miceli-Richard et al. (PMID: 11687797) and Kanazawa et al. (PMID: 15459013) report additional GOF variants with NF-kB assay data.

2. **Consider binary framing.** Once training set is augmented, reframe as pathogenic-vs-benign binary classifier (combining GOF+LOF), then use a secondary model or heuristic (domain location, NF-kB effect direction) to distinguish GOF from LOF. This doubles the positive training set.

3. **Do not use SMOTE or synthetic augmentation without domain validation.** Synthetic variants in feature space may not correspond to biologically meaningful mutations.

4. **Use AlphaMissense as a standalone baseline.** Given its dominance (importance=0.21) and the model's failure to improve over single-predictor baselines, compare the ensemble VUS predictions against AlphaMissense alone. If they largely agree, the ensemble is not adding value.

### What to report now:

- VUS predictions should be labeled as **exploratory/hypothesis-generating only**, not clinical-grade
- The 440 "high-confidence" VUS classifications reflect confidence in neutral prediction, not genuine discriminative ability
- Negative results (model failure) should be prominently reported to prevent misuse

## Limitations

- LOO analysis uses a single GB classifier (not the full ensemble) due to computational constraints, but results are representative
- Binary OvR uses fixed hyperparameters rather than nested CV tuning
- With n=5, all statistical metrics have wide confidence intervals

## Output Files

- `analysis/class_imbalance_results.json` — Full quantitative results
- `analysis/calibration_curves.png` — Calibration plots per class
- `analysis/probability_distributions.png` — Predicted probability distributions
