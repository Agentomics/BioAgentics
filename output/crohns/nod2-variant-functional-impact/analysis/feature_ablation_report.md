# Feature Ablation Study: AlphaMissense Dominance and Single-Predictor Collapse

**Task:** 856
**Project:** nod2-variant-functional-impact
**Date:** 2026-03-19
**Analyst:** analyst

## Summary

AlphaMissense has the highest feature importance (0.21) but is **counterproductive** for 3-class NOD2 prediction. AlphaMissense alone yields AUC=0.474 (below random chance), and removing it from the ensemble **improves** AUC from 0.768 to 0.820. The conventional pathogenicity predictors (AlphaMissense, REVEL, PolyPhen2, SIFT, CADD) are essentially useless for GOF/neutral/LOF classification because they predict binary pathogenicity, not functional direction.

The most informative features are structural and domain-based: **domain features alone achieve AUC=0.881 with GOF recall=0.80 and LOF recall=1.00** — dramatically outperforming the full 24-feature ensemble. This reflects the known biology: GOF variants cluster in the NACHT domain (constitutive NF-kB activation), LOF variants cluster in the LRR domain (impaired MDP sensing).

## Key Numbers

| Configuration | AUC | GOF Recall | LOF Recall |
|--------------|-----|------------|------------|
| Full model (24 features) | 0.768 | 0.40 | 0.00 |
| AlphaMissense only | 0.474 | 0.60 | 0.00 |
| **Without AlphaMissense** | **0.820** | **0.60** | **0.00** |
| **Domain features only (6)** | **0.881** | **0.80** | **1.00** |
| Structure features only (3) | 0.812 | 0.40 | 0.00 |
| pLDDT alone | 0.821 | 0.40 | 0.40 |
| girdin_interface_distance alone | 0.730 | 0.40 | 0.80 |
| VarMeter2 only (3) | 0.783 | 0.40 | 0.20 |
| Predictor scores only (9) | 0.424 | 0.20 | 0.00 |
| Girdin features only (3) | 0.630 | 0.40 | 0.20 |

## Ablation: Removing Feature Groups

| Removed Group | AUC | Delta vs Full |
|--------------|-----|---------------|
| None (full) | 0.768 | — |
| Structure | 0.870 | +0.102 |
| Girdin | 0.863 | +0.095 |
| VarMeter2 | 0.823 | +0.055 |
| Domain | 0.727 | -0.041 |
| Predictors | 0.643 | -0.125 |

**Removing structure and girdin features *increases* AUC.** This paradox indicates overfitting: with only 5 GOF/LOF examples, the model memorizes noise in continuous features rather than learning generalizable patterns.

## Feature Correlation

AlphaMissense is highly correlated with other pathogenicity predictors:
- REVEL: r=0.711
- PolyPhen2 HVar: r=0.706
- PhyloP 100way: r=0.647
- CADD phred: r=0.527
- PolyPhen2 HDiv: r=0.519

This redundancy means the 9 predictor features effectively provide ~2-3 independent signals. All measure evolutionary constraint / predicted pathogenicity — none distinguish GOF from LOF.

## Interpretation

### Why predictors fail for 3-class

AlphaMissense, REVEL, and similar tools are trained to distinguish pathogenic from benign variants. For NOD2:
- **GOF variants** (Blau syndrome, NACHT domain) → pathogenic
- **LOF variants** (Crohn's disease, LRR domain) → pathogenic
- **Neutral variants** → benign

The predictors correctly identify both GOF and LOF as "pathogenic" but cannot distinguish between them. Worse, they conflate functional direction, making the 3-class problem harder.

### Why domain features excel

With n=5 per class, the strongest signal is the simplest: **where is the variant?**
- GOF variants: 4/5 in NACHT domain (p.Arg334Gln, p.Arg334Trp, p.Leu469Phe) or WH (p.Glu383Lys)
- LOF variants: 3/5 in LRR domain (p.Arg702Trp, p.Gly908Arg, p.Leu1007fs)
- Neutral: distributed across all domains

Domain location is a near-perfect separator for these 10 examples. However, **this performance will not generalize** to larger training sets because many VUS in the NACHT domain are likely neutral, and the 5/5/301 training set cannot teach the model this nuance.

### Girdin_interface_distance for LOF

Notably, girdin_interface_distance alone achieves LOF recall=0.80 (4/5). The canonical CD LOF variants are in the LRR C-terminal region near the girdin binding interface, so proximity to this interface is a strong LOF signal. This validates the Ghosh et al. JCI 2025 finding bioinformatically.

## Recommendations

1. **Remove or downweight pathogenicity predictors.** AlphaMissense, REVEL, PolyPhen2, SIFT, CADD, PhyloP, PhastCons, and GERP collectively hurt 3-class performance. Consider:
   - Using a 2-stage approach: predictors for pathogenic-vs-benign, then structural features for GOF-vs-LOF
   - Or removing predictors entirely until training set is larger

2. **Prioritize domain + structural features.** Domain location, pLDDT, girdin_interface_distance, and ripk2_interface_distance capture the biological signal. A minimal model (domain + girdin_interface_distance + pLDDT) may outperform the full ensemble.

3. **Do not interpret AlphaMissense importance=0.21 as "most informative."** GB importance measures split utility, which correlates with variance in the feature, not predictive value. AlphaMissense varies more than one-hot domain features, so GB splits on it frequently — but the splits don't improve class separation.

4. **After training set augmentation (Task 896):** re-evaluate whether predictors become useful with more rare-class examples. With n≥20 per class, the model may be able to learn a joint signal.

## Limitations

- Ablation uses fixed GB hyperparameters (not nested CV), so absolute AUC values differ slightly from the original evaluation
- With n=5 per rare class, all AUC estimates have high variance (~±0.15)
- SHAP analysis was not performed due to the instability of the underlying model (SHAP values would be unreliable with 5 positive examples)

## Output Files

- `analysis/feature_ablation_results.json` — Full ablation AUC table
- `analysis/feature_ablation_aucs.png` — Ablation summary bar chart
- `analysis/feature_correlation_matrix.png` — Feature correlation heatmap
