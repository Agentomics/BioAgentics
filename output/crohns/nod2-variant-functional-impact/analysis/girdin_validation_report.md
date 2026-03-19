# Validation of girdin_interface_distance as Novel Pathogenicity Feature

**Task:** 857
**Project:** nod2-variant-functional-impact
**Date:** 2026-03-19
**Analyst:** analyst

## Summary

girdin_interface_distance is the 2nd most important feature (importance=0.18) and alone achieves LOF recall=0.80 in the ablation study. However, **statistical validation fails**: Mann-Whitney LOF vs neutral p=0.79 (not significant). Only 2/5 LOF training variants are near the girdin interface (G908R at 14.2A, L1007fs at 0.0A); the other 3 (R702W, L682F, R587C) are 60-67A away. The feature's predictive power is **confounded with LRR domain membership** — variants near the girdin interface are in the LRR C-terminal, and domain location is the true signal. Girdin remains a biologically plausible and novel mechanistic feature (Ghosh et al. JCI 2025), but the current training set is too small to validate it independently of domain position.

## Key Numbers

| Metric | Value |
|--------|-------|
| Feature importance rank | 2nd (0.18) |
| LOF recall (girdin alone) | 0.80 (4/5) in ablation |
| Mann-Whitney LOF vs neutral | U=94.5, p=0.79 (NS) |
| Mann-Whitney GOF vs neutral | U=93.0, p=0.84 (NS) |
| LOF variants near girdin (<20A) | 2/5 |
| LOF variants far from girdin (>50A) | 3/5 |
| VUS within 10A of girdin interface | 72/665 |
| Training predictions changed by girdin | 0/311 |

## 1. Girdin Distance by Functional Class

### LOF Variants
| Variant | Source | Girdin Dist (A) | Position | Interpretation |
|---------|--------|-----------------|----------|----------------|
| p.Leu1007fs | ClinVar/CD | 0.0 | 1007 (LRR C-term) | Directly at interface |
| p.Gly908Arg | ClinVar/CD | 14.2 | 908 (LRR) | Near interface |
| p.Leu682Phe | SURF | 60.9 | 682 (WH/LRR boundary) | Far from interface |
| p.Arg702Trp | ClinVar/CD | 66.8 | 702 (LRR N-term) | Far from interface |
| p.Arg587Cys | SURF | 62.8 | 587 (NACHT) | Far from interface |

Only L1007fs and G908R support the girdin proximity hypothesis. R702W — the most common CD-associated NOD2 variant — is 67A from the girdin interface, suggesting its LOF mechanism is NOT through girdin disruption but through impaired MDP sensing in the LRR more broadly.

### GOF Variants
| Variant | Girdin Dist (A) | Position |
|---------|-----------------|----------|
| p.Arg334Gln | 54.0 | 334 (NACHT) |
| p.Arg334Trp | 54.0 | 334 (NACHT) |
| p.Leu469Phe | 52.7 | 469 (NACHT) |
| p.Glu383Lys | 49.9 | 383 (NACHT) |
| p.Pro268Ser | 14.5 | 268 (linker/NACHT) |

GOF variants are uniformly distant from the girdin interface (49-54A), except P268S. This is expected — GOF variants cause constitutive NF-kB activation via the NACHT domain, not via girdin binding.

## 2. Domain Confounding

girdin_interface_distance is correlated with LRR domain membership because the girdin binding interface is located at the LRR C-terminal. The feature's predictive power in the ablation study (LOF recall=0.80) reflects this confound:

- LRR domain variants tend to have lower girdin_interface_distance
- The 2 LOF variants correctly detected by girdin alone (L1007fs, G908R) are both in the LRR C-terminal
- **But so are neutral variants**: I939V (9.5A), M950V (16.0A), V955I (12.8A) are all neutral and equally close to girdin

## 3. VUS Predictions Near Girdin Interface

72 VUS are within 10A of the girdin interface. Most are LRR C-terminal variants. Of these:
- 4 predicted LOF (all LRR): p.Leu1007Phe, p.Asn1021Lys, p.Arg1019Leu, p.Leu1039Phe
- These are biologically plausible LOF candidates based on proximity to both the girdin interface AND the canonical L1007fs frameshift

**Concern:** 45 VUS in the LRR are predicted GOF (high confidence), which is biologically implausible — GOF mutations in the LRR are rare and mechanistically unexpected. This suggests the model's GOF predictions in LRR are false positives driven by feature interactions, not biology.

## 4. False Positive Check

Adding girdin features changes **0/311** training predictions — the feature is dominated by other signals (domain, pLDDT) in the full model context. This means girdin_interface_distance contributes to probability calibration but not to hard class assignments. Its influence is subtle, not decisive.

## 5. Supporting Literature Assessment

**Ghosh et al. JCI 2025** reported:
- NOD2 binds girdin (CCDC88A) via the LRR C-terminal region
- This interaction modulates macrophage polarization via NF-kB/RANK pathway crosstalk
- A 53-gene macrophage polarization signature was identified

Our analysis is **partially consistent**:
- L1007fs (at the interface) and G908R (near the interface) are the two most functionally severe CD LOF variants
- Their girdin proximity provides a plausible additional mechanism beyond impaired MDP sensing

But **not fully validated**:
- R702W (the most common CD variant) is far from the interface, suggesting its LOF mechanism is independent of girdin
- No functional data specifically links R702W or other LOF variants to disrupted girdin binding
- The 53-gene macrophage signature has not been tested as an orthogonal validation for our predictions

## 6. Recommendation

### For the current model:
- **Keep girdin_interface_distance** as a feature — it adds biological interpretability and captures a real spatial signal
- **Do not claim girdin validation** in the manuscript without larger training sets and functional confirmation
- Label girdin-proximal LOF predictions as "consistent with girdin disruption hypothesis" rather than "validated"

### For further validation:
1. **Functional assay data needed:** Test whether VUS near the girdin interface (L1007F, N1021K, R1019L/G) show impaired girdin binding in co-IP or Y2H assays
2. **Orthogonal validation:** Check the 53-gene macrophage polarization signature (Ghosh et al.) against single-cell data from NOD2-variant patient samples
3. **Expanded LOF set:** With more LOF variants (Task 896), re-test whether girdin distance remains significant after controlling for domain

### Novel finding assessment:
**Partially validated.** girdin_interface_distance captures a biologically plausible signal (LRR C-terminal proximity), but its discriminative value is confounded with domain location and not statistically significant with n=5. This is a **promising lead, not a confirmed finding**.

## Limitations

- n=5 per rare class: all statistical tests are underpowered (power <0.10)
- Girdin distance is computed from static AlphaFold structure; protein dynamics and conformational changes are not captured
- No experimental validation of predicted girdin-disrupting VUS
- The SURF study (PMID 41357180) functional data was not directly compared

## Output Files

- `analysis/girdin_validation_report.md` — This report
- `analysis/girdin_distance_scatter.png` — Girdin distance scatter plots
