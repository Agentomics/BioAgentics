# VUS Classification Confidence and Reliability Analysis

**Task:** 858
**Project:** nod2-variant-functional-impact
**Date:** 2026-03-19
**Analyst:** analyst

## Summary

440/665 VUS (66%) are classified with high confidence (P>0.8), but this metric is misleading. **367/440 high-confidence predictions (83%) are neutral** — the model is confident at doing what it does best (predicting neutral) and unreliable for what matters most (identifying GOF/LOF). Of the 125 GOF predictions, **64 (51%) are outside the NACHT/WH domains and biologically suspect** — including 19 high-confidence GOF predictions in the LRR domain, where GOF mutations are rare and mechanistically unexpected. LOF predictions are mostly low-confidence (33/47). The current VUS classifications should be treated as **exploratory hypotheses, not clinical-grade predictions**.

## Key Numbers

| Metric | Value |
|--------|-------|
| Total VUS | 665 |
| High-confidence (P>0.8) | 440 (66.2%) |
| Moderate (0.6-0.8) | 58 (8.7%) |
| Low (<0.6) | 167 (25.1%) |
| Predicted GOF | 125 (18.8%) |
| Predicted LOF | 47 (7.1%) |
| Predicted neutral | 493 (74.1%) |
| GOF in NACHT/WH (plausible) | 61 |
| GOF outside NACHT/WH (suspect) | 64 |
| VUS with pathogenicity score >0.5 | 181 |

## 1. Confidence Distribution

### By Class
| Class | High | Moderate | Low | Total |
|-------|------|----------|-----|-------|
| GOF | 63 (50%) | 22 (18%) | 40 (32%) | 125 |
| LOF | 10 (21%) | 4 (9%) | 33 (70%) | 47 |
| neutral | 367 (74%) | 32 (7%) | 94 (19%) | 493 |

LOF predictions are overwhelmingly low-confidence (70%). The model has no reliable LOF detection capability — consistent with the LOF recall=0.0 finding from Task 855.

### Threshold Assessment
The P>0.8 threshold is set arbitrarily and untested against calibration data. Given the class imbalance analysis (Task 855), these probabilities are **not calibrated**: the model assigns P(neutral)>0.8 to almost everything, including variants that are functionally pathogenic. The threshold measures model confidence in neutral, not classification quality.

## 2. Low-Confidence VUS Clustering

167 low-confidence VUS cluster in the LRR and C-terminal regions:
- **LRR: 74 (44%)** — the region with mixed LOF/neutral signals
- **WH: 29 (17%)**
- **NACHT: 27 (16%)**
- **Linker: 27 (16%)**

Mean residue position for low-confidence VUS: 622 (vs 450 for high-confidence). The model is most uncertain in the C-terminal half of the protein, where the biological signal is most complex (LRR contains both disease-causing LOF and benign variation).

## 3. GOF Prediction Plausibility

### Known Biology
NOD2 GOF mutations cause Blau syndrome / early-onset sarcoidosis. All known GOF variants affect the NACHT domain (nucleotide-binding oligomerization), causing constitutive NF-kB activation independent of MDP ligand binding. GOF in the LRR is biologically unexpected because the LRR senses MDP — disruption reduces signaling (LOF), not enhances it.

### Assessment of 125 GOF predictions

| Domain | GOF Predicted | High-Confidence GOF | Plausibility |
|--------|--------------|---------------------|--------------|
| NACHT | 39 | 25 | **Plausible** — known GOF domain |
| WH | 22 | 12 | **Plausible** — adjacent to NACHT, mechanism possible |
| LRR | 45 | 19 | **Suspect** — GOF in LRR is rare and mechanistically unexpected |
| Linker | 18 | 7 | **Uncertain** — limited data on linker GOF |
| CARD1 | 1 | 0 | **Suspect** — CARD mediates protein interaction, not catalysis |

**51% of GOF predictions (64/125) are outside the expected GOF domains.** The 19 high-confidence GOF predictions in the LRR are particularly concerning — these likely reflect the model's overfitting to the small training set rather than genuine biological signal.

### Specific suspect predictions
- p.Leu746Pro (LRR, GOF, high-conf) — Proline substitution at LRR N-terminal; more likely structural disruption (LOF)
- p.Gly905Glu (LRR, GOF, high-conf) — Adjacent to known LOF variant G908R; GOF is implausible
- p.Glu748Asp (LRR, GOF, high-conf) — Conservative substitution in LRR; more likely neutral or mild LOF

## 4. LOF Predictions

47 LOF predictions, heavily concentrated in LRR (31/47 = 66%). This is biologically appropriate — the LRR is the primary LOF domain. However:
- Only 10/47 are high-confidence
- Only 8 in LRR are high-confidence
- 16 LOF predictions are outside LRR (8 in WH, 6 in NACHT, 2 in CARD1)

The LOF-in-NACHT predictions deserve investigation — some NACHT variants might impair ATP hydrolysis needed for NF-kB signaling activation, creating a mechanistic LOF distinct from MDP-sensing loss.

## 5. Highest-Pathogenicity VUS

The top 20 VUS by pathogenicity score (P(GOF)+P(LOF)) are all predicted GOF with P≈1.0, mostly in NACHT domain. p.Arg334Pro stands out — same residue as the known Blau GOF variant p.Arg334Gln, making this the most biologically credible novel finding. Other high-value predictions in NACHT (Cys483Tyr, Phe488Ser, Tyr514His) affect conserved positions near the Walker A/B motifs and are strong GOF candidates.

### Top GOF candidates (biologically supported):
1. **p.Arg334Pro** (NACHT) — Same residue as validated Blau GOF R334Q/R334W
2. **p.Leu456Pro** (NACHT) — Near validated GOF L469F
3. **p.Cys483Tyr** (NACHT) — Near Walker B motif
4. **p.Tyr514His** (NACHT) — Conserved NACHT position

## 6. Reliability Assessment

### What the model does well:
- Identifies neutral VUS with reasonable confidence
- GOF predictions in the NACHT domain have biological support
- Pathogenicity score correctly ranks R334P (same position as known GOF) highest

### What the model does poorly:
- LOF detection (recall=0.0 in CV, 70% of LOF predictions are low-confidence)
- GOF predictions outside NACHT/WH (likely false positives)
- Probability calibration (probabilities reflect class imbalance, not true posterior)

### Clinical actionability:
- **0 VUS can be confidently reclassified** based on this model alone
- The 61 NACHT/WH GOF predictions are hypothesis-generating for functional assay prioritization
- The 31 LRR LOF predictions are biologically plausible but lack statistical support

## Limitations

- No external validation against independently characterized variants
- Leave-one-out (Task 855) showed GOF LOO accuracy=1/5, LOF=0/5
- Confidence thresholds are arbitrary and not calibrated
- ClinVar conflict analysis not performed (would require ClinVar API query)

## Output Files

- `analysis/vus_confidence_report.md` — This report
- `analysis/vus_confidence_distribution.png` — Probability histograms and scatter plots
- `analysis/vus_confidence_by_position.png` — Confidence by residue position
- `analysis/vus_confidence_results.json` — Summary statistics
