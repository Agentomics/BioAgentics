# Statistical Robustness Assessment — GAS Molecular Mimicry Results

**Date:** 2026-03-19
**Analyst:** analyst (pandas_pans division)

## 1. Structural Mimicry (TM-score)

- **Successful alignments:** 16 (via AlphaFold structures)
- **TM >= 0.5 (same fold):** 7/16 (43.8%)
- **Unique targets with TM >= 0.5:** 2 — ENO2 (max TM=0.662) and GAPDH (max TM=0.658)
- **Binomial test vs random:** p = 1.25e-08, 21.9x enrichment over background rate of 2%

**Caveat:** These pairs were pre-selected by DIAMOND at pident >= 40%. Sequence-similar proteins are expected to share structural folds. The binomial test overstates significance because the "random" background (TM=0.02) does not apply to sequence-filtered pairs. The Spearman correlation between pident and TM-score is r = -0.346 (p = 0.19), suggesting structural similarity is not simply a function of sequence identity in this dataset — the relationship is more nuanced.

**Interpretation:** The TM-score results confirm that sequence mimicry translates to structural mimicry for ENO2 and GAPDH. Notably, B4DHP5/HSPA6 (top-ranked target) has low TM-scores (0.08-0.17) despite 49% pident, meaning its mimicry operates at the epitope/peptide level rather than whole-fold structural similarity.

## 2. B-cell Epitope Overlap

- **Pairs with overlap:** 15/40 (37.5%)
- **Targets with overlap:** 3/8 — B4DHP5/HSPA6 (12 pairs, score 0.606), B4DHW5/MCCC1-like (3 pairs, score 0.582), P09104/ENO2 (2 pairs, score 0.580)
- **Targets without overlap:** GAPDH, PSMC6, B4DHP7, B4DHZ4, AK9

**Background estimation:** For two random proteins of ~400 residues sharing 40% identity, B-cell epitope co-localization depends on epitope density (~2-5 per 100 residues) and the length of the aligned region. A formal permutation test would require randomly pairing proteins of matched length and identity — not feasible without a larger dataset of similar pairs.

**Interpretation:** The 3 targets with epitope overlap are precisely the top-3 ranked targets. This is partly circular (epitope contributes 20% of composite score), but the fact that epitope overlap concentrates in specific targets rather than being uniformly distributed suggests genuine biological signal. GAPDH's lack of B-cell epitope overlap despite being a confirmed autoantigen suggests it may operate through T-cell (MHC-II) mediated mechanisms.

## 3. MHC-II Cross-Reactive Binding

- **Cross-reactive combos:** 93/120 (77.5%) of GAS-human-allele combinations
- **Targets with cross-reactivity:** 6/8
- **Targets without:** B4DHZ4 (1 serotype only, 31aa alignment) and Q5TCS8/AK9 (2 serotypes, 56aa alignment)

**Background:** For random 9-mer peptides, ~2-5% bind HLA-DR with IC50 < 500nM. The high cross-reactivity rate (77.5%) is EXPECTED given pident >= 40% — similar sequences generate similar peptide pools with similar MHC binding. This result confirms a necessary condition (MHC-II presentation of mimicry peptides) rather than demonstrating enrichment.

**Interpretation:** MHC-II cross-reactivity is nearly universal among mimicry hits, providing the mechanistic basis for T-cell activation by mimicry peptides. The 2 targets without cross-reactivity (B4DHZ4, AK9) have very short alignments (31 and 56 residues), limiting the peptide pool available for MHC binding.

## 4. Serotype Conservation

- **Fully conserved (6/6 serotypes):** 6/8 (75.0%)
- **Non-conserved:** AK9 (2/6: M18, M5), B4DHZ4 (1/6: M12)
- **Binomial test vs 60% core genome rate:** p = 0.32

**Interpretation:** The 75% conservation rate is modestly enriched over the ~60% GAS core genome but the test is severely underpowered (n=8). High conservation is expected: the mimicry targets are housekeeping proteins (DnaK, enolase, GAPDH) that are core genome members by definition. This supports the biological model but is not independently informative.

The 2 non-conserved targets represent potential serotype-specific mimicry, but both have low composite scores (AK9: 0.120, B4DHZ4: 0.208), suggesting they are weak mimicry candidates.

## 5. Effect Size Summary

| Metric | Observed | Expected Random | Fold Enrichment | p-value | Reliable? |
|--------|----------|----------------|-----------------|---------|-----------|
| TM >= 0.5 | 7/16 (44%) | ~2% | 21.9x | 1.2e-08 | Inflated (pre-selected) |
| B-cell epitope overlap | 3/8 targets | Unknown | N/A | N/A | Qualitative only |
| MHC-II cross-reactivity | 6/8 targets | ~2-5% random peptides | N/A | N/A | Expected given pident |
| Conservation 6/6 | 6/8 (75%) | ~60% core | 1.25x | 0.32 | Underpowered |

## 6. Limitations

1. **Small sample size (n=8 targets):** Limits statistical power for all tests. Individual p-values should be interpreted cautiously.
2. **Pre-selection bias:** All targets were filtered by DIAMOND at pident >= 40% and E-value <= 1e-3. Comparisons to "random" backgrounds are misleading.
3. **No permutation-based null:** A proper statistical test would require sampling random GAS-human protein pairs of matched length and running the full pipeline to generate a null distribution. This was not performed.
4. **Computational predictions:** Epitope and MHC-II predictions are computational approximations, not experimental measurements. Classical epitope scales (Parker, Kolaskar, Emini, Chou-Fasman) have ~60-70% accuracy.
5. **Only 3 HLA alleles tested:** MHC-II predictions limited to DRB1*04:01, *07:01, *01:01. Broader allele coverage could change results for individual targets.

## 7. Conclusion

**The statistical evidence is strongest when viewed as convergent multi-modal support rather than individual significant tests.** No single metric provides definitive statistical proof of molecular mimicry. However, the top targets (B4DHP5/HSPA6 and ENO2) are supported by multiple independent lines of evidence:

- **HSPA6:** sequence identity (49.3%), B-cell epitope overlap (12 pairs), MHC-II cross-reactivity (6.0 max score across 3 alleles), full serotype conservation (6/6), invasive-phase classification. Low structural TM-score (0.17) indicates epitope-level rather than fold-level mimicry.
- **ENO2:** sequence identity (49.1%), B-cell epitope overlap (2 pairs), MHC-II cross-reactivity (5.9 max score), structural mimicry (TM=0.66), full conservation, published literature support.

The convergence of evidence across modalities, combined with literature cross-reference validation (GAPDH positive control recovery, ENO2 as known autoantigen), provides confidence that the top-ranked targets represent genuine mimicry candidates worthy of experimental validation.
