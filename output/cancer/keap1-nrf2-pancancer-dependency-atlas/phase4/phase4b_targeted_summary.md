# KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 4b: Targeted Drug Analysis

## 1. Targeted Drug Sensitivity (KEAP1/NRF2-altered vs WT)

### Drug Availability in PRISM 24Q2

| Drug | Mechanism | Status |
|------|-----------|--------|
| BAY-2402234 | DHODH inhibitor | **FOUND** |
| ATR inhibitor 2 | ATR inhibitor | **FOUND** |
| brequinar | DHODH inhibitor | NOT FOUND |
| ceralasertib (AZD6738) | ATR inhibitor | NOT FOUND |
| berzosertib (M6620) | ATR inhibitor | NOT FOUND |
| elimusertib (BAY1895344) | ATR inhibitor | NOT FOUND |
| prexasertib | CHK1 inhibitor | NOT FOUND |
| SRA737 | CHK1 inhibitor | NOT FOUND |

### Sensitivity Results

| Drug | Mechanism | Context | d | p-value | FDR | n_mut | n_wt |
|------|-----------|---------|---|---------|-----|-------|------|
| BAY-2402234 | DHODH inhibitor | Pan-cancer | -0.240 | 1.30e-01 | 3.25e-01 | 38 | 574 |
| BAY-2402234 | DHODH inhibitor | Lung | -0.305 | 1.92e-01 | 3.85e-01 | 17 | 73 |
| BAY-2402234 | DHODH inhibitor | Bladder/Urinary Tract | 0.421 | 4.32e-01 | 6.77e-01 | 4 | 14 |
| BAY-2402234 | DHODH inhibitor | Esophagus/Stomach | -0.120 | 6.67e-01 | 6.77e-01 | 7 | 34 |
| BAY-2402234 | DHODH inhibitor | Uterus | -0.430 | 5.02e-01 | 6.77e-01 | 5 | 20 |
| ATR inhibitor 2 | ATR inhibitor | Pan-cancer | -0.073 | 6.74e-01 | 6.77e-01 | 38 | 561 |
| ATR inhibitor 2 | ATR inhibitor | Lung | -0.407 | 8.21e-02 | 3.25e-01 | 17 | 69 |
| ATR inhibitor 2 | ATR inhibitor | Bladder/Urinary Tract | 1.479 | 2.37e-04 | 2.37e-03 * | 4 | 14 |
| ATR inhibitor 2 | ATR inhibitor | Esophagus/Stomach | 0.198 | 6.77e-01 | 6.77e-01 | 7 | 34 |
| ATR inhibitor 2 | ATR inhibitor | Uterus | 0.447 | 1.23e-01 | 3.25e-01 | 5 | 20 |

## 2. KL-Subtype PRISM Screen (STK11+KRAS co-mutant vs WT)

- Compounds screened: **6790**
- Significant (|d|>0.3, FDR<0.1): **2**
- Sensitizing in KL-subtype: **0**

### Top KL-subtype sensitizers (by effect size, uncorrected)

| Compound | d | p-value | FDR | n_kl |
|----------|---|---------|-----|------|
| BRD-K03294269 | -2.514 | 6.37e-02 | 9.22e-01 | 5 |
| GDC-0927 | -1.493 | 1.17e-01 | 9.27e-01 | 6 |
| BRD-K79315489 | -1.420 | 1.57e-01 | 9.27e-01 | 4 |
| Olinciguat | -1.419 | 1.48e-01 | 9.27e-01 | 6 |
| Tepilamide fumarate | -1.380 | 1.02e-01 | 9.27e-01 | 6 |
| BRD-K20313525 | -1.374 | 7.83e-03 | 7.15e-01 | 5 |
| BRD-K08748705 | -1.330 | 1.53e-01 | 9.27e-01 | 5 |
| BRD-K59456551 | -1.310 | 9.40e-02 | 9.27e-01 | 4 |
| T-025 | -1.272 | 1.24e-01 | 9.27e-01 | 6 |
| BRD-A55312468 | -1.271 | 6.14e-02 | 9.17e-01 | 4 |

## 3. UGDH CRISPR Dependency (Phase 2 Chronos Data)

| Context | d | FDR | n_altered | n_wt |
|---------|---|-----|-----------|------|
| Bladder/Urinary Tract | 0.240 | 9.577e-01 | 4 | 17 |
| Uterus | -0.011 | 1.000e+00 | 3 | 23 |
| Lung | -0.120 | 9.984e-01 | 18 | 87 |
| Head and Neck | -0.361 | 9.809e-01 | 5 | 40 |
| Esophagus/Stomach | 0.417 | 9.856e-01 | 6 | 40 |
| Pan-cancer (pooled) | -0.013 | 9.770e-01 | 46 | 796 |

## 4. NRF2 Transcriptional Activity vs UXS1 Dependency

- Signature genes: NQO1, GCLM, TXNRD1, HMOX1, AKR1C1
- All lines: Spearman r=-0.173, p=6.93e-09, n=1112
- KEAP1/NRF2-altered only: Spearman r=-0.445, p=2.81e-03, n=43

## 5. Convergence Summary

The UXS1/pyrimidine and ATR/replication-stress axes are the two priority
therapeutic hypotheses from Phases 1-3. This analysis tests them with
pharmacological (PRISM drug sensitivity) and transcriptional (NRF2 activity
score) data to assess drug-target convergence.
