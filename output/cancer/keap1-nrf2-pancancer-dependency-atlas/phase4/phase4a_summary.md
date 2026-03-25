# KEAP1/NRF2 Pan-Cancer Dependency Atlas - Phase 4a: PRISM Drug Screen

## Cohort
- KEAP1/NRF2-altered cell lines in PRISM: **38**
- WT cell lines in PRISM: **590**
- Compounds screened: **6790**

## Summary Statistics
- Significant compounds (|d|>0.3, FDR<0.1): **4**
  - Sensitizing in altered (d<0): **0**
  - Resistance in altered (d>0): **4**

## Top 30 Sensitizing Hits (d<0, more sensitive in KEAP1/NRF2-altered)

| Rank | Compound | d | p-value | FDR | n_mut | n_wt |
|------|----------|---|---------|-----|-------|------|

## Top Resistance Hits (d>0, less sensitive in KEAP1/NRF2-altered)

| Rank | Compound | d | p-value | FDR | n_mut | n_wt |
|------|----------|---|---------|-----|-------|------|
| 1 | BRD-K06426971 | 0.962 | 5.06e-06 | 3.44e-02 | 32 | 408 |
| 2 | walrycin-B | 0.619 | 5.25e-05 | 9.82e-02 | 38 | 570 |
| 3 | S63845 | 0.373 | 2.01e-05 | 6.81e-02 | 38 | 574 |
| 4 | BRD-K68379918 | 0.355 | 5.79e-05 | 9.82e-02 | 31 | 347 |

## Notes
- MOA and target annotations are not in the PRISM 24Q2 treatment metadata.
  Phase 4b targeted analysis will annotate specific compound classes.
- Negative d = compound kills KEAP1/NRF2-altered cells more than WT (potential therapeutic).
- Welch t-test used for p-values; Benjamini-Hochberg for FDR correction.
