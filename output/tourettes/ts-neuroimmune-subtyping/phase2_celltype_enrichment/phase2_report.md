# Phase 2: Immune Cell-Type Enrichment Analysis

**Gene-level results from Phase 1b**: 19141 genes
**Methods**: MAGMA-style continuous association, Wilcoxon top-decile competitive test
**Specificity reference**: DICE (Schmiedel et al. 2018, Cell)

## Grouped Cell Types (9 types) — Continuous Association

| Cell Type | Beta | SE | t-stat | P-value | FDR | Bonferroni |
|-----------|------|----|---------|---------|----|------------|
| DC | 0.0606 | 0.0289 | 2.094 | 1.8114e-02 | 0.163 | 0.163 |
| Monocyte | 0.0155 | 0.0275 | 0.565 | 2.8597e-01 | 0.860 | 1.000 |
| NK | 0.0043 | 0.0267 | 0.160 | 4.3648e-01 | 0.860 | 1.000 |
| CD8_T | -0.0025 | 0.0396 | -0.064 | 5.2548e-01 | 0.860 | 1.000 |
| Gamma_delta_T | -0.0036 | 0.0477 | -0.075 | 5.2973e-01 | 0.860 | 1.000 |
| CD4_T | -0.0088 | 0.0477 | -0.184 | 5.7309e-01 | 0.860 | 1.000 |
| Th17 | -0.0324 | 0.0500 | -0.648 | 7.4165e-01 | 0.923 | 1.000 |
| B_cell | -0.0403 | 0.0302 | -1.334 | 9.0882e-01 | 0.923 | 1.000 |
| Treg | -0.0584 | 0.0410 | -1.427 | 9.2322e-01 | 0.923 | 1.000 |

## Grouped Cell Types — Top-Decile Competitive Test

| Cell Type | N Genes | Mean Z (set) | Mean Z (rest) | Wilcoxon Z | P-value | FDR |
|-----------|---------|-------------|--------------|-----------|---------|-----|
| CD4_T | 1828 | 0.088 | 0.073 | 1.647 | 4.9810e-02 | 0.448 |
| DC | 1894 | 0.092 | 0.072 | 1.059 | 1.4476e-01 | 0.486 |
| Th17 | 1900 | 0.080 | 0.074 | 0.986 | 1.6215e-01 | 0.486 |
| CD8_T | 1824 | 0.073 | 0.074 | 0.517 | 3.0260e-01 | 0.649 |
| NK | 1863 | 0.072 | 0.074 | 0.357 | 3.6069e-01 | 0.649 |
| Monocyte | 1884 | 0.074 | 0.074 | -0.123 | 5.4899e-01 | 0.823 |
| B_cell | 1842 | 0.061 | 0.076 | -1.019 | 8.4600e-01 | 0.922 |
| Treg | 1858 | 0.065 | 0.075 | -1.217 | 8.8829e-01 | 0.922 |
| Gamma_delta_T | 1888 | 0.061 | 0.076 | -1.421 | 9.2241e-01 | 0.922 |

## Individual Subtypes (25 types) — Continuous Association

| Cell Type | Beta | SE | t-stat | P-value | FDR |
|-----------|------|----|---------|---------|----|
| Plasmablasts | 0.3920 | 0.1010 | 3.881 | 5.2215e-05 | 0.001 |
| Plasmacytoid_DC | 0.1009 | 0.0450 | 2.243 | 1.2441e-02 | 0.155 |
| Treg_memory | 0.2047 | 0.0982 | 2.085 | 1.8546e-02 | 0.155 |
| Th2 | 0.1074 | 0.0806 | 1.333 | 9.1206e-02 | 0.522 |
| Th1 | 0.1366 | 0.1086 | 1.257 | 1.0434e-01 | 0.522 |
| Myeloid_DC | 0.0391 | 0.0484 | 0.807 | 2.0988e-01 | 0.875 |
| NK_memory | 0.0447 | 0.0679 | 0.658 | 2.5523e-01 | 0.910 |
| CD8_T | 0.0448 | 0.0895 | 0.500 | 3.0839e-01 | 0.910 |
| CD8_T_naive | 0.0258 | 0.0697 | 0.369 | 3.5592e-01 | 0.910 |
| Monocyte | 0.0136 | 0.0390 | 0.348 | 3.6389e-01 | 0.910 |
| NK_mature | 0.0073 | 0.0620 | 0.119 | 4.5282e-01 | 0.952 |
| CD8_T_effector_memory | -0.0073 | 0.0840 | -0.086 | 5.3441e-01 | 0.952 |
| NK_immature | -0.0044 | 0.0314 | -0.139 | 5.5544e-01 | 0.952 |
| CD8_T_central_memory | -0.0113 | 0.0584 | -0.193 | 5.7663e-01 | 0.952 |
| CD4_T_effector | -0.0182 | 0.0604 | -0.301 | 6.1822e-01 | 0.952 |
| B_bulk | -0.0297 | 0.0673 | -0.442 | 6.7079e-01 | 0.952 |
| CD4_T_memory | -0.0581 | 0.1265 | -0.459 | 6.7699e-01 | 0.952 |
| Gamma_delta_T | -0.0475 | 0.0850 | -0.558 | 7.1170e-01 | 0.952 |
| Tfh | -0.0590 | 0.0966 | -0.610 | 7.2911e-01 | 0.952 |
| CD4_T_naive | -0.0684 | 0.0961 | -0.711 | 7.6160e-01 | 0.952 |
| Th17 | -0.1147 | 0.0950 | -1.207 | 8.8624e-01 | 0.975 |
| B_memory | -0.0947 | 0.0712 | -1.330 | 9.0824e-01 | 0.975 |
| Treg_naive | -0.1226 | 0.0701 | -1.749 | 9.5984e-01 | 0.975 |
| Treg | -0.1167 | 0.0634 | -1.839 | 9.6706e-01 | 0.975 |
| B_naive | -0.0685 | 0.0349 | -1.962 | 9.7510e-01 | 0.975 |

## Method Agreement

No cell types reached P<0.05 in both methods simultaneously.

## Priority Cell Types (per research plan)

- **Th17**: beta=-0.0324, t=-0.648, P=7.4165e-01, FDR=0.923
- **NK**: beta=0.0043, t=0.160, P=4.3648e-01, FDR=0.860
- **CD4_T**: beta=-0.0088, t=-0.184, P=5.7309e-01, FDR=0.860

## Power Note

The 2019 TS GWAS (N=14,307) has limited power for cell-type enrichment. Null results are expected and do not exclude immune cell-type involvement. Future re-analysis with the 2024 TSAICG GWAS (N=19,138) is planned.
