# Phase 1b: Lymphocytic Enrichment Method Validation

**GWAS**: Yu et al. 2019 PGC TS (N=14,307; 4,819 cases + 9,488 controls)
**Genes tested**: 19141
**Methods**: MAGMA competitive regression, Wilcoxon rank-sum competitive test


## Lymphocyte Gene Sets

| Gene Set | N Genes | MAGMA Z | MAGMA P | MAGMA FDR | Wilcoxon Z | Wilcoxon P | Wilcoxon FDR |
|----------|---------|---------|---------|-----------|------------|------------|--------------|
| DICE_B_cell | 1842 | -0.98 | 8.374e-01 | 0.953 | -1.02 | 8.460e-01 | 0.966 |
| DICE_CD4_T | 1828 | 1.06 | 1.441e-01 | 0.953 | 1.65 | 4.981e-02 | 0.697 |
| DICE_CD8_T | 1824 | -0.06 | 5.244e-01 | 0.953 | 0.52 | 3.026e-01 | 0.966 |
| DICE_Gamma_delta_T | 1888 | -1.00 | 8.414e-01 | 0.953 | -1.42 | 9.224e-01 | 0.966 |
| DICE_NK | 1863 | -0.14 | 5.548e-01 | 0.953 | 0.36 | 3.607e-01 | 0.966 |
| DICE_Th17 | 1900 | 0.46 | 3.241e-01 | 0.953 | 0.99 | 1.622e-01 | 0.757 |
| DICE_Treg | 1858 | -0.71 | 7.610e-01 | 0.953 | -1.22 | 8.883e-01 | 0.966 |
| KEGG_HEMATOPOIETIC_CELL_LINEAGE | 78 | -0.86 | 8.051e-01 | 0.953 | -1.31 | 9.043e-01 | 0.966 |
| NK_INNATE_LYMPHOID | 35 | -1.24 | 8.924e-01 | 0.953 | -1.82 | 9.656e-01 | 0.966 |
| REACTOME_ADAPTIVE_IMMUNE_SYSTEM | 115 | -1.67 | 9.528e-01 | 0.953 | -1.37 | 9.149e-01 | 0.966 |
| lymphocyte_union | 9918 | -0.99 | 8.380e-01 | 0.953 | -0.49 | 6.873e-01 | 0.966 |
| lymphocyte_union_plus_msigdb | 10010 | -1.18 | 8.802e-01 | 0.953 | -0.73 | 7.683e-01 | 0.966 |

## Non-Lymphocyte Controls

| Gene Set | N Genes | MAGMA Z | MAGMA P | MAGMA FDR | Wilcoxon Z | Wilcoxon P | Wilcoxon FDR |
|----------|---------|---------|---------|-----------|------------|------------|--------------|
| DICE_DC | 1894 | 1.35 | 8.864e-02 | 0.953 | 1.06 | 1.448e-01 | 0.757 |
| DICE_Monocyte | 1884 | -0.02 | 5.091e-01 | 0.953 | -0.12 | 5.490e-01 | 0.966 |

## Method Agreement

No gene sets reached P<0.05 in both methods simultaneously.


## Power Note

The 2019 TS GWAS (N=14,307) has limited power for gene-set enrichment analyses. 
Null results are expected and do not exclude lymphocytic involvement. 
Future re-analysis with the 2024 TSAICG GWAS (N=19,138) is planned.

