# Pan-Cancer MTAP/PRMT5 Synthetic Lethality Atlas

**Date:** March 17, 2026
**Project:** pancancer-mtap-prmt5-atlas
**Status:** Documentation
**Data source:** DepMap 25Q3, TCGA Pan-Cancer Atlas, C-CAT (Suzuki et al., ESMO Open 2025)
**Pipeline:** `src/pancancer_mtap_prmt5_atlas/` (scripts 01–07 + phase3 modules)
**Validation:** Approved — journal #507 (validation_scientist, March 17, 2026)

---

## Executive Summary

This atlas ranks 16 cancer types by the strength of PRMT5 synthetic lethality (SL) in MTAP-deleted cell lines, using DepMap 25Q3 CRISPR dependency data. It identifies the tumor types most likely to respond to MTA-cooperative PRMT5 inhibitors (vopimetostat, AMG 193, BMS-986504, AZD3470, IDE892) and estimates addressable patient populations from TCGA.

**Key findings:**

- **5 cancer types** have ROBUST PRMT5 SL (confirmed by FDR correction, leave-one-out stability, and permutation testing): **Breast** (d = −1.73), **Lung** (d = −1.23), **Soft Tissue** (d = −1.05), **Esophagus/Stomach** (d = −0.97), and **CNS/Brain** (d = −0.68).
- **1 cancer type** is classified MARGINAL: **Pancreas** (d = −0.45). PRMT5 is broadly essential in pancreatic cancer regardless of MTAP status, weakening the SL differential.
- **Breast cancer** is the #1 underexplored opportunity — the strongest PRMT5 SL effect (exceeding NSCLC) with ~10,163 MTAP-deleted patients/year in the US, yet no PRMT5 inhibitor trials are enrolling breast cancer patients.
- **GBM** shows stronger PRMT5 SL (d = −0.74, FDR = 0.031) than the CNS/Brain aggregate (d = −0.68), supporting the brain-penetrant PRMT5 inhibitor TNG456 (NCT06810544).
- **MAT2A** SL is not significant in any cancer type (0/16 FDR-significant) and is uncorrelated with PRMT5 SL (Spearman r = −0.11). The IDE892+IDE397 (PRMT5+MAT2A) combination has strongest rationale in Head & Neck and Liver only.
- **Lung cancer** has the largest addressable MTAP-deleted population (~37,494 patients/year) and the strongest clinical validation (BMS-986504 ORR 29% in NSCLC).

---

## Important Caveats

The following limitations must be considered when interpreting all results in this atlas:

1. **Small sample sizes for top-ranked types.** Breast has only N = 8 MTAP-deleted lines, Soft Tissue N = 9, and Esophagus/Stomach N = 10. Large effect sizes may be inflated by small sample sizes. Leave-one-out analysis confirms sign consistency but confidence intervals are wide (Breast CI: −2.83 to −0.90).

2. **DepMap 2D culture limitations.** CRISPR dependency scores in 2D culture do not capture the tumor microenvironment (TME), immune interactions, stromal contributions, or in vivo pharmacokinetics/pharmacodynamics. Effect sizes reflect intrinsic genetic dependency only.

3. **Monotherapy vs. combination clinical reality.** All advancing PRMT5 inhibitor trials test combinations (PRMT5i + chemotherapy, IO, or RAS inhibitors). DepMap CRISPR knockout measures monotherapy genetic dependency and does not directly predict combination benefit.

4. **N = 2 for DepMap–clinical ORR comparison.** Only Lung (BMS-986504 ORR 29%) and Pancreas (vopimetostat ORR 25%) have per-histology clinical ORR data. This is insufficient to establish a predictive correlation between DepMap effect size and clinical response. The atlas provides **hypothesis-generating rankings, not clinical predictions**.

5. **C-CAT cross-population bias.** TCGA MTAP deletion frequencies were cross-validated against the C-CAT dataset (Suzuki et al., ESMO Open 2025, N = 51,828), which is derived from a Japanese cohort. MTAP deletion frequencies may have population-specific variation. The largest correction (Bladder: TCGA 26.0% vs. C-CAT 11.0%) should be interpreted with this caveat.

6. **MTAP is a weaker biomarker in pancreatic cancer.** 11/31 MTAP-intact PDAC lines show strong PRMT5 dependency (dep < −1.2). PRMT5 is broadly essential in PDAC regardless of MTAP status. For vopimetostat pivotal trial biomarker strategy, PRMT5 dependency score or SDMA levels may be needed as complementary biomarkers beyond MTAP status alone.

7. **Breast cancer subtype data gap.** ER+/HER2+/TNBC stratification is unavailable in DepMap metadata. The strong aggregate effect (d = −1.73) may be driven by a specific breast cancer subtype. This is critical for clinical translation — PRMT5 inhibitor trials would need to specify a breast cancer subtype.

---

## 1. Methodology

### 1.1 MTAP Deletion Classification

Cell lines from DepMap 25Q3 were classified by MTAP copy number status using a threshold of < 0.5 copy number ratio (validated in our prior NSCLC analysis). Expression data confirmed a bimodal distribution. Cancer types with ≥ 5 MTAP-deleted AND ≥ 5 MTAP-intact lines qualified for analysis. **16 of 31 cancer types** met this threshold.

MTAP deletion frequencies in DepMap ranged from 54.9% (CNS/Brain) to 15.6% (Ovary/Fallopian Tube) among qualifying types.

### 1.2 Effect Size Computation

For each qualifying cancer type:
- **PRMT5 and MAT2A CRISPR dependency** compared between MTAP-deleted and MTAP-intact groups using Mann-Whitney U tests (two-sided).
- **Cohen's d** computed with pooled standard deviation and Bessel's correction.
- **95% bootstrap confidence intervals** (1,000 iterations, seed = 42, percentile method).
- **Benjamini-Hochberg FDR correction** across 16 cancer types (threshold: FDR < 0.05).

### 1.3 Robustness Validation

For the top 6 cancer types (FDR < 0.05 or d < −0.5):
- **Leave-one-out (LOO) analysis:** Removed each cell line individually and recomputed d. A result is sign-consistent if all LOO values remain negative.
- **Permutation testing:** 10,000 permutations of MTAP labels; computed empirical p-value.
- **Classification:** ROBUST requires FDR < 0.05, LOO sign consistency, AND permutation p < 0.05. MARGINAL means FDR-significant but permutation p ≥ 0.05.

### 1.4 Subtype Analysis

Cancer types where subtype-level biology matters (e.g., CNS/Brain) were stratified by OncotreeSubtype. The same statistical pipeline was applied within each subtype (≥ 5 deleted + ≥ 5 intact required). BH FDR correction was applied across all 14 testable subtypes.

### 1.5 Patient Population Estimates

TCGA Pan-Cancer Atlas MTAP copy number data (32 cancer types, 10,443 patients) provided homozygous deletion rates. These were multiplied by US cancer incidence (SEER 2024) to estimate eligible patients/year. Where available, C-CAT data (N = 51,828) provided cross-validation.

### 1.6 Clinical Concordance

DepMap PRMT5 SL rankings were compared to available clinical ORR data from vopimetostat, AMG 193, and BMS-986504 trials. Due to limited per-histology clinical data (N = 2 tumor types with ORR), this analysis is hypothesis-generating only.

---

## 2. PRMT5 Synthetic Lethality Rankings

### 2.1 Full Rankings

| Rank | Cancer Type | d | 95% CI | FDR | N (del/intact) | Classification |
|------|-------------|------|---------|------|-----------------|----------------|
| 1 | **Breast** | −1.73 | [−2.83, −0.90] | 0.003 | 8 / 45 | ROBUST |
| 2 | **Lung** | −1.23 | [−1.75, −0.76] | 4.9e-6 | 26 / 100 | ROBUST |
| 3 | **Soft Tissue** | −1.05 | [−1.86, −0.40] | 0.040 | 9 / 37 | ROBUST |
| 4 | **Esophagus/Stomach** | −0.97 | [−1.78, −0.31] | 0.032 | 10 / 59 | ROBUST |
| 5 | Ovary/Fallopian Tube | −0.90 | [−1.87, −0.12] | 0.123 | 7 / 52 | — |
| 6 | Kidney | −0.85 | [−2.12, +0.27] | 0.193 | 5 / 30 | — |
| 7 | Head and Neck | −0.71 | [−1.49, −0.04] | 0.193 | 8 / 69 | — |
| 8 | **CNS/Brain** | −0.68 | [−1.14, −0.27] | 0.013 | 37 / 52 | ROBUST |
| 9 | Myeloid | −0.64 | [−1.29, −0.07] | 0.105 | 16 / 27 | — |
| 10 | Liver | −0.62 | [−1.57, +0.24] | 0.353 | 5 / 19 | — |
| 11 | Bladder/Urinary Tract | −0.49 | [−1.33, +0.33] | 0.628 | 8 / 26 | — |
| 12 | Skin | −0.47 | [−0.99, −0.02] | 0.123 | 17 / 58 | — |
| 13 | **Pancreas** | −0.45 | [−1.40, +0.10] | 0.037 | 17 / 31 | MARGINAL |
| 14 | Lymphoid | −0.18 | [−0.93, +0.52] | 0.680 | 13 / 80 | — |
| 15 | Pleura | −0.12 | [−0.90, +0.75] | 0.750 | 8 / 13 | — |
| 16 | Bone | +0.01 | [−0.63, +0.72] | 0.680 | 14 / 35 | — |

*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/prmt5_forest_plot.png`*

### 2.2 ROBUST Cancer Types

Five cancer types passed all three robustness criteria (FDR, LOO, permutation):

**Breast (d = −1.73):** The strongest PRMT5 SL effect across all cancer types — stronger than the NSCLC reference (d = −1.23 at the lineage level). All 8 MTAP-deleted breast lines show PRMT5 dependency stronger than the intact median (−1.04). No outliers detected. LOO analysis shows HCC1806 (DCIS, dep = −0.92) as the least dependent; removing it *strengthens* d to −2.07. Permutation p < 0.001. However, N = 8 limits statistical power and the confidence interval is wide (−2.83 to −0.90). Histologically: 5 IDC, 2 IBC-NOS, 1 DCIS. ER/HER2/TNBC status unavailable in DepMap (see Caveat #7).

**Lung (d = −1.23):** The most statistically robust result (FDR = 4.9e-6) with the largest sample size (N = 126). LOO is extremely stable (max change = 0.15). Permutation p < 0.001. Clinically validated: BMS-986504 ORR 29% in NSCLC, vopimetostat included in histology-selective cohort (49% ORR). TCGA: 15.7% MTAP homozygous deletion (LUSC 19.5%, LUAD 12.1%), validated by C-CAT (14.3%). This represents ~37,494 eligible US patients/year — the largest addressable population.

**Soft Tissue (d = −1.05):** Strong effect confirmed by permutation p = 0.009. LOO is sign-consistent. FDR = 0.040. N = 9 deleted lines limits power. TCGA: 9.5% MTAP homdel, ~1,289 eligible patients/year. No PRMT5 inhibitor clinical data exists for sarcoma — vopimetostat histology-selective cohort explicitly excluded sarcoma.

**Esophagus/Stomach (d = −0.97):** Permutation p = 0.005. LOO sign-consistent. FDR = 0.032. AMG 193 included esophageal cancer in Phase 1 but no tumor-specific ORR reported. TCGA: 12.7% homdel (ESCA 20.9%, STAD 9.4%), ~6,086 eligible patients/year. C-CAT concordant (9.0%, within 3.7 pp).

**CNS/Brain (d = −0.68):** The most stable result by LOO (max change = 0.056) due to large N (N = 89). Permutation p < 0.001. FDR = 0.013. TCGA: 26.6% homdel (GBM 42.4%, LGG 8.8%), ~6,759 eligible patients/year. TNG456 (brain-penetrant PRMT5i) is enrolling GBM patients (NCT06810544). See Section 2.3 for GBM subtype analysis.

### 2.3 GBM Subtype Finding

At the OncotreeSubtype level, **Glioblastoma** (GBM) shows stronger PRMT5 SL than the CNS/Brain aggregate:

| Subtype | d | 95% CI | FDR | N (del/intact) |
|---------|------|---------|------|-----------------|
| Glioblastoma | −0.74 | [−1.41, −0.19] | 0.031 | 25 / 25 |
| Astrocytoma | −0.21 | — | n.s. | 12 / 27 |

Astrocytoma (d = −0.21) dilutes the CNS/Brain aggregate. The GBM-specific result is well-powered (balanced N = 25+25), FDR-significant across 14 subtypes, and confirmed by permutation (p = 0.005). TCGA shows 42.4% MTAP homozygous deletion in GBM — the highest of any individual TCGA study. This supports prioritizing GBM for brain-penetrant PRMT5 inhibitor development.

### 2.4 Pancreas: MARGINAL Classification

Pancreas (d = −0.45) reaches FDR significance (0.037) but fails permutation testing (p = 0.067). The weak SL differential is explained by **PRMT5 being broadly essential in pancreatic cancer regardless of MTAP status**: 11 of 31 MTAP-intact PDAC lines have strong PRMT5 dependency (dep < −1.2). The intact group has high variance (SD = 0.52 vs. 0.28 for deleted), with many intact lines as dependent as deleted lines.

This has direct biomarker implications. MTAP deletion alone is a weaker discriminating biomarker in PDAC than in other tumor types. The vopimetostat 2L PDAC ORR (25%) is consistent with this moderate SL differential — lower than the histology-selective ORR (49%) that reflects tumor types with stronger SL.

All 17 MTAP-deleted PDAC lines are histologically PDAC — histology dilution does not explain the weak effect.

---

## 3. MAT2A Synthetic Lethality Analysis

### 3.1 MAT2A Rankings

No cancer type shows FDR-significant MAT2A SL (0/16). The top-ranked types by MAT2A effect size are:

| Rank | Cancer Type | MAT2A d | FDR | PRMT5 d |
|------|-------------|---------|------|---------|
| 1 | Liver | −0.77 | 0.512 | −0.62 |
| 2 | Head and Neck | −0.69 | 0.232 | −0.71 |
| 3 | Skin | −0.62 | 0.232 | −0.47 |
| 4 | Esophagus/Stomach | −0.42 | 0.475 | −0.97 |
| 5 | Pancreas | −0.40 | 0.279 | −0.45 |

*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/mat2a_forest_plot.png`*

### 3.2 PRMT5 vs. MAT2A Comparison

PRMT5 and MAT2A SL effects are **uncorrelated** across cancer types (Spearman r = −0.11, p = 0.70). PRMT5 shows stronger SL than MAT2A in 12/16 cancer types.

*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/prmt5_vs_mat2a_scatter.png`*

### 3.3 Dual-Target Classification

Using a threshold of d < −0.5 for "strong" SL:

| Classification | Cancer Types | Count |
|----------------|-------------|-------|
| PRMT5-only | Breast, Lung, Soft Tissue, Esoph/Stomach, Ovary, Kidney, CNS/Brain, Myeloid | 8 |
| Dual-target (PRMT5 + MAT2A) | **Head and Neck**, **Liver** | 2 |
| MAT2A-only | Skin | 1 |
| Neither | Bladder, Pancreas, Lymphoid, Pleura, Bone | 5 |

**Implication for IDE892+IDE397 (PRMT5+MAT2A) combination:** The dual-target rationale is strongest in Head & Neck (PRMT5 d = −0.71, MAT2A d = −0.69) and Liver (PRMT5 d = −0.62, MAT2A d = −0.77). Critically, the top PRMT5 SL cancer types (Breast, Lung, Soft Tissue) are **not** dual-target candidates — they show weak or absent MAT2A SL.

---

## 4. 9p21 Co-Deletion Landscape

### 4.1 Co-Deletion Rates

CDKN2A is co-deleted with MTAP in 100% of MTAP-deleted lines across all 16 cancer types — confirming universal co-deletion. CDKN2B co-deletion ranges from 66.7% (Head and Neck) to 100% (Pleura, Bladder, Esophagus/Stomach, Breast, Liver). Full 9p21 locus deletion (including DMRTA1 and IFN cluster) varies more widely:

| Cancer Type | DMRTA1 co-del | IFNA2 co-del | IFNB1 co-del |
|-------------|---------------|--------------|--------------|
| Pleura | 87.5% | 50.0% | 25.0% |
| Liver | 80.0% | 60.0% | 40.0% |
| Pancreas | 76.5% | 52.9% | 41.2% |
| Kidney | 71.4% | 28.6% | 14.3% |
| Ovary | 71.4% | 42.9% | 28.6% |
| Breast | 70.0% | 50.0% | 40.0% |
| Myeloid | 70.0% | 50.0% | 40.0% |
| Bladder | 62.5% | 12.5% | 12.5% |
| CNS/Brain | 56.4% | 43.6% | 30.8% |
| Lung | 51.5% | 36.4% | 27.3% |

*Full table: `output/cancer/pancancer-mtap-prmt5-atlas/9p21_codeletion_rates.csv`*
*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/9p21_codeletion_heatmap.png`*

### 4.2 Deletion Extent and PRMT5 Dependency

Cell lines were classified by 9p21 deletion extent:
- **CDKN2A-only** (MTAP intact): N = 82
- **CDKN2A+MTAP** (focal deletion): N = 148
- **Full 9p21** (CDKN2A+MTAP+DMRTA1+IFN cluster): N = 69

PRMT5 dependency increases with deletion extent (Kruskal-Wallis H = 27.6, p = 1.0e-6):

| Deletion Extent | Median PRMT5 Dependency |
|-----------------|------------------------|
| CDKN2A-only | −1.091 |
| CDKN2A+MTAP (focal) | −1.258 |
| Full 9p21 | −1.306 |

However, the difference between focal and full 9p21 deletion is not statistically significant (Mann-Whitney p = 0.275). The primary driver of PRMT5 dependency is MTAP loss, not the extent of surrounding 9p21 deletion.

MAT2A shows a similar but weaker trend (Kruskal-Wallis p = 0.033; focal vs. full p = 0.082).

*Visualizations: `output/cancer/pancancer-mtap-prmt5-atlas/deletion_extent_prmt5_boxplot.png`, `deletion_extent_mat2a_boxplot.png`*

---

## 5. Patient Population Estimates

### 5.1 TCGA MTAP Deletion Frequencies

| Cancer Type | TCGA Study | N Profiled | Homdel % | Any Loss % | US Incidence | Eligible/Year |
|-------------|-----------|-----------|----------|-----------|-------------|---------------|
| GBM | GBM | 575 | 42.4% | 73.9% | — | — |
| Mesothelioma | MESO | 87 | 32.2% | 55.2% | 2,500 | 804 |
| Bladder | BLCA | 408 | 26.0% | 56.6% | 83,190 | 21,613 |
| Pancreas | PAAD | 183 | 21.9% | 58.5% | 66,440 | 14,522 |
| Esophageal | ESCA | 182 | 20.9% | 68.1% | — | — |
| Skin (Melanoma) | SKCM | 367 | 19.6% | 73.8% | 100,640 | 19,744 |
| Lung (Squamous) | LUSC | 487 | 19.5% | 76.0% | — | — |
| Lung (combined) | LUSC+LUAD | 998 | 15.7% | 66.0% | 238,340 | 37,494 |
| Head and Neck | HNSC | 517 | 14.5% | 49.7% | 55,970 | 8,119 |
| Esoph/Stomach (combined) | ESCA+STAD | 620 | 12.7% | 50.2% | 47,770 | 6,086 |
| Soft Tissue (Sarcoma) | SARC | 253 | 9.5% | 41.9% | 13,590 | 1,289 |
| Breast | BRCA | 1,070 | 3.3% | 33.0% | 310,720 | 10,163 |
| CNS/Brain (combined) | GBM+LGG | 1,086 | 26.6% | 55.2% | 25,400 | 6,759 |

*Full table: `output/cancer/pancancer-mtap-prmt5-atlas/tcga_mtap_frequencies.csv`*
*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/population_bar_chart.png`*

### 5.2 TCGA–C-CAT Cross-Validation

Eight cancer types had both TCGA and C-CAT (Suzuki et al., ESMO Open 2025, N = 51,828) deletion frequencies. Concordance is moderate (Spearman r = 0.66, p = 0.076).

| Cancer Type | TCGA Homdel % | C-CAT Homdel % | Difference |
|-------------|--------------|----------------|------------|
| Lung | 15.7% | 14.3% | 1.4 pp |
| Breast | 3.3% | 3.0% | 0.3 pp |
| Pancreas | 21.9% | 18.4% | 3.5 pp |
| Esoph/Stomach | 12.7% | 9.0% | 3.7 pp |
| Head and Neck | 14.5% | 10.0% | 4.5 pp |
| Biliary Tract | 11.1% | 15.6% | 4.5 pp |
| Ovary | 2.6% | 3.0% | 0.4 pp |
| **Bladder** | **26.0%** | **11.0%** | **15.0 pp** |

Bladder/Urinary Tract is the sole discordant type (15 percentage point difference). C-CAT (N = 51,828) is likely more accurate than TCGA (N = 408) for Bladder, though cross-population bias may contribute (see Caveat #5). Patient population estimates for Bladder should be treated with caution.

---

## 6. Clinical Concordance

### 6.1 Available Clinical Data

| Drug | Company | Phase | Per-Histology ORR Available |
|------|---------|-------|--------------------------|
| Vopimetostat | Tango | Pivotal (PDAC) | Pan-cancer 27%, selective 49%, 2L PDAC 25% |
| AMG 193 | Amgen | Phase 1 | Pan-cancer 21.4%. Responses in 8 types (sqNSCLC, nsqNSCLC, PDAC, biliary, esophageal, renal, gallbladder, ovarian Sertoli-Leydig) |
| BMS-986504 | BMS | Phase 2/3 | NSCLC 29% (N=35). MountainTAP-29 (1L NSCLC) and MountainTAP-30 (1L PDAC) initiated |
| AZD3470 | AstraZeneca | Phase 1/2a | No data disclosed (overdue, est. Feb 2026 completion) |
| IDE892 | IDEAYA | Phase 1 | FPI March 2026. No efficacy data yet |
| TNG456 | Tango | Phase 1 | Enrolling GBM (NCT06810544). No data yet |

### 6.2 DepMap vs. Clinical ORR

Only two tumor types have per-histology ORR data:

| Cancer Type | DepMap PRMT5 d | Clinical ORR | Drug |
|-------------|---------------|-------------|------|
| Lung (NSCLC) | −1.23 | 29% | BMS-986504 |
| Pancreas (PDAC) | −0.45 | 25% | Vopimetostat |

The direction is concordant: stronger DepMap SL (Lung d = −1.23 > Pancreas d = −0.45) corresponds to higher ORR (29% > 25%). However, **N = 2 is insufficient to establish a predictive correlation.** These are different drugs, different dosing, different patient populations, and different lines of therapy. This concordance is hypothesis-generating only (see Caveat #4).

The vopimetostat histology-selective cohort (49% ORR, excluding PDAC/NSCLC/sarcoma, 13 histologies) provides directional support: tumor types other than PDAC and NSCLC respond well, consistent with the atlas finding that Pancreas ranks 13th while several other types rank higher.

*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/clinical_concordance_plot.png`*

---

## 7. Integrated Priority Rankings

### 7.1 Priority Matrix

Cancer types were ranked by combining PRMT5 SL effect size rank and patient population rank:

| Cancer Type | SL Rank | Pop Rank | Combined Rank | Notes |
|-------------|---------|----------|---------------|-------|
| **Lung** | 2 | 1 | **2** | Clinically validated, largest population |
| **Breast** | 1 | 5 | **5** | Strongest SL, underexplored |
| Bladder | 11 | 2 | 22 | Weak SL, large population |
| Esoph/Stomach | 4 | 8 | 32 | ROBUST SL, moderate population |
| Soft Tissue | 3 | 11 | 33 | ROBUST SL, small population |
| Skin | 12 | 3 | 36 | Weak SL |
| Head and Neck | 7 | 6 | 42 | Moderate SL, dual-target candidate |
| Pancreas | 13 | 4 | 52 | MARGINAL, large population |
| CNS/Brain | 8 | 7 | 56 | ROBUST, GBM subtype stronger |
| Kidney | 6 | 10 | 60 | Moderate SL, low homdel rate |
| Ovary | 5 | 14 | 70 | Strong SL, small MTAP-del population |

*Visualization: `output/cancer/pancancer-mtap-prmt5-atlas/priority_matrix.png`*

### 7.2 Trial Expansion Recommendations

**Underexplored cancer types** (strong PRMT5 SL, no clinical PRMT5i data):

| Priority | Cancer Type | PRMT5 d | FDR | Eligible/Year | Rationale |
|----------|-------------|---------|------|---------------|-----------|
| **HIGH** | **Breast** | −1.73 | 0.003 | 10,163 | Strongest SL, large population, no trials |
| **HIGH** | **Soft Tissue** | −1.05 | 0.040 | 1,289 | ROBUST SL, excluded from vopimetostat selective cohort |
| MEDIUM | Ovary | −0.90 | 0.123 | 516 | Strong SL, small population |
| MEDIUM | Kidney | −0.85 | 0.193 | 1,344 | AMG 193 showed renal response |
| MEDIUM | Liver | −0.62 | 0.353 | 1,134 | Dual-target candidate (MAT2A d = −0.77) |
| MEDIUM | Myeloid | −0.64 | 0.105 | 124 | Small addressable population |

*Data: `output/cancer/pancancer-mtap-prmt5-atlas/underexplored_cancer_types.csv`*

---

## 8. Competitive Landscape (March 2026)

Six MTA-cooperative PRMT5 inhibitors are in clinical development:

| Compound | Company | Stage | Key Data | Tumor Types |
|----------|---------|-------|----------|-------------|
| **Vopimetostat** | Tango | Pivotal planned (2L PDAC) | Pan-cancer ORR 27%, selective 49%, PDAC 25% | PDAC, NSCLC, histology-agnostic |
| **AMG 193** | Amgen | Phase 1 | ORR 21.4% pan-cancer | 8 responding histologies |
| **BMS-986504** | BMS | Phase 2/3 | ORR 29% NSCLC (Phase 1); MountainTAP-29 (1L NSCLC), -30 (1L PDAC) | NSCLC, PDAC |
| **AZD3470** | AstraZeneca | Phase 1/2a (PRIMROSE) | No data disclosed (overdue) | Solid tumors, heme |
| **IDE892** | IDEAYA | Phase 1 (FPI March 2026) | ~1,400x MTA selectivity, single-digit nM potency | NSCLC, PDAC |
| **TNG456** | Tango | Phase 1 | Brain-penetrant | GBM |

**Combination strategies evolving rapidly:**
- **PRMT5i + RAS inhibitor:** Vopimetostat + ERAS-0015 (pan-RAS molecular glue) enrolling in MTAP-del PDAC and MTAP-del/RAS-mut NSCLC. BMS-986504 + KRAS inhibitor shown preclinically in MTAP-del KRAS-mut PDAC. Nearly all MTAP-del PDAC harbor co-occurring RAS mutations.
- **PRMT5i + MAT2A inhibitor:** IDE892 + IDE397 combo FPI targeted mid-2026. Preclinical data showed durable complete responses at sub-monotherapy doses.
- **PRMT5i + IO/chemo:** MountainTAP-29 (BMS-986504 + pembrolizumab + chemo, 1L NSCLC, N = 590) and MountainTAP-30 (BMS-986504 + nab-P/gem, 1L PDAC, N = 470) are the first Phase 2/3 randomized PRMT5i trials.
- **CDKN2A targeting:** IDEAYA targeting development candidate nomination H2 2026, IND H1 2027. >80% prevalence in PDAC. Relevant given universal CDKN2A co-deletion with MTAP.

---

## 9. Interpretation and Next Steps

### 9.1 Key Biological Insights

1. **PRMT5 SL is tissue-specific, not uniform.** Effect sizes range from d = −1.73 (Breast) to d = +0.01 (Bone). This 1.74 standard deviation spread means tumor type is a critical variable for PRMT5 inhibitor efficacy — consistent with vopimetostat showing 49% ORR in selected histologies vs. 27% pan-cancer.

2. **PRMT5 and MAT2A SL are independent.** The lack of correlation (r = −0.11) means PRMT5 and MAT2A target different aspects of MTAP-deletion vulnerability. Most tumor types are PRMT5-only responders. The IDE892+IDE397 combination has strongest rationale in Head & Neck and Liver, not the top PRMT5 SL types.

3. **Pancreas is an imperfect index tumor for PRMT5 inhibitors.** Despite being the lead indication for vopimetostat (pivotal trial) and BMS-986504 (MountainTAP-30), PDAC has only MARGINAL PRMT5 SL (d = −0.45, rank 13/16). This is because PRMT5 is broadly essential in PDAC regardless of MTAP status. Breast, Lung, Soft Tissue, and Esophagus/Stomach all show stronger SL differentials.

4. **GBM has a distinct and actionable PRMT5 SL signal.** GBM-specific analysis (d = −0.74) is stronger than the CNS/Brain aggregate (d = −0.68) and has the highest MTAP homozygous deletion rate of any TCGA study (42.4%). The brain-penetrant PRMT5 inhibitor TNG456 is well-positioned.

5. **MTAP loss, not deletion extent, drives PRMT5 dependency.** Full 9p21 locus deletion does not create significantly stronger PRMT5 vulnerability than focal CDKN2A+MTAP deletion (p = 0.275). This simplifies biomarker strategy: MTAP deletion status is sufficient.

### 9.2 Recommended Follow-Up

- **Breast cancer PRMT5i trial exploration.** The strongest SL signal with ~10,000 eligible US patients/year and no clinical data. Breast cancer subtype resolution (ER+/HER2+/TNBC) is the critical next step — this may require linking DepMap cell lines to external molecular subtype annotations.
- **Soft tissue/sarcoma clinical signal.** ROBUST SL (d = −1.05) but explicitly excluded from vopimetostat histology-selective cohort. Warrants dedicated clinical investigation.
- **Per-histology clinical ORR data.** Vopimetostat per-histology ORR breakdowns (within the 49% selective cohort) and AMG 193 per-histology data would enable direct validation of atlas rankings. Expected at ASCO 2026 (May 30–Jun 3).
- **RAS co-mutation analysis.** The ERAS-0015+vopimetostat combination trial motivates quantifying RAS co-mutation rates by tumor type to predict which histologies benefit most from PRMT5i+RAS inhibitor combinations.

---

## Data Availability

All output files are in `output/cancer/pancancer-mtap-prmt5-atlas/`:

| File | Description |
|------|-------------|
| `prmt5_effect_sizes.csv` / `.json` | PRMT5 SL effect sizes by cancer type |
| `prmt5_forest_plot.png` | Forest plot of PRMT5 SL rankings |
| `mat2a_effect_sizes.csv` | MAT2A SL effect sizes by cancer type |
| `mat2a_forest_plot.png` | Forest plot of MAT2A SL rankings |
| `prmt5_vs_mat2a_scatter.png` | PRMT5 vs. MAT2A SL comparison |
| `dual_target_comparison.json` | Dual-target classification |
| `cancer_type_summary.csv` | Master summary of qualifying cancer types |
| `9p21_codeletion_rates.csv` | Co-deletion rates by cancer type |
| `9p21_codeletion_heatmap.png` | Co-deletion heatmap |
| `deletion_extent_by_cancer_type.csv` | Deletion extent classification |
| `deletion_extent_dependency.json` | Deletion extent vs. dependency stats |
| `deletion_extent_prmt5_boxplot.png` | PRMT5 dependency by deletion extent |
| `deletion_extent_mat2a_boxplot.png` | MAT2A dependency by deletion extent |
| `tcga_mtap_frequencies.csv` | TCGA MTAP deletion frequencies |
| `tcga_vs_ccat_validation.json` | TCGA–C-CAT cross-validation |
| `patient_population_estimates.csv` | Addressable patient estimates |
| `population_bar_chart.png` | Patient population bar chart |
| `priority_ranking.csv` | Integrated priority rankings |
| `priority_matrix.png` | Priority matrix visualization |
| `clinical_concordance.json` | Clinical concordance data |
| `clinical_concordance_plot.png` | DepMap vs. clinical ORR |
| `underexplored_cancer_types.csv` | Underexplored opportunities |
| `trial_expansion_recommendations.json` | Trial expansion priorities |
| `all_cell_lines_classified.csv` | Full cell line classification |

---

*Pipeline: `src/pancancer_mtap_prmt5_atlas/` (scripts 01–07 + phase3 modules)*
*Analyst validation: journal #384*
*Validation scientist approval: journal #507*
*Research plan: `plans/cancer/pancancer-mtap-prmt5-atlas.md`*
