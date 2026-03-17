# Pan-Cancer MTAP/PRMT5 Synthetic Lethality Atlas

## Objective
Rank all cancer types by PRMT5 synthetic lethal effect size in MTAP-deleted cell lines to predict which tumor types will respond best to MTA-cooperative PRMT5 inhibitors (vopimetostat, AMG 193, IDE892, BMS-986504, AZD3470), and estimate the addressable patient population per cancer type.

## Background
Our mtap-prmt5-nsclc-sl initiative demonstrated that PRMT5 SL in NSCLC (Cohen's d = -1.19, p = 2.0e-6) is STRONGER than the pan-cancer average (d = -0.69). This tissue-specificity has direct clinical implications: vopimetostat shows 49% ORR in a "histology-selective" cohort vs 27% pan-cancer, confirming that tumor type determines PRMT5i efficacy.

Five MTA-cooperative PRMT5 inhibitors are now in clinical trials across multiple cancer types, but no systematic comparison of PRMT5 SL strength across tumor types exists using modern DepMap 25Q3 data. The pivotal vopimetostat trial targets pancreatic cancer first — is that the strongest PRMT5 SL tumor type, or are there better candidates?

Key prior findings from mtap-prmt5-nsclc-sl:
- PRMT5 is the ONLY genuine genome-wide SL hit for MTAP deletion (FDR=0.012 in NSCLC)
- No co-mutation modulates the SL relationship (all p > 0.5)
- GSK3326595 (substrate-competitive) shows no MTAP-selective effect — supports MTA-cooperative mechanism
- MTAP homozygous deletion: 15.6% of NSCLC (LUAD 12.0%, LUSC 18.6%)

## Data Sources
- **DepMap 25Q3:** CRISPR dependency, copy number (MTAP deletion classification), expression (MTAP/PRMT5 bimodal validation), mutations
- **TCGA pan-cancer:** MTAP deletion frequencies per cancer type (32 cancer types, 10,443 patients — already downloaded by data_curator, task #70)
- **Clinical benchmarks:** Vopimetostat (27%/49% ORR), BMS-986504 (29% NSCLC ORR), AMG 193 (~21% pan-cancer)
- **mtap-prmt5-nsclc-sl outputs:** NSCLC analysis as reference/validation

## Methodology

### Phase 1: Pan-Cancer MTAP Classification
- Classify MTAP deletion status across ALL DepMap cell lines using copy number (threshold < 0.5 ratio, validated in NSCLC analysis)
- Cross-validate with expression (bimodal distribution expected)
- Report: MTAP deletion frequency per cancer type, minimum N requirements
- Filter: cancer types with ≥5 MTAP-deleted AND ≥5 MTAP-intact lines (statistical power)

### Phase 2: Cancer-Type-Specific PRMT5 and MAT2A SL Effect Sizes
- For each qualifying cancer type:
  - Compute PRMT5 CRISPR dependency: MTAP-deleted vs intact (Mann-Whitney U, Cohen's d)
  - Compute MAT2A CRISPR dependency: MTAP-deleted vs intact (Mann-Whitney U, Cohen's d)
  - Compute 95% CI on effect sizes for both targets
- Rank cancer types by PRMT5 SL strength (Cohen's d)
- Rank cancer types by MAT2A SL strength (Cohen's d)
- Compare MAT2A vs PRMT5 rankings — do the same cancer types benefit from both targets?
- Forest plot visualization: PRMT5 and MAT2A effect sizes ± CI across all cancer types
- Compare each cancer type to NSCLC reference (PRMT5 d = -1.19)
- Identify cancer types where PRMT5 SL is STRONGER than NSCLC
- Note: Ensure glioma/GBM lines are included if they meet N≥5 threshold (TNG456 brain-penetrant PRMT5i entering clinic, NCT06810544)
- Rationale for MAT2A: IDEAYA IDE892+IDE397 combination trial (Q2 2026) targets both PRMT5 and MAT2A. If MAT2A SL varies by cancer type differently from PRMT5, this identifies which tumors benefit from dual vs mono inhibition

### Phase 3: 9p21 Co-Deletion Landscape and Deletion-Extent Classification
- For each cancer type, quantify co-deletion rates:
  - CDKN2A (always co-deleted with MTAP in NSCLC — universal or tissue-specific?)
  - CDKN2B
  - IFNA/IFNB cluster (immune implications)
  - DMRTA1
- Do co-deletion patterns vary across cancer types?
- **Deletion-extent classification:** Classify each cell line by 9p21 deletion extent:
  - CDKN2A-only (MTAP intact)
  - CDKN2A+MTAP (focal)
  - Full 9p21 locus (CDKN2A+MTAP+DMRTA1+IFN cluster)
- Compare PRMT5 and MAT2A dependency profiles across deletion-extent groups
- Test whether full locus deletion creates additional vulnerabilities beyond MTAP-specific PRMT5 SL
- Rationale: IDEAYA CDKN2A IND expected end-2026 — understanding deletion-extent-specific vulnerabilities informs triple-pathway targeting (PRMT5+MAT2A+CDKN2A)

### Phase 4: TCGA Patient Population Estimates
- Use TCGA pan-cancer MTAP CN deletion frequencies (data/tcga/pancancer_mtap_cn/, task #103 approved)
- For each cancer type: MTAP homozygous deletion rate × US incidence → eligible patients/year
- Rank by addressable patient population
- Cross-validate TCGA frequencies with Suzuki et al. C-CAT dataset (51,828 patients, ESMO Open 2025): PDAC 18.4%, biliary 15.6%, lung 14.3%
- Combine with Phase 2 ranking: prioritize cancer types with BOTH strong SL AND large patient populations

### Phase 5: Clinical Concordance Analysis
- Cross-reference DepMap PRMT5 SL ranking with available clinical data:
  - Vopimetostat: which histologies were in the 49% ORR "selective" cohort vs excluded?
  - BMS-986504: any tumor-type-specific response rates?
  - AMG 193: any tumor-type breakdowns?
- Assess whether DepMap SL strength predicts clinical response magnitude
- Identify "underexplored" cancer types: strong DepMap SL but no clinical data yet

## Expected Outputs
- Cancer-type ranking by PRMT5 SL effect size (forest plot)
- Cancer-type ranking by MAT2A SL effect size (forest plot)
- PRMT5 vs MAT2A SL comparison (scatter plot, dual-target opportunity matrix)
- TCGA MTAP deletion frequencies per cancer type (with C-CAT cross-validation)
- Patient population estimates per cancer type
- 9p21 co-deletion landscape and deletion-extent classification across cancer types
- Clinical concordance analysis
- Priority recommendations for PRMT5i trial expansion and dual-target (PRMT5+MAT2A) opportunities

## Success Criteria
- At least 8 cancer types with ≥5 MTAP-deleted lines (statistical power)
- At least 5 cancer types with significant (p < 0.05) PRMT5 SL effect
- Identification of ≥2 cancer types with PRMT5 SL comparable to or stronger than NSCLC
- Concordance between DepMap ranking and available clinical response data (directional agreement)

## Labels
drug-candidate, genomic, drug-screening, novel-finding, clinical
