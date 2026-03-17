# PLAN: MTAP/PRMT5 Synthetic Lethality in NSCLC

## Objective

Determine whether MTAP-deleted NSCLC cell lines show enhanced PRMT5 dependency in DepMap 25Q3, and identify co-mutation contexts (KRAS alleles, STK11, KEAP1, TP53) that modulate this synthetic lethal relationship to guide patient selection for PRMT5 inhibitor trials.

## Background

MTAP (methylthioadenosine phosphorylase) is co-deleted with CDKN2A on chromosome 9p21 in ~15-18% of NSCLC. MTAP loss leads to accumulation of methylthioadenosine (MTA), which selectively inhibits PRMT5 enzymatic activity, creating a well-validated synthetic lethal dependency. PRMT5 MTA-cooperative inhibitors (IDE892/AMG 193) exploit this by being activated by accumulated MTA — they are more potent in MTAP-deleted cells.

IDE892 entered Phase 1 in March 2026 for advanced solid tumors including NSCLC. However, it remains unclear which NSCLC molecular subtypes show the strongest PRMT5 dependency within the MTAP-deleted population. Co-occurring mutations in KRAS, STK11, KEAP1, and TP53 may modulate the SL relationship. Understanding these interactions could identify the highest-responding patient subpopulations for ongoing clinical trials.

Key gap: Most MTAP/PRMT5 SL studies are pan-cancer. A focused NSCLC analysis incorporating the KP/KL/KOnly subtype framework has not been done.

## Data Sources

All data already available locally:
- **DepMap 25Q3 CRISPRGeneEffect.csv** — PRMT5 and related gene dependency scores across ~1,900 cell lines
- **DepMap 25Q3 PortalOmicsCNGeneLog2.csv** — MTAP copy number for deletion calling (log2 ratio < -1)
- **DepMap 25Q3 OmicsSomaticMutations.csv** — MTAP/KRAS/STK11/KEAP1/TP53 mutation status
- **DepMap 25Q3 OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv** — MTAP expression for validation
- **DepMap 25Q3 Model.csv** — Cell line cancer type annotations
- **PRISM 24Q2** — Drug sensitivity data; check for GSK3326595 (PRMT5i) or related compounds
- **TCGA LUAD/LUSC** — Already downloaded (`data/tcga/luad/`, `data/tcga/lusc/`): MTAP deletion frequency across NSCLC subtypes
- **TCGA patient subtypes** — `data/tcga/nsclc_patient_subtypes.csv`: KP/KL/KOnly classification

## Methodology

### Phase 1: MTAP-PRMT5 Dependency Validation
1. Identify MTAP-deleted cell lines in DepMap using copy number (log2 < -1) and/or expression (bottom quartile) thresholds
2. Subset to NSCLC lines (~95 available) and classify MTAP status (deleted/intact)
3. Compare PRMT5 CRISPR dependency scores between MTAP-deleted vs MTAP-intact NSCLC lines (Wilcoxon rank-sum test)
4. Compute effect size (Cohen's d) and 95% CI
5. Pan-cancer positive control: validate the same SL relationship across all DepMap lines (known strong effect)

### Phase 2: Co-Mutation Modulation Analysis
1. Within MTAP-deleted NSCLC lines, stratify by KRAS (allele-specific: G12C/G12D/G12V/WT), STK11, KEAP1, TP53
2. Test whether each co-mutation significantly modulates PRMT5 dependency (Kruskal-Wallis across subgroups)
3. Build multivariate model: PRMT5_dependency ~ MTAP_status + KRAS_allele + STK11 + KEAP1 + TP53 + interaction terms
4. Report significant interaction terms with effect sizes

### Phase 3: Extended SL Network
1. Test related methionine salvage pathway genes for dependency in MTAP-deleted NSCLC: PRMT1, PRMT7, MAT2A, MAT2B, SRM, SMS
2. Genome-wide differential dependency screen: identify ALL genes with significantly stronger dependency in MTAP-deleted vs intact NSCLC (FDR < 0.05)
3. PRISM validation: if GSK3326595 or another PRMT5i is in PRISM, validate with drug sensitivity data

### Phase 4: TCGA Patient Population Estimate
1. Map MTAP deletion frequency in TCGA LUAD and LUSC using CN and mutation data
2. Stratify by KP/KL/KOnly subtypes — is MTAP deletion enriched in specific oncogenic contexts?
3. Estimate patient population: MTAP-deleted × favorable co-mutation context = PRMT5i-eligible NSCLC patients
4. Survival analysis: MTAP deletion prognostic impact in TCGA NSCLC

## Expected Outputs

- Statistical quantification of PRMT5 SL in MTAP-deleted vs intact NSCLC (effect size, p-value, N)
- Co-mutation modifier map: which molecular contexts enhance or suppress PRMT5 dependency
- Extended SL gene list beyond PRMT5 for MTAP-deleted NSCLC
- TCGA patient population estimate for PRMT5i-eligible NSCLC subpopulations
- Drug sensitivity validation (if PRISM data available)
- Visualization: dependency score distributions, co-mutation interaction plots, TCGA subtype frequencies

## Success Criteria

1. PRMT5 shows significantly stronger dependency (p < 0.01, Cohen's d > 0.5) in MTAP-deleted vs intact NSCLC lines
2. At least one co-mutation context significantly modulates PRMT5 dependency (p < 0.05)
3. TCGA analysis confirms ≥10% NSCLC patients carry MTAP deletion
4. Findings extend existing MTAP/PRMT5 literature with NSCLC-specific subtype stratification not previously reported

## Labels

drug-candidate, genomic, drug-screening, novel-finding
