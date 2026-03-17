# CDKN2A Deletion Pan-Cancer Dependency Atlas

## Objective

Map gene dependencies and drug sensitivities in CDKN2A-deleted vs intact cell lines across all cancer types in DepMap 25Q3, to identify which tumor types show the strongest CDK4/6 dependency, discover novel non-CDK4/6 vulnerabilities, and inform expansion of CDK4/6 inhibitor indications beyond breast cancer.

## Background

CDKN2A (encoding p16/INK4a and p14/ARF) is the second most commonly deleted tumor suppressor in human cancer (~10-15% pan-cancer, >30% in GBM, mesothelioma, pancreatic, bladder). Homozygous deletion eliminates the p16-CDK4/6-RB cell cycle checkpoint, creating a well-characterized synthetic lethal relationship with CDK4/6.

**CDK4/6 inhibitor landscape (March 2026):**
- **Palbociclib** (Pfizer), **ribociclib** (Novartis), **abemaciclib** (Lilly): FDA-approved for HR+/HER2- breast cancer
- CDK4/6i trials in CDKN2A-deleted GBM, mesothelioma, liposarcoma, NSCLC — mixed results
- Patient selection beyond breast cancer is unclear — which CDKN2A-deleted cancer types actually depend on CDK4/6?

**Key gaps this atlas addresses:**
1. No systematic pan-cancer ranking of CDK4/6 dependency strength in CDKN2A-deleted lines
2. Unknown whether non-CDK4/6 vulnerabilities emerge in CDKN2A-deleted contexts (RB1-independent, ARF/MDM2 axis)
3. Co-mutation contexts (RB1 co-loss, TP53 status) that modulate CDK4/6 dependency are uncharacterized at scale
4. PRISM drug sensitivity data may reveal CDK4/6i selectivity patterns not captured by CRISPR

**Cross-project links:**
- TP53 atlas: CDKN2A encodes p14/ARF which activates p53 via MDM2 inhibition. CDKN2A deletion disrupts both RB and p53 pathways. Cross-reference with TP53 allele-specific dependencies.
- NSCLC project: CDKN2A deletion common in NSCLC (~25% squamous). May identify NSCLC subgroup-specific vulnerabilities.

## Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv — CDK4, CDK6, RB1, CDK2, CCND1, CCNE1, E2F1, MDM2 and genome-wide dependency scores
- **DepMap copy number:** PortalOmicsCN.csv — CDKN2A copy number for deletion calling (log2 CN ≤ -1.0 for deep deletion)
- **DepMap mutations:** OmicsSomaticMutations.csv — RB1, TP53, CDKN2B co-mutation status
- **DepMap model annotations:** Model.csv — cancer type classification
- **PRISM 24Q2:** CDK4/6 inhibitor sensitivity (palbociclib, ribociclib, abemaciclib) + genome-wide drug screen
- **TCGA:** CDKN2A deletion frequency by cancer type, co-occurring alterations, survival associations

## Methodology

### Phase 1 — CDKN2A Deletion Classifier
- Classify all DepMap lines by CDKN2A status using copy number data (deep deletion vs diploid)
- Cross-validate with CDKN2A expression (deleted lines should have near-zero expression)
- Annotate RB1 co-loss (important: RB1 loss abolishes CDK4/6 dependency)
- Annotate TP53 status (ARF pathway context)
- Report qualifying cancer types (≥5 CDKN2A-deleted + ≥10 intact with CRISPR data)
- Power analysis per cancer type

### Phase 2 — CDK4/6 Dependency Effect Sizes by Cancer Type
- For each qualifying type: compare CDK4, CDK6 CRISPR dependency in CDKN2A-deleted vs intact
- Statistics: Cohen's d, Mann-Whitney U, bootstrap 95% CI (1000 iterations, seed=42), BH FDR
- **Critical control:** Stratify by RB1 status — CDK4/6 dependency requires intact RB1
- Control genes: CDK2, CDK1 (should NOT show CDKN2A-selective dependency)
- Classify ROBUST/MARGINAL/NOT SIGNIFICANT per cancer type
- Leave-one-out robustness for small-N types

### Phase 3 — Genome-Wide CDKN2A-Selective Dependency Screen
- Screen all ~18K genes for CDKN2A-deleted-specific dependencies per qualifying cancer type
- BH FDR < 0.05, |Cohen's d| > 0.3 thresholds
- Pathway enrichment (cell cycle, RB/E2F, MDM2/p53, CDK signaling)
- SL benchmark enrichment with known CDKN2A SL pairs
- Anti-correlation analysis for novel SL candidates

### Phase 4 — PRISM Drug Sensitivity
- CDK4/6 inhibitors (palbociclib, ribociclib, abemaciclib): CDKN2A-deleted vs intact sensitivity
- Genome-wide PRISM screen for CDKN2A-selective drug sensitivities
- CRISPR-PRISM concordance for CDK4/6 axis
- Identify non-CDK4/6 drugs with CDKN2A selectivity

### Phase 5 — TCGA Clinical Integration
- CDKN2A deletion frequency per cancer type from TCGA
- Estimate addressable patient populations (CDKN2A-del frequency × cancer incidence)
- Priority ranking: SL effect size × population size × druggability
- Co-mutation landscape by cancer type
- Map to active CDK4/6i clinical trials (ClinicalTrials.gov)

## Expected Outputs

- Pan-cancer ranking of CDK4/6 dependency in CDKN2A-deleted contexts
- Cancer-type-specific CDKN2A vulnerability catalogs
- RB1 co-loss impact quantification (biomarker for CDK4/6i exclusion)
- Novel non-CDK4/6 therapeutic targets in CDKN2A-deleted cancers
- PRISM drug sensitivity profiles
- Clinical population estimates and trial recommendations

## Success Criteria

- CDK4/6 positive control: ROBUST SL in ≥2 cancer types with RB1-intact context
- ≥1 novel non-CDK4/6 dependency identified (FDR < 0.05, |d| > 0.5) in ≥2 cancer types
- RB1 co-loss clearly abolishes CDK4/6 dependency (negative control)
- Clinical population estimates for ≥5 cancer types

## Labels

genomic, drug-screening, drug-candidate, novel-finding, clinical
