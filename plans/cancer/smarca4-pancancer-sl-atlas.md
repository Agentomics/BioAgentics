# SMARCA4/BRG1-Deficient Pan-Cancer Synthetic Lethality Atlas

## Objective
Map synthetic lethal dependencies in SMARCA4 (BRG1)-deficient cell lines across all cancer types in DepMap 25Q3, to identify which tumor types respond best to SMARCA2 degraders and discover novel therapeutic targets beyond the SMARCA2/BRG1 paralog axis.

## Background
SMARCA4 (BRG1) is a core catalytic subunit of the SWI/SNF chromatin remodeling complex, lost in 5-10% of NSCLC, nearly all SCCOHT (small cell carcinoma of the ovary, hypercalcemic type), and subsets of bladder, endometrial, and other cancers. SMARCA4 loss creates a well-validated synthetic lethal dependency on its paralog SMARCA2 (BRM) — BRM is the remaining catalytic subunit and becomes essential when BRG1 is absent.

**Clinical momentum:** Multiple SMARCA2 degraders are entering the clinic:
- **PRT3789** (Prelude Therapeutics): Phase 1/2, SMARCA2 degrader for SMARCA4-mutant solid tumors
- **FHD-609** (Foghorn Therapeutics): Phase 1, BRM degrader for SMARCA4-loss synovial sarcoma and SCCOHT
- **GS-5829** and other BET inhibitors showing preclinical SWI/SNF-mutant activity

**Gap:** While SMARCA2 is the canonical SL target, it is unknown:
1. Which cancer types beyond NSCLC and SCCOHT show the strongest differential SMARCA2 dependency
2. Whether other genes show SMARCA4-specific SL that could serve as combination partners
3. How co-occurring mutations (TP53, KRAS, STK11) modulate the SL relationship
4. What the addressable patient population is per cancer type

This atlas directly parallels our ARID1A-SL work — both are SWI/SNF complex members, and findings may reveal shared vs. distinct SWI/SNF vulnerabilities.

## Data Sources
- **DepMap 25Q3 CRISPR (CRISPRGeneEffect.csv):** Genome-wide dependency scores
- **DepMap 25Q3 OmicsSomaticMutations.csv:** SMARCA4 mutations (LOF: frameshift, nonsense, splice site)
- **DepMap 25Q3 OmicsCNGene.csv:** SMARCA4 homozygous deletions
- **DepMap 25Q3 OmicsExpressionProteinCodingGenesTPMLogp1.csv:** SMARCA4/SMARCA2 expression confirmation
- **DepMap 25Q3 Model.csv:** Cell line metadata (OncotreeLineage, OncotreePrimaryDisease, OncotreeSubtype)
- **PRISM 24Q2 drug sensitivity:** Drug response in SMARCA4-deficient lines
- **TCGA pan-cancer:** SMARCA4 mutation/deletion frequencies per cancer type
- **PDB/UniProt:** SMARCA2 degrader binding sites for structural context

## Methodology

### Phase 1: SMARCA4 Cell Line Classification
- Classify DepMap lines as SMARCA4-deficient (LOF mutation OR homozygous deletion) vs. intact
- Confirm with expression: SMARCA4 TPM < 1.0 in deficient lines
- Annotate by OncotreeLineage and OncotreeSubtype (learn from MTAP atlas — use both levels)
- Report N per cancer type; require minimum 5 deficient + 10 intact for inclusion

### Phase 2: Known SL Effect Sizes
- Calculate SMARCA2 dependency differential (deficient vs. intact) per cancer type
- Mann-Whitney U test, Cohen's d effect size, bootstrap 95% CI
- BH-FDR correction across cancer types
- Rank cancer types by SMARCA2 SL strength
- Add ARID1A, ARID1B, PBRM1 (other SWI/SNF members) as secondary known targets

### Phase 3: Genome-Wide SL Screen
- For each cancer type with adequate N: screen all genes for differential dependency
- Same statistical framework (Mann-Whitney U, Cohen's d, FDR)
- Identify novel SMARCA4-specific dependencies beyond SMARCA2
- Cross-reference hits against SL benchmark database (1,476 pairs from nsclc-depmap-targets)
- Pathway enrichment of top hits (chromatin remodeling, transcription, DNA repair)

### Phase 4: PRISM Drug Validation
- Map SL genes to PRISM compounds
- Check if SMARCA4-deficient lines show differential drug sensitivity
- Priority: EZH2 inhibitors (tazemetostat), BET inhibitors, HDAC inhibitors — known SWI/SNF-relevant drug classes

### Phase 5: TCGA Clinical Integration
- SMARCA4 mutation/deletion frequencies per cancer type from TCGA
- Calculate addressable patient populations
- Co-mutation context: KRAS, TP53, STK11, KEAP1 in SMARCA4-mutant patients
- Survival analysis: SMARCA4-mutant vs WT by cancer type

## Expected Outputs
- Pan-cancer ranking of SMARCA4-SMARCA2 SL strength by tumor type
- Novel SL gene candidates beyond SMARCA2 per cancer type
- PRISM drug sensitivity analysis for SMARCA4-deficient lines
- TCGA-based patient population estimates
- Co-mutation landscape affecting SL relationships
- Comparison with ARID1A atlas: shared vs. distinct SWI/SNF vulnerabilities

## Success Criteria
1. SMARCA2 SL confirmed in >= 3 cancer types with FDR < 0.05
2. >= 2 novel SL genes identified beyond SMARCA2 with biological plausibility
3. PRISM drug sensitivity concordance for at least 1 drug class
4. Addressable patient population estimated for all cancer types with significant SL

## Labels
genomic, drug-screening, novel-finding, drug-candidate
