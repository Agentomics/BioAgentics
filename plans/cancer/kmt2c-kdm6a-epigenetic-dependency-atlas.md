# KMT2C/KDM6A Pan-Cancer Epigenetic Modifier Dependency Atlas

## Objective

Map synthetic lethal dependencies created by loss of the histone modifiers KMT2C (MLL3, H3K4 methyltransferase) and KDM6A (UTX, H3K27 demethylase) across cancer types, identifying actionable therapeutic vulnerabilities in tumors with epigenetic modifier loss.

## Background

KMT2C (MLL3) is the 5th most frequently mutated gene across cancers (~7-8% pan-cancer), with particularly high rates in bladder (20-25%), endometrial (15-20%), breast (5-8%), and colorectal cancers. KDM6A (UTX) is mutated in ~5% of cancers overall but reaches 25-30% in bladder cancer. Both are tumor suppressors with no approved targeted therapies for their loss-of-function contexts.

KMT2C is a histone H3K4 mono-methyltransferase in the COMPASS complex. Its loss disrupts enhancer priming and differentiation programs. KDM6A is an H3K27 demethylase that opposes Polycomb (PRC2/EZH2) silencing — its loss leads to aberrant H3K27me3 accumulation.

**Gap filled:** Our portfolio extensively covers SWI/SNF chromatin remodelers (ARID1A, SMARCA4, SWI/SNF convergence) but has no coverage of histone-modifying enzyme dependencies. KMT2C/KDM6A extends our TSG atlas into a new functional category with distinct biology (histone marks vs chromatin remodeling).

**Clinical relevance:**
- No targeted therapies approved for KMT2C- or KDM6A-loss tumors
- EZH2 inhibitors (tazemetostat) have theoretical rationale in KDM6A-loss (PRC2/EZH2 hyperactivity)
- Bladder cancer has high rates of both mutations and limited therapeutic options
- KDM6A is X-linked, creating sex-specific vulnerability patterns relevant to precision medicine

## Data Sources

- **DepMap 25Q3 CRISPR (CRISPRGeneEffect.csv):** ~1,000 cell lines, gene-level dependency scores. Expected: ~70-80 KMT2C-mutant, ~50 KDM6A-mutant lines.
- **DepMap 25Q3 Mutations (OmicsSomaticMutations.csv):** Classify KMT2C/KDM6A loss-of-function (truncating, splice site, homozygous deletion).
- **DepMap 25Q3 Expression (OmicsExpressionTPMLogp1.csv):** Expression-based confirmation of functional loss.
- **DepMap 25Q3 Copy Number (OmicsCNGene.csv):** Homozygous deletion detection.
- **PRISM Repurposing (PRISM_Repurposing_25Q2):** Drug sensitivity screen for pharmacological validation.
- **TCGA Pan-Cancer (local data/tcga/):** Mutation frequency, co-alteration patterns, expression validation across 33 cancer types.

## Methodology

### Phase 1: Cohort Classification
- Classify DepMap lines as KMT2C-mutant (LOF mutations, homozygous deletion, expression loss) vs KMT2C-intact
- Separately classify KDM6A-mutant vs KDM6A-intact
- Annotate cancer type, sex (for KDM6A X-linkage analysis), and co-occurring mutations (especially TP53, ARID1A, PIK3CA)
- Apply MTAP-correction methodology (proven in CDKN2A atlas) if significant co-alteration confounds exist

### Phase 2: Genome-Wide Dependency Screen
- For each modifier (KMT2C, KDM6A): compute differential CRISPR dependency between mutant vs intact lines
- Apply cancer-type-aware analysis (stratify by lineage to control tissue confounding)
- Identify SL hits: Cohen's d ≥ 0.3, FDR < 0.05, consistent across ≥ 2 cancer types
- Special focus: EZH2/PRC2 complex dependencies in KDM6A-loss (mechanistic prior)
- Special focus: COMPASS complex member dependencies in KMT2C-loss

### Phase 3: Pathway Enrichment & Biological Interpretation
- Pathway enrichment of SL hits (KEGG, Reactome, GO)
- Compare KMT2C vs KDM6A dependency profiles — identify convergent vs divergent epigenetic dependencies
- Cross-reference with SWI/SNF dependencies (ARID1A, SMARCA4) for chromatin modifier convergence
- Biological network analysis: protein-protein interaction enrichment

### Phase 4: Drug Sensitivity (PRISM)
- Map PRISM drug sensitivity differences for KMT2C-mutant and KDM6A-mutant lines
- Focus on: EZH2 inhibitors (tazemetostat), HDAC inhibitors, BET inhibitors, DOT1L inhibitors
- Apply MTAP/co-alteration correction as needed
- CRISPR-PRISM concordance analysis for validated drug-target-genotype associations

### Phase 5: TCGA Clinical Validation
- Validate SL gene expression patterns in TCGA (is the SL target expressed in relevant tumors?)
- Co-alteration landscape: KMT2C/KDM6A co-mutations with other TSGs
- Sex-stratified analysis for KDM6A (X-linkage effects)
- Survival correlations where sample sizes permit

## Expected Outputs

1. Classified cohort files: KMT2C-mutant/intact, KDM6A-mutant/intact line annotations
2. SL dependency tables: genome-wide differential dependency scores for each modifier
3. Pathway enrichment results with convergence/divergence analysis
4. PRISM drug sensitivity profiles with co-alteration corrections
5. TCGA validation tables: expression, co-alteration, survival
6. Cross-reference with SWI/SNF atlas results: chromatin modifier convergence map
7. Comprehensive research report with clinical translation discussion

## Success Criteria

1. ≥ 10 SL dependencies per modifier (Cohen's d ≥ 0.3, FDR < 0.05)
2. ≥ 1 druggable SL dependency validated by PRISM concordance
3. Pathway enrichment at FDR < 0.01 for ≥ 1 biologically coherent pathway
4. KMT2C vs KDM6A convergence/divergence clearly characterized
5. ≥ 50% of top SL targets show TCGA expression confirmation in relevant cancer types

## Labels

genomic, drug-screening, novel-finding, biomarker, high-priority
