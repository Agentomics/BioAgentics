# TP53 Hotspot Allele-Specific Dependency Atlas

## Objective
Determine whether different TP53 hotspot mutations (R175H, R248W, R273H, G245S, R249S, Y220C, R282W) create distinct gene dependencies in DepMap 25Q3, to identify allele-specific therapeutic vulnerabilities and predict which gain-of-function mutations are most targetable by emerging p53 reactivators and combination strategies.

## Background
TP53 is the most commonly mutated gene in human cancer (~50% of all tumors). However, TP53 mutations are heterogeneous — different hotspot missense mutations have distinct gain-of-function (GOF) properties:

- **R175H (structural):** Complete loss of DNA binding, GOF via protein aggregation, chromatin remodeling
- **R248W (contact):** Direct DNA contact mutation, retains partial structure, GOF via altered transcriptional programs
- **R273H (contact):** Similar to R248W, different transcriptional targets
- **G245S (structural):** Loop L3 destabilization, GOF via CREB1/MMP activation
- **R249S (structural):** Aflatoxin-associated, hepatocellular carcinoma enriched
- **Y220C (structural):** Creates druggable surface pocket — PC14586 (rezatapopt/Pyxis) targets this specific allele
- **R282W (structural):** H2 helix destabilization

**Clinical relevance:**
- **Rezatapopt (PC14586):** Y220C-specific p53 reactivator in Phase 1/2, ORR 25% in Y220C tumors
- **Eprenetapopt (APR-246/PRIMA-1^MET):** Thiol-reactive, stabilizes mutant p53, preferentially active on structural mutants (R175H, G245S, Y220C)
- **KT-253 (MDM2 degrader):** For TP53-WT tumors with MDM2 overexpression
- **Combination strategies:** GOF-specific vulnerabilities could guide rational combinations (e.g., R175H-dependent cells may have unique druggable dependencies)

**Gap:** While TP53 mutation status is universally tracked, allele-specific dependency analysis using DepMap is largely unexplored. Most studies treat TP53 as binary (mutant vs. WT). DepMap has hundreds of TP53-mutant lines with known alleles, enabling the first systematic allele-level analysis. This follows our successful allele-specific framework from PIK3CA and CRC KRAS projects.

## Data Sources
- **DepMap 25Q3 CRISPRGeneEffect.csv:** Genome-wide dependency scores
- **DepMap 25Q3 OmicsSomaticMutations.csv:** TP53 mutation calls with specific alleles
- **DepMap 25Q3 OmicsCNGene.csv:** TP53 copy number (LOH context)
- **DepMap 25Q3 OmicsExpressionProteinCodingGenesTPMLogp1.csv:** TP53 expression levels (GOF mutants often overexpressed)
- **DepMap 25Q3 Model.csv:** Cell line metadata
- **PRISM 24Q2:** Drug sensitivity by TP53 allele
- **TCGA pan-cancer:** TP53 allele frequencies by cancer type
- **IARC TP53 Database:** Comprehensive TP53 mutation registry for validation

## Methodology

### Phase 1: TP53 Allele Classifier
- Extract all TP53 mutations from OmicsSomaticMutations.csv
- Classify into: R175H, R248W, R273H, G245S, R249S, Y220C, R282W, other_missense, truncating (frameshift/nonsense), TP53_WT
- Annotate LOH status from copy number data (heterozygous vs. biallelic inactivation)
- Annotate TP53 expression level (GOF mutants often show protein accumulation proxy)
- Report N per allele per cancer type; require minimum 10 lines per allele for pan-cancer analysis
- Expected: R175H and R248W will have largest N (most common hotspots)

### Phase 2a: Mutant vs. WT Genome-Wide Screen
- Compare all TP53-mutant vs. TP53-WT: genome-wide dependency screen
- Mann-Whitney U, Cohen's d, BH-FDR correction
- Identify pan-TP53-mutation dependencies (shared across all alleles)
- Key checkpoint: MDM2 should show INVERSE dependency (essential in WT, not mutant) — positive control

### Phase 2b: Allele-Specific Dependencies
- For each hotspot with N >= 10: compare allele-specific vs. all-other-TP53-mutant dependencies
- Focus on R175H vs. R248W/R273H (structural vs. contact) as primary contrast
- Identify GOF-specific vulnerabilities per allele
- Key biology to check: HSP90 chaperone dependency (R175H needs HSP90 for stabilization), CREBBP/EP300 for transcriptional GOF mutants
- Cross-reference against SL benchmark database

### Phase 3: PRISM Drug Sensitivity by Allele
- Map PRISM drug sensitivity stratified by TP53 allele
- Priority compounds: eprenetapopt (structural mutant preference), HDACi (GOF-dependent), proteasome inhibitors (protein homeostasis), HSP90 inhibitors (R175H structural dependency)
- Check if allele-specific dependencies predict drug sensitivity

### Phase 4: TCGA Allele Frequency Integration
- TP53 allele frequencies per cancer type from TCGA
- Calculate addressable populations per allele per cancer type
- Known enrichments to validate: R249S in hepatocellular carcinoma (aflatoxin), R175H in high-grade serous ovarian
- Co-mutation landscape per allele: which oncogenic drivers co-occur with which TP53 alleles?

## Expected Outputs
- Pan-cancer map of TP53 allele-specific gene dependencies
- Structural vs. contact mutant dependency profiles
- GOF-specific druggable vulnerabilities per hotspot
- PRISM drug sensitivity stratification by TP53 allele
- TCGA allele frequency and addressable population per cancer type
- Rational combination predictions for p53 reactivators

## Success Criteria
1. MDM2 positive control: significantly more essential in TP53-WT vs. mutant (FDR < 0.05)
2. >= 1 allele-specific dependency identified that distinguishes structural vs. contact mutants (FDR < 0.05)
3. TP53 allele-drug sensitivity correlation in PRISM for at least 1 compound class
4. Biologically coherent allele-specific pathways (HSP90 for R175H, transcription for R248W/R273H)

## Labels
genomic, drug-screening, novel-finding, drug-candidate, clinical
