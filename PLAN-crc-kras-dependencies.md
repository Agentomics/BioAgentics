# CRC KRAS Allele-Specific Dependency Atlas

## Objective
Identify KRAS allele-specific therapeutic vulnerabilities in colorectal cancer using DepMap 25Q3, to inform patient selection for allele-specific (sotorasib G12C, zoldonrasib/MRTX1133 G12D) and pan-RAS (ERAS-0015, elironrasib) inhibitors entering CRC clinical trials.

## Background
CRC has a fundamentally different KRAS allele landscape from NSCLC: G12D (47%) and G13D (35%) dominate vs G12C (21%) in NSCLC. KRAS G12D-selective inhibitors (zoldonrasib FDA BTD, MRTX1133 Phase 2) are entering pivotal CRC trials. Pan-RAS agents (ERAS-0015, elironrasib FDA BTD) cover all alleles. No systematic analysis exists comparing allele-specific dependencies across CRC KRAS variants in DepMap 25Q3.

Data curator confirmed feasibility (task #81): 60 CRC cell lines with CRISPR data (48 COAD, 12 READ), 34 KRAS-mutant (56.7%), allele diversity: G12D(16), G13D(12), G12V(8), G12C(7), Q61H(3), A146T(3). Co-mutations: APC 83%, TP53 80%, PIK3CA 35%, BRAF 30%, SMAD4 20%. MSI-H: 10/60. BRAF V600E mutual exclusivity with KRAS confirmed.

## Data Sources
- **DepMap 25Q3:** CRISPR dependency (CRISPRGeneEffect), expression (OmicsExpressionProteinCodingGenesTPMLogp1), mutations (OmicsSomaticMutations), copy number (PortalOmicsCNGeneLog2)
- **PRISM 24Q2:** Drug sensitivity for CRC lines
- **TCGA COAD/READ:** Mutation frequencies, co-mutation patterns, survival data
- **Clinical benchmarks:** Sotorasib CRC (CodeBreaK 101: ORR 9.7% monotherapy, ~30% combo), MRTX1133 Phase 1/2 (enrolling), zoldonrasib Phase 1 (CRC cohort)

## Methodology

### Phase 1: CRC Line Classification
- Extract all CRC lines from DepMap 25Q3
- Classify by KRAS allele: G12D, G13D, G12V, G12C, Q61H, A146T, WT
- Annotate co-mutations: APC, TP53, PIK3CA, BRAF V600E, SMAD4, MSI status
- Annotate CMS (consensus molecular subtype) if available from expression data
- Quality control: verify KRAS mutation calls against expression, check for compound mutations

### Phase 2: Allele-Specific Differential Dependency Analysis
- For each KRAS allele group (≥5 lines) vs KRAS-WT:
  - Compute differential CRISPR dependency (Mann-Whitney U, Cohen's d)
  - FDR correction across all genes
  - Identify allele-specific dependencies (significant in one allele, not others)
- Key comparisons:
  - G12D vs G13D (most common alleles, different downstream signaling)
  - G12C vs other alleles (sotorasib-eligible vs not)
  - All KRAS-mut vs WT (shared KRAS dependencies)
- Co-mutation stratification: PIK3CA co-mutation × KRAS allele interaction
- MSI-H vs MSS within KRAS-mutant: separate biology?

### Phase 3: PRISM Drug Sensitivity by KRAS Allele
- Extract PRISM drug sensitivity for CRC lines
- Differential drug sensitivity across KRAS alleles
- Identify drugs with allele-selective efficacy
- Cross-reference: do allele-specific dependencies have corresponding drug sensitivities?
- Key drugs to check: MEK inhibitors (trametinib, selumetinib), ERK inhibitors, SHP2 inhibitors, mTOR inhibitors, EGFR antibodies (cetuximab resistance in KRAS-mut)

### Phase 4: TCGA CRC Validation
- KRAS allele frequencies in TCGA COAD and READ
- Co-mutation patterns by allele (APC, TP53, PIK3CA, SMAD4)
- Survival analysis by KRAS allele (if sufficient sample sizes)
- Estimate patient populations per allele for trial enrollment projections

### Phase 5: Therapeutic Strategy Matrix
- For each KRAS allele group, compile:
  - Top allele-specific dependencies (druggable?)
  - Drug sensitivity profile
  - Suggested combination strategies
  - Patient population estimate
- Compare CRC findings to NSCLC allele-specific biology from nsclc-depmap-targets

## Expected Outputs
- Allele-specific dependency rankings per KRAS variant
- Drug sensitivity profiles by allele
- Co-mutation interaction effects
- CRC vs NSCLC KRAS biology comparison
- Patient selection framework for KRAS-targeted trials

## Success Criteria
- At least 3 allele-specific dependencies with FDR < 0.05
- G12D vs G13D biological distinction confirmed computationally
- PRISM drug-dependency correlation > 0.3 for at least one allele-drug pair
- CRC-NSCLC divergence identified for shared KRAS alleles (especially G12C)

## Labels
genomic, drug-screening, drug-candidate, novel-finding, clinical
