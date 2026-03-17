# PIK3CA Allele-Specific Pan-Cancer Dependency Atlas

## Objective
Determine whether different PIK3CA hotspot mutations (H1047R, E545K, E542K, and others) create distinct gene dependencies across cancer types in DepMap 25Q3, to predict allele-specific therapeutic vulnerabilities beyond PI3K pathway inhibition.

## Background
PIK3CA (encoding the p110α catalytic subunit of PI3K) is one of the most commonly mutated oncogenes in human cancer. Activating mutations occur in: breast (35%), endometrial (30%), cervical (25%), colorectal (20%), head and neck (15%), gastric (10-15%), bladder (10%), hepatocellular (5%), and NSCLC (5%). Three hotspot clusters account for ~80% of mutations:

- **H1047R/L** (kinase domain): strongest PI3Kα kinase activity, preferentially activates AKT, drives membrane-proximal signaling
- **E545K** (helical domain): intermediate activity, gains p85-independent signaling, may preferentially activate SGK over AKT
- **E542K** (helical domain): similar to E545K but with distinct structural effects on p85 binding

Recent work has shown allele-specific differences:
- Vasan et al. (Nature 2019): H1047R is more oncogenic than E545K in breast cancer
- Croessmann et al. (Cancer Discovery 2020): differential response to PI3Ki by mutation location
- Inavolisib Phase III (INAVO120): showed benefit across PIK3CA mutations but subgroup analysis suggested allele-dependent magnitude
- Alpelisib (SOLAR-1): FDA-approved for PIK3CA-mutant HR+/HER2- breast, but not all mutations respond equally
- **TRIUMPH trial (PMID 39901702, Cancer Res Treatment 2025): H1047R patients had PFS 1.6 vs 7.3 months (p=0.017) compared to helical-domain + amplification patients on alpelisib in HNSCC — 4.6x worse PFS. Direct clinical evidence for allele-specific response.**
- **H1047R-selective inhibitors: pyridopyrimidinone scaffold (J Med Chem 2024, DOI 10.1021/acs.jmedchem.4c00078) — compound 17 achieved tumor regression via cryptic C-terminal pocket unique to H1047R**
- **Inavolisib Phase 2 shows activity across PIK3CA-mutated cancers beyond breast (Targeted Oncology, 2025)**

No systematic DepMap analysis has examined whether different PIK3CA alleles create different gene dependencies. This is directly analogous to our KRAS allele-specific approach in CRC — and PIK3CA has significantly more mutant cell lines than KRAS alleles in CRC.

## Data Sources
- **DepMap 25Q3 CRISPRGeneEffect** — genome-wide CRISPR dependency scores
- **DepMap 25Q3 OmicsMutationsProfile** — PIK3CA allele-level mutations
- **DepMap 25Q3 Model** — cell line metadata, cancer type classification
- **PRISM 24Q2** — drug sensitivity (alpelisib, inavolisib, capivasertib, everolimus available)
- **TCGA PanCancer Atlas** — PIK3CA allele frequencies per cancer type (cBioPortal)
- **ClinicalTrials.gov** — PI3K pathway clinical trials by indication

## Methodology

### Phase 1: PIK3CA Allele Classification
- Parse OmicsMutationsProfile for all PIK3CA mutations. Classify by:
  - Hotspot group: H1047R/L (kinase), E545K (helical), E542K (helical), C420R (C2), other activating, other/VUS
  - Zygosity: heterozygous vs homozygous/LOH (if detectable)
- Map allele distribution by cancer type
- Require ≥5 PIK3CA-mutant AND ≥5 WT lines per cancer type for powered analysis
- For allele-vs-allele comparisons: require ≥5 per allele group (H1047R vs E545K/E542K)

### Phase 2: Mutation-Level Dependency Analysis
**2a. PIK3CA-mutant vs WT per cancer type:**
- Genome-wide differential dependency (Mann-Whitney U + Cohen's d + BH-FDR)
- PIK3CA itself should be top dependency (positive control)
- Identify cancer-type-specific mutant dependencies

**2b. Allele-specific dependencies (pan-cancer pooled):**
- H1047R vs E545K/E542K: differential dependency across all cancer types
- Kinase domain vs helical domain aggregate comparison
- Identify allele-specific SL candidates

**2c. Allele-specific within cancer types** (where powered):
- Breast cancer likely has enough H1047R and E545K lines for intra-type comparison
- Endometrial, CRC may also qualify

### Phase 3: PRISM Drug Sensitivity by Allele
- PI3Kα-selective: alpelisib, inavolisib (if in PRISM)
- Pan-PI3K: buparlisib, pictilisib
- AKT: capivasertib, ipatasertib
- mTOR: everolimus
- CDK4/6: palbociclib, ribociclib, abemaciclib (clinically combined with PI3Ki)
- Test differential drug sensitivity by PIK3CA allele
- Compute CRISPR-drug concordance (do genetic dependencies predict drug sensitivity?)

### Phase 4: TCGA Allele Frequencies
- PIK3CA mutation frequencies already downloaded: `data/tcga/pancancer_pik3ca/pik3ca_mutation_frequencies.csv` (32 cancer types, 10,443 patients, 13.1% pan-cancer rate; top: UCEC 50.1%, BRCA 32.6%, CESC 28.5%, COADREAD 27.5%, BLCA 22.0%)
- Key observation: allele distribution varies by cancer type — BRCA is kinase-domain dominant (H1047R), CESC/BLCA/COADREAD are helical-domain dominant (E545K). This directly affects allele-specific analysis power per cancer type.
- Calculate per-allele patient populations by cancer type
- Cross-validate with DepMap cell line allele distributions
- Identify alleles with large patient populations but limited therapeutic options

### Phase 5: Clinical Concordance
- Map DepMap allele-specific findings to clinical trial data:
  - SOLAR-1 (alpelisib): response by PIK3CA allele if available
  - INAVO120 (inavolisib): allele subgroup analysis
  - CAPItello-291 (capivasertib): PIK3CA allele response
- Identify allele-specific therapeutic strategies
- Generate recommendation matrix: per-allele, per-cancer-type therapeutic prioritization

## Expected Outputs
- Pan-cancer PIK3CA allele frequency map (DepMap + TCGA)
- Allele-specific dependency ranking tables
- Forest plots of PI3K pathway dependency by allele
- Genome-wide novel dependencies by allele (if powered)
- PRISM drug sensitivity by allele
- Patient population estimates per allele per cancer type
- Allele-specific therapeutic strategy matrix

## Success Criteria
1. ≥6 cancer types with sufficient PIK3CA-mutant lines for powered analysis
2. PIK3CA itself confirmed as top dependency in mutant vs WT (positive control, d > 0.5)
3. ≥1 statistically significant allele-specific dependency (H1047R vs E545K, FDR < 0.1 or nominal p < 0.01 with effect size d > 0.5)
4. PI3K pathway drug sensitivity in PRISM shows allele-specific differential in ≥1 drug class

## Labels
genomic, drug-screening, novel-finding, drug-candidate, clinical
