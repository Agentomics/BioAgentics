# ARID1A-Loss Pan-Cancer Synthetic Lethality Atlas

## Objective
Systematically rank all cancer types by synthetic lethal vulnerability in ARID1A-mutant/deleted cell lines using DepMap 25Q3, to identify which tumor types respond best to EZH2 inhibitors and discover novel SL-based therapeutic strategies.

## Background
ARID1A (AT-rich interaction domain 1A) is a subunit of the SWI/SNF (BAF) chromatin remodeling complex and one of the most frequently mutated tumor suppressor genes across human cancers. Loss-of-function mutations occur at high frequencies: ovarian clear cell carcinoma (~50%), endometrial carcinoma (~40%), gastric cancer (~30%), bladder cancer (~20%), cholangiocarcinoma (~15%), hepatocellular carcinoma (~10%), and colorectal cancer (~10%).

ARID1A loss disrupts chromatin accessibility and transcriptional regulation, creating dependencies on:
- **ARID1B** (paralog, residual SWI/SNF function — paralog SL)
- **EZH2** (Polycomb repressive complex — antagonistic SL)
- **ATR/ATRIP** (DNA damage repair pathway — replication stress SL)
- **HDAC6** (histone deacetylase — chromatin compensation)
- **PI3K/AKT** (signaling pathway — co-activated with ARID1A loss)
- **BRD2/BET** (bromodomain — chromatin reader compensation)

Tazemetostat (EZH2 inhibitor) is FDA-approved for epithelioid sarcoma with INI1/SMARCB1 loss (a related SWI/SNF deficiency) and follicular lymphoma. However, ARID1A-specific therapeutic strategies are less developed. Clinical trials testing EZH2i, ATRi, and PI3Ki in ARID1A-mutant tumors exist for individual cancer types but no cross-cancer comparison of SL strength has been performed to guide trial expansion.

Our pan-cancer atlas pipeline (proven with MTAP/PRMT5) can systematically quantify SL relationships across cancer types and identify where the clinical opportunity is strongest.

## Data Sources
- **DepMap 25Q3 CRISPRGeneEffect** — genome-wide CRISPR dependency scores (~18,000 genes × ~1,100 lines)
- **DepMap 25Q3 OmicsMutationsProfile** — ARID1A mutation status (LOF mutations)
- **DepMap 25Q3 PortalOmicsCNGeneLog2** — ARID1A copy number (homozygous deletions)
- **DepMap 25Q3 Model** — cell line metadata, cancer type classification
- **PRISM 24Q2** — drug sensitivity profiles (EZH2 inhibitors, ATR inhibitors available)
- **TCGA PanCancer Atlas** — ARID1A mutation frequencies per cancer type (via cBioPortal API)

## Methodology

### Phase 1: ARID1A Classification
- Classify all DepMap cell lines by ARID1A status: LOF mutation (nonsense, frameshift, splice site) OR homozygous deletion
- Cross-validate with ARID1A expression (bimodal distribution expected)
- Map cancer type distribution. Require ≥5 ARID1A-mutant AND ≥5 WT lines per cancer type for powered analysis
- Annotate co-occurring SWI/SNF mutations (SMARCA4, SMARCB1, ARID2, PBRM1) as potential confounders

### Phase 2: Known SL Target Effect Sizes
Per qualifying cancer type, compute dependency scores (Mann-Whitney U + Cohen's d + BH-FDR) in ARID1A-mutant vs WT for:
- **EZH2** (primary positive control — well-established SL)
- **ARID1B** (paralog SL — most direct)
- **USP8** (deubiquitinase → FGFR2 degradation → STAT3 dephosphorylation — claimed MORE selective than EZH2i in OCCC; Saito et al., npj Precision Oncology 2025)
- **BRD2/BRD4** (bromodomain reader — ARID1A loss → loss of HDAC1 binding → H4K acetylation → BRD4-driven transcription; Nagarajan et al., Nature Genetics 2020)
- **HDAC1/HDAC2/HDAC3** (histone deacetylases — pan-HDAC inhibitor sensitivity shown in OCCC; Bitler et al., Cell Reports 2018)
- **HDAC6** (chromatin compensation)
- **ATR/ATRIP** (replication stress SL)
- **PARP1/PARP2** (DNA damage repair deficiency SL)
- **HSP90AA1** (proteostatic stress SL)
- **PIK3CA/AKT1/MTOR** (co-activated pathway)
- Generate forest plot ranked by EZH2 SL strength. Flag cancer types where USP8/BRD4/HDAC > EZH2 — these may represent tumor-type-specific SL hierarchies (literature suggests different dominant SL partners per cancer type).

### Phase 3: Genome-Wide SL Screen
- For each qualifying cancer type: genome-wide differential dependency (ARID1A-mut vs WT), Mann-Whitney U + Cohen's d + BH-FDR
- Identify cancer-type-specific SL hits (FDR < 0.05)
- Identify universal SL hits (significant in ≥3 cancer types)
- Cross-reference with SWI/SNF pathway genes to identify functional relationships
- Flag potential novel targets not previously associated with ARID1A loss

### Phase 4: PRISM Drug Validation
- Test whether ARID1A-mutant lines show differential sensitivity to:
  - Tazemetostat / other EZH2 inhibitors (if in PRISM)
  - ATR inhibitors (berzosertib, ceralasertib — if in PRISM)
  - PI3K inhibitors (alpelisib, inavolisib)
  - HDAC inhibitors (vorinostat, panobinostat)
  - BET inhibitors (JQ1, OTX015)
- Correlate PRISM drug sensitivity with CRISPR SL effect sizes
- Per cancer type: rank available drugs by predicted efficacy in ARID1A-mutant context

### Phase 5: TCGA Population & Clinical Concordance
- ARID1A mutation frequencies already downloaded: `data/tcga/pancancer_arid1a/arid1a_mutation_deletion_frequencies.csv` (32 cancer types, 10,443 patients, 8.2% pan-cancer rate; top: UCEC 44.5%, STAD 27.3%, BLCA 26.3%)
- NOTE: ARID1A is predominantly LOF mutations, not homdel (unlike MTAP). Phase 1 classification must capture both LOF + homdel. SKCM has low LOF ratio — UV-passenger mutations may inflate mutation count.
- Estimate ARID1A-mutant patient population per cancer type per year
- Map DepMap SL ranking × patient population → combined clinical impact score
- Cross-reference with ClinicalTrials.gov: which cancer types have SL-based trials? Which are underexplored?
- Identify the top 3-5 cancer types with strong DepMap SL, large patient populations, and no active SL-targeted trials

## Expected Outputs
- Pan-cancer ARID1A SL ranking table (effect sizes, FDR, sample sizes per cancer type)
- Forest plots for EZH2, ARID1B, and top novel SL genes
- Genome-wide SL hit list per cancer type
- PRISM drug concordance analysis
- TCGA patient population estimates for ARID1A-mutant tumors
- Clinical trial gap analysis: underexplored cancer types for SL-based therapy
- Recommended trial expansion priorities

## Success Criteria
1. ≥8 cancer types with sufficient ARID1A-mutant lines for powered analysis (≥5 mut + ≥5 WT)
2. EZH2 SL confirmed as positive control in ≥3 cancer types (d > 0.5, p < 0.05)
3. ≥2 novel SL genes identified beyond EZH2/ARID1B with FDR < 0.05 in ≥1 cancer type
4. ≥1 underexplored cancer type identified with strong SL but no active SL-targeted clinical trials

## Labels
genomic, drug-screening, novel-finding, drug-candidate
