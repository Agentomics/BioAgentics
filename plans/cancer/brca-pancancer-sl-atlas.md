# BRCA1/2 Pan-Cancer Synthetic Lethality Atlas

## Objective

Map gene dependencies and drug sensitivities in BRCA1/2-deficient vs proficient cell lines across all cancer types in DepMap 25Q3, to identify non-homologous recombination (non-HR) therapeutic vulnerabilities beyond PARP inhibition and discover combination strategies for PARPi-resistant patients.

## Background

BRCA1/2 are the most clinically actionable tumor suppressors in oncology. Six PARP inhibitors are FDA-approved (olaparib, niraparib, rucaparib, talazoparib, veliparib, fuzuloparib), but 30-50% of BRCA-mutant patients either don't respond or develop acquired resistance. Known resistance mechanisms include BRCA reversion mutations, 53BP1/SHLD complex loss, RAD51C promoter demethylation, and drug efflux pump upregulation — but therapeutic options after PARPi failure remain limited.

Most synthetic lethality research in BRCA-deficient cancers focuses on the HR pathway itself (RAD51, ATR, CHK1, WEE1). A systematic genome-wide screen across all cancer types could reveal non-HR dependencies — metabolic rewiring, epigenetic vulnerabilities, or immune signaling dependencies — that are overlooked by candidate-gene approaches.

BRCA1 mutations occur in ~5% of breast, ~15% of high-grade serous ovarian (HGSOC), ~3% of prostate, and ~5% of pancreatic cancers. BRCA2 mutations have similar frequencies. Combined, BRCA1/2 mutations affect an estimated 80,000+ new US cancer patients annually.

## Data Sources

- **DepMap 25Q3**: CRISPR gene effect scores (~18,000 genes × ~1,800 cell lines), mutation calls, copy number, expression
- **PRISM 24Q2**: Drug sensitivity (olaparib BRD:BRD-K82859696, talazoparib BRD:BRD-K01337880, niraparib BRD:BRD-K28907609, rucaparib, ceralasertib/AZD6738, adavosertib/AZD1775)
- **TCGA**: Pan-cancer mutation frequencies, co-mutation landscape, survival data for BRCA1/2-mutant patients
- **ClinicalTrials.gov**: Active PARPi combination trials for clinical context

## Methodology

### Phase 1: BRCA1/2 Classification & Annotation

- Classify all DepMap cell lines by BRCA1 and BRCA2 status separately:
  - **BRCA-deficient**: LOF mutations (frameshift, nonsense, splice-site) OR homozygous deletions
  - **BRCA-proficient**: No damaging mutations, no deep deletions
  - Exclude lines with VUS (missense only) from both groups
- Co-annotate with:
  - TP53 status (critical: BRCA1/TP53 co-mutation is near-universal in HGSOC)
  - PTEN status (cross-reference with PTEN atlas)
  - 53BP1/SHLD complex status (SHLD1, SHLD2, SHLD3, MAD2L2) — known PARPi resistance markers
  - HR pathway gene status (PALB2, RAD51C, RAD51D, ATM, ATR, CHEK2)
- Report qualifying cancer types (≥10 BRCA-deficient lines) for powered analysis
- Optionally compute BRCAness/HRD signature as orthogonal grouping

### Phase 2: HR Pathway Dependency Validation

- For each qualifying cancer type, compute:
  - Mann-Whitney U test: BRCA-deficient vs BRCA-proficient gene effect scores
  - Cohen's d effect size (pooled SD)
  - BH-FDR correction per cancer type
- **Positive controls** (must validate or flag methodology issues):
  - PARP1: should show differential dependency (BRCA-deficient more sensitive to PARP loss)
  - RAD51 paralogs (RAD51C, RAD51D, XRCC2, XRCC3): SL with BRCA loss
  - POLQ (PolTheta): known SL with HR deficiency (ART4215/novobiocin Phase 2)
  - RPA1: ssDNA protection, essential when HR is compromised
- **Negative controls:**
  - PARP1 dependency should be REDUCED in lines with 53BP1/SHLD loss (PARPi resistance restoration)
- Stratify BRCA1 vs BRCA2 separately — they have distinct biology (BRCA1: HR + DNA end resection; BRCA2: RAD51 loading)

### Phase 3: Genome-Wide Non-HR Dependency Screen

- Test all ~18,000 genes for differential dependency in BRCA-deficient vs proficient lines
- Dual threshold: FDR < 0.05 AND |Cohen's d| > 0.3
- Categorize hits by pathway:
  - DNA repair (non-HR): base excision repair, mismatch repair, Fanconi anemia
  - Metabolic: one-carbon metabolism (MTHFD2, SHMT2), iron/ROS homeostasis
  - Epigenetic: histone modifiers, chromatin remodelers (cross-reference SWI/SNF atlases)
  - Cell cycle: replication stress response (CDC7, MCM complex, CDK12)
  - Immune/signaling: cGAS-STING pathway (BRCA-deficient cells have increased cytosolic DNA)
- Compare BRCA1-specific vs BRCA2-specific vs shared dependencies
- Per-cancer-type analysis for qualifying types

### Phase 4: PRISM Drug Sensitivity

- Test PARPi sensitivity by BRCA status:
  - Olaparib, talazoparib, niraparib, rucaparib (if available in PRISM)
  - Expected: BRCA-deficient lines more sensitive
- Test DDR combination agents:
  - ATR inhibitors (ceralasertib/AZD6738)
  - WEE1 inhibitors (adavosertib/AZD1775)
  - CHK1 inhibitors
  - DNA-PKcs inhibitors (AZD7648 — cross-reference with WRN-MSI atlas)
- Correlate Phase 3 genetic dependencies with drug sensitivity (gene-drug concordance)
- Identify drugs that selectively kill BRCA-deficient lines via non-HR mechanisms

### Phase 5: TCGA Clinical Integration

- Pan-cancer BRCA1/2 mutation frequencies from TCGA
- Co-mutation landscape per cancer type (TP53, PIK3CA, PTEN, KRAS)
- Estimate addressable patient populations (SEER incidence × TCGA mutation rates)
- Cross-reference with TP53 atlas: how does TP53 co-mutation modulate BRCA dependencies?
- Cross-reference with PTEN atlas: PI3K pathway status in BRCA-deficient tumors

## Expected Outputs

- BRCA1/2 classification for all DepMap cell lines with co-mutation annotations
- Effect size tables and forest plots for HR pathway genes by cancer type
- Genome-wide dependency rankings (separate for BRCA1, BRCA2, combined)
- PRISM PARPi and DDR drug sensitivity analysis
- TCGA patient population estimates and co-mutation landscape
- Prioritized non-HR therapeutic targets with drug mapping

## Success Criteria

- Positive controls validate: RAD51 paralogs and POLQ show SL with BRCA loss (FDR < 0.05, d < -0.3)
- At least 5 non-HR novel dependencies identified with FDR < 0.05
- PARPi sensitivity correlates with BRCA status in PRISM
- At least 3 cancer types qualify for powered analysis (≥10 BRCA-deficient lines each)
- BRCA1 vs BRCA2 comparison reveals at least some distinct dependencies

## Labels

genomic, drug-screening, drug-candidate, novel-finding, clinical
