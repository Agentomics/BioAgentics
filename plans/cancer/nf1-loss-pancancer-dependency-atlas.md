# NF1 Loss Pan-Cancer Dependency Atlas

## Objective

Map synthetic lethal dependencies specific to NF1-loss cancers across all DepMap tumor types to identify novel therapeutic targets beyond MEK inhibition, with particular focus on malignant peripheral nerve sheath tumors (MPNST) and NF1-mutant subtypes of melanoma, glioma, and NSCLC.

## Background

NF1 (neurofibromin 1) encodes a RAS-GAP that negatively regulates RAS signaling by catalyzing GTP hydrolysis on RAS proteins. NF1 loss-of-function mutations constitutively activate the RAS/MAPK pathway without direct RAS oncogene mutations, representing a distinct mechanism of RAS pathway hyperactivation.

**NF1 alteration prevalence:**
- MPNST: ~90% (hallmark alteration, often biallelic loss)
- Cutaneous melanoma: ~15% (enriched in non-UV/desmoplastic subtypes)
- GBM/glioma: ~15% (mesenchymal subtype)
- NSCLC: ~10% (enriched in adenocarcinoma, KP-like)
- CRC: ~5%
- NF1 syndrome patients: germline heterozygous, ~10% lifetime MPNST risk

**Therapeutic landscape (March 2026):**
- **Selumetinib** (AstraZeneca): FDA-approved 2020 for NF1 plexiform neurofibromas in pediatric patients — NOT for MPNST or other NF1-mutant cancers
- **MEK inhibitors:** Limited single-agent activity in MPNST (mepesertinib Phase 2: ORR <10%). Acquired resistance via PI3K/mTOR pathway activation is rapid
- **mTOR inhibitors:** Everolimus showed some activity in NF1-related tumors; MEK+mTOR combos (selumetinib+sirolimus) under investigation
- **Key gap:** MPNST has no effective targeted therapy. 5-year survival is ~40% for localized, <15% for metastatic. Standard of care remains surgery + doxorubicin/ifosfamide with poor response rates

**Why this atlas is needed:**
1. NF1-loss activates RAS pathway through a fundamentally different mechanism than direct RAS mutations — dependencies may differ from KRAS-mutant tumors
2. No systematic pan-cancer DepMap analysis of NF1-loss dependencies exists
3. MPNST is a devastating cancer with zero effective targeted therapies — urgent unmet need
4. NF1 co-occurs with TP53 loss in MPNST (>75% of cases) — combinatorial dependencies may exist
5. Our published KRAS/SMARCA4/ARID1A atlases provide direct comparison framework for distinguishing NF1-specific from shared RAS pathway dependencies

**Cross-project links:**
- Published atlases (BRCA, ARID1A, SMARCA4, MTAP-PRMT5, PIK3CA): framework for methodology and cross-comparison
- NSCLC-DepMap targets: NF1 mutations in NSCLC create a non-KRAS RAS-activated subtype; cross-reference findings
- SWI/SNF metabolic convergence: some NF1-mutant tumors have co-occurring SWI/SNF alterations

## Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv — genome-wide dependency scores across all cell lines
- **DepMap 25Q3 mutations:** OmicsSomaticMutations.csv — NF1 truncating/missense mutations, TP53, CDKN2A co-mutation status
- **DepMap 25Q3 copy number:** PortalOmicsCNGeneLog2.csv — NF1 copy number for deletion calling
- **DepMap 25Q3 expression:** OmicsExpression.csv — NF1 expression for validation of loss-of-function
- **DepMap model annotations:** Model.csv — cancer type classification, MPNST lines identification
- **PRISM 24Q2:** Drug sensitivity data for NF1-loss stratification (MEKi, mTORi, PI3Ki, CDKi)
- **TCGA:** NF1 alteration frequency by cancer type, co-occurring alterations, survival data

## Methodology

### Phase 1 — NF1 Loss Classifier
- Classify all DepMap lines by NF1 status: deep deletion (log2 CN ≤ -1.0), truncating mutations (nonsense, frameshift, splice), or both
- Cross-validate with NF1 expression (lost lines should have low/absent expression)
- Annotate co-mutations: TP53 (critical for MPNST), CDKN2A, RAS family (KRAS/NRAS/HRAS — exclude lines with concurrent RAS mutations to isolate NF1-specific effects)
- Report qualifying cancer types (≥5 NF1-lost + ≥10 NF1-intact with CRISPR data)
- Special attention to MPNST cell line availability — even if few lines exist, characterize their dependency profiles individually
- Power analysis per cancer type

### Phase 2 — RAS/MAPK Pathway Dependency Baseline
- For each qualifying type: compare RAS pathway gene dependencies (BRAF, RAF1, MAP2K1, MAP2K2, MAPK1, MAPK3, SOS1, GRB2, SHP2/PTPN11) in NF1-lost vs intact
- Quantify the degree of MAPK pathway addiction — is it stronger or weaker than in KRAS-mutant lines?
- Statistics: Cohen's d, Mann-Whitney U, bootstrap 95% CI (1000 iterations, seed=42), BH FDR
- This establishes the baseline for identifying dependencies BEYOND the expected MAPK axis

### Phase 3 — Genome-Wide Differential Dependency Analysis
- Compute effect sizes for ALL gene dependencies comparing NF1-lost vs NF1-intact, both pan-cancer and per-tumor-type
- Apply negative-effect filter: only consider genes where NF1-lost lines show MORE dependency (negative CRISPR score direction)
- Key hypothesis-driven gene sets to examine:
  - **mTOR/PI3K cross-talk:** PIK3CA, PIK3CB, AKT1, AKT2, MTOR, RICTOR, RPTOR
  - **Cell cycle regulators:** CDK4, CDK6, CDK2, CCND1, RB1
  - **Epigenetic regulators:** PRC2 complex (EZH2, EED, SUZ12), BRD4, DOT1L
  - **DNA damage response:** ATR, CHK1, WEE1, PARP1 (NF1 has reported roles in DNA repair)
  - **Metabolic dependencies:** OXPHOS, glutamine metabolism
  - **RAS feedback regulators:** SPRY1/2/4, DUSP4/6, ERF, RASA2
- Rank hits by |Cohen's d| × -log10(FDR) composite score

### Phase 4 — Drug Target Mapping
- Cross-reference top dependencies with druggable targets: FDA-approved, clinical-stage, preclinical compounds
- Priority targets: compounds that could be combined with MEK inhibitors
- Map PRISM 24Q2 drug sensitivity data: which compounds show NF1-loss-selective sensitivity?
- Identify candidates suitable for MPNST clinical trials

### Phase 5 — Cross-Atlas Comparison & Clinical Context
- Compare NF1-loss dependency profile with KRAS-mutant dependency profile from published work
- Identify: (a) shared RAS-pathway dependencies, (b) NF1-specific dependencies, (c) KRAS-specific dependencies
- MPNST-specific analysis: even with limited cell lines, generate target recommendations
- Cross-reference with NF1 clinical trial landscape (ClinicalTrials.gov)
- Generate per-tumor-type therapeutic recommendations

## Expected Outputs

- Ranked list of NF1-loss-specific dependencies per tumor type (effect size heatmap)
- NF1-loss vs KRAS-mutant dependency comparison matrix
- Druggable target prioritization table with clinical-stage compounds
- MPNST-specific therapeutic target recommendations
- Publication-ready figures and summary

## Success Criteria

- Identify ≥5 NF1-loss-specific dependencies with |d| > 0.5 and FDR < 0.1 in ≥2 cancer types
- At least 2 hits with druggable targets (clinical-stage or FDA-approved compounds)
- Clear differentiation from KRAS-mutant dependency profiles (≥3 NF1-specific hits not shared with KRAS)
- MPNST-specific findings with therapeutic potential (even from individual cell line characterization)

## Labels

genomic, novel-finding, drug-candidate, high-priority
