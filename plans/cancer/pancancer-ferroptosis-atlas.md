# PLAN: Pan-Cancer Ferroptosis Vulnerability Atlas

## Objective

Map ferroptosis pathway gene dependencies across all cancer types in DepMap 25Q3, stratified by NRF2/KEAP1 mutation status, to identify which cancer types beyond NSCLC are most susceptible to ferroptosis-inducing therapies and where therapeutic opportunities exist.

## Background

Ferroptosis is a regulated cell death mechanism driven by iron-dependent lipid peroxidation. Cancer cells evade ferroptosis through multiple defense mechanisms: the GPX4/GSH axis, the FSP1/CoQ10 axis, SAT1 suppression, SHMT1/2 antioxidant defense, and GLS1-mediated glutathione synthesis. The NRF2/KEAP1 pathway is a master regulator — KEAP1 loss-of-function or NFE2L2 gain-of-function constitutively activates NRF2, which simultaneously upregulates multiple ferroptosis defense genes.

Our nsclc-depmap-targets project identified a 5-layer KEAP1-mutant ferroptosis defense model in NSCLC: (1) FSP1/CoQ, (2) GPX4/SLC7A11, (3) SAT1 suppression, (4) SHMT1/2, (5) GLS1. Wu et al. (Nature, Nov 2025) demonstrated that FSP1 inhibition with icFSP1 reduces KP LUAD tumor growth by ~80% in vivo. Multiple ferroptosis-targeting drugs are entering development: icFSP1 (FSP1 inhibitor), GPX4 inhibitors, CB-839/telaglenastat (GLS1 inhibitor), and ferroptosis-inducing combinations (HDACi + ferroptosis inducers).

The critical unanswered question: Which cancer types beyond NSCLC have the strongest ferroptosis pathway dependencies, and how does NRF2/KEAP1 status modulate ferroptosis vulnerability across different tumor contexts? This determines where ferroptosis-targeting therapies should be tested.

## Data Sources

Available locally (DepMap 25Q3):
- **CRISPRGeneEffect.csv** — ~1,900 cell lines across all cancer types, includes all ferroptosis genes
- **OmicsSomaticMutations.csv** — NFE2L2 and KEAP1 mutation status
- **Model.csv** — Cancer type annotations (>30 lineages)
- **OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv** — Ferroptosis gene expression levels
- **PRISM 24Q2** — Drug sensitivity data; check for erastin, RSL3, or ferroptosis-modulating compounds

Requires download:
- **TCGA pan-cancer mutation frequencies** — NFE2L2/KEAP1 mutation rates across 33 tumor types (available from GDC or cBioPortal)

## Methodology

### Phase 1: Pan-Cancer Ferroptosis Dependency Map
1. Extract CRISPR dependency scores for ferroptosis gene panel across all DepMap cell lines:
   - **Ferroptosis defense (pro-survival):** FSP1/AIFM2, GPX4, SLC7A11, GLS1, GCLC, GCLM, TXNRD1, NQO1, FTH1, HMOX1
   - **Ferroptosis promotion (pro-death when active):** ACSL4, LPCAT3, SAT1, NCOA4, TFRC, ALOX15
   - **Metabolic modulators:** SHMT1, SHMT2, MTHFD2, CBS
2. Compute per-cancer-type statistics: mean dependency, fraction of lines showing dependency (score < -0.5), interquartile range
3. Rank cancer types by ferroptosis pathway vulnerability (composite score across all defense genes)
4. Hierarchical clustering of cancer types by full ferroptosis gene dependency profile

### Phase 2: NRF2/KEAP1 Stratification
1. Classify all DepMap cell lines by NRF2/KEAP1 status:
   - KEAP1-mutant (loss-of-function → NRF2 constitutively active)
   - NFE2L2-mutant (gain-of-function → NRF2 constitutively active)
   - Double wild-type (NRF2 normally regulated)
2. Within each cancer type (where sample size permits, N ≥ 5 per group), compare ferroptosis dependencies between NRF2-active vs NRF2-WT
3. Test whether NRF2 activation uniformly protects against ferroptosis or shows cancer-type-specific effects
4. Identify cancer types where NRF2 status most strongly modulates ferroptosis vulnerability (largest effect sizes)

### Phase 3: Therapeutic Opportunity Ranking
1. Obtain pan-cancer NRF2/KEAP1 mutation frequencies from TCGA (download or use published data)
2. For each cancer type, compute therapeutic opportunity score: ferroptosis vulnerability × NRF2-WT patient fraction
3. Rank cancer types by therapeutic opportunity for:
   - **FSP1 inhibitors (icFSP1):** cancers with strongest FSP1/AIFM2 dependency
   - **GPX4 inhibitors:** cancers with strongest GPX4 dependency
   - **Multi-target approach:** cancers requiring combination strategies (NRF2-active, multiple defense layers)
4. PRISM validation: if ferroptosis-related compounds are in PRISM, validate computational predictions with drug sensitivity

### Phase 4: Cross-Cancer Comparison with NSCLC
1. Use NSCLC ferroptosis profile from nsclc-depmap-targets as reference
2. Identify "ferroptosis-analogous" cancer types — non-NSCLC cancers whose ferroptosis dependency profile most closely resembles KEAP1-mutant NSCLC
3. These cancers may benefit from the same therapeutic strategies (icFSP1, HDACi + ferroptosis, ATR + ferroptosis modulation)
4. Document cancer types with novel ferroptosis profiles not seen in NSCLC — potential new biology

## Expected Outputs

- Pan-cancer ferroptosis vulnerability ranking across all DepMap cancer types (~30 lineages)
- NRF2/KEAP1 × ferroptosis interaction map by cancer type
- Therapeutic opportunity scores for FSP1i, GPX4i, GLS1i, and combination approaches
- "Ferroptosis analog" cancer type identification (similar vulnerability profile to KEAP1-mutant NSCLC)
- Heatmap visualization: ferroptosis gene dependencies by cancer type and NRF2 status
- Summary table: top 5 cancer types for each class of ferroptosis-targeting therapy

## Success Criteria

1. At least 5 cancer types beyond NSCLC show significant ferroptosis pathway dependencies (mean dependency score < -0.3 for ≥2 ferroptosis defense genes)
2. NRF2/KEAP1 status significantly modulates ferroptosis vulnerability in ≥3 cancer types
3. Identification of ≥2 cancer types where ferroptosis-targeting therapies are currently unexplored clinically but computationally predicted as promising
4. Drug sensitivity data (if available in PRISM) validates computational predictions (r > 0.3 between dependency and drug IC50)

## Labels

novel-finding, drug-screening, genomic, high-priority
