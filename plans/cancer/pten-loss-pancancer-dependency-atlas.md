# PTEN Loss Pan-Cancer Dependency Atlas

## Objective

Map gene dependencies and drug sensitivities in PTEN-lost (deleted or truncated) vs intact cell lines across all cancer types in DepMap 25Q3, to discover vulnerabilities beyond PI3K/AKT/mTOR pathway inhibition and identify cancer types where PTEN loss creates the strongest therapeutic opportunities.

## Background

PTEN is the third most commonly altered tumor suppressor in human cancer (~8-10% pan-cancer, with high frequency in endometrial ~65%, GBM ~35%, prostate ~20%, breast ~10%). PTEN loss constitutively activates PI3K/AKT signaling, but the dependency landscape beyond direct pathway inhibition is poorly characterized at scale.

**Therapeutic landscape (March 2026):**
- **Capivasertib** (AstraZeneca): FDA-approved Nov 2023 for PIK3CA/AKT1/PTEN-altered HR+/HER2- breast cancer (CAPItello-291)
- **Alpelisib** (Novartis): PI3Kα-selective, approved for PIK3CA-mutant breast cancer — does NOT address PTEN-loss directly
- **Inavolisib** (Genentech): PI3Kα-selective mutant degrader, approved Oct 2024 for PIK3CA-mutant breast
- **Ipatasertib** (Roche): AKT inhibitor, Phase 3 in prostate (IPATential150 — mixed results in PTEN-loss subgroup)
- **Key gap:** PI3K/AKT pathway inhibitors have modest single-agent efficacy in PTEN-lost cancers. What co-dependencies could enable effective combinations?

**Why this atlas is needed:**
1. PTEN loss and PIK3CA mutation both activate PI3K signaling but through distinct mechanisms — dependencies may differ
2. PTEN has phosphatase-independent functions (nuclear PTEN, chromosome stability) — non-PI3K vulnerabilities may exist
3. No pan-cancer ranking of AKT/mTOR dependency strength in PTEN-lost lines
4. Combination strategies are urgently needed for PTEN-lost tumors

**Cross-project links:**
- PIK3CA allele dependencies atlas: same pathway, different genetic mechanism. Cross-reference to identify shared vs divergent dependencies.
- TP53 atlas: PTEN and TP53 co-alteration is common and prognostically adverse. Interaction effects on dependencies?

## Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv — AKT1, AKT2, MTOR, PIK3CA, PIK3CB, RICTOR, RPTOR, and genome-wide dependency scores
- **DepMap copy number:** PortalOmicsCN.csv — PTEN copy number for deletion calling
- **DepMap mutations:** OmicsSomaticMutations.csv — PTEN truncating/missense mutations, PIK3CA, AKT1, TP53 co-mutation status
- **DepMap model annotations:** Model.csv — cancer type classification
- **PRISM 24Q2:** AKT inhibitors, PI3K inhibitors, mTOR inhibitors sensitivity by PTEN status + genome-wide screen
- **TCGA:** PTEN alteration frequency by cancer type, co-occurring alterations, survival data

## Methodology

### Phase 1 — PTEN Loss Classifier
- Classify all DepMap lines by PTEN status: deep deletion (CN ≤ -1.0), truncating mutation (nonsense, frameshift, splice), or both
- Cross-validate with PTEN expression (lost lines should have low expression)
- Annotate PIK3CA co-mutation (double-hit PI3K activation)
- Annotate TP53, RB1 co-alteration status
- Report qualifying cancer types (≥5 PTEN-lost + ≥10 PTEN-intact with CRISPR data)
- Power analysis per cancer type

### Phase 2 — PI3K/AKT Pathway Dependency Effect Sizes
- For each qualifying type: compare AKT1, AKT2, MTOR, PIK3CA, PIK3CB, RICTOR, RPTOR dependency in PTEN-lost vs intact
- Statistics: Cohen's d, Mann-Whitney U, bootstrap 95% CI (1000 iterations, seed=42), BH FDR
- **Key test:** PIK3CB (p110β) should be selectively essential in PTEN-null (known biology — PTEN-null cells depend on p110β rather than p110α)
- Control: PIK3CA should NOT show PTEN-selective dependency (PIK3CA is the gain-of-function partner)
- Classify ROBUST/MARGINAL/NOT SIGNIFICANT per cancer type
- Leave-one-out robustness

### Phase 3 — Genome-Wide PTEN-Selective Dependency Screen
- Screen all ~18K genes for PTEN-lost-specific dependencies per qualifying cancer type
- BH FDR < 0.05, |Cohen's d| > 0.3 thresholds
- Pathway enrichment (PI3K/AKT/mTOR, insulin signaling, DNA damage response, chromatin remodeling)
- SL benchmark enrichment
- Anti-correlation analysis for novel SL candidates
- **Special focus:** non-PI3K dependencies (PTEN nuclear functions, chromosome stability)

### Phase 4 — PRISM Drug Sensitivity
- PI3K pathway inhibitors (capivasertib, alpelisib, everolimus, etc.): PTEN-lost vs intact sensitivity
- Genome-wide PRISM screen for PTEN-selective drug sensitivities
- CRISPR-PRISM concordance for AKT/mTOR axis
- Identify combination drug candidates (non-PI3K drugs with PTEN selectivity)

### Phase 5 — TCGA Clinical Integration
- PTEN alteration frequency per cancer type from TCGA
- Estimate addressable patient populations (PTEN-loss frequency × cancer incidence)
- Priority ranking: dependency effect size × population size × druggability
- Cross-reference with PIK3CA atlas: shared vs divergent vulnerabilities
- Map to active AKT/PI3K/mTOR clinical trials in PTEN-selected populations

## Expected Outputs

- Pan-cancer ranking of AKT/mTOR dependency in PTEN-lost contexts
- PIK3CB positive control validation (known PTEN-null dependency)
- Cancer-type-specific PTEN vulnerability catalogs
- Novel non-PI3K therapeutic targets in PTEN-lost cancers
- PRISM drug sensitivity profiles with combination candidates
- Clinical population estimates and trial recommendations
- Cross-reference with PIK3CA atlas (shared pathway, distinct mechanisms)

## Success Criteria

- PIK3CB positive control: ROBUST PTEN-selective dependency in ≥2 cancer types
- AKT1/AKT2 show PTEN-selective dependency (pathway validation)
- ≥1 novel non-PI3K/AKT dependency identified (FDR < 0.05, |d| > 0.5) in ≥2 cancer types
- Clear differentiation from PIK3CA-mutant dependency profile
- Clinical population estimates for ≥5 cancer types

## Labels

genomic, drug-screening, drug-candidate, novel-finding, clinical
