# RB1-Loss Pan-Cancer Dependency Atlas

## Objective

Map synthetic lethal dependencies specific to RB1-loss cancers across all DepMap tumor types, to identify targetable vulnerabilities in SCLC, neuroendocrine prostate cancer, bladder cancer, and other RB1-null malignancies — with clinical alignment to CDK2 inhibitors and emerging SL targets.

## Background

RB1 (retinoblastoma protein) is a master cell cycle regulator that restrains E2F-driven transcription. RB1 loss deregulates the G1/S checkpoint, causing constitutive E2F activation and cyclin E/CDK2 overactivity. RB1 is among the most frequently inactivated tumor suppressors across cancer types.

**RB1 loss prevalence:**
- **SCLC:** ~95% biallelic RB1 inactivation (near-universal, defines the disease)
- **Bladder cancer:** ~15-20% (enriched in neuroendocrine variant)
- **Prostate cancer:** ~10-15% in treatment-naive; ~30-50% in neuroendocrine prostate cancer (NEPC)
- **Triple-negative breast cancer (TNBC):** ~10-15%
- **Osteosarcoma:** ~60-70%
- **Retinoblastoma:** ~100% (but pediatric, small population)

**Known SL vulnerabilities in RB1-loss:**
- **CDK2:** RB1 loss creates cyclin E/CDK2 overactivity. CDK2 inhibitors show selective efficacy in RB1-deficient preclinical models. INX-315 (Incyte), PF-07220060 (Pfizer) in Phase 1/2 clinical trials specifically enrolling CDK2-dependent (CCNE1-amplified and RB1-loss) tumors.
- **Aurora kinases (AURKA/AURKB):** Mitotic checkpoint compensation for G1/S checkpoint loss. Alisertib (AURKA) showed activity in SCLC.
- **CHK1/WEE1:** Replication stress from deregulated S-phase entry. RB1-loss cells are hypersensitive to S-phase checkpoint inhibitors.
- **Casein Kinase 2 (CK2):** Recent discovery (Science Advances 2024) — RB1 loss in TNBC/HGSC creates SL dependency on CK2 when treated with replication-perturbing agents (carboplatin, gemcitabine, PARPi).
- **TTK/MPS1:** Spindle assembly checkpoint kinase. RB1-loss cells depend on mitotic fidelity checkpoints.
- **E2F targets:** FOXM1, MYBL2 as downstream mediators of RB1-loss proliferation.

**Clinical gap:** SCLC has a 5-year survival rate <7%. No targeted therapies are approved. Platinum + etoposide + IO (atezolizumab/durvalumab) is standard, but nearly all patients relapse. CDK2 inhibitors represent the first SL-based targeted therapy for RB1-null cancers.

**Cross-project links:**
- CDKN2A project: CDKN2A/p16 is upstream of CDK4/6-RB1. CDKN2A deletion bypasses p16 control of CDK4/6 → RB1 phosphorylation. Code and methodology from the CDKN2A atlas directly transfers (differential dependency analysis, effect size quantification, PRISM validation).
- TP53 project: TP53 and RB1 are co-inactivated in SCLC. Our TP53 allele-specific dependencies may intersect with RB1-loss context.

## Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv — CDK2, AURKA, AURKB, CHK1, WEE1, CSNK2A1 (CK2), TTK dependency scores
- **DepMap model annotations:** Model.csv — cancer type, subtype, RB1 status
- **DepMap mutations:** OmicsSomaticMutations.csv — RB1 truncating mutations, deletions
- **DepMap copy number:** OmicsCNGene.csv — RB1 deep deletions, CCNE1 amplification
- **DepMap expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv — RB1 expression (proxy for functional loss including epigenetic silencing)
- **PRISM 24Q2:** Drug sensitivity for CDK2 inhibitors (if available), CDK4/6 inhibitors, Aurora inhibitors, CHK1/WEE1 inhibitors
- **TCGA:** RB1 alteration frequency by cancer type, survival analysis, co-occurring mutations (cBioPortal pan-cancer)

## Methodology

### Phase 1 — RB1-Loss Classifier
- Classify all DepMap cell lines by RB1 functional status using multi-modal approach:
  - RB1 truncating/frameshift mutations (biallelic)
  - RB1 deep deletion (CN < 0.5)
  - RB1 expression loss (bottom quartile, proxy for epigenetic silencing)
- Cross-validate: RB1-loss lines should show CDK4/6 inhibitor resistance (positive control — no RB to phosphorylate)
- Report qualifying cancer types (>=5 RB1-loss + >=10 RB1-intact lines with CRISPR data)
- Annotate CCNE1 amplification as co-occurring feature (CDK2 dependency intensifier)
- Expected qualifying types: SCLC, Bladder, TNBC, Prostate, Osteosarcoma, potentially Ovarian/Endometrial

### Phase 2 — Candidate Dependency Analysis
- For established SL candidates (CDK2, AURKA, AURKB, CHK1, WEE1, CSNK2A1/CK2, TTK, FOXM1):
  - Compare CRISPR dependency in RB1-loss vs RB1-intact per qualifying cancer type
  - Statistics: Cohen's d effect size, Mann-Whitney U test, bootstrap 95% CI (1000 iterations, seed=42)
  - BH FDR correction
  - Classify as ROBUST / MARGINAL / NOT SIGNIFICANT per tumor type
- Negative controls: CDK4, CDK6 (should NOT show SL with RB1-loss — no target to phosphorylate)
- **Key question:** Which known SL dependencies are universal vs tissue-specific?

### Phase 3 — Genome-Wide RB1-Loss Dependency Screen
- Screen all ~18K genes for RB1-loss-specific dependencies per qualifying cancer type
- Identify novel RB1-loss SL partners beyond known cell cycle/DDR targets
- Fisher enrichment against SL benchmark datasets
- Pathway enrichment analysis (KEGG, Reactome) — expect cell cycle, DNA replication, mitotic checkpoints
- Anti-correlated dependency pairs (genes gained as dependencies in RB1-loss)
- Compare with CDKN2A atlas findings: which dependencies are shared (pathway-level) vs unique to RB1 loss?

### Phase 4 — PRISM Drug Sensitivity Validation
- Map relevant drug classes in PRISM to RB1 status:
  - CDK2 inhibitors (if available), CDK4/6 inhibitors (negative control — should lose efficacy)
  - Aurora kinase inhibitors (alisertib, barasertib)
  - CHK1 inhibitors (prexasertib), WEE1 inhibitors (adavosertib, azenosertib)
  - Platinum agents and etoposide (SCLC standard of care)
  - PARPi (cross-reference with CK2 SL finding — RB1-loss + PARPi → CK2 dependency)
- CRISPR-PRISM concordance: do gene dependencies predict drug sensitivity?
- CDK4/6i resistance as positive control for RB1-loss classification accuracy

### Phase 5 — Clinical Integration & CDK2 Landscape
- RB1 alteration frequency per cancer type from TCGA pan-cancer
- Estimate addressable patient populations per cancer type per year
- Survival analysis: RB1-loss vs RB1-intact by cancer type
- **CDK2 inhibitor clinical landscape:** INX-315 (Incyte Phase 1/2), PF-07220060 (Pfizer Phase 1), others
  - Map trial enrollment criteria, tumor type focus, biomarker selection (CCNE1 amp vs RB1 loss)
  - Predict which tumor types have highest CDK2 dependency from DepMap × highest RB1-loss prevalence
- Priority matrix: RB1-loss SL strength × population size × drug availability
- Co-alteration analysis: RB1 + TP53, RB1 + MYC amplification, RB1 + CCNE1 amplification as co-dependency modifiers

## Expected Outputs

1. Pan-cancer RB1-loss dependency atlas: effect sizes per known SL target per tumor type
2. Classification of tumor types as ROBUST / MARGINAL / NOT SIGNIFICANT for each SL dependency
3. Novel RB1-loss-specific dependencies from genome-wide screen
4. Comparison with CDKN2A atlas: shared pathway-level vs RB1-unique dependencies
5. Drug sensitivity validation mapped to CRISPR dependencies
6. CDK2 inhibitor clinical alignment: which tumor types should trials prioritize?
7. Population-weighted priority ranking for RB1-loss targeted therapy development
8. Co-alteration dependency modifiers (TP53 co-loss, CCNE1 co-amplification)

## Success Criteria

- **CDK2 positive control:** Significant SL effect (FDR<0.05) for CDK2 in >=2 RB1-loss cancer types (expect SCLC, TNBC)
- **CDK4/6 negative control:** CDK4/CDK6 dependency is LOST in RB1-null lines (validates classifier)
- **Novel findings:** >=1 novel RB1-loss SL dependency beyond known cell cycle targets
- **SCLC actionability:** Identify >=3 targetable dependencies in SCLC with drugs in clinical development
- **CDKN2A comparison:** Distinguish RB1-specific from CDKN2A-pathway-shared dependencies
- **Clinical concordance:** CDK2 dependency ranking aligns with CDK2 inhibitor trial tumor type selection

## Labels

genomic, drug-screening, drug-candidate, novel-finding, clinical, high-priority
