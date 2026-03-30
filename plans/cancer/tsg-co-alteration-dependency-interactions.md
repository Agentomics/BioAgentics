# Tumor Suppressor Co-Alteration Dependency Interactions

**Initiative:** tsg-co-alteration-dependency-interactions
**Division:** cancer
**Type:** Catalyst — Cross-Project Synthesis
**Source Projects:** pten-loss-pancancer-dependency-atlas, rb1-loss-pancancer-dependency-atlas, arid1a-pancancer-sl-atlas, tp53-hotspot-allele-dependencies, pik3ca-allele-dependencies, brca-pancancer-sl-atlas, pancancer-mtap-prmt5-atlas
**Labels:** catalyst, novel-finding, high-priority, clinical

---

## Objective

**Hypothesis:** Pairwise co-loss of tumor suppressor genes from our atlas series creates dependency interaction effects — synergistic, antagonistic, or redirective — that are not predictable from individual atlas results. Specifically: (1) at least 2 of the top 5 most frequent co-alteration pairs show non-additive interaction effects on SL dependencies, and (2) TP53 co-mutation status systematically modulates SL effect sizes across atlases because TP53-mutant cells lose p53-mediated apoptotic responses to metabolic stress.

This is falsifiable: if all tested co-alteration pairs show purely additive (independent) effects, then individual atlas results can be safely combined without interaction corrections, and the hypothesis fails.

## Rationale

Every published atlas in this division treats its target gene as if it exists in isolation. Real tumors carry 4-7 driver mutations simultaneously. Multiple atlases explicitly flag this gap:

1. **PTEN atlas:** "Co-alteration confounding (TP53 69%, RB1 21%) not fully deconvolved." 21% of PTEN-loss tumors also have RB1 loss. PTEN creates PI3Kb/AKT dependency; RB1 creates CDK2 dependency. Does a PTEN+RB1 double-loss tumor respond to BOTH AKT and CDK2 inhibitors? No one has tested this.

2. **RB1 atlas:** "Upstream vs downstream pathway disruption creates largely distinct dependency landscapes" (comparing CDKN2A). If same-pathway disruption at different points creates different dependencies, cross-pathway co-loss is even less predictable.

3. **ARID1A atlas:** PIK3CA co-mutation does NOT attenuate EZH2 dependency — one tested interaction shows independence. But does ARID1B (the dominant SL) hold in ARID1A+PIK3CA double-mutant cells?

4. **RB1 atlas:** CCNE1 co-amplification ADDITIVELY intensifies CDK2 dependency (d=-2.280). Proof that co-alterations CAN modify SL effect sizes.

5. **Breast stratification project:** Found 39.4% ARID1A x PIK3CA overlap in breast cancer, exceeding the 30% non-overlap threshold. These overlapping patients need interaction data to guide therapy.

6. **TP53 atlas:** 616 shared dependencies across all TP53 mutations. TP53 co-mutation rates exceed 60% in many atlas targets (69% of PTEN-loss, common in BRCA tumors). TP53-mutant cells lose the MDM2/p53 apoptotic checkpoint — any SL dependency that partially works through p53-mediated cell death would be WEAKER in TP53-co-mutant contexts. This could explain why SL effect sizes vary across cancer types: cancer types with high TP53 mutation rates may show attenuated SL effects for p53-dependent mechanisms.

**The unconventional insight:** The division has been building single-gene vulnerability atlases. The RD plans more single-gene atlases (KRAS, STK11). But the clinical reality is multi-gene. This initiative questions whether the atlas framework itself is sufficient, or whether interaction effects make individual atlas results unreliable in real tumors. A 30% failure rate on this hypothesis is acceptable — even negative results (all interactions additive) would validate the current atlas approach.

## Data Sources

- **TCGA Pan-Cancer Atlas:** Co-mutation frequencies for MTAP, ARID1A, SMARCA4, PIK3CA, PTEN, RB1, BRCA1/2, TP53, KEAP1, NF1, CDKN2A (11 genes, 55 pairwise combinations)
- **DepMap 25Q3:** CRISPR dependency data for double-mutant cell lines
- **Published atlas findings:** Individual gene SL dependency gene lists and effect sizes from 6 published atlases
- **Documentation-stage atlases:** RB1, PTEN, TP53, ferroptosis, WRN-MSI, CRC-KRAS findings for additional interaction candidates

## Methodology

### Phase 1: Co-Alteration Frequency Matrix
Using TCGA pan-cancer data, compute pairwise co-alteration frequencies for all 55 combinations of the 11 atlas target genes. Rank by:
- Absolute co-occurrence frequency (patient count)
- Clinical actionability (both genes have atlas-identified drug targets)
- DepMap representativeness (double-mutant lines available)

### Phase 2: DepMap Double-Mutant Identification
For the top 10 co-alteration pairs by frequency x actionability, identify DepMap 25Q3 cell lines carrying BOTH alterations. Minimum N=5 double-mutant lines required per pair to proceed.

Priority pairs (based on atlas findings):
1. **PTEN-loss + RB1-loss** (~21% co-occurrence in PTEN-loss tumors)
2. **ARID1A-mut + PIK3CA-mut** (39.4% overlap in breast; tested in ARID1A atlas for EZH2 only)
3. **TP53-mut + PTEN-loss** (69% co-occurrence in PTEN-loss tumors)
4. **TP53-mut + BRCA1/2-mut** (common in breast/ovarian)
5. **TP53-mut + RB1-loss** (SCLC hallmark, ~95% + ~95%)

### Phase 3: Dependency Interaction Analysis
For each qualifying pair, test:
1. **Independence test:** Do SL dependencies from BOTH individual atlases hold in double-mutant lines? (Two-way ANOVA: gene_A_status x gene_B_status on dependency scores for each atlas-identified SL target)
2. **Interaction test:** Is there a statistically significant interaction term? (Synergistic: double-mutant shows STRONGER dependency than either single. Antagonistic: double-mutant shows WEAKER dependency.)
3. **Emergence test:** Do double-mutant lines show novel dependencies NOT present in either single-mutant atlas?

### Phase 4: TP53 Modulation Analysis
Across ALL published atlases: stratify cancer-type-level SL effect sizes by TP53 mutation frequency. Test whether cancer types with high TP53 mutation rates show systematically weaker SL effects, suggesting p53-dependent apoptosis contributes to observed SL.

### Phase 5: Clinical Implication Framework
For each non-additive interaction:
- Identify the combination therapy implied (e.g., AKT inhibitor + CDK2 inhibitor for PTEN+RB1)
- Estimate the co-alteration patient population
- Flag interactions where one dependency masks another (contraindicated combinations)

## Success Criteria

1. **Power:** At least 3 of the top 5 co-alteration pairs have >=5 double-mutant lines in DepMap
2. **Interaction detection:** At least 2 pairs show non-additive interaction effects (interaction term p<0.05, uncorrected — hypothesis-generating threshold appropriate for catalyst work)
3. **TP53 modulation:** Correlation between cancer-type TP53 mutation rate and atlas SL effect size is r<-0.3 for at least 2 published atlases
4. **Actionability:** At least 1 interaction pair has both targets druggable with approved or clinical-trial agents

## Risk Assessment

- **Primary risk:** DepMap double-mutant sample sizes may be too small (<5 lines per pair). Mitigation: focus on the most common co-alterations; TP53 co-mutation should have adequate power given its high frequency.
- **Secondary risk:** Co-alterations may be confounded by cancer type (e.g., PTEN+RB1 mostly in prostate). Mitigation: include cancer type as a covariate; report both pan-cancer and within-type results.
- **Tertiary risk:** All interactions may be additive (boring). Mitigation: this is still a valuable negative result — it validates the atlas framework and justifies combining individual atlas drug recommendations without interaction corrections.
- **What we learn if it fails:** If dependencies are consistently additive, our atlas approach is validated. If sample sizes are too small, we define the minimum dataset size needed for future interaction studies, guiding DepMap/PRISM expansion priorities.
