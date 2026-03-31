# cd-microbiome-stromal-mediation

## Objective

Test the falsifiable hypothesis that **gut microbiome composition mediates the relationship between mucosal inflammation and stromal tissue proportion in Crohn's disease**, and that this microbiome-to-stroma axis explains why biopsy microbiome classifiers (AUC=0.90) massively outperform bulk transcriptomic classifiers (AUC=0.71) for predicting biologic treatment response.

**Primary hypothesis (H1):** Microbiome composition (specifically A. muciniphila abundance and M. gnavus abundance) significantly predicts deconvolved stromal proportion in CD tissue, after adjusting for inflammation severity.

**Secondary hypothesis (H2):** Microbiome-metabolome subtypes (from microbiome-metabolome-subtyping project) map to distinct stromal proportion states, with Subtype 0 (high A. muciniphila) showing lower stromal content than Subtype 1.

## Rationale

Five independent projects converged on a single mechanistic chain that nobody anticipated:

1. **cd-epithelial-reserve-treatment-stratification (PIVOT):** Stromal proportion (fibroblast + myofibroblast + endothelial) is the strongest cell-type predictor of anti-TNF response (AUC=0.667), beating epithelial (0.526) and immune (0.616) proportions. Lower stroma → better response. Consistent across 3 cohorts.

2. **cd-gpx4-ferroptosis-convergence:** A. muciniphila upregulates GPX4 and downregulates ACSL4 (from literature). GPX4 correlates with fibrosis markers (rho=0.549-0.723). ACSL4 drives fibroblast→epithelial paracrine ferroptosis. Mechanism: A. muciniphila → GPX4/ACSL4 → ferroptosis modulation → fibroblast survival → stromal remodeling.

3. **microbiome-metabolome-subtyping:** Identified reproducible CD subtypes with distinct A. muciniphila and Bifidobacterium profiles. GSH (GPX4 substrate) and tryptophan-NAD pathways differentiate subtypes.

4. **anti-tnf-response-prediction:** Bulk transcriptomic classifier ceiling ~0.71. Biopsy microbiome (Zafeiropoulou et al., 2026) achieves 0.90. The AUC gap demands a mechanistic explanation.

5. **Literature:** Zafeiropoulou (Front Cell Infect Microbiol 2026): M. gnavus = non-response, Blautia = response in biopsy 16S. These taxa are also key players in HMP2 microbiome subtypes.

**The unconventional insight:** Everyone assumes microbiome predicts treatment response through direct immune modulation. We propose it works through tissue architecture instead — the microbiome shapes stromal composition, and it is the stroma that determines drug efficacy. This inverts the standard "microbiome → immune → response" model to "microbiome → stroma → response."

## Data Sources

**Primary dataset: HMP2/IBDMDB** (already curated by cd-flare-longitudinal-prediction)
- Host RNA-seq from mucosal biopsies (for deconvolution)
- 16S rRNA + metagenomic sequencing (microbiome composition)
- Metabolomics (including untargeted LC-MS)
- Clinical metadata (disease activity scores, some treatment annotations)
- Longitudinal design: can test temporal relationships

**Deconvolution reference: GSE134809** (already processed by il23-single-cell-response-atlas and cd-epithelial-reserve-treatment-stratification)
- scRNA-seq atlas of CD ileum used as cell-type reference for NNLS deconvolution

**Microbiome subtypes:** Output from microbiome-metabolome-subtyping project (CMTF-derived subtypes with 19-species NC classifier)

## Methodology

### Phase 1: HMP2 Deconvolution (Development)
1. Apply NNLS deconvolution to HMP2 host RNA-seq using GSE134809 scRNA-seq reference (same method as epithelial reserve project)
2. Extract per-sample cell-type proportions: epithelial, stromal (fibroblast + myofibroblast + endothelial), immune
3. Quality check: compare deconvolution results with known CD inflammation patterns
4. Output: per-sample stromal proportion estimates for all HMP2 CD subjects

### Phase 2: Microbiome-Stromal Associations (Analysis)
1. **Univariate tests:** Correlate A. muciniphila abundance, M. gnavus abundance, Blautia abundance, and alpha diversity with stromal proportion (Spearman, FDR-corrected)
2. **Multivariate model:** PERMANOVA / distance-based regression: does microbiome beta diversity explain stromal proportion variance after adjusting for calprotectin, CRP, and biopsy location?
3. **Subtype mapping:** Assign HMP2 CD samples to microbiome subtypes using 19-species NC classifier from microbiome-metabolome-subtyping. Compare stromal proportion between subtypes (Wilcoxon, effect size).
4. **Temporal analysis:** In longitudinal pairs, does microbiome change precede stromal proportion change? (Granger-style or cross-lagged correlation)

### Phase 3: Mediation Analysis (Analysis)
1. Formal mediation test: inflammation → microbiome composition → stromal proportion (using HMP2 calprotectin as inflammation proxy)
2. Alternative mediation: microbiome → metabolites (tryptophan, glutathione) → stromal proportion
3. Sensitivity: test reverse direction (stromal → microbiome) to assess bidirectionality
4. Identify specific taxa driving the mediation effect

### Phase 4: Integration (Validation)
1. Cross-reference mediation findings with Zafeiropoulou's anti-TNF predictive taxa
2. Test whether A. muciniphila → stromal pathway explains GPX4-fibrosis correlation
3. If HMP2 has treatment annotations: test whether microbiome-predicted stromal proportion associates with treatment outcomes
4. Compare predictive power: raw microbiome vs microbiome-mediated stromal score

## Success Criteria

1. **SC1 (Primary):** A. muciniphila and/or M. gnavus abundance significantly correlates with deconvolved stromal proportion (|rho| > 0.25, FDR < 0.05)
2. **SC2:** Microbiome subtypes show significantly different stromal proportions (Wilcoxon p < 0.05, effect size d > 0.3)
3. **SC3:** Formal mediation analysis shows significant indirect effect (microbiome → stromal proportion pathway, bootstrap p < 0.05)
4. **SC4 (Stretch):** Taxa driving mediation overlap with Zafeiropoulou's anti-TNF predictive taxa (≥2 of top 5 taxa shared)

Meeting SC1 + SC2 = hypothesis supported. Meeting SC3 = strong evidence. SC4 = clinical translation pathway confirmed.

## Risk Assessment

**Most likely failure mode:** Microbiome and stromal proportion are both correlated with inflammation severity but have no independent relationship. Adjusting for inflammation (calprotectin) could eliminate the association.

**Mitigation:** Phase 2 explicitly adjusts for inflammation. A null result after adjustment is still informative — it would mean microbiome predicts response through immune, not stromal, mechanisms, which is useful knowledge.

**Second risk:** HMP2 deconvolution quality may be poor (bulk RNA-seq from heterogeneous biopsies). The epithelial reserve project already validated NNLS on anti-TNF cohorts; HMP2 may behave differently.

**Mitigation:** Phase 1 includes quality checks. If deconvolution fails, fall back to fibrosis/EMT gene scores (which don't require deconvolution).

**Third risk:** HMP2 has limited treatment response data, so Phase 4 may be underpowered.

**What we learn from failure:** If microbiome does NOT predict stromal proportion, it means the AUC gap between microbiome and transcriptomic classifiers is NOT explained by tissue architecture — the microbiome captures a different signal entirely. This would redirect the field toward direct microbiome-immune interactions instead.

## Labels

catalyst, cross-project, novel-finding, microbiome, high-priority
