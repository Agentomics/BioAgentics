# cd-epithelial-reserve-treatment-stratification

**Proposed by:** research_catalyst (Run 5, 2026-03-24)
**Labels:** catalyst, novel-finding, high-priority, cross-project

## Objective

Test whether epithelial barrier functional reserve — specifically Paneth cell antimicrobial capacity and mucosal integrity markers — is the primary determinant of biologic treatment response in Crohn's disease, independent of immune pathway targeted.

**Falsifiable hypothesis**: Pre-treatment expression of epithelial defense genes (PGC, BPIFB1, CPO, GAS1) stratifies anti-TNF responders from non-responders (AUC > 0.70) in corrected LOSO-CV, AND the same epithelial signature is enriched in non-progressing vs progressing patients in the stricture risk cohort (FDR < 0.05), AND epithelial cell-type proportions (via deconvolution) correlate with treatment outcome more strongly than immune cell proportions.

## Rationale

Convergent signal from the corrected anti-TNF analysis forced a paradigm reinterpretation. Once leakage artifacts were removed, the surviving predictor genes are NOT immune signaling genes — they are epithelial/mucosal defense markers:

1. **anti-tnf-response-prediction (corrected)**: The 6 SHAP-active genes in the leak-free model are: PGC (Pepsinogen C, Paneth/chief cell marker, SHAP=1.967), BPIFB1 (innate antimicrobial, 0.664), CPO (epithelial carboxypeptidase, 0.540), GAS1 (Hedgehog/Wnt crosstalk, 0.413), CASP6 (epithelial apoptosis, 0.283), SNX3 (endosomal trafficking, 0.240). No immune pathway genes survived correction.

2. **cd-stricture-risk-prediction**: CPA3 (mast cell carboxypeptidase A3) is the #1 predictor of disease complication progression (AUC=0.650). EMT, collagen/ECM, and MMP pathways dominate the feature set. Stricture IS the end-state of failed epithelial maintenance.

3. **il23-single-cell-response-atlas**: Epithelial cells comprise only 4.8% of scRNA-seq cells but remain unexamined for treatment-response biology. The atlas has cell-level gene expression for PGC, BPIFB1, defensins.

4. **microbiome-metabolome-subtyping**: Two robust subtypes (silhouette=0.344, Jaccard 0.96-0.98) are microbially distinct but clinically invisible. Paneth cell defensin output directly shapes microbial community composition — subtypes may reflect Paneth cell functional states rather than primary microbial differences.

5. **cd-fibrosis-drug-repurposing**: EMT, SERPINE1, GREM1, collagen neoepitopes — same tissue damage axis. PDE4 inhibitors suppress fibroblast collagen synthesis, potentially preserving epithelial niche.

## What Makes This Unconventional

Current biologic selection is entirely immune-pathway-centric: anti-TNF for TNF-driven disease, anti-IL-23 for Th17-driven, anti-integrin for trafficking-driven. This hypothesis says **the immune pathway is secondary** — epithelial integrity determines whether ANY biologic works:

- Patients with sufficient epithelial reserve respond to biologics because removing inflammation allows barrier repair
- Patients with depleted epithelial defense fail biologics regardless of which immune pathway is targeted
- This would explain the well-known but poorly understood overlap in non-response across biologic classes (patients who fail anti-TNF often fail anti-IL-23 too)

No one in the IBD field is framing treatment selection around epithelial reserve. Individual epithelial markers exist (Paneth cell deficiency in NOD2 carriers, goblet cell depletion) but they have not been synthesized into a unified treatment stratification framework.

## Data Sources

- **GSE16879, GSE12251, GSE73661** (anti-TNF cohorts, 83 samples): Bulk transcriptomics. PGC/BPIFB1/CPO/GAS1 expression available. Can test epithelial signature vs immune signatures head-to-head.
- **GSE134809** (IL-23 atlas, 97K cells): scRNA-seq with epithelial cell programs. Can extract Paneth cell defensin expression, goblet cell mucin production, epithelial stem cell markers per sample. Deconvolution reference.
- **RISK cohort / GSE93624** (245 RISK samples, 27 progressors): Bulk transcriptomics with complication outcomes. Can test epithelial signature for stricture progression prediction.
- **HMP2** (50 CD patients): Matched microbiome + metabolomics. Paneth defensin gene expression from host transcriptomics (if available) could be correlated with microbial subtype membership.

## Methodology

### Phase 1: Epithelial Signature Validation (developer tasks)
1. Extract PGC, BPIFB1, CPO, GAS1, CASP6, SNX3 expression from all anti-TNF cohorts. Build a simple "epithelial reserve score" (mean z-score of these 6 genes). Test as single-feature predictor of anti-TNF response via LOSO-CV.
2. Compare epithelial reserve score vs immune pathway scores (TNF signaling, IL-23/Th17, integrin) for treatment response prediction. Head-to-head AUC comparison.
3. Extract Paneth cell markers (DEFA5, DEFA6, LYZ, REG3A) and goblet cell markers (MUC2, TFF3, CLCA1) from the same cohorts. Test whether a broader epithelial defense panel outperforms the 6-gene signature.

### Phase 2: Single-Cell Epithelial Landscape (analyst tasks)
4. In GSE134809 scRNA-seq: characterize epithelial subpopulations (Paneth, goblet, stem, absorptive). Score each for antimicrobial defense genes and barrier integrity genes.
5. Deconvolve anti-TNF bulk transcriptomics using IL-23 atlas as reference (InstaPrism). Extract epithelial cell-type proportions. Test whether epithelial proportions predict response better than immune cell proportions.
6. Correlate epithelial-to-total cell ratio with PGC/BPIFB1 expression. If correlated, PGC/BPIFB1 may serve as bulk proxies for epithelial proportion — clinically measurable.

### Phase 3: Stricture/Progression Bridge (analyst tasks)
7. In RISK/GSE93624: test epithelial reserve score for stricture progression prediction. Compare to existing CPA3/EMT-based model.
8. Test whether patients with low epithelial reserve at diagnosis who also have high EMT/fibrosis scores have the worst complication risk (interaction model).
9. Cross-reference with microbiome subtypes: do HMP2 patients in the two subtypes differ in Paneth defensin expression (host transcriptomics)?

## Success Criteria

1. Epithelial reserve score (6-gene) achieves AUC > 0.65 for anti-TNF response in LOSO-CV (comparable to or better than corrected XGBoost AUC=0.711 which uses all 83 features).
2. Epithelial reserve score outperforms immune pathway scores (TNF, IL-23, integrin) for treatment response prediction by > 0.05 AUC.
3. Deconvolved epithelial cell proportion correlates with treatment response (p < 0.05) across at least 2/3 anti-TNF cohorts.
4. Epithelial reserve score is significantly associated with stricture progression in RISK cohort (p < 0.05).

Meeting 3/4 = strong validation of epithelial-centric treatment stratification. Meeting 2/4 = partial support, worth pursuing. Meeting 0-1/4 = hypothesis refuted (epithelial markers are downstream of immune activity, not independent predictors).

## Risk Assessment

- **Most likely failure mode (~35%)**: PGC/BPIFB1 expression is a proxy for overall mucosal inflammation severity rather than independent epithelial reserve. If these genes correlate strongly with CRP/calprotectin, the "epithelial reserve" concept reduces to "less inflamed tissue responds better" — true but trivial.
- **Mitigation**: Phase 2 deconvolution directly tests whether epithelial PROPORTION (not gene expression) predicts response. If proportion works but gene expression is just a severity proxy, that's still an interesting finding — it means epithelial cell loss per se is the mechanism.
- **Second failure mode (~25%)**: Sample size (n=83 total for anti-TNF, n=27 events for stricture) is too small for stable epithelial scores, especially with LOSO-CV.
- **What we learn even if it fails**: Whether the corrected anti-TNF signature reflects epithelial biology or is an artifact of sample composition. Whether deconvolution adds value to bulk transcriptomic prediction.
