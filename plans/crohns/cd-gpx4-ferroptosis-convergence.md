# cd-gpx4-ferroptosis-convergence

**Proposed by:** research_catalyst (Run 4, 2026-03-23)
**Labels:** catalyst, novel-finding, high-priority, cross-project

## Objective

Test whether GPX4-regulated ferroptosis is a mechanistic bridge between epithelial cell death, ILC2 dysfunction, and fibrotic progression in Crohn's disease, and whether epithelial GPX4 expression stratifies patients into prognostically distinct treatment-response trajectories.

**Falsifiable hypothesis**: GPX4 expression in epithelial and/or ILC2 populations is significantly lower in anti-TNF non-responders compared to responders (effect size > 0.5, FDR < 0.05), AND correlates with ferroptosis pathway activation scores AND fibrosis gene signatures in the same samples.

## Rationale

Convergent signal across 4+ independent projects in the crohns division:

1. **il23-single-cell-response-atlas**: GPX4-dependent ferroptosis governs ILC2 homeostasis (Liu et al. Cell Mol Immunol 2026). ILC2s are the second-most IL-23-responsive population. Loss of ILC2 function via ferroptosis could shift the balance toward pro-inflammatory programs.

2. **anti-tnf-response-prediction**: The corrected 30-gene non-response signature contains three regulated cell death genes that converge on ferroptosis biology:
   - GSDMC (#1 SHAP feature): gasdermin-mediated pyroptosis shares lipid peroxidation/membrane pore mechanisms with ferroptosis
   - NFS1 (Fe-S cluster biosynthesis): directly regulates intracellular iron availability, a key ferroptosis trigger
   - ACAD9 (mitochondrial complex I assembly): mitochondrial dysfunction drives oxidative stress → lipid peroxidation → ferroptosis

3. **cd-fibrosis-drug-repurposing (literature)**: Verstockt et al. (Gastroenterology 2026, PMID 41850538) — reduced epithelial GPX4 expression predicts postoperative CD recurrence across 3 independent cohorts. GPX4 is identified as a druggable target.

4. **cd-flare-longitudinal-prediction**: Pre-flare metabolic signatures (urate, 3-hydroxybutyrate) and succinotype suggest metabolic/oxidative stress precedes clinical flare — consistent with ferroptosis-driven tissue damage.

## Data Sources

- **GSE134809** (IL-23 atlas, 97K cells): GPX4 expression available per cell type. Can score ferroptosis pathway and correlate with ILC2 function, epithelial integrity.
- **GSE16879, GSE12251, GSE73661** (anti-TNF cohorts, 83 samples total): Bulk transcriptomics. GPX4 expression measurable. Can correlate with response status and the existing 30-gene signature.
- **GSE282122** (Thomas et al. Nature Immunology 2024): ~1M cells, 216 biopsies, longitudinal anti-TNF scRNA-seq. Just added to IL-23 atlas pipeline — could provide single-cell resolution of GPX4 in treatment context.
- **Fibrosis cohorts**: Any transcriptomic data used by cd-stricture-risk-prediction or cd-fibrosis-drug-repurposing.

## Methodology

### Phase 1: GPX4 Expression Landscape (developer tasks)
1. Extract GPX4 expression from GSE134809 annotated h5ad. Score per cell type (especially epithelial, ILC2, inflammatory macrophage). Compare IL-23-high vs IL-23-low samples.
2. Build ferroptosis gene signature (GPX4, SLC7A11, ACSL4, LPCAT3, ALOX15, TFRC, FTH1, FTL, NFS1) and score across all cell types.
3. Extract GPX4 expression from anti-TNF bulk cohorts. Test association with treatment response (responder vs non-responder).

### Phase 2: Cross-Compartment Analysis (analyst tasks)
4. In GSE134809: correlate epithelial GPX4 with ILC2 ferroptosis score, fibrosis gene module (ILK, COL1A1, ACTA2, FAP), and inflammatory gene module within the same samples.
5. In anti-TNF cohorts: test whether GPX4 expression improves prediction beyond the existing 30-gene signature. Run LOSO-CV with GPX4 alone and GPX4 + top features.
6. Cross-reference: do samples with low epithelial GPX4 in scRNA-seq show the same innate-dominant/Th17-absent profile as anti-TNF non-responders?

### Phase 3: Ferroptosis-Fibrosis Bridge (analyst tasks)
7. In stricture/fibrosis data: test GPX4 association with fibrotic progression markers.
8. Build a "ferroptosis-fibrosis index" combining GPX4, NFS1, GSDMC, ACAD9, ILK — test as multivariate predictor.
9. Check if iron metabolism genes (TFRC, FTH1, FTL, HAMP, SLC40A1) differ between CD subtypes from microbiome-metabolome project.

## Success Criteria

1. GPX4 is differentially expressed between anti-TNF responders and non-responders (p < 0.05, |logFC| > 0.5 or |Cohen's d| > 0.5) in at least 2/3 cohorts.
2. Ferroptosis pathway score is significantly elevated in at least one cell type in IL-23-high vs IL-23-low CD samples (FDR < 0.05).
3. GPX4 expression correlates with fibrosis gene module (|rho| > 0.3, p < 0.05) in at least one dataset.
4. The ferroptosis-fibrosis index (criterion 8) achieves AUC > 0.65 for treatment response prediction when added to existing features.

Meeting 2/4 criteria = partial validation worth publishing. Meeting 3/4 = strong validation. Meeting 0-1/4 = hypothesis refuted (still publishable as negative result documenting the investigation).

## Risk Assessment

- **Most likely failure mode (~40%)**: GPX4 expression is insufficiently variable in bulk transcriptomics to stratify patients. Epithelial GPX4 signal may be diluted in bulk data (works at single-cell level but not bulk).
- **Mitigation**: Phase 1 starts with single-cell data where GPX4 signal is strongest. If bulk fails but scRNA-seq succeeds, the finding is still valuable — just harder to translate to clinical biomarker.
- **Second failure mode (~20%)**: GSDMC finding may not survive the anti-TNF pipeline re-run with leakage fix. If GSDMC drops out, the pyroptosis-ferroptosis connection weakens.
- **What we learn even if it fails**: Whether ferroptosis markers have any variance in CD tissue, and whether regulated cell death modality is orthogonal to immune signaling as a predictor.
