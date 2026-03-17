# Transcriptomic Biomarker Panel for PANDAS/PANS Differential Diagnosis

## Objective
Develop a computational transcriptomic classifier that distinguishes PANDAS/PANS from primary pediatric OCD, tic disorders, and healthy controls using peripheral blood gene expression signatures.

## Background
PANDAS/PANS diagnosis remains clinical and subjective — no validated laboratory biomarker reliably differentiates these autoimmune-driven neuropsychiatric conditions from primary psychiatric disorders. The Cunningham Panel measures select autoantibodies but has debated specificity. Peripheral blood transcriptomics captures the systemic immune activation signature that should distinguish autoimmune-mediated from non-autoimmune neuropsychiatric presentations. Multiple GEO datasets exist for related autoimmune and neuroinflammatory conditions, but no one has built a computational classifier specifically targeting the PANDAS/PANS vs primary psychiatric distinction.

## Data Sources
- **GEO Datasets:**
  - Pediatric autoimmune encephalitis blood transcriptomics (e.g., GSE datasets on anti-NMDAR encephalitis)
  - Pediatric OCD cohort gene expression studies
  - Systemic autoimmune/inflammatory conditions in children (JIA, SLE, rheumatic fever)
  - Streptococcal infection immune response profiles
- **ImmPort:** Pediatric immune profiling studies
- **GTEx:** Baseline expression references for immune genes
- **MSigDB:** Immune and inflammatory gene sets for pathway enrichment

## Methodology
1. **Data Acquisition & Harmonization:** Download and normalize GEO datasets using standard pipelines (RMA/quantile normalization for microarray; DESeq2/edgeR for RNA-seq). Apply batch correction (ComBat/limma) across studies.
2. **Differential Expression Analysis:** Identify DEGs between autoimmune neuropsychiatric (proxy for PANDAS/PANS) vs primary psychiatric vs healthy controls using limma-voom or DESeq2. **MANDATORY: All DE analyses must be run sex-stratified as well as combined** — Rahman SS et al. 2025 (PMID 41254741) demonstrated striking sex-associated differences in monocyte phenotypes in PANS, particularly during recovery. Collapsing sexes risks masking diagnostic signatures.
3. **Weighted Gene Co-expression Network Analysis (WGCNA):** Identify gene modules correlated with autoimmune neuropsychiatric phenotype. Focus on modules enriched in immune/inflammatory pathways.
4. **Feature Selection & Classifier Training:** Use recursive feature elimination + cross-validation to select top discriminatory genes. Include **monocyte M1/M2 polarization markers** (CD14, CD16, CD163, CD206, HLA-DR, CCR2) as candidate classifier features — Rahman SS et al. 2025 showed M1-like inflammatory monocytes elevated during flare and M2-like anti-inflammatory monocytes enriched in recovery, making M1/M2 ratio a strong candidate discriminator. Train Random Forest, XGBoost, and logistic regression classifiers. Evaluate with nested cross-validation (AUC, sensitivity, specificity). **Train sex-stratified models in addition to combined models** — report performance for each.
5. **IFNγ-Target Gene Features:** Include IFNγ-responsive gene sets (HALLMARK_INTERFERON_GAMMA_RESPONSE from MSigDB) as mandatory candidate classifier features. Shammas G et al. 2026, Neuron (PMID 41448185) showed IFNγ drives persistent epigenetic chromatin closing in neurons during encephalitis — peripheral leukocyte IFNγ-target gene expression may serve as proxies for CNS epigenetic damage. Cross-reference with Keating NTI164 EPIC v2.0 methylation data (if deposited to GEO/ArrayExpress) to identify IFNγ-responsive genes with concordant epigenetic and transcriptomic changes in PANS leukocytes. This connects the classifier to the IFNγ persistence mechanism that explains symptom chronicity.
6. **Pathway Enrichment:** GSEA and over-representation analysis on classifier genes to validate biological plausibility (expect complement, Th17, BBB, autoantibody, **IFNγ-epigenetic** pathways).
6. **External Validation:** Hold out one dataset for independent validation. Compare classifier performance to Cunningham Panel components.

## Expected Outputs
- Ranked gene signature panel (target: 20-50 genes) for PANDAS/PANS vs primary psychiatric distinction
- Trained ML classifier with cross-validated performance metrics
- Pathway enrichment analysis revealing key biological axes
- Comparison with Cunningham Panel-associated genes
- Publication-ready figures: volcano plots, ROC curves, heatmaps, pathway networks

## Success Criteria
- Classifier AUC ≥ 0.80 on held-out validation set
- Gene signature enriched in biologically plausible immune/inflammatory pathways
- Signature includes at least some genes not in current Cunningham Panel (demonstrating added value)

## Labels
biomarker, novel-finding, autoimmune, immunology, clinical, high-priority
