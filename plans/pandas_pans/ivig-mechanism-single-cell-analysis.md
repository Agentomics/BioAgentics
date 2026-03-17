# IVIG Treatment Mechanism Single-Cell Analysis

## Objective
Characterize cell-type-specific immune changes induced by IVIG treatment in PANS using single-cell RNA sequencing, and identify pre-treatment immune cell signatures that predict treatment response.

## Background
IVIG (intravenous immunoglobulin) is the primary immunomodulatory treatment for moderate-to-severe PANS/PANDAS, but it is expensive (~$10K per infusion), access is contested (AAP report vs. PANDAS Network/PPN rebuttals), and response is variable. Han VX et al. (Neurology N&NI 2025; 12(6):e200467) published the first single-cell RNA sequencing study of PANS patients pre- and post-IVIG, profiling 144,470 cells across 5 PANS children and 4 healthy controls. Key findings: pre-IVIG PANS showed downregulated defense response, innate immunity, and secretory granules; post-IVIG these abnormalities reversed; histone modification pathways were downregulated in neutrophils/NK cells after treatment.

No computational re-analysis has been performed on this dataset. A systematic single-cell analysis could reveal: (1) which immune cell types are most dysregulated in PANS, (2) cell-type-specific IVIG mechanism of action, (3) pre-treatment signatures predicting treatment response, and (4) convergence with our other initiatives (cytokine networks, autoantibody targets).

## Data Sources
- **Primary:** Han VX et al. IVIG scRNA-seq dataset — Neurology N&NI 2025, DOI: 10.1212/NXI.0000000000200467. 144,470 cells, 11 cell clusters, 5 PANS + 4 controls, pre/post-IVIG timepoints. **GEO accession to be confirmed by data_curator** — may be deposited under GSE293230 or a separate accession.
- **Companion:** Han VX et al. bulk RNA-seq — Mol Psychiatry 2025, GEO: GSE293230. PANS (n=32) vs non-PANS NDD (n=68) vs controls (n=58). For cross-validation of cell-type deconvolution signatures.
- **Reference:** PBMCpedia harmonized PBMC scRNA-seq database (bioRxiv 2025) for standardized cell type annotation and healthy baseline.
- **Validation context:** Cunningham Panel data (CaMKII sensitivity 88%, specificity 83-92%) for correlating autoantibody-related gene modules with clinical response.

## Methodology
### Phase 1: Data Acquisition & QC
- Confirm GEO deposition of IVIG scRNA-seq data. If not deposited, contact authors (institutional affiliation: Kids Neuroscience Centre, Sydney).
- Download and QC: cell filtering (nFeature, nCount, mitochondrial %), doublet removal (scDblFinder), ambient RNA correction (SoupX).
- Integrate with PBMCpedia reference for standardized cell type annotation.

### Phase 2: Cell-Type Resolution Analysis
- Re-cluster and annotate cell types at fine resolution (beyond original 11 clusters). Focus on monocyte subsets (classical/intermediate/non-classical), T cell subsets (Th1/Th2/Th17/Treg), NK cell subtypes, and B cell maturation states.
- Differential abundance analysis: which cell types change in proportion pre vs post IVIG? PANS vs controls?
- Pseudobulk differential expression per cell type (pre-IVIG vs control, post-IVIG vs control, pre vs post).

### Phase 3: Pathway & Network Analysis
- Gene set enrichment per cell type using GO, KEGG, Reactome.
- **Autophagy pathway scoring:** Specifically assess ATG7, UVRAG, and BECN1 expression pre/post IVIG in monocyte clusters. Singh SC et al. 2026 (PMID 41822482) showed IVIG upregulates core autophagy genes in monocytes with cell-type-specific selective autophagy programs. This connects to mTOR→autophagy axis (Fronticelli Baldelli G 2025, PMID 41462744).
- **S100A12-TLR4-MYD88 axis assessment:** Evaluate S100A12 expression in CD14+CD16 monocytes as a candidate predictor of IVIG non-response. Feng C et al. 2025 (PMID 41394864) identified this axis in IVIG-non-responsive Kawasaki disease, along with T/NK exhaustion and disrupted Tfh-B cell coordination.
- **Cell-cell communication analysis:** Apply CellChat or NicheNet framework (as in Singh et al.) to characterize how IVIG remodels immune cell communication networks.
- Cross-reference with cytokine-network-flare-prediction cytokine gene sets (IL-17, TNF-α, IL-6, TGF-β1, complement).
- Cross-reference with autoantibody-target-network-mapping pathways — do IVIG-responsive cell types express autoantibody target-related genes?
- Histone modification pathway deep-dive in neutrophils/NK cells (novel finding from original paper).

### Phase 4: Treatment Response Signatures
- Identify gene modules that distinguish pre-IVIG PANS from controls (disease signature) and that normalize post-IVIG (treatment-responsive signature).
- Score disease and treatment-responsive modules across all cells.
- If clinical response data is available: correlate module scores with symptom improvement.
- Define a minimal gene signature for predicting IVIG responders.

### Phase 5: Integration & Cross-Initiative Validation
- Deconvolve bulk RNA-seq (GSE293230) using single-cell reference to estimate cell type proportions in the larger PANS cohort.
- Validate treatment-responsive gene modules in deconvolved bulk data.
- Map IVIG-responsive pathways onto autoantibody target network (from autoantibody-target-network-mapping).
- Check whether HLA-associated genes (from hla-peptide-presentation-modeling) show cell-type-specific expression changes.

## Expected Outputs
1. Fine-resolution immune cell atlas of PANS (pre/post-IVIG + controls)
2. Cell-type-specific differential expression profiles and pathway enrichment
3. IVIG mechanism model: which cell types respond, which pathways change, temporal dynamics
4. Pre-treatment gene signature predicting IVIG response (candidate biomarker panel)
5. Cell type deconvolution reference for bulk RNA-seq PANS studies
6. Integration analysis linking IVIG mechanism to autoantibody/cytokine/HLA findings

## Success Criteria
- Recover known IVIG effects (reduced pro-inflammatory monocytes, histone modification changes) from re-analysis
- Identify at least 3 cell-type-specific gene modules that distinguish PANS from controls
- Define a treatment-responsive signature of ≤50 genes that normalizes post-IVIG
- Successfully deconvolve GSE293230 bulk data using the single-cell reference

## Labels
biomarker, immunology, clinical, novel-finding, high-priority, drug-repurposing
