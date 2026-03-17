# IL-23/Th17 Single-Cell Response Atlas

## Objective
Build a single-cell transcriptomic atlas of the IL-23/Th17 axis in Crohn's disease mucosa to identify cell-type-specific gene programs predictive of response to IL-23-targeting biologics (ustekinumab, risankizumab).

## Background
The IL-23/Th17 pathway is a validated therapeutic target — ustekinumab and risankizumab are first-line biologics for moderate-severe CD with response rates of 40-60%. No validated biomarkers exist to predict which patients will respond. Single-cell RNA-seq of CD mucosa is now available from multiple studies but hasn't been systematically analyzed for IL-23 pathway biology or treatment response prediction. Most IL-23 studies focus on bulk tissue or animal models, missing the cell-type-specific resolution needed to understand heterogeneous responses. Understanding which cell populations express IL-23 pathway genes and how they differ between responders and non-responders could enable precision biologic selection.

## Data Sources
- **GSE134809** (Kong et al.) — Single-cell RNA-seq of CD ileal tissue, inflamed and non-inflamed
- **GSE150392** (Martin et al.) — Single-cell atlas of inflamed human gut with treatment metadata
- **GSE207617** — Single-cell CD with clinical/treatment annotations
- **Single Cell Portal** (Broad Institute) — Additional IBD single-cell datasets
- **CellxGene** — Curated single-cell datasets with standardized annotations

## Methodology
1. **Data acquisition & QC**: Download raw count matrices from 3-4 single-cell studies. Filter cells (>500 genes, <20% mito), filter genes (expressed in >10 cells). Doublet removal (Scrublet).
2. **Integration & batch correction**: Harmonize datasets using scVI or Harmony. Evaluate integration quality with LISI scores.
3. **Cell-type annotation**: Reference-based transfer learning using Azimuth gut atlas. Manual validation with canonical markers. Focus on: Th17 cells, ILC3s, macrophages, dendritic cells, fibroblasts — all IL-23-responsive lineages.
4. **IL-23/Th17 pathway mapping**: Score cells for IL-23 pathway gene modules (IL23R, IL12RB1, RORC, IL17A, IL17F, IL22, CCR6, STAT3, JAK2, TYK2, IL23A, IL12B). Identify cell types with highest pathway activity.
5. **Differential analysis**: Compare inflamed vs non-inflamed tissue per cell type. Identify differentially abundant cell populations and differentially expressed gene programs.
6. **Response prediction** (if treatment data available): Differential abundance and expression between responders and non-responders. Build cell-type proportion + gene module score classifier.
7. **Gene regulatory network**: Infer transcription factor regulons active in IL-23-responsive cells using SCENIC/pySCENIC. Identify master regulators beyond RORC.

## Expected Outputs
- Annotated multi-study single-cell atlas of IL-23/Th17 pathway in CD mucosa
- Cell-type-specific gene programs associated with active inflammation vs remission
- Ranked list of cell types by IL-23 pathway activity and inflammatory contribution
- Candidate biomarker signatures for IL-23 blockade response (if treatment data sufficient)
- Transcription factor regulon analysis of IL-23-responsive cells
- Visualizations: UMAP plots, gene module heatmaps, cell proportion bar charts, pathway activity scores

## Success Criteria
- Successfully integrate ≥3 single-cell datasets with >50,000 total cells after QC
- Identify ≥2 cell-type-specific gene programs significantly enriched in active CD vs remission (FDR <0.05)
- Characterize IL-23 pathway activity across ≥5 cell types with quantitative module scores
- If treatment response data available: classifier AUC >0.7 for responder vs non-responder
- Identify ≥1 novel transcription factor regulon in IL-23-responsive cell populations

## Labels
immunology, genomic, biomarker, novel-finding, high-priority
