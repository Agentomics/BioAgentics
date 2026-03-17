# IL-23/Th17 Single-Cell Response Atlas

## Objective
Build a single-cell transcriptomic atlas of the IL-23/Th17 axis in Crohn's disease mucosa to identify cell-type-specific gene programs predictive of response to the IL-23 inhibitor mechanism class (risankizumab, guselkumab, ustekinumab), enabling patient stratification across the expanding IL-23p19 therapeutic landscape.

## Background
The IL-23/Th17 pathway is a validated therapeutic target with multiple biologics now in clinical use. The SEQUENCE trial (NEJM 2024) confirmed risankizumab superiority over ustekinumab (endoscopic remission 31.8% vs 16.2%), but ~2/3 of risankizumab-treated patients still do NOT achieve endoscopic remission — this non-remission rate is the concrete clinical gap. J&J has initiated a guselkumab vs risankizumab head-to-head trial, and guselkumab received FDA approval for subcutaneous induction in UC. With multiple IL-23p19 inhibitors now available and 2025 ACG guidelines no longer requiring anti-TNF failure before IL-23 inhibitors, predicting which patients respond to this drug class (vs other mechanisms) is clinically urgent. This atlas targets the shared IL-23/Th17 pathway signature rather than drug-specific features, aiming to predict mechanism-class response. Single-cell RNA-seq of CD mucosa is now available from multiple studies but hasn't been systematically analyzed for IL-23 pathway biology or treatment response prediction. Most IL-23 studies focus on bulk tissue or animal models, missing the cell-type-specific resolution needed to understand heterogeneous responses.

## Data Sources
- **GSE134809** (Kong et al.) — Single-cell RNA-seq of CD ileal tissue, inflamed and non-inflamed
- **GSE150392** (Martin et al.) — Single-cell atlas of inflamed human gut with treatment metadata
- **GSE207617** — Single-cell CD with clinical/treatment annotations
- **Single Cell Portal** (Broad Institute) — Additional IBD single-cell datasets
- **CellxGene** — Curated single-cell datasets with standardized annotations
- **IBDverse** (1.1M cells, 111 CD + 232 controls) — Largest scRNA-seq resource for ileal CD. Candidate primary cell-type reference for annotation (pending data_curator evaluation, task #231)

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
- Candidate biomarker signatures for IL-23 mechanism-class response (if treatment data sufficient) — framed as pathway-level predictors applicable across risankizumab, guselkumab, and future IL-23p19 inhibitors
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
