# TS Striatal Interneuron Molecular Pathology

## Objective
Perform deep re-analysis of the first single-cell transcriptomic and epigenomic atlas of the Tourette syndrome basal ganglia to characterize interneuron subtype-specific vulnerability, identify disrupted regulatory networks, and nominate druggable targets.

## Background
Wang et al. (Biological Psychiatry, Jan 2025; PMID 39892689) published the first single-cell-resolution study of the TS caudate nucleus, analyzing snRNA-seq and snATAC-seq from 6 TS vs. 6 control postmortem brains. Key findings include ~50% interneuron loss, MSN mitochondrial dysfunction, decreased synaptic gene expression in surviving interneurons, and microglial immune activation. However, the published analysis used broad cell-type categories and did not deeply characterize which specific interneuron subclasses are most vulnerable, what epigenomic changes precede or accompany cell loss, or how these changes connect to known TS genetic risk loci. A complementary human striatal interneuron atlas (Nature Communications, Jul 2024; GSE151761, GSE152058) provides a 14-subclass taxonomy that can serve as reference.

## Data Sources
- **Wang et al. 2025** — snRNA-seq + snATAC-seq, TS caudate (6 TS, 6 control). GEO accession TBD (data_curator to locate)
- **GSE151761 / GSE152058** — Human dorsal striatum snRNA-seq, healthy reference atlas with 14 interneuron subclasses
- **Allen Human Brain Atlas** — ISH and microarray expression for spatial validation
- **TSAICG GWAS summary statistics** — for linking single-cell findings to genetic risk
- **Gene Ontology / Reactome / KEGG** — pathway enrichment databases
- **DrugBank / ChEMBL** — for target druggability assessment

## Methodology

### Phase 1: Data Acquisition & Integration
- Obtain Wang et al. snRNA-seq + snATAC-seq data from GEO
- Download reference atlas datasets (GSE151761, GSE152058)
- Quality control: filter cells, normalize, batch correction (Harmony/scVI)
- Integrate TS and control samples with reference atlas for unified cell-type annotation

### Phase 2: Interneuron Subtype-Resolution Analysis
- Map Wang et al. interneurons onto the 14-subclass taxonomy
- Quantify per-subclass abundance changes in TS vs. control
- Identify which subclasses drive the ~50% overall interneuron loss
- Test whether cholinergic (ChAT+), parvalbumin (PV+), or somatostatin (SST+) interneurons are differentially affected
- Assess whether loss reflects cell death, dedifferentiation, or developmental deficit
- **Include de novo variant risk genes** (PPP5C, EXOC1, GXYLT1) in subtype analysis — these lack functional characterization; expression patterns across interneuron subclasses may predict their mechanisms [task #292]
- **Leverage mesoscale striatal atlas (GSE303705, 1.1M cells)** for spatial context: map interneuron loss onto 6-zone striatal subdivision to test whether pathology is zone-specific [task #382]

### Phase 3: Transcriptomic Dysregulation Profiling
- Differential expression analysis at subclass resolution (pseudobulk DESeq2)
- Focus on surviving interneurons: what gene programs are disrupted?
- Characterize MSN mitochondrial dysfunction: which specific complexes/pathways?
- WGCNA co-expression networks within TS vs. control interneurons
- Gene regulatory network inference (SCENIC/pySCENIC) for transcription factor dysregulation

### Phase 4: Epigenomic Integration
- Analyze snATAC-seq for cell-type-specific chromatin accessibility changes
- Identify differentially accessible peaks in TS interneurons and MSNs
- Motif enrichment in disrupted peaks (transcription factor binding sites)
- Link GWAS risk variants to cell-type-specific regulatory elements
- Integrate with Hi-C data to connect distal regulatory elements to target genes

### Phase 5: Target Nomination & Druggability
- Prioritize genes/pathways by convergence: genetic risk + expression dysregulation + epigenomic disruption
- Assess druggability of top targets using DrugBank/ChEMBL
- Compare against existing TS drug targets (D2R, VMAT2, alpha-2 agonists)
- Nominate novel targets, particularly in interneuron survival/function pathways

## Expected Outputs
- Interneuron subtype-resolution cell census of TS vs. control caudate
- Per-subclass differential expression and pathway enrichment results
- Epigenomic regulatory network maps for affected cell types
- GWAS-to-single-cell variant-to-gene-to-cell-type maps
- Ranked list of druggable targets with convergent evidence
- Visualization: UMAP plots, heatmaps, network diagrams

## Success Criteria
- Successfully identify ≥2 differentially affected interneuron subclasses
- Transcriptomic and epigenomic changes converge on interpretable biological pathways
- ≥3 GWAS risk loci map to cell-type-specific regulatory elements in affected populations
- At least 2 novel druggable targets nominated with multi-omic support

## Labels
genomic, multi-omics, novel-finding, biomarker, high-priority, promising
