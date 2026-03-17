# CSTC Circuit Gene Expression Atlas for Tourette Syndrome

## Objective
Profile the spatial and cell-type-specific expression of Tourette syndrome risk genes across the cortico-striato-thalamo-cortical (CSTC) circuit to identify circuit nodes and cell types most likely disrupted in TS.

## Background
The CSTC circuit is the primary neural circuit implicated in tic generation. Motor and limbic loops through the basal ganglia — involving cortex, striatum (caudate/putamen), globus pallidus, subthalamic nucleus, and thalamus — show structural and functional alterations in TS neuroimaging studies. However, which specific cell types and circuit nodes are most vulnerable to TS-associated genetic variation is unknown. By mapping TS risk gene expression across CSTC regions at single-cell resolution, we can pinpoint where genetic risk converges anatomically and cellularly.

## Data Sources
- **Allen Human Brain Atlas (AHBA)** — bulk microarray data across ~3,700 brain samples from 6 donors, covering all CSTC regions
- **BrainSpan** — developmental transcriptome (prenatal through adult) for temporal expression trajectories
- **Human striatum single-cell RNA-seq** — GEO datasets (e.g., GSE118020 Tran et al. 2019, GSE160936 Zhu et al. 2021) for striatal cell-type resolution
- **PsychENCODE single-cell data** — cortical cell-type expression
- **TS risk gene lists** — TSAICG GWAS genes, rare variant genes (SLITRK1, HDC, NRXN1, CNTN6, WWC1), and curated TS candidate gene sets

## Methodology
1. **AHBA spatial mapping**: Extract expression of TS risk genes across CSTC regions (prefrontal cortex, motor cortex, caudate, putamen, GPe, GPi, STN, thalamus). Compute regional enrichment scores
2. **Developmental trajectory analysis**: Use BrainSpan to characterize when TS risk genes are most highly expressed during development (critical period identification)
3. **Single-cell deconvolution**: Map TS gene expression onto striatal cell types (D1/D2 medium spiny neurons, cholinergic interneurons, parvalbumin interneurons, astrocytes, oligodendrocytes)
4. **Co-expression network analysis**: WGCNA on AHBA data within CSTC regions to identify co-expression modules enriched for TS genes
5. **Circuit vulnerability scoring**: Integrate spatial, temporal, and cell-type data to score each CSTC node for TS genetic vulnerability
6. **Comparison with neuroimaging findings**: Cross-reference expression-based vulnerability with published TS structural MRI and fMRI findings (ENIGMA-TS)

## Expected Outputs
- CSTC regional expression heatmap for TS risk genes
- Developmental expression trajectory plots (critical windows for TS gene activity)
- Cell-type specificity profiles (which cell types express TS genes most)
- Co-expression modules enriched for TS genes with hub gene identification
- Circuit vulnerability score per CSTC node
- Integration figure overlaying genetic vulnerability with neuroimaging alterations

## Success Criteria
- Clear regional enrichment pattern showing TS genes preferentially expressed in specific CSTC nodes (not uniform)
- Cell-type specificity for D1 vs D2 MSNs or specific interneuron populations
- At least one co-expression module significantly enriched for TS genes (permutation p < 0.01)
- Concordance between expression-based vulnerability and published neuroimaging findings

## Labels
genomic, neuroimaging, novel-finding, high-priority
