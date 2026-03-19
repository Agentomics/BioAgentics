# CSTC Circuit Gene Expression Atlas for Tourette Syndrome

## Objective
Profile the spatial and cell-type-specific expression of Tourette syndrome risk genes across the cortico-striato-thalamo-cortical (CSTC) circuit to identify circuit nodes and cell types most likely disrupted in TS.

## Background
The CSTC circuit is the primary neural circuit implicated in tic generation. Motor and limbic loops through the basal ganglia — involving cortex, striatum (caudate/putamen), globus pallidus, subthalamic nucleus, and thalamus — show structural and functional alterations in TS neuroimaging studies. However, which specific cell types and circuit nodes are most vulnerable to TS-associated genetic variation is unknown. By mapping TS risk gene expression across CSTC regions at single-cell resolution, we can pinpoint where genetic risk converges anatomically and cellularly.

## Data Sources
- **Allen Human Brain Atlas (AHBA)** — bulk microarray data across ~3,700 brain samples from 6 donors, covering all CSTC regions
- **BrainSpan** — developmental transcriptome (prenatal through adult) for temporal expression trajectories
- **Human striatum single-cell RNA-seq** — GEO datasets (e.g., GSE118020 Tran et al. 2019, GSE160936 Zhu et al. 2021) for striatal cell-type resolution
- **Mesoscale striatal atlas (GSE303705)** — 1.1M cells, 19 donors, Slide-tags spatial transcriptomics, 6-zone striatal subdivision with neuron-astrocyte spatial coordination data [NEW Mar 2026]
- **3 new basal ganglia spatial transcriptomics atlases (Jan-Mar 2026)** — developing human BG multi-omic atlas (snm3C-seq + spatial), additional datasets identified by literature_reviewer [NEW]
- **TS prefrontal cortex snRNA-seq (bioRxiv Jan 2026; doi:10.64898/2026.01.14.699521)** — snRNA-seq of TS PFC revealing stress-related activation patterns. Complements Wang et al. basal ganglia data by providing cortical cell-type resolution for the "C" (cortical) node of the CSTC circuit [NEW task #932]
- **Thalamo-frontal DBS-EEG (Mol Psychiatry; doi:10.1038/s41380-025-03220-9)** — alpha-band (8-12 Hz) thalamo-frontal connectivity negatively correlates with TS severity and drops before tic onset. Provides electrophysiological cross-validation of cortical-thalamic circuit disruption for comparison with expression-based vulnerability scoring [NEW task #932]
- **PsychENCODE single-cell data** — cortical cell-type expression
- **TS risk gene lists** — TSAICG GWAS genes, rare variant genes (SLITRK1, HDC, NRXN1, CNTN6, WWC1), and curated TS candidate gene sets

## Additional Gene Sets (NEW Mar 2026)
- **Iron homeostasis pathway**: TF, TFR1 (TFRC), FTH1, FTL, IRP1 (ACO1), IRP2 (IREB2), HAMP, SLC40A1 — informed by 7T MRI study (Brain Communications 2025) showing circuit-wide iron depletion in caudate, pallidum, STN, thalamus, red nucleus, and substantia nigra, with D1 receptor correlation to tic severity
- **Hippo signaling pathway**: WWC1/KIBRA, YAP1, TAZ (WWTR1), LATS1, LATS2, MST1 (STK4), MST2 (STK3) — informed by WWC1 W88C functional study (Science Advances Mar 2025) showing developmental-stage-specific dopamine dysregulation

## Methodology
1. **AHBA spatial mapping**: Extract expression of TS risk genes across CSTC regions (prefrontal cortex, motor cortex, caudate, putamen, GPe, GPi, STN, thalamus). Compute regional enrichment scores
2. **Developmental trajectory analysis**: Use BrainSpan to characterize when TS risk genes are most highly expressed during development (critical period identification)
3. **Single-cell deconvolution**: Map TS gene expression onto striatal cell types (D1/D2 medium spiny neurons, cholinergic interneurons, parvalbumin interneurons, astrocytes, oligodendrocytes)
4. **Mesoscale striatal zone-specific profiling**: Use GSE303705 to map TS gene expression across 6 striatal zones [NEW]
5. **Iron homeostasis pathway spatial profiling**: Map iron pathway gene expression across CSTC nodes; test concordance with 7T MRI iron depletion pattern [NEW]
6. **Co-expression network analysis**: WGCNA on AHBA data within CSTC regions to identify co-expression modules enriched for TS genes
7. **Circuit vulnerability scoring**: Integrate spatial, temporal, and cell-type data to score each CSTC node for TS genetic vulnerability
8. **PFC cell-type expression profiling** [NEW task #932]: Map TS risk gene expression onto PFC cell types using the new TS PFC snRNA-seq (bioRxiv Jan 2026). Identify stress-responsive cell populations with elevated TS gene expression. Compare PFC cell-type vulnerability with striatal cell-type vulnerability (Wang et al.) to determine whether TS genetic risk is preferentially cortical or subcortical at the cell-type level
9. **Thalamo-frontal electrophysiology cross-validation** [NEW task #932]: Use alpha-band (8-12 Hz) thalamo-frontal connectivity findings (Mol Psychiatry) to cross-validate the expression-based CSTC vulnerability scores. Test whether the thalamic and frontal regions showing reduced alpha connectivity are the same regions showing highest TS gene expression vulnerability. Pre-tic alpha drop provides temporal resolution for circuit disruption that expression data alone cannot
10. **DBS optimal fiber tract profiling** [NEW]: Multi-center DBS study (n=115, 12 centers, medRxiv Feb 2026) identified three optimal fiber tracts for tic improvement: ansa lenticularis, fasciculus lenticularis, and posterior intralaminar-lentiform projections (explaining 19% of variance). Prioritize expression profiling of TS risk genes along these three tract corridors. OCD response maps diverge from tic maps in thalamus — examine whether different gene sets are expressed in tic-responsive vs. OCD-responsive thalamic regions [task #359]
7. **Comparison with neuroimaging findings**: Cross-reference expression-based vulnerability with published TS structural MRI and fMRI findings (ENIGMA-TS)

## Expected Outputs
- CSTC regional expression heatmap for TS risk genes
- Developmental expression trajectory plots (critical windows for TS gene activity)
- Cell-type specificity profiles (which cell types express TS genes most)
- Iron homeostasis pathway spatial expression map across CSTC nodes [NEW]
- Co-expression modules enriched for TS genes with hub gene identification
- Circuit vulnerability score per CSTC node
- PFC cell-type vulnerability profile from TS PFC snRNA-seq [NEW task #932]
- Cortical vs. subcortical vulnerability comparison (PFC snRNA-seq vs. Wang et al. caudate)
- Alpha-band electrophysiology concordance analysis with expression-based vulnerability [NEW task #932]
- Integration figure overlaying genetic vulnerability with neuroimaging alterations

## Success Criteria
- Clear regional enrichment pattern showing TS genes preferentially expressed in specific CSTC nodes (not uniform)
- Cell-type specificity for D1 vs D2 MSNs or specific interneuron populations
- At least one co-expression module significantly enriched for TS genes (permutation p < 0.01)
- Concordance between expression-based vulnerability and published neuroimaging findings
- Iron pathway expression profile concordant with 7T MRI iron depletion pattern [NEW]

## Labels
genomic, neuroimaging, novel-finding, high-priority
