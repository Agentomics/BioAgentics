# PANS Genetic Variant Pathway Enrichment Analysis

## Objective
Perform pathway enrichment analysis on ultra-rare genetic variants identified in PANS patients to map disease-associated biological pathways and cross-reference with neuroinflammation gene expression data.

## Background
A 2022 exome/WGS study (Scientific Reports, s41598-022-15279-3) identified ultra-rare genetic variants in PANS patients. This work has now been expanded and peer-reviewed: Vettiatil D et al. 2026, "Ultrarare Variants in DNA Damage Repair and Mitochondrial Genes in Pediatric Acute-Onset Neuropsychiatric Syndrome," Dev Neurosci (PMID 41662332). The validated cohort (n=32) identifies 14 pathogenic/likely pathogenic variants in DDR genes: CUX1, USP45, PARP14, UVSSA, EP300, TREX1, SAMHD1, STK19, MYT1L, TEP1, PIDD1, ADNP, FANCD2, RAD54L. Key new additions: TREX1 (cGAS-STING pathway), SAMHD1 (innate immunity/dNTP regulation), ADNP (chromatin remodeling/neuroprotection). The study also identifies a novel gut microbiome-genetic variant axis in PANS pathogenesis.

GEO dataset GSE102482 (LPS-treated microglia) was used in the original study to examine PANS candidate gene expression in neuroinflammatory contexts. Combining genetic variant data with expression data could reveal convergent biological pathways underlying PANS susceptibility.

The field currently lacks a systematic computational analysis connecting PANS genetic findings to functional pathways. Most genetic studies in PANDAS/PANS are underpowered for GWAS, but pathway-level aggregation of rare variants can reveal biological signals that individual variant analyses miss.

## Data Sources
- **Genetic variants:** Validated 14-gene DDR/mitochondrial list from Vettiatil et al. 2026, Dev Neurosci (PMID 41662332); original variant list from Scientific Reports s41598-022-15279-3 (supplementary tables)
- **Expression data:** GEO dataset GSE102482 — LPS-treated mouse microglia transcriptomics
- **Pathway databases:** KEGG, Reactome, Gene Ontology (GO), DisGeNET
- **Protein interaction networks:** STRING, BioGRID
- **Immune gene sets:** ImmuneSigDB, MSigDB Hallmark gene sets (interferon, inflammatory response, complement)

## Methodology
1. **Variant curation:** Extract all ultra-rare variants from the published study. Map to genes. Annotate with functional impact scores (CADD, PolyPhen-2, SIFT from published data).
2. **Pathway enrichment:** Run over-representation analysis (ORA) and gene set enrichment analysis (GSEA) against KEGG, Reactome, and GO. Use clusterProfiler or equivalent.
3. **Neuroinflammation cross-reference:** Analyze GSE102482 for differential expression of PANS variant genes under LPS stimulation. Identify which PANS candidate genes are upregulated in neuroinflammatory conditions.
4. **Network analysis:** Build protein-protein interaction subnetwork of PANS variant genes using STRING. Identify hub genes and network modules. Overlay with known autoimmune and neuropsychiatric pathway annotations.
5. **Immune pathway focus:** Specifically test enrichment in innate immunity (TLR signaling, complement), adaptive immunity (T cell activation, Th17 differentiation), BBB-related pathways, and cGAS-STING/type I interferon signaling (via TREX1, SAMHD1 connections).
6. **mTOR convergence analysis:** Evaluate mTOR pathway (Fronticelli Baldelli G et al. 2025, PMID 41462744) as a convergence node connecting infection→inflammation→CNS dysfunction. mTOR links to autophagy (ATG7/UVRAG from IVIG initiative) and immune regulation — test whether PANS variant genes cluster on mTOR-related pathways.

## Expected Outputs
- Ranked list of enriched pathways from PANS rare variants
- Overlap analysis: PANS variant genes that are also differentially expressed in neuroinflammation (GSE102482)
- PPI network visualization with hub gene identification
- Comparison of PANS enriched pathways vs known PANDAS/PANS mechanisms (molecular mimicry, complement, Th17)

## Success Criteria
- Identification of at least 3 significantly enriched pathways (FDR < 0.05) from PANS variant genes
- At least 1 novel pathway not previously associated with PANDAS/PANS
- Convergence between genetic variant pathways and neuroinflammation expression data

## Labels
genomic, autoimmune, immunology, novel-finding, promising
