# TS GWAS Functional Annotation & Pathway Convergence

## Objective
Systematically annotate Tourette syndrome GWAS risk loci to identify convergent biological pathways and prioritize causal genes across the cortico-striato-thalamo-cortical circuit.

## Background
The Tourette Syndrome Association International Consortium for Genetics (TSAICG) published the largest TS GWAS meta-analysis identifying genome-wide significant and suggestive loci. However, translating statistical associations into biological mechanisms remains a major gap. Functional annotation using eQTL mapping, chromatin interaction data, and gene-set analysis can bridge this gap. Prior work has identified candidate genes (FLT3, MECR, MEIS1, SLITRK1, HDC, NRXN1) but systematic pathway convergence analysis across all loci is lacking.

## Data Sources
- **TSAICG GWAS summary statistics** — publicly available from the Psychiatric Genomics Consortium (PGC) data portal
- **GTEx v8** — eQTL data for brain regions (caudate, putamen, nucleus accumbens, frontal cortex, cerebellum)
- **BrainSpan** — developmental transcriptome for temporal expression profiling
- **Hi-C chromatin interaction data** — PsychENCODE/ENCODE for brain-specific chromatin loops
- **GWAS Catalog** — cross-trait pleiotropy with OCD, ADHD, ASD

## Methodology
1. **SNP-to-gene mapping**: Positional mapping (10kb window), eQTL mapping (GTEx brain tissues, FDR < 0.05), chromatin interaction mapping (Hi-C brain data)
2. **Gene-set analysis**: MAGMA gene-set analysis against MSigDB, Gene Ontology, KEGG, Reactome pathways
3. **Cell-type enrichment**: MAGMA-celltype/LDSC-SEG using single-cell RNA-seq from human striatum and cortex
4. **Pathway convergence**: Identify pathways enriched across multiple loci, focusing on dopaminergic signaling, synaptic adhesion, axon guidance, and neurodevelopmental processes
5. **Cross-disorder comparison**: Compare enriched pathways with OCD, ADHD, and ASD GWAS to identify TS-specific vs shared pathways
6. **Causal gene prioritization**: Integrate positional, eQTL, chromatin, and pathway evidence to rank candidate causal genes per locus

## Expected Outputs
- Annotated locus table with prioritized causal genes per GWAS locus
- Pathway enrichment results with FDR-corrected significance
- Cell-type enrichment profiles (which neuronal subtypes are implicated)
- Cross-disorder pathway comparison heatmap
- Prioritized gene list for downstream functional studies

## Success Criteria
- Identification of at least 3 convergent pathways enriched at FDR < 0.05
- Cell-type resolution showing enrichment in specific neuronal populations (e.g., medium spiny neurons, interneurons)
- Novel candidate genes not previously linked to TS in at least 2 loci
- Cross-disorder analysis revealing TS-specific pathway signatures

## Labels
genomic, high-priority, novel-finding
