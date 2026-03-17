# TS Rare-Common Variant Convergence Analysis

## Objective
Systematically integrate rare variant evidence with common variant GWAS findings in Tourette syndrome to identify convergent biological pathways and high-confidence causal genes.

## Background
TS genetics spans both rare variants (SLITRK1, HDC, NRXN1, CNTN6, copy number variants) and common variants (TSAICG GWAS: BCL11B, NDFIP2, RBM26 via MAGMA). These have been studied independently. In autism and schizophrenia, rare-common variant convergence analysis has identified high-confidence pathways (synaptic function, chromatin remodeling) that neither variant class could establish alone. No equivalent systematic analysis exists for TS.

The TSAICG GWAS (journal #446) found enrichment in loss-of-function intolerant genes and high-confidence neurodevelopmental disorder genes, suggesting substantial overlap with the rare variant architecture. Identifying convergent pathways would prioritize the highest-confidence biological mechanisms for therapeutic targeting and provide the strongest evidence for causal gene nomination.

## Data Sources
- **TSAICG GWAS summary statistics** — PGC portal (task #235)
- **TS rare variant genes** — Literature curation:
  - Established: SLITRK1, HDC, NRXN1, CNTN6
  - Candidate: CELSR3, WWC1, ASH1L, OPRK1, FN1
  - TS-associated CNV regions (16p13.11, 22q11.2, NRXN1 deletions)
  - De novo variant genes from trio studies
- **gnomAD v4** — pLI, LOEUF, missense Z-scores
- **STRING v12** — PPI network
- **Gene Ontology, KEGG, Reactome** — Pathway databases
- **Cross-disorder rare variant genes** — Autism (SFARI Gene), OCD, ADHD for specificity analysis

## Methodology

### Phase 1 — Rare Variant Gene Curation
Systematic literature review to compile all TS-implicated rare variant genes. For each gene:
- Evidence type: de novo, segregation, functional, CNV, case report
- Evidence strength: strong (replicated, functional validation), moderate (multiple independent reports), suggestive (single report, no functional data)
- Variant type: LoF, missense, structural
- Pathway membership

### Phase 2 — Constraint Analysis
Map TS rare variant genes AND GWAS-implicated genes (MAGMA gene-based significant + eQTL-mapped) onto gnomAD constraint landscape:
- Compare pLI/LOEUF distributions against genome-wide background
- Test whether rare variant genes and GWAS genes show similar constraint profiles
- Identify whether TS genes are enriched for LoF-intolerant genes (extending TSAICG finding)

### Phase 3 — Network Connectivity Analysis
Build PPI subnetwork containing rare variant genes + GWAS-implicated genes using STRING v12:
- Compute network proximity metrics (shortest path, random walk, diffusion kernel)
- Test whether rare and common variant genes cluster in the same network neighborhoods vs. random expectation (permutation test, n=10,000)
- Identify bridge genes connecting rare and common variant clusters
- Visualize network topology

### Phase 4 — Pathway Convergence
Independent pathway enrichment for rare variant gene set and GWAS gene set:
- GO Biological Process, KEGG, Reactome
- Rank pathways by convergence strength (Fisher's combined probability test)
- Identify pathways enriched in BOTH gene sets independently
- Focus on pathways relevant to CSTC circuit biology

### Phase 5 — Cross-Disorder Specificity
Compare TS convergent pathways with convergent pathways from:
- Autism (SFARI genes + GWAS)
- OCD (rare variants + GWAS)
- ADHD (rare variants + GWAS)
Identify TS-specific convergent pathways vs. shared neurodevelopmental pathways.

## Expected Outputs
- Curated, evidence-scored TS rare variant gene list (≥15 genes)
- Constraint profile comparison: rare variant genes vs. GWAS genes vs. background
- PPI network analysis: rare-common variant connectivity with significance testing
- Ranked convergent pathways with Fisher's combined p-values
- Cross-disorder specificity scores for each convergent pathway
- High-confidence causal gene shortlist (supported by both rare and common variant evidence)

## Success Criteria
- ≥15 TS rare variant genes identified with strong/moderate evidence
- Significant network clustering of rare and common variant genes (permutation p < 0.05)
- ≥3 pathways showing significant convergence across variant types (combined p < 0.01)
- ≥5 genes supported by both rare and common variant evidence
- ≥1 TS-specific convergent pathway distinguishable from autism/OCD/ADHD

## Labels
genomic, novel-finding, high-priority
