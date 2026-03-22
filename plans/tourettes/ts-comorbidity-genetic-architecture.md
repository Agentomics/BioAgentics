# TS Comorbidity Genetic Architecture

## Objective
Dissect the shared vs. Tourette-specific genetic architecture across psychiatric comorbidities using genomic structural equation modeling and stratified polygenic risk scores to identify genetically-defined TS subtypes.

## Background
Tourette syndrome shows extensive psychiatric comorbidity — ~50% have ADHD, ~40% OCD, with additional overlap with ASD, anxiety, and depression. Grotzinger et al. (Nature 649:406-415, 2026; doi:10.1038/s41586-025-09820-3) analyzed 1,056,201 cases across 14 psychiatric disorders and identified 5 genomic factors explaining ~66% of genetic variance: **Schizophrenia-Bipolar (SB), Internalizing, Neurodevelopmental, Compulsive,** and **Substance Use**. TS uniquely **dual-loads on the Neurodevelopmental and Compulsive factors** and retains ~87% trait-specific residual variance, making it the most subtype-rich disorder in the panel. The paper also reports 238 pleiotropic loci with factor-specific effects and curated factor-level GWAS summary statistics that can seed our analyses. No study has yet decomposed TS genetic risk into these factor-defined components to build subtypes or clinical prediction tools, which is exactly the gap this initiative fills.

## Data Sources
- **TSAICG GWAS summary statistics** — 9,619 TS cases, 981,048 controls (PGC portal)
- **Grotzinger et al. 2026 GenomicSEM package** — factor loadings, 5-factor GWAS summary stats, and 238 pleiotropic loci tables (PGC/GWAS Catalog/CDG3)
- **Individual disorder GWAS** — OCD (IOCDF-GC), ADHD (PGC), ASD (PGC), MDD (PGC)
- **UK Biobank** — LD reference panel, phenotype data for PRS validation
- **GWAS Catalog** — curated association data for replication

## Methodology

### Phase 1: Genetic Correlation Matrix
- Compute LD score regression genetic correlations between TS and all major psychiatric disorders
- Partition genetic correlations by functional annotation (coding, regulatory, brain-expressed) using stratified LDSC
- Identify which genomic regions drive TS-OCD vs. TS-ADHD correlations

### Phase 2: Genomic SEM Factor Analysis
- Lock in the Grotzinger 5-factor model (SB, Internalizing, Neurodevelopmental, Compulsive, Substance Use) as the primary framework
- Fit alternative TS loading structures: (a) compulsive-only, (b) compulsive + neurodevelopmental (published dual-loading), (c) free-loading on all factors; compare fit statistics
- Quantify subtype variance components: TS-Neurodevelopmental vs. TS-Compulsive vs. TS-residual (87% unique variance target)
- Extract factor loadings, factor-specific SNP effects, and residual GWAS for downstream PRS/pathway modules

### Phase 3: Stratified Polygenic Risk Scores
- Build TS PRS stratified by Grotzinger factor components:
  - **TS-Compulsive PRS** — SNP weights from Compulsive factor GWAS (+ pleiotropic loci flagged for compulsive loadings)
  - **TS-Neurodevelopmental PRS** — SNP weights from Neurodevelopmental factor GWAS (+ pleiotropic loci flagged for neurodevelopmental loadings)
  - **TS-specific PRS** — residual GWAS weights capturing the 87% TS-unique variance
- Map PRS strata back to clinical phenotypes to define TS-Compulsive vs. TS-Neurodevelopmental subtype membership probabilities
- Validate in UK Biobank by testing association with tic-related phenotypes and comorbidity patterns
- **PRS-PheWAS validation framework** [NEW]: TS PRS-PheWAS study (Translational Psychiatry 2023) found 57 traits associated with TS PRS but aggregate PRS did NOT predict OCD, ADHD, or ASD severity. This directly validates the need for factor-specific PRS decomposition — factor-specific PRS should recover comorbidity associations that aggregate PRS misses. The 57 significant trait associations provide a validation set [task #360]
- **DBS dissociation as testable prediction** [NEW]: DBS tic vs. OCD response dissociation (pallidum shared, thalamus divergent) suggests compulsive factor genes should enrich in pallidal cell types while TS-specific factor genes enrich in thalamic cell types — testable via cross-reference with CSTC atlas project [task #360]

### Phase 4: Pathway-Level Decomposition
- MAGMA gene-set analysis on each PRS stratum
- Identify pathways unique to each comorbidity axis:
  - Compulsive axis: cortico-striatal glutamate signaling? Serotonergic?
  - Neurodevelopmental axis: synaptic development? Dopaminergic?
  - TS-specific: CSTC circuit? Interneuron development?
- Compare against the 238 pleiotropic loci from the cross-disorder study

### Phase 5: Subtype Definition & Clinical Implications
- Cluster TS genetic risk profiles into discrete subtypes
- Map subtypes to predicted comorbidity patterns
- Identify subtype-specific drug targets using pathway results

## Cross-Project Interfaces
- **ts-rare-variant-convergence:** Compare rare-variant carriers against PRS-defined TS-Neurodevelopmental vs. TS-Compulsive subtypes to test whether de novo burden concentrates in the neurodevelopmental factor.
- **ts-neuroimmune-subtyping:** Evaluate whether immune/tissue-specific enrichments fall outside the 5 factors, indicating an orthogonal immune axis that needs integration.
- **ts-drug-repurposing-network:** Feed subtype-specific pathway hits into drug response predictions (dopaminergic for neurodevelopmental vs. serotonergic/CBIT potentiators for compulsive).

## Expected Outputs
- Genetic correlation matrix and factor model for TS across 14 disorders
- Stratified PRS weights (TS-OCD, TS-ADHD, TS-specific)
- Pathway maps for each genetic subtype
- Subtype classification framework with clinical predictions

## Success Criteria
- Successfully decompose TS genetic variance into ≥2 distinct comorbidity-associated components
- Stratified PRS explain more phenotypic variance than unstratified TS PRS
- Pathway analysis reveals biologically distinct mechanisms for each subtype
- At least one subtype suggests a novel therapeutic hypothesis

## Labels
comorbidity, genomic, novel-finding, biomarker, clinical, high-priority
