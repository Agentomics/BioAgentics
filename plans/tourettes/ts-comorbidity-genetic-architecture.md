# TS Comorbidity Genetic Architecture

## Objective
Dissect the shared vs. Tourette-specific genetic architecture across psychiatric comorbidities using genomic structural equation modeling and stratified polygenic risk scores to identify genetically-defined TS subtypes.

## Background
Tourette syndrome shows extensive psychiatric comorbidity — ~50% have ADHD, ~40% OCD, with additional overlap with ASD, anxiety, and depression. A landmark 2025 Nature study (doi:10.1038/s41586-025-09820-3) analyzed 1,056,201 cases across 14 psychiatric disorders and identified 5 genomic factors explaining ~66% of genetic variance. Genomic SEM places TS in a "compulsive disorders" factor with OCD and anorexia nervosa, but TS also shares substantial genetic overlap with ADHD and neurodevelopmental disorders. No study has systematically decomposed TS genetic risk into comorbidity-specific components or used this decomposition to define biologically meaningful subtypes.

## Data Sources
- **TSAICG GWAS summary statistics** — 9,619 TS cases, 981,048 controls (PGC portal)
- **Cross-disorder GWAS summary statistics** — Nature 2025, 14 psychiatric disorders (PGC/GWAS Catalog)
- **Individual disorder GWAS** — OCD (IOCDF-GC), ADHD (PGC), ASD (PGC), MDD (PGC)
- **UK Biobank** — LD reference panel, phenotype data for PRS validation
- **GWAS Catalog** — curated association data for replication

## Methodology

### Phase 1: Genetic Correlation Matrix
- Compute LD score regression genetic correlations between TS and all major psychiatric disorders
- Partition genetic correlations by functional annotation (coding, regulatory, brain-expressed) using stratified LDSC
- Identify which genomic regions drive TS-OCD vs. TS-ADHD correlations

### Phase 2: Genomic SEM Factor Analysis
- Replicate the 5-factor model from the Nature 2025 study
- Fit alternative models testing whether TS loads on multiple factors (compulsive + neurodevelopmental)
- Extract TS-specific genetic variance not captured by shared factors (residual GWAS)

### Phase 3: Stratified Polygenic Risk Scores
- Build TS PRS stratified by comorbidity-associated loci:
  - TS-OCD shared PRS (loci in compulsive factor)
  - TS-ADHD shared PRS (loci in neurodevelopmental factor)
  - TS-specific PRS (residual loci)
- Validate in UK Biobank by testing association with tic-related phenotypes and comorbidity patterns

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
