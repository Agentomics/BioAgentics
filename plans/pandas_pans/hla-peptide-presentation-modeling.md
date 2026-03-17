# HLA-GAS Peptide Presentation Modeling

## Objective
Computationally predict which Group A Streptococcus (GAS) peptides are preferentially presented by PANDAS/PANS-associated HLA alleles, linking genetic susceptibility to molecular mimicry and explaining why only a subset of GAS-infected children develop autoimmune neuropsychiatric disease.

## Background
Only a fraction of children with GAS infections develop PANDAS, strongly implicating host genetic factors — particularly HLA alleles — in disease susceptibility. Several HLA associations have been reported: DRB1*01 and DRB1*04 in rheumatic fever (the closest autoimmune analog), restricted HLA DR-DQ haplotypes in PANS-ASD overlap (CIRS phenotype), and the 2025 OCD GWAS (Nature Genetics, 30 loci) identified MHC region associations, genetically linking OCD to autoimmune mechanisms. Meanwhile, GAS molecular mimicry is the leading pathogenic mechanism for PANDAS. No study has computationally linked these two lines of evidence by predicting which GAS peptides are preferentially presented by susceptibility HLA alleles to T cells, potentially initiating the autoimmune cascade.

## Data Sources
- **HLA alleles**: Published PANDAS/PANS HLA associations from literature, rheumatic fever HLA data (DRB1*01, DRB1*04, DQB1*0602), restricted DR-DQ haplotypes from CIRS/PANS-ASD study, Allele Frequency Net Database (AFND) for population frequencies
- **GAS proteomes**: UniProt reference proteomes for serotypes M1 (SF370), M3, M5, M12, M18 — shared resource with gas-molecular-mimicry-mapping project
- **Prediction tools**: NetMHCpan 4.1 (MHC-I binding prediction), NetMHCIIpan 4.3 (MHC-II binding prediction), IEDB T-cell epitope prediction tools
- **Cross-reference**: Molecular mimicry hits from gas-molecular-mimicry-mapping project, known PANDAS autoantibody targets (DRD1, DRD2, tubulin, CaMKII, GM1, glycolytic enzymes)
- **Autoimmune disease HLA data**: GWAS Catalog, OCD GWAS MHC loci (Nature Genetics 2025)

## Methodology

### Phase 1: HLA Susceptibility Allele Compilation
1. Systematic extraction of all published PANDAS/PANS HLA associations
2. Include rheumatic fever HLA data as proxy (shared GAS etiology)
3. Extract restricted HLA DR-DQ haplotypes from CIRS/PANS-ASD overlap study
4. Cross-reference with OCD GWAS MHC loci to identify converging alleles
5. Define control allele set: common HLA alleles without autoimmune neuropsychiatric associations

### Phase 2: GAS Peptide Library Generation
1. In silico digestion of GAS proteomes into 8-11mer peptides (MHC-I) and 15mer peptides (MHC-II)
2. Prioritize M protein, C5a peptidase, streptolysin O, and other virulence factors
3. Share peptide library with gas-molecular-mimicry-mapping project

### Phase 3: HLA-Peptide Binding Prediction
1. Run NetMHCpan 4.1 for all GAS peptides against PANDAS-associated MHC-I alleles and control alleles
2. Run NetMHCIIpan 4.3 for all GAS peptides against PANDAS-associated MHC-II alleles (critical — CD4+ T cell help drives autoantibody production) and control alleles
3. Classify peptides as strong binders (<0.5% rank), weak binders (<2% rank), or non-binders

### Phase 4: Differential Presentation Analysis
1. Identify GAS peptides with significantly stronger binding to susceptibility alleles vs control alleles (>10-fold affinity difference)
2. Statistical enrichment: are GAS virulence proteins over-represented among differentially presented peptides?
3. Serotype comparison: which M protein variants generate the most differentially presented peptides?

### Phase 5: Mimicry-HLA Integration
1. Cross-reference differentially presented GAS peptides with molecular mimicry candidates from gas-molecular-mimicry-mapping
2. For peptides that are both mimicry hits AND differentially presented: identify the human protein target
3. Focus on known autoantibody targets: do DRD1/DRD2/tubulin/CaMKII-mimicking GAS peptides bind preferentially to susceptibility HLA alleles?

### Phase 6: Population-Level Risk Modeling
1. Model PANDAS susceptibility risk based on HLA allele frequencies across populations using AFND data
2. Compare predicted risk profiles with reported PANDAS prevalence patterns
3. Identify high-risk HLA haplotype combinations

## Expected Outputs
- Ranked list of GAS peptides preferentially presented by PANDAS-associated HLA alleles
- Overlap analysis with molecular mimicry candidates (Venn diagram + statistical test)
- HLA-peptide-human protein network linking genetic susceptibility → peptide presentation → molecular mimicry → autoimmune targets
- Serotype-specific risk profiles (which GAS strains are most likely to trigger PANDAS via HLA-restricted mimicry)
- Population frequency analysis of susceptibility allele combinations
- Visualizations: binding affinity heatmaps, network diagrams, population risk maps

## Success Criteria
- At least 5 GAS peptides show >10-fold preferential binding to PANDAS-associated HLA alleles vs controls
- Overlap between differentially presented peptides and mimicry candidates is statistically significant (Fisher's exact test, p<0.05)
- Known autoantibody targets (DRD1/2, tubulin) are represented in the final HLA-mimicry network
- M protein N-terminal hypervariable region peptides are enriched among differential binders (consistent with known autoimmunogenic region)
- Results generate testable hypotheses about specific GAS peptide + HLA allele combinations driving PANDAS

## Labels
genomic, autoimmune, immunology, novel-finding, high-priority
