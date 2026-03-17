# Microbiome-Metabolome Integration for Crohn's Disease Subtyping

## Objective
Integrate metagenomic and metabolomic profiles from Crohn's disease patients to identify reproducible disease subtypes with distinct biological mechanisms and potential therapeutic implications.

## Background
Crohn's disease is clinically heterogeneous — patients vary in disease location (ileal, colonic, ileocolonic), behavior (inflammatory, stricturing, penetrating), and treatment response. Gut microbiome dysbiosis is a hallmark feature, but single-omic studies capture only partial biology. The HMP2/IBDMDB project generated matched metagenomic, metatranscriptomic, metabolomic, and clinical data from IBD patients over time, providing a unique opportunity for multi-omic disease subtyping. Identifying biologically coherent subtypes could guide treatment selection and predict disease course.

## Data Sources
- **HMP2/IBDMDB**: Longitudinal multi-omics from 132 subjects (65 CD, 38 UC, 27 non-IBD controls). Metagenomics, metabolomics, metatranscriptomics, serology. Available at ibdmdb.org and via HMP DACC.
- **curatedMetagenomicData (Bioconductor)**: Standardized metagenomic profiles from multiple IBD cohorts for validation (Pasolli et al. 2017)
- **MetaHIT**: European metagenome cohort including IBD samples (ERP002061)
- **PRISM cohort metabolomics**: Fecal metabolomic profiles from Franzosa et al. 2019 (doi:10.1038/s41564-018-0306-4)

## Methodology
1. **Data acquisition and QC**: Download HMP2 metagenomic species/pathway abundances and metabolomic profiles. Filter low-prevalence features, normalize (CLR for metagenomics, log-transform for metabolomics).
2. **Single-omic exploration**: Characterize CD vs control differences in each omic layer. Identify differentially abundant species (Faecalibacterium, Roseburia, E. coli) and metabolites (SCFAs, bile acids, acylcarnitines). **Modality prioritization** (validated by pediatric 6-omics study, Commun Med 2025): Feature importance hierarchy in integrated CD models — fecal bacteria 40%, fecal metabolites 22%, fecal proteins 16%, plasma metabolites 12%, fecal fungi 6%, urine metabolites 4%. Fecal compartment dominates (84% combined importance). Our metagenomics + metabolomics focus captures the top two modalities (62% combined). Use this hierarchy to guide feature weighting: bacterial features should be given proportionally higher weight in integration. **Three candidate metabolic axes for subtype separation:**
   - *(1) Bifidobacterium-TCDCA axis* (central anchor): B. catenulatum → TCDCA production → M1 suppression + Treg dominance. Replicated across infliximab response (LDA AUC 80.5%) and pediatric disease discrimination. Subtypes may separate along Bifidobacterium-high/low axis with distinct therapeutic implications.
   - *(2) Tryptophan-NAD axis* (Nap et al., Nat Commun 2025, PMID 40456745): Elevated microbial tryptophan catabolism depletes circulating tryptophan → impairs host NAD biosynthesis → disrupted nitrogen homeostasis, polyamine/glutathione metabolism. GEM modeling predicts saccharide removal dietary intervention restores metabolic homeostasis. Tryptophan, kynurenine, NAD precursors (nicotinamide riboside, nicotinic acid) as candidate metabolomic features; tryptophan pathway gene abundances (e.g., tnaA, kynA) as metagenomic features.
   - *(3) BCAA axis* (Gut Microbes 2025, doi:10.1080/19490976.2025.2527863): CD-specific leucine/valine synthesis perturbation detected at diagnosis in inception cohort (pre-treatment, eliminating pharmacological confounders). Distinct from tryptophan-NAD axis — may define a separate CD subpopulation. Leucine, valine, and BCAA biosynthesis pathway abundances as candidate features.
   These three axes may define mechanistically distinct CD subtypes: Bifidobacterium-depleted/bile-acid-deficient, tryptophan-catabolism-dominant, and BCAA-perturbed. If subtypes separate along these axes, subtype-specific therapeutic strategies emerge (FMT/probiotics for axis 1, dietary saccharide modification for axis 2, BCAA supplementation for axis 3).
3. **Multi-omic integration** (multi-method approach per Communications Biology 2025 benchmark, doi:10.1038/s42003-025-08515-9):
   - *Primary biomarker detection*: **CMTF** (coupled matrix and tensor factorization) — consistently outperformed traditional methods in both dimension reduction AND biomarker detection in the 19-method benchmark (Commun Biol 2025). Most relevant for our subtyping goals where reproducible, biologically distinct subtypes are the target. CMTF handles coupled microbiome-metabolome matrices natively and is robust to noise typical of clinical IBD cohorts.
   - *Primary latent factor discovery*: **MOFA2** for interpretable latent factors capturing shared variance across metagenomics and metabolomics. Best at recapitulating data variability but highly variable (0–100% explained variance) depending on signal strength, noise, and sample size — complement with CMTF for robustness.
   - *Baseline comparator*: **RDA** (redundancy analysis) — most consistent performance across all benchmark scenarios (avg 52% explained variance). Use as a reliable baseline to calibrate CMTF and MOFA2 results.
   - *Complementary feature selection*: **sPLS-Regression** (mixOmics) for multivariate feature selection — best performer in benchmark for exploiting inter-omics correlations. Use to identify the specific species-metabolite pairs driving each CMTF/MOFA2 factor.
   - *Compositional-aware selection*: **CODA-LASSO** for univariate feature selection — scale-invariant and preserves sub-compositionality of microbiome data. Use to identify individual microbial taxa with strongest metabolite associations without compositional bias.
   - *Exploratory*: Consider mechanistic metabolic modeling (Nap et al., Nature Comms 2025, doi:10.1038/s41467-025-60233-2) to map which CMTF/MOFA2 latent factors correspond to specific metabolic derangements (tryptophan-NAD axis, BCAA pathways, bile acid deconjugation, SCFA pathways). This establishes causal metabolic relationships complementing statistical associations.
   - Benchmark code available: https://github.com/lmangnier/Benchmark_Integration_Metagenomics_Metabolomics
4. **Subtype discovery**: Cluster CD patients based on integrated latent factors using consensus clustering. Determine optimal number of subtypes via silhouette analysis and gap statistic.
5. **Subtype characterization**: For each subtype, identify — top driving species and metabolites, associated clinical features (Montreal classification, CRP, fecal calprotectin), pathway enrichment (MetaCyc/KEGG).
6. **Validation**: Test subtype reproducibility in curatedMetagenomicData cohorts. Assess association with treatment response where data available.

## Expected Outputs
- 2-4 reproducible Crohn's disease subtypes defined by microbiome-metabolome signatures
- Subtype-specific biomarker panels (species + metabolites)
- Network visualization of species-metabolite correlations per subtype
- Clinical feature associations per subtype (disease location, severity, treatment history)
- MOFA latent factor loadings showing which features drive subtype separation

## Success Criteria
- At least 2 subtypes with silhouette score > 0.3 and distinct biological signatures
- Subtypes replicate in at least one external validation cohort
- At least one subtype shows significant association with a clinical outcome (disease behavior, treatment response, or calprotectin levels)

## Labels
microbiome, multi-omics, biomarker, novel-finding, high-priority
