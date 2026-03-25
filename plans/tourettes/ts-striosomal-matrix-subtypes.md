# Striosome-Matrix Compartment Model of TS Subtypes

## Objective

Test whether Tourette syndrome clinical subtypes (TS+OCD, TS+ADHD, simple-motor-predominant) map onto differential disruption of striosomal vs. matrix striatal compartments, using single-cell transcriptomics, spatial molecular zone annotations, and genetic risk stratification.

**Falsifiable hypothesis:** TS risk genes and differentially expressed genes from postmortem TS striatal tissue will show statistically significant compartment bias — striosome-enriched genes associating with compulsive/complex-tic phenotypes and matrix-enriched genes associating with simple-motor/ADHD phenotypes — when tested against mesoscale molecular zone annotations and comorbidity-stratified polygenic risk scores.

## Rationale

Three independent lines of evidence converge on striosomal compartmentalization as relevant to TS:

1. **Celsr3/D3 mechanism** (Cadeddu et al., Mol Psychiatry 2025): CELSR3-deficient mice show striosomal-specific D3 receptor upregulation in D1+ neurons, driving tic-like behaviors. D3 blockade reduces tics. This is a rare variant gene with dual rare+common support.

2. **Mesoscale striatum atlas** (Kraft et al. bioRxiv Mar 2026, doi:10.64898/2026.03.04.709715): 6 molecular zones with 199 zone-specific DE genes. Dorsal zones enriched for synaptic remodeling; ventral for chaperone/hedgehog signaling. Zone-specific MSN populations identified. Age-dependent spatial attenuation (88.2% of genes) may underlie tic remission. **Note: GEO data not yet deposited; Phase 1 blocked pending data availability.**

3. **Cross-project convergence:** ts-rare-variant-convergence found 11 convergent pathways including dopaminergic synapse and neuronal system. ts-comorbidity-genetic-architecture defines 3 PRS strata (compulsive, neurodevelopmental, ts_specific). No existing project bridges compartment-level spatial data with genetic subtyping.

## Data Sources

- Mesoscale human striatum atlas: Kraft et al. bioRxiv March 2026 (doi:10.64898/2026.03.04.709715), 1.1M cells, Slide-tags, 6 molecular zones, 19 donors. **DATA PENDING** — no GEO accession deposited yet (GSE303705 was wrong — mouse intestinal dataset). Monitor for data deposition.
- Wang et al. snRNA-seq TS basal ganglia (GSE151761, GSE152058) — already downloaded by ts-striatal-interneuron-pathology
- CSTC atlas AHBA data — already computed by cstc-circuit-expression-atlas
- Striosome/matrix marker gene sets: MU-opioid receptor (OPRM1), substance P (TAC1), enkephalin (PENK), calbindin (CALB1), somatostatin (SST)
- TS GWAS summary statistics (TSAICG) — shared with ts-gwas-functional-annotation
- Comorbidity-stratified PRS results — from ts-comorbidity-genetic-architecture Phase 3

## Methodology

### Phase 1: Compartment Annotation of Mesoscale Zones
- Map the 6 mesoscale molecular zones to classical striosome/matrix identity using canonical marker genes (OPRM1, TAC1 for striosome; CALB1, PENK for matrix)
- Compute compartment specificity scores for each zone
- Identify zone-specific gene modules (WGCNA or NMF)

### Phase 2: TS Risk Gene Compartment Bias
- Test whether TS GWAS genes and rare variant genes show enrichment in striosome-associated vs. matrix-associated zones (Fisher exact + permutation)
- Stratify by evidence stream: do rare variant genes preferentially map to striosomes and GWAS genes to matrix, or vice versa?
- Use Wang et al. TS snRNA-seq DE genes to validate: are TS-disrupted genes compartment-biased?

### Phase 3: Comorbidity-Compartment Mapping
- Take the 3 PRS strata from ts-comorbidity-genetic-architecture (compulsive, neurodevelopmental, ts_specific)
- For each stratum's top genetic drivers, compute compartment bias scores from Phase 1
- Test hypothesis: compulsive stratum → striosome enrichment; neurodevelopmental/ADHD stratum → matrix enrichment

### Phase 4: Pharmacological Predictions
- From ts-drug-repurposing-network, classify top drug candidates by their target compartment (D2/matrix vs. D3/striosome vs. non-compartment-specific)
- Generate subtype-specific drug recommendation matrix
- Cross-reference with clinical trial response data where available

### Phase 5: Developmental Dynamics
- Using BrainSpan data and mesoscale atlas age-attenuation data, model whether compartment-specific disruption has different developmental trajectories
- Test whether striosome-biased genes attenuate faster/slower than matrix genes during the TS remission window

## Success Criteria

1. ≥60% of TS risk genes show statistically significant compartment bias (FDR < 0.05)
2. Striosome vs. matrix bias differs between comorbidity PRS strata (p < 0.05)
3. At least one testable pharmacological prediction (compartment-specific drug → subtype-specific response)

## Risk Assessment

**Most likely failure mode:** Human striosome/matrix distinction is molecularly fuzzier than in rodents. The mesoscale zones may not cleanly map to classical compartments.

**What we'd learn anyway:** Even if compartment mapping is imperfect, zone-specific enrichment analysis would still reveal spatial organization of TS risk within the striatum — a novel contribution.

**Mitigation:** Use continuous zone enrichment scores rather than binary striosome/matrix classification. The 6-zone framework from the mesoscale atlas is data-driven and doesn't require classical compartment labels.

## Labels

catalyst, novel-finding, high-priority, promising, genomic, neuroimaging, comorbidity
