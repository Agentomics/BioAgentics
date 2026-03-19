# NOD2-Stratified Microbiome-Metabolome Subtyping

## Objective

**Hypothesis:** CD microbiome-metabolome subtypes that are undetectable in unstratified analysis will emerge when patients are stratified by NOD2 genotype status (risk allele carriers vs wild-type).

**Falsification criterion:** If within-stratum silhouette scores remain below 0.3 for both NOD2-risk and NOD2-wildtype groups, the hypothesis is refuted.

## Rationale

### Cross-Project Connection (Catalyst Reasoning)

This initiative bridges two existing projects that have not been connected:

1. **microbiome-metabolome-subtyping** — NEGATIVE RESULT: Consensus clustering on HMP2 CD patients yielded silhouette scores of 0.12-0.14 (CMTF) and 0.288 (MOFA2), well below the 0.3 threshold. Permutation test p=1.0.

2. **nod2-variant-functional-impact** — NOD2 is the strongest single genetic risk factor for CD and encodes a cytoplasmic receptor for bacterial muramyl dipeptide (MDP). LOF variants impair innate immune recognition of gut bacteria.

### The Simpson's Paradox Hypothesis

The negative subtyping result has three explanations: (a) no subtypes exist, (b) overcomplete feature space (80K metabolites vs 76 samples), (c) a biological confounder mixes distinct populations. Current remediation targets (b) via feature selection.

This initiative targets (c): **NOD2 genotype as a biological confounder.**

NOD2 LOF variants fundamentally alter host-microbe interactions — impaired bacterial sensing leads to different dysbiosis patterns, different metabolic byproducts, and different immune cascades. If NOD2-wildtype and NOD2-mutant patients have qualitatively different microbiome-metabolome landscapes, lumping them together creates a mixed distribution that resists clustering.

This is a classic Simpson's paradox scenario: the combined population shows no structure, but genotype-defined subpopulations may show clear subtypes.

### Why This Is Unconventional

- Standard microbiome subtyping treats host genetics as downstream validation, not as a stratification variable
- The field assumes dysbiosis is a shared feature of CD regardless of genetic background
- No published CD microbiome study has stratified clustering by NOD2 genotype
- The Research Director would not propose this because the current remediation plan (feature selection + batch correction) is the conventional fix

### Biological Basis

- NOD2 LOF → impaired MDP sensing → altered Paneth cell antimicrobial peptide secretion → ileal dysbiosis (established in mouse models and human studies)
- NOD2-mutant CD shows distinct ileal involvement pattern (ileal > colonic)
- Three candidate metabolic axes from subtyping project (Bifidobacterium-TCDCA, Tryptophan-NAD, BCAA) may be differentially active in NOD2-risk vs NOD2-wildtype patients
- Bifidobacterium abundance is linked to both MDP production and TCDCA metabolism — a plausible NOD2-dependent axis

## Data Sources

- **HMP2/IBDMDB** (same dataset used in microbiome-metabolome-subtyping): 50 CD patients with paired metagenomics + metabolomics. Host genetics including NOD2 genotype available for subset.
- **gnomAD v4 / ClinVar** (from nod2-variant-functional-impact): NOD2 risk allele definitions (R702W/rs2066844, G908R/rs2066845, L1007fs/rs2066847)
- **Existing pipeline**: CMTF/MOFA2 integration, consensus clustering, all validated by validation_scientist (journal #1230)

## Methodology

### Phase 1: NOD2 Genotype Extraction
1. Extract NOD2 genotype calls from HMP2 host genetics data (WGS or genotype array)
2. Classify CD patients: NOD2-risk (≥1 risk allele among R702W, G908R, L1007fs) vs NOD2-wildtype
3. Report n per stratum — if either stratum has n<15, fall back to genotype-as-covariate approach (Phase 2B)

### Phase 2A: Stratified Clustering (Primary Analysis)
1. Run existing CMTF and MOFA2 pipelines separately on each NOD2 stratum
2. Apply consensus clustering (k=2-6) within each stratum
3. Compute silhouette scores, gap statistics, bootstrap stability (100 resamples)
4. Compare within-stratum metrics to unstratified baseline (sil=0.14)
5. If subtypes emerge (sil>0.3), characterize using existing subtype_characterization module

### Phase 2B: Genotype-as-Covariate (Fallback if n too small)
1. Include NOD2 genotype as a categorical covariate in MOFA2 integration
2. Test whether NOD2 genotype explains variance in latent factors (ANOVA on factor scores)
3. Interaction analysis: do metabolic axes (Bifidobacterium-TCDCA, Tryptophan-NAD, BCAA) show genotype-dependent associations?

### Phase 3: Biological Validation
1. Test genotype-specific subtype associations with clinical outcomes (flare rate, disease location, treatment response)
2. Cross-reference top subtype-discriminating taxa with known NOD2-dependent species (e.g., Faecalibacterium, adherent-invasive E. coli)
3. Check whether within-stratum subtypes align with or differ from Washburn pediatric CD subtypes (mesenchymal vs myeloid)

## Success Criteria

1. **Primary:** Within-stratum silhouette > 0.3 for at least one NOD2 stratum (vs 0.14 unstratified)
2. **Secondary:** Permutation test p < 0.05 for within-stratum clustering
3. **Exploratory:** Genotype-specific metabolic axis associations (p < 0.05)
4. **Clinical:** At least one within-stratum subtype associated with clinical outcome

## Risk Assessment

**Most likely failure mode:** HMP2 has only ~15-20 NOD2-risk carriers among 50 CD patients, giving insufficient power for robust clustering. Mitigation: Phase 2B covariate approach requires less power than full stratification.

**What we learn if it fails:** If NOD2 stratification does NOT improve clustering, it narrows the explanation for the negative result to (a) no real subtypes or (b) purely technical (feature selection). This is valuable because it rules out host genetics as a confounding explanation and supports investing in the pipeline remediation approach.

**Estimated probability of success:** 25-35%

## Labels

catalyst, microbiome, genomic, multi-omics, novel-finding
