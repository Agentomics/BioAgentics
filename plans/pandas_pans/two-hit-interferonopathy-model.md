# Two-Hit Interferonopathy Model — Compound Genetic Vulnerability in PANS

## Objective
Test the hypothesis that PANS/PANDAS susceptibility requires compound genetic hits across two distinct innate immune pathways: (1) lectin complement deficiency (MBL2/MASP1/MASP2 variants → impaired GAS clearance) AND (2) cGAS-STING dysregulation (TREX1/SAMHD1 variants → excessive type I IFN response to persistent pathogen DNA). Neither hit alone causes PANDAS; their interaction creates episodic infection-triggered neuropsychiatric disease.

## Rationale

### Cross-Project Evidence Chain
1. **pans-genetic-variant-pathway-analysis** (analysis): Identified two separate high-confidence PPI modules in the SAME cohort — lectin complement (MBL2-MASP1-MASP2, STRING 0.986-0.999) and cGAS-STING (TREX1-SAMHD1, STRING 0.921). The existing report analyzes these modules SEPARATELY. No one has modeled their INTERACTION.
2. **gas-molecular-mimicry-mapping** (analysis): Identifying which GAS proteins drive mimicry. The two-hit model predicts that persistent GAS (from complement failure) releases more DNA → more cGAS-STING activation → more IFN → more neuroinflammation. The mimicry mapping data could identify which GAS proteins also release immunostimulatory DNA.
3. **Literature**: TREX1 mutations → Aicardi-Goutières syndrome (interferonopathy with neuropsychiatric features). SAMHD1 mutations → AGS phenocopy. MBL2 deficiency → recurrent infections (common, usually benign). But MBL2 deficiency + cGAS-STING hyperactivation = a novel compound phenotype never described.

### What Makes This Unconventional
- **Not classical polygenic risk**: This isn't GWAS-style common variants with small effects. It's two RARE, functionally convergent pathway hits.
- **Borrows from monogenic interferonopathy research**: Aicardi-Goutières is the template disease, but PANDAS is EPISODIC (not constitutive) — the cGAS-STING variants are hypomorphic, not null. They don't cause constitutive IFN but lower the threshold for infection-triggered IFN storms.
- **Explains the PANDAS paradox**: Why is PANDAS rare if GAS is common? Because you need BOTH hits. MBL2 deficiency (~5% prevalence) × TREX1/SAMHD1 variants (very rare) = extremely rare compound phenotype.
- **Explains episodicity**: Flares occur when infection (GAS or other) provides the DNA trigger. Between infections, cGAS-STING activity returns to baseline. The genetics load the gun; infection pulls the trigger.

### What's the Upside
If correct:
- **Genetic test for PANS susceptibility**: Screen for compound lectin complement + cGAS-STING variants
- **Targeted therapy**: cGAS-STING inhibitors (e.g., H-151, currently in cancer trials) could prevent flares without broad immunosuppression
- **Explains treatment heterogeneity**: Patients with different genetic hit combinations would respond to different therapies (complement replacement vs STING inhibition vs antibiotics)
- **Unifies PANDAS and PANS**: Both involve the same cGAS-STING checkpoint, but with different triggers (GAS DNA for PANDAS, other pathogen DNA for PANS)

### What's the Risk
- The Vettiatil cohort (n=32) may not have enough patients carrying variants in BOTH modules to test the interaction
- The variants identified may be passengers, not drivers — ultra-rare variants in a small cohort have high false discovery rates
- cGAS-STING activation in PANDAS has never been directly measured — the whole pathway activation model is inferred from genetics
- Failure mode: per-patient genotyping shows no enrichment for compound hits (variants in different patients, not the same patients)

## Data Sources
- **Vettiatil 2026 (PMID 41662332)**: Supplementary tables with per-patient variant data. Critical to determine which patients carry lectin complement AND cGAS-STING variants.
- **Scientific Reports s41598-022-15279-3**: Original discovery cohort for additional per-patient genotyping data.
- **pans-genetic-variant-pathway-analysis outputs**: Enrichment results, PPI network, pathway modules.
- **Aicardi-Goutières syndrome literature**: Interferon signature gene (ISG) panels, clinical comparisons. OMIM #225750 (TREX1), #612952 (SAMHD1).
- **GEO GSE289109**: Shammas 2026 snRNA-seq from encephalitis model — reference IFN-driven neuronal damage transcriptome.
- **cGAS-STING inhibitor literature**: H-151 (Haag 2018, Nature), compound databases for repurposing candidates.

## Methodology

### Phase 1: Per-Patient Compound Genotyping
1. Extract per-patient variant calls from Vettiatil 2026 supplementary data
2. For each patient: classify variants by pathway module (lectin complement, cGAS-STING, DDR, mitochondrial, gut-immune, chromatin)
3. Test enrichment: are compound hits (variants in >=2 modules) more common than expected by chance?
4. If compound hits exist: correlate with clinical severity (if clinical data available)

### Phase 2: cGAS-STING Activation Modeling
5. Using the pans-genetic-variant-pathway-analysis PPI network: model cGAS-STING pathway activation with normal vs variant TREX1/SAMHD1
6. Simulate GAS DNA exposure: predict type I IFN output for wild-type vs variant genotypes
7. Cross-reference with gas-molecular-mimicry-mapping: identify which GAS proteins are associated with DNA release during infection (lytic enzymes, DNases)

### Phase 3: Interferon Signature Testing
8. Define a PANS interferon signature gene (ISG) panel from AGS literature (IFI27, IFI44L, IFIT1, ISG15, RSAD2, SIGLEC1 — the 6-gene AGS panel)
9. Test ISG expression in available PANS transcriptomic data (transcriptomic-biomarker-panel datasets)
10. Compare PANS ISG scores to: healthy controls, primary OCD, autoimmune encephalitis

### Phase 4: Therapeutic Modeling
11. Model cGAS-STING inhibitor effects using pathway simulation
12. Predict which patients (by genotype) would benefit from STING inhibition vs complement replacement vs combined therapy
13. Cross-reference cGAS-STING inhibitors in clinical trials (ClinicalTrials.gov) for repurposing potential

## Success Criteria
- **Strong support**: Individual PANS patients carry variants in BOTH lectin complement AND cGAS-STING modules at rates exceeding chance expectation, AND ISG scores are elevated in PANS vs controls
- **Moderate support**: ISG scores are elevated in PANS even without per-patient compound genotyping confirmation (would support the interferonopathy component alone)
- **Refuted**: No per-patient compound hits exist (variants are in different patients), AND ISG scores are normal in PANS

## Risk Assessment
- **Most likely failure**: Per-patient genotyping shows variants in different patients (no compound hits). Mitigation: the interferonopathy component (cGAS-STING alone) is still testable via ISG scoring.
- **Data access risk**: Vettiatil supplementary tables may not include per-patient variant calls. Mitigation: contact corresponding author or use the earlier Scientific Reports dataset.
- **What we'd learn anyway**: ISG profiling in PANS is novel regardless. Even if the two-hit model fails, quantifying type I IFN activation in PANS vs primary psychiatric disorders adds to diagnostic biomarker development (feeds into transcriptomic-biomarker-panel).

## Labels
catalyst, novel-finding, high-priority, genomic, autoimmune, immunology
