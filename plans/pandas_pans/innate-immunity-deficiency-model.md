# Innate Immunity Deficiency Model — PANS as Primary Innate Failure

## Objective
Test the hypothesis that PANS/PANDAS reflects a primary DEFICIENCY of innate immunity (lectin complement, trained immunity) with secondary compensatory adaptive immune overshoot (autoantibodies), rather than primary autoimmune hyperactivation. This paradigm inversion reframes the disease mechanism and predicts different treatment strategies.

## Rationale

### Cross-Project Evidence Chain
1. **pans-genetic-variant-pathway-analysis** (analysis): The #1 enriched pathway is lectin complement (MBL2/MASP1/MASP2, FDR=1.12e-5). These are LOSS-OF-FUNCTION variants that IMPAIR innate pathogen clearance, not enhance immunity.
2. **ivig-mechanism-single-cell-analysis** (development): Han VX scRNA-seq shows PANS patients have DOWNREGULATED defense response, innate immunity, and secretory granules. REDUCED TNF/IL-6 responses to TLR stimulation. The pre-treatment immune profile is DEFICIENT, not hyperactive.
3. **cytokine-network-flare-prediction** (analysis): If cytokine profiles during flares show innate deficiency signatures rather than classical autoimmune activation (Th1/Th17 excess), this supports the model.
4. **Literature**: Totè 2025 shows PANDAS has elevated zonulin/LPS (gut barrier permeability) — consistent with impaired mucosal innate defense allowing bacterial translocation.

### What Makes This Unconventional
The entire PANDAS/PANS field is built on the autoimmune model: GAS triggers cross-reactive antibodies → neuronal damage. This hypothesis says: the autoantibodies are REAL but SECONDARY. The PRIMARY defect is that these children cannot clear GAS properly (lectin complement failure), leading to prolonged antigen exposure that overwhelms immune tolerance and produces autoantibodies as a byproduct. The disease driver is innate immune failure, not adaptive immune excess.

No published study has framed PANDAS/PANS as an innate immune deficiency disorder. This is a genuine paradigm inversion.

### What's the Upside
If correct, this changes treatment strategy entirely:
- **Current approach**: immunosuppression (IVIG, steroids, plasmapheresis) to dampen autoimmunity
- **Predicted approach**: enhance innate clearance (prophylactic antibiotics, lectin complement replacement, trained immunity stimulation) to PREVENT the autoimmune cascade
- Explains why IVIG Phase III missed primary OCD endpoint (can't reverse CNS damage) but improved CGI-I (temporarily boosts pathogen clearance via passive antibodies)
- Explains why prophylactic antibiotics work in some PANDAS patients (reduce burden that innate system can't handle)
- Predicts that MBL2 serum levels could be a susceptibility biomarker

### What's the Risk
- Han VX "reduced responses" may be an artifact of treatment history or small sample size (n=5 PANS)
- MBL2 deficiency is common (~5% of population) — most MBL2-deficient children don't get PANDAS, so additional factors are needed
- The autoimmune component may be both primary AND secondary (not mutually exclusive)
- Failure mode: innate immune markers in PANS patients are normal, and the observed downregulation is post-hoc rather than causal

## Data Sources
- **Existing division data**: Re-analyze ivig-mechanism scRNA-seq pipeline outputs for innate immunity gene modules (NK cell, neutrophil, monocyte defense genes). Re-analyze transcriptomic-biomarker-panel for innate vs adaptive immune gene expression ratios.
- **pans-genetic-variant-pathway-analysis**: Extract per-patient variant data from Vettiatil 2026 supplementary tables. Determine whether patients with lectin complement variants have distinct clinical phenotypes.
- **External**: MBL2 deficiency prevalence studies, trained immunity literature (Netea et al., Cell Host & Microbe), Aicardi-Goutières syndrome innate immune profiles for comparison.
- **GEO**: GSE289109 (Shammas encephalitis snRNA-seq) as reference for neuronal damage signatures independent of peripheral immune state.

## Methodology

### Phase 1: Innate Immune Signature Extraction
1. From ivig-mechanism scRNA-seq data: quantify expression of lectin complement genes (MBL2, MASP1, MASP2, FCN1/2/3, COLEC11) across all cell types in PANS vs controls
2. Compute innate-to-adaptive immune gene ratio from transcriptomic-biomarker pipeline
3. Re-analyze cytokine network data: classify each cytokine as innate (IL-1β, TNF-α, IL-6, IL-8) vs adaptive (IL-4, IL-13, IL-17, IFN-γ) and compute directional changes during flares

### Phase 2: Genetic-Immune Correlation
4. Extract per-patient lectin complement variant status from Vettiatil 2026 data
5. Test association: patients with MBL2/MASP variants → lower innate immune gene expression?
6. Compare innate immune profiles in patients with/without cGAS-STING variants (TREX1/SAMHD1)

### Phase 3: Model Testing
7. Build "innate deficiency score" from Phase 1-2 features
8. Test whether innate deficiency score predicts: disease severity, IVIG response, flare frequency
9. Compare PANDAS (strep-triggered, Totè: elevated LPS/zonulin) vs PANS (non-strep, Totè: normal) innate profiles — if the model is correct, PANDAS should show more profound lectin complement deficiency

### Phase 4: Mechanistic Modeling
10. Network analysis: model how lectin complement failure → prolonged GAS exposure → epitope spreading → autoantibody diversification (connects to autoantibody-target-network-mapping)
11. Predict which autoantibody targets would emerge from prolonged vs acute GAS exposure using gas-molecular-mimicry-mapping outputs

## Success Criteria
- **Strong support**: PANS patients show statistically significant reduction in innate immunity gene modules compared to controls, AND lectin complement variant carriers show more severe innate deficiency
- **Moderate support**: Innate deficiency scores correlate with clinical features (flare frequency, IVIG response) even without genetic correlation
- **Refuted**: Innate immune gene expression in PANS patients is normal or elevated compared to controls

## Risk Assessment
- **Most likely failure**: Sample sizes are too small (n=5 in Han VX) to detect meaningful innate immune differences. Mitigation: combine with transcriptomic-biomarker data when available.
- **Conceptual risk**: Innate and adaptive immunity are deeply intertwined — a clean "deficiency → overshoot" separation may be an oversimplification. The model may need refinement to a more nuanced "dysregulation" framing.
- **What we'd learn anyway**: Even if refuted, quantifying innate immune gene expression in PANS patients is novel and publishable. The innate-to-adaptive ratio analysis could improve biomarker panels.

## Labels
catalyst, novel-finding, high-priority, autoimmune, immunology, biomarker
