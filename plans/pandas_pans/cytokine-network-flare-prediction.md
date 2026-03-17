# Cytokine Network Meta-Analysis for PANDAS/PANS Flare Prediction

## Objective
Perform a computational meta-analysis of cytokine/chemokine profiles across published PANDAS/PANS studies to define the immune signature of disease flares and build a predictive model for flare onset.

## Background
PANDAS/PANS flares — sudden-onset exacerbations of OCD, tics, anxiety, and other neuropsychiatric symptoms — are the hallmark of the disease and the primary source of morbidity. Individual studies have measured various cytokines during flares vs. remission but with small sample sizes and inconsistent panels (some measure Th1/Th2, others Th17, others complement). No meta-analysis has computationally integrated these fragmented datasets to define a consensus cytokine signature. Understanding the cytokine network architecture during flares could identify therapeutic targets (e.g., IL-17 inhibitors, TNF-α blockers) and enable flare prediction from routine bloodwork.

## Data Sources
- **Published Literature:** Systematic extraction of cytokine/chemokine measurements from PANDAS/PANS clinical studies (PubMed search: PANDAS OR PANS + cytokine/interleukin/chemokine/inflammatory markers)
- **GEO:** Immune cell profiling datasets from pediatric autoimmune neuroinflammation
- **ImmPort:** Pediatric immune monitoring studies with longitudinal cytokine data
- **ClinicalTrials.gov:** IVIG and plasmapheresis trial data with cytokine endpoints
- **ImmunoGlobe / InnateDB:** Cytokine-cytokine interaction networks for pathway modeling

## Methodology
1. **Systematic Data Extraction:** Search PubMed for all PANDAS/PANS studies reporting cytokine/chemokine levels. Extract: analyte, measurement method, sample type (serum/CSF/plasma), flare vs. remission status, sample size, mean/median, SD/IQR, p-values.
2. **Meta-Analysis:** For each cytokine measured in ≥3 studies, compute pooled effect sizes (standardized mean difference, random-effects model). Assess heterogeneity (I², Cochran's Q). Generate forest plots. Sensitivity analysis by sample type and measurement method.
3. **Cytokine Network Construction:** Build a cytokine interaction network using ImmunoGlobe/InnateDB. Overlay meta-analysis results (upregulated = red, downregulated = blue) to identify activated network modules during flares.
4. **Immune Axis Classification:** Classify the flare signature by immune axis: Th1 (IFN-γ, TNF-α), Th2 (IL-4, IL-13), Th17 (IL-17A, IL-22), regulatory (IL-10, TGF-β), innate (IL-1β, IL-6, TNF-α). Determine dominant axis. **IL-6 is the central hub node** — Pozzilli et al. showed no other cytokine was elevated without concurrent IL-6 in 112 pediatric neuroinflammatory patients (approved task #363). **NLRP3→IL-1β is an independent parallel arm**, not subordinate to autoantibody signaling (Luo et al., approved task #363).
5. **IFNγ Epigenetic Persistence Layer:** Shammas G et al. 2026, Neuron (PMID 41448185) demonstrated IFNγ drives persistent chromatin closing in neurons via epigenetic mechanisms — synaptopathy OUTLASTS the immune response. Add an **"epigenetic memory" node downstream of IFNγ** in the network model. IFNγ must be modeled not only as an acute Th1 flare mediator but as a driver of PERSISTENT symptoms via irreversible/slowly-reversible epigenetic damage to neuronal synaptic gene expression. This mechanism explains the central PANS paradox: why neuropsychiatric symptoms persist after peripheral inflammation resolves. The network should distinguish acute cytokine effects (hours-days) from epigenetic persistence effects (weeks-months). Human encephalitis tissue confirms the finding.
5. **Predictive Modeling:** Using datasets with longitudinal samples, train a predictive model for flare onset from pre-flare cytokine levels. Methods: logistic regression, random forest, time-series modeling where temporal data available.
6. **Treatment Response Analysis:** Where available, compare cytokine profiles pre/post IVIG, plasmapheresis, and antibiotic treatment to identify which cytokine shifts predict treatment response.

## Expected Outputs
- Consensus cytokine signature of PANDAS/PANS flares (forest plots for each analyte)
- Cytokine interaction network map highlighting activated modules during flares
- Immune axis classification (Th1/Th2/Th17/innate balance during flares)
- Flare prediction model (if sufficient longitudinal data)
- Treatment-response cytokine markers
- Identification of candidate cytokine-targeted therapies

## Success Criteria
- Include data from ≥10 published studies in meta-analysis
- Identify ≥3 cytokines with significant pooled effect sizes (flare vs. remission)
- Network analysis reveals coherent immune module activation (not random)
- Flare signature is distinguishable from general infection-driven inflammation

## Labels
biomarker, immunology, clinical, novel-finding, promising
