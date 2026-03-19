# TS Developmental Trajectory Modeling

## Objective
Model the neurodevelopmental trajectory of Tourette syndrome to identify molecular and circuit-level factors underlying tic onset, peak severity, and spontaneous remission — the defining clinical feature that ~60% of TS patients experience by adulthood.

## Background
Tourette syndrome has a characteristic developmental arc: tic onset at mean age 6.4 years, peak severity at 10-12 years, and spontaneous improvement in approximately 60% of patients by age 18 (Bloch et al. 2006, Groth et al. 2017). This temporal pattern is among the most distinctive features of any neuropsychiatric disorder and implies that TS pathophysiology is intimately tied to specific neurodevelopmental windows. Yet no study has computationally mapped TS risk genes to developmental expression trajectories to explain WHY tics follow this pattern or WHAT distinguishes the ~40% of patients whose tics persist into adulthood.

Key observations:
1. **Tic onset window (5-7y)** coincides with a period of active corticostriatal circuit maturation and dopaminergic innervation of the striatum
2. **Peak severity (10-12y)** aligns with pubertal onset and a surge in gonadal steroid hormones that modulate dopamine signaling
3. **Remission window (15-18y)** corresponds to cortical maturation (especially prefrontal), myelination completion, and GABAergic inhibitory circuit refinement
4. **Persistence (~40%)** may reflect a failure of compensatory circuit maturation, potentially due to disrupted interneuron development
5. **Pediatric DBS > adult DBS** (70% vs 56% improvement at 60 months) — further evidence of a developmental window for intervention

This initiative aims to connect TS genetic architecture (GWAS risk loci, rare variant genes) with developmental neurobiology using public transcriptomic atlases to build a computational model of why TS follows its characteristic temporal pattern.

## Data Sources
- **BrainSpan Atlas** — Developmental transcriptomics covering 8 post-conception weeks (PCW) through 40 years across 16 brain regions including striatum (STR), mediodorsal thalamus (MD), and multiple cortical areas. Freely available via API and bulk download.
- **Allen Brain Atlas (adult)** — Spatial gene expression across adult human brain for baseline comparison.
- **GTEx v8** — Adult tissue expression for brain regions (caudate, putamen, nucleus accumbens, cortex, cerebellum) and non-brain tissues for specificity.
- **TS GWAS risk genes** — From TSAICG 2024 (9,619 cases): genome-wide significant loci and suggestive loci. Top genes: BCL11B, NDFIP2, RBM26 + SLITRK1, HDC, CNTN6 from prior studies.
- **TS rare variant genes** — From ts-rare-variant-convergence project: NRXN1, SLITRK1, HDC, plus de novo genes PPP5C, EXOC1, GXYLT1.
- **Wang et al. 2025 cell-type markers** — Striatal interneuron and MSN markers from TS caudate snRNA-seq.
- **Published natural history cohorts** — Meta-analytic tic trajectory data from Bloch 2006 (N=46 longitudinal), Groth 2017 (N=314 Danish registry), EMTICS COURSE cohort (N=715).

## Methodology

### Phase 1 — TS Risk Gene Developmental Expression Profiling
Map expression trajectories of all TS GWAS risk genes and rare variant genes across BrainSpan developmental timepoints in striatum, thalamus, and cortex. Cluster genes by temporal expression pattern:
- **Early-peak genes** — high expression in fetal/infant period, declining with age
- **Onset-window genes** — expression peak at 4-8 years
- **Adolescent-peak genes** — expression peak at 12-18 years
- **Flat/constitutive genes** — stable expression across development

Identify which temporal clusters are enriched for TS risk genes relative to genome-wide background. Use WGCNA to identify co-expression modules with developmentally dynamic patterns.

### Phase 2 — Critical Period Gene Module Analysis
From Phase 1 modules, identify gene sets whose expression dynamics match the clinical trajectory of TS:
- **Module A (onset):** Genes that upregulate in striatum at ~5-7 years — candidate drivers of tic emergence
- **Module B (peak):** Genes peaking at 10-12 years in CSTC circuit — candidate mediators of tic severity
- **Module C (remission):** Genes that increase in late adolescence in cortex/striatum — candidate mediators of compensatory inhibition
Test enrichment of TS GWAS genes in each module. Perform pathway analysis (GO, KEGG) on each module to identify biological processes.

### Phase 3 — Cell-Type Developmental Dynamics
Using Wang et al. 2025 cell-type markers (MSN, cholinergic interneuron, parvalbumin interneuron, microglia), decompose BrainSpan bulk expression into estimated cell-type proportions across development. Test whether:
- Cholinergic interneuron signature peaks during the tic-onset window
- Parvalbumin interneuron signature increases during remission window (inhibitory circuit maturation)
- D2-MSN:D1-MSN ratio shifts across development and correlates with tic trajectory
- Microglia activation signature has a developmental component

### Phase 4 — Persistence vs. Remission Molecular Model
Integrate Phases 1-3 to build a computational model predicting molecular features of persistent vs. remitting TS:
- **Remission model:** PV interneuron maturation + cortical GABA strengthening → increased inhibitory control over CSTC circuit
- **Persistence model:** Disrupted PV interneuron development (consistent with ts-striatal-interneuron-pathology findings) → failure of compensatory inhibition
- **Striatal zonation attenuation model [NEW]:** Mesoscale striatal atlas (bioRxiv Mar 2026; doi:10.64898/2026.03.04.709715; 1.1M cells, 19 donors) shows 6 molecular zones with age-dependent spatial attenuation — 88.2% of zone-specific genes lose spatial differentiation in older donors. Dorsal zones (CSTC-relevant) show greatest age-related change. This spatial attenuation may provide a molecular substrate for tic remission: dorsal-zone MSN transcriptomic convergence during adolescence could reduce aberrant CSTC circuit activity. Test whether dorsal-zone gene attenuation trajectory correlates with remission-window timing.
- Identify specific genes whose developmental trajectories are most predictive of the persistence/remission boundary
- Cross-reference with gonadal steroid hormone receptor expression (AR, ESR1, ESR2) to test hormonal modulation hypothesis

### Phase 5 — Therapeutic Window Prediction
Use the developmental model to identify optimal intervention windows:
- At what developmental stage would DBS be most effective? (Compare with pediatric > adult DBS data)
- Which gene modules represent druggable targets during specific developmental windows?
- Could early intervention during the onset window prevent tic escalation?
- Generate testable predictions for clinical validation

## Expected Outputs
- Developmental expression heatmaps for all TS risk genes across BrainSpan timepoints
- WGCNA co-expression modules with developmental temporal signatures
- Cell-type proportion estimates across development in striatum
- Enrichment statistics: TS GWAS genes vs. onset/peak/remission gene modules
- Computational model of persistence vs. remission molecular features
- Therapeutic window predictions with specific gene targets

## Success Criteria
- ≥1 developmentally dynamic gene module significantly enriched for TS risk genes (p < 0.05, FDR-corrected)
- Temporal expression pattern of enriched module matches known TS clinical trajectory (onset, peak, or remission)
- Cell-type deconvolution reveals developmentally dynamic interneuron signature correlating with remission
- Model generates ≥3 testable predictions about persistence vs. remission that can be validated against published cohort data

## Labels
clinical, genomic, novel-finding, high-priority, promising
