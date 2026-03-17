# CD Flare Prediction from Longitudinal Multi-Omics

## Objective

Predict Crohn's disease flares before clinical onset using longitudinal multi-omic trajectory features, enabling preemptive treatment intensification.

## Background

50-60% of CD patients experience relapsing-remitting disease courses. Current monitoring relies on reactive markers (CRP, fecal calprotectin) that detect inflammation after it manifests clinically. No validated pre-flare signatures exist. The HMP2/IBDMDB longitudinal cohort provides a unique opportunity: matched metagenomics, metabolomics, host transcriptomics, serology, and clinical activity scores collected over ~1 year with dense sampling intervals. Most microbiome-CD studies are cross-sectional and miss temporal dynamics that precede flares.

## Data Sources

1. **HMP2/IBDMDB** (primary): 132 subjects (67 CD, 38 UC, 27 non-IBD controls), ~2,965 stool/biopsy samples over 1 year. Includes:
   - Metagenomics (MetaPhlAn species profiles, HUMAnN pathway abundances)
   - Metabolomics (untargeted, ~8,000 features)
   - Host transcriptomics (from biopsies at subset of timepoints)
   - Serology (ASCA, anti-CBir1, anti-OmpC)
   - Clinical activity (HBI scores, clinical metadata)
   - Download: https://ibdmdb.org/ and https://www.nature.com/articles/s41586-019-1237-9
2. **RISK cohort longitudinal subset** (validation, if available): Pediatric CD with longitudinal sampling
3. **curatedMetagenomicData** (external validation): Any longitudinal CD cohorts with >3 timepoints per patient
4. **PRJNA1216747** (validation): Inception cohort of 73 treatment-naive CD patients, 1-year longitudinal 16S rRNA + 324 metabolic pathways, biologic exposure annotated (PMC12456514). Baseline dysbiosis correlated with disease severity and predicted biologic need. Aerobic respiration pathways correlated with CRP/calprotectin — aligns with the anaerobe-depletion/aerobic-shift framework.
5. **Foli-seq / fecal exfoliome** (future extension): Non-invasive host transcriptomic profiling from stool via fecal exfoliated mRNA (Nature Biotechnology 2025, PMID 41249836). Captures temporal immune dynamics, stratifies IBD by severity, reveals host-microbe interaction networks — all without endoscopy. Not yet applied to flare prediction, but its dense temporal resolution is exactly what longitudinal monitoring requires. Note as validation/extension technology when cohort data becomes available.

## Methodology

### Phase 1: Data Preparation & Flare Event Definition
- Define flare events from HBI trajectory: sustained HBI increase ≥3 points or transition from HBI <5 to HBI ≥5
- Define pre-flare windows: 2-week and 4-week windows before each flare event
- Define stable windows: ≥4 weeks of HBI <5 with no subsequent flare within 4 weeks
- Pair pre-flare vs stable windows per patient to create classification instances

### Phase 2: Temporal Feature Engineering
- **Microbiome volatility**: Bray-Curtis dissimilarity between consecutive samples, Shannon diversity trajectory slopes, species-level abundance change rates
- **Metabolomic trajectories**: Sliding-window slopes for bile acids, SCFAs, tryptophan metabolites, acylcarnitines. Rate-of-change features for top 200 most variable metabolites
- **Pathway dynamics**: HUMAnN pathway abundance change rates, focusing on oxidative stress, butyrate production, sulfur metabolism
- **Host features**: If biopsy transcriptomics available near flare events, compute inflammatory module scores (TNF, IL-17, IFN-gamma pathways)
- **Serologic trends**: ASCA/anti-CBir1 trajectory slopes (if longitudinal serology available)

### Phase 3: Pre-Flare Signature Discovery
- Within-patient paired analysis: pre-flare vs stable features per individual (controls for inter-individual variation)
- Feature selection: stability selection via randomized lasso across bootstrap samples
- Multi-omic integration (two-stage approach):
  - **Stage 1 — MOFA2** (primary): Latent factor discovery across omic layers. Test whether latent factor trajectories shift before flares. MOFA2 handles missing data and heterogeneous data types but treats samples independently — it does not model temporal structure.
  - **Stage 1 — MintTea** (complementary module discovery): Sparse generalized CCA (sgCCA) to identify disease-associated multi-omic modules with flare status as the outcome variable. May capture cross-omic correlations missed by MOFA2's unsupervised approach. GitHub: efratmuller/MintTea. Nature Comms 2024 (doi:10.1038/s41467-024-46888-3).
  - **Stage 2 — Temporal modeling of latent factors**: Feed MOFA2 latent factor trajectories (or MintTea module scores) into temporal models in Phase 4. This two-stage design separates multi-omic integration (Stage 1) from temporal prediction (Stage 2), keeping each stage tractable.

### Phase 4: Predictive Modeling
- **Sliding-window classifier**: Extract feature windows at each timepoint, classify as "pre-flare" or "stable"
- Models: XGBoost (primary), logistic regression (interpretable baseline), GRU (handles variable-length time-series — relevant for HMP2's uneven sampling intervals; preferred over LSTM for smaller datasets). **DBN option**: Dynamic Bayesian Networks (from PALM pipeline, Briefings in Bioinformatics 2025 review) for directed temporal causal structure — captures which omic features drive which others across timepoints, providing mechanistic interpretability that black-box models lack. Trade-off: DBN may overfit with n=67 CD patients; use as an exploratory/interpretive complement to XGBoost, not as primary classifier.
- Validation: Leave-one-patient-out cross-validation (critical for temporal data to avoid leakage)
- Calibration: Platt scaling for probability outputs
- Comparison baselines: calprotectin-only, CRP-only, microbiome diversity-only

### Phase 5: Clinical Utility Analysis
- Define clinically meaningful lead time: can the model detect pre-flare state ≥2 weeks before HBI criteria met?
- Net benefit analysis (decision curve) for different risk thresholds
- SHAP feature importance: which omic features contribute most to pre-flare prediction?

## Expected Outputs

1. Defined pre-flare multi-omic signature (top features ranked by SHAP importance)
2. Temporal risk score model with probability calibration
3. Comparison of prediction lead times (2-week vs 4-week windows)
4. Feature importance by omic layer (which data types are most informative?)
5. Visualization: patient-level risk trajectories overlaid with clinical events

## Success Criteria

- Leave-one-patient-out AUC > 0.70 for pre-flare classification at 2-week lead time
- Multi-omic model outperforms calprotectin-only baseline by ≥0.10 AUC
- Identification of ≥5 pre-flare features with consistent directionality
- At least 2 omic layers contribute meaningfully to prediction (SHAP importance >10% each)

## Labels

clinical, microbiome, multi-omics, biomarker, novel-finding, high-priority
