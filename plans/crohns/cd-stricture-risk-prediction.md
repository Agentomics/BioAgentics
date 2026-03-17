# CD Stricture Risk Prediction

## Objective
Develop a multi-feature predictive model for stricturing (B2) and penetrating (B3) disease behavior in Crohn's disease by integrating genomic, transcriptomic, and clinical features to enable early risk stratification at diagnosis.

## Background
Approximately 30-50% of CD patients progress from inflammatory (B1) to stricturing (B2) or penetrating (B3) phenotype within 10 years of diagnosis, often requiring surgical intervention. Early identification of high-risk patients could enable more aggressive upfront therapy (early biologic initiation, combination therapy) to prevent irreversible bowel damage. Known risk factors include NOD2 variants, young age at diagnosis, ileal location, and perianal disease, but these clinical features alone achieve only ~60% AUC. Molecular features — particularly mucosal gene expression related to fibrosis pathways (TGFβ, Wnt, collagen deposition, MMP activity) — remain underexplored as prognostic markers. The RISK pediatric cohort uniquely provides genetic, transcriptomic, and longitudinal clinical data with disease behavior progression tracked over time.

## Data Sources
- **RISK cohort (GSE57945)** — Pediatric CD inception cohort: ileal gene expression (RNA-seq), genotyping, clinical metadata including Montreal classification behavior progression at follow-up. ~300 CD patients.
- **IBDGC GWAS summary statistics** — Risk variants associated with stricturing/penetrating phenotype (from subphenotype GWAS analyses)
- **GSE16879** (Arijs et al.) — Mucosal gene expression with clinical phenotype annotations, potential external validation
- **ClinVar/gnomAD** — Variant frequencies for CD-associated fibrosis genes (NOD2, TGFβR, SMAD3, MUC19)
- **MSigDB** — Curated fibrosis and ECM gene sets for pathway-level features

## Methodology
1. **Phenotype extraction**: Parse RISK cohort clinical metadata to identify patients who progressed to B2/B3 vs those who remained B1 at latest follow-up. Define binary outcome (progressor vs non-progressor) and time-to-event if dates available.
2. **Feature engineering**:
   - *Genetic*: Polygenic risk score from CD-associated stricturing variants (NOD2, ATG16L1, IRGM, LRRK2, MUC19). Individual variant genotypes for top hits. **Cross-reference 90 new GWAS risk loci** (medRxiv 2025, 63,415 IBD cases) against spatial modules from Nature Genetics stricturing study (doi:10.1038/s41588-025-02225-y) — risk loci enriched in inflammatory fibroblast/lymphoid follicle spatial zones may carry stricture-specific risk and should be weighted in the PRS.
   - *Transcriptomic*: Fibrosis pathway scores (TGFβ, Wnt, collagen, MMP, EMT signatures from MSigDB). Top differentially expressed genes between progressors and non-progressors. **Priority candidate genes from IBD 2025 8-gene stricture panel** (doi:10.1093/ibd/izaf026): LY96, AKAP11, SRM, GREM1, EHD2, SERPINE1, HDAC1, FGF2. GREM1 (BMP antagonist, fibroblast-specific, AUC ≥0.75) and SERPINE1 (PAI-1, ECM remodeling) are highest priority for stability selection given cell-type specificity and biological plausibility. **TL1A pathway genes**: TNFSF15 (TL1A) and TNFRSF25 (DR3) expression as candidate features. TL1A signals through DR3 to drive both inflammation and fibrosis simultaneously — a dual mechanism distinct from other cytokine pathways. TNFSF15 has established GWAS associations with CD. Anti-TL1A antibodies (duvakitug Phase 2b: 48% endoscopic response vs 13% placebo; tulisokibart Phase 2a: significant endoscopic response) are advancing to Phase 3, making TL1A pathway activity a therapeutically actionable feature. Patients with active TL1A-driven fibrotic programs may be identifiable from transcriptomics and stratified for emerging anti-TL1A therapies.
   - *Clinical*: Age at diagnosis, disease location (L1-L3), perianal disease, baseline CRP, baseline calprotectin if available.
3. **Model training**: Gradient boosted trees (XGBoost) + logistic regression ensemble. Nested 5-fold cross-validation for hyperparameter tuning and unbiased performance estimation.
4. **Feature importance**: SHAP values to rank predictive features and quantify relative contribution of genetic, transcriptomic, and clinical feature groups.
5. **Baseline comparison**: Compare full model vs clinical-only baseline vs genetic-only baseline to quantify added value of transcriptomic features.
6. **External validation**: Evaluate on GSE16879 if sufficient phenotype annotation exists for stricturing behavior.

## Expected Outputs
- Predictive model for B2/B3 disease progression with cross-validated performance metrics
- SHAP-based feature importance ranking showing relative contribution of each feature domain
- Comparison of full model vs clinical-only and genetic-only baselines
- External validation performance (if feasible)
- Risk stratification framework: low/medium/high risk groups with distinct progression rates
- Identification of novel molecular markers in top predictive features

## Success Criteria
- Cross-validated AUC >0.75 in RISK cohort (full model)
- Full model outperforms clinical-only baseline (expected AUC ~0.60) by ≥0.10
- At least one molecular feature (genetic or transcriptomic) in top 10 predictive features
- External validation AUC >0.65 (if validation cohort available)
- Clinically interpretable risk groups with ≥2-fold difference in progression rates between high and low risk

## Labels
clinical, genomic, biomarker, high-priority, promising
