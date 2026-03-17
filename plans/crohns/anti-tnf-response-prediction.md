# Anti-TNF Response Prediction from Pre-treatment Transcriptomics

## Objective
Develop a gene expression-based classifier to predict response to anti-TNF therapy (infliximab, adalimumab) in Crohn's disease patients using pre-treatment mucosal or blood transcriptomic profiles.

## Background
Anti-TNF biologics are first-line advanced therapy for moderate-to-severe Crohn's disease, but 30-40% of patients are primary non-responders. Early identification of non-responders could spare patients ineffective treatment, reduce costs, and accelerate time to effective therapy (e.g., vedolizumab, ustekinumab, or JAK inhibitors). Several small studies have identified candidate gene signatures (TREM1, oncostatin M signaling, OSMR pathway), but no robust, validated classifier exists. Aggregating multiple public datasets and applying modern machine learning could yield a clinically useful predictor.

## Data Sources
- **GSE16879**: Mucosal biopsies from 24 CD patients before/after infliximab, with response annotation (Arijs et al. 2009)
- **GSE12251**: Colonic biopsies from 22 CD patients pre-infliximab with response data (Arijs et al. 2009)
- **GSE73661**: Mucosal biopsies from 73 IBD patients (including CD) pre-anti-TNF treatment (Haberman et al. 2014)
- **GSE100833**: Blood transcriptomics from IBD patients on anti-TNF therapy
- **RISK cohort (GSE57945)**: Pediatric CD treatment-naive ileal biopsies with follow-up data
- **E-MTAB-7604**: Mucosal transcriptomics from the PANTS anti-TNF study

## Methodology
1. **Data acquisition**: Download CEL/raw files from GEO/ArrayExpress. Process with consistent pipeline (RMA for microarray, salmon for RNA-seq). Map to common gene symbols.
2. **Batch correction**: Apply ComBat-seq/ComBat across studies. Verify batch removal via PCA and UMAP visualization while preserving biological signal (responder/non-responder separation).
3. **Differential expression**: Identify genes differentially expressed between responders and non-responders pre-treatment (limma/DESeq2). Perform pathway enrichment (GSEA with Hallmark and Reactome gene sets).
4. **Feature selection**: Use stability selection (randomized lasso) and recursive feature elimination to identify a compact gene signature (target: 10-30 genes). Cross-reference with known biology (TNF signaling, Th17, innate immunity, fibrosis). **Candidate feature sets to evaluate:**
   - *pediCD-TIME vector genes* (eLife Nov 2025, doi:10.7554/eLife.91792): Composite cellular vector (T cell, innate lymphocyte, myeloid, epithelial states) predicting anti-TNF response with r=-0.87 correlation. Validated in E-MTAB-7604 (adult, n=43) — one of our training datasets. Extract gene signature via ARBOL package (GitHub, R and Python). Note convergence with GIMATS module (Martin et al. 2019). Key caveat: anti-TNF treatment pushes pediatric cellular ecosystem toward adult, treatment-refractory state — treatment-naïve vs post-treatment samples may perform differently.
   - *Cell-type deconvolution features*: Use InstaPrism (fast BayesPrism implementation, Bioinformatics 2024, doi:10.1093/bioinformatics/btae440) — top-performing deconvolution method per Briefings in Bioinformatics 2025 benchmark — with pediCD-TIME single-cell reference (254,911 cells, Broad Single Cell Portal) or IBDverse atlas (1.1M cells) to estimate cell-type proportions from bulk data as additional classifier features.
   - *HLA-DQA1*05 genotype*: Validated predictor of anti-drug antibody formation for infliximab and adalimumab (PMC12133927). Include if available in training cohorts.
5. **Classifier training**: Train and compare multiple models — elastic net logistic regression, random forest, XGBoost. Use leave-one-study-out cross-validation to assess generalization across cohorts.
6. **Signature analysis**: Interpret final gene signature biologically. Map to cell types using single-cell reference atlases (e.g., Smillie et al. 2019 UC atlas). Assess overlap with OSMR/oncostatin M pathway.

## Expected Outputs
- Validated multi-gene classifier predicting anti-TNF response in CD
- Gene signature with biological interpretation (pathways, cell types)
- Cross-study performance metrics (AUC, sensitivity, specificity per study)
- Comparison with existing published signatures
- Risk stratification visualization (predicted probability vs actual outcome)

## Success Criteria
- Leave-one-study-out AUC > 0.75 (clinically meaningful improvement over random)
- Gene signature enriched in biologically interpretable pathways
- Consistent performance across mucosal biopsy datasets (AUC > 0.70 in each)
- Signature size <= 30 genes (practical for clinical translation)
- **Benchmark**: Demonstrate complementary value beyond adalimumab ML model (PLOS One 2025, doi:10.1371/journal.pone.0331447) which achieved AUC=0.935 at 48 weeks using clinical/lab features only (calprotectin, CRP, hemoglobin). Our transcriptomic classifier should either exceed this in a comparable setting or show additive value when combined with clinical features.

## Labels
biomarker, clinical, immunology, high-priority, promising
