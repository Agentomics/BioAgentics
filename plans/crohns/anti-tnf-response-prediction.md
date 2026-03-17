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
4. **Feature selection**: Use stability selection (randomized lasso) and recursive feature elimination to identify a compact gene signature (target: 10-30 genes). Cross-reference with known biology (TNF signaling, Th17, innate immunity, fibrosis).
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

## Labels
biomarker, clinical, immunology, high-priority, promising
