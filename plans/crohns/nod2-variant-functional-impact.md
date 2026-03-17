# NOD2 Variant Functional Impact Prediction

## Objective
Build a computational model to predict the functional impact of NOD2 variants on innate immune signaling and Crohn's disease risk, enabling classification of variants of uncertain significance (VUS).

## Background
NOD2 is the strongest single-gene risk factor for Crohn's disease. The three major risk variants (R702W, G908R, L1007fsinsC) are well-characterized, but hundreds of rare NOD2 variants exist with unknown clinical significance. Current GWAS approaches identify risk loci but cannot distinguish functional from benign rare variants. A predictive model combining structural, evolutionary, and clinical features could classify VUS and improve genetic counseling for IBD patients.

## Data Sources
- **ClinVar**: Annotated NOD2 variants with clinical significance ratings (accession: VCV000005321 and related)
- **gnomAD v4**: Population allele frequencies for NOD2 variants
- **IBDGC GWAS summary statistics**: Association data from the International IBD Genetics Consortium (doi:10.1038/ng.3359)
- **AlphaFold DB**: Predicted NOD2 protein structure (UniProt: Q9HC29)
- **UniProt/InterPro**: Domain annotations, evolutionary conservation scores
- **dbNSFP**: Pre-computed functional prediction scores (CADD, REVEL, PolyPhen-2, SIFT)

## Methodology
1. **Data collection**: Aggregate all known NOD2 variants from ClinVar, gnomAD, and IBDGC. Annotate with allele frequency, clinical significance, domain location.
2. **Feature engineering**: For each variant compute — evolutionary conservation (PhyloP, GERP++), structural features (distance to active site, solvent accessibility from AlphaFold), domain impact (LRR, NACHT, CARD domains), existing predictor scores.
3. **Training set construction**: Use ClinVar pathogenic/benign variants as labeled training data. Apply stringent filtering to avoid circular predictions.
4. **Model development**: Train ensemble classifier (gradient boosting + logistic regression) to predict pathogenicity. Use nested cross-validation to avoid overfitting.
5. **Validation**: Hold-out validation against recently reclassified ClinVar variants. Compare performance to individual predictors (CADD, REVEL alone).
6. **VUS classification**: Apply model to all NOD2 VUS and rank by predicted pathogenicity.

## Expected Outputs
- Trained NOD2 variant pathogenicity classifier with performance metrics (AUC, sensitivity, specificity)
- Ranked list of NOD2 VUS with predicted functional impact scores
- Feature importance analysis showing which structural/evolutionary features drive pathogenicity
- Visualization of variants mapped onto NOD2 3D structure colored by predicted impact

## Success Criteria
- Model AUC > 0.90 on held-out ClinVar variants
- At least 50 VUS classified with high confidence (probability > 0.8 or < 0.2)
- Novel structural insights into which NOD2 domains are most sensitive to variation

## Labels
genomic, immunology, biomarker, novel-finding, high-priority
