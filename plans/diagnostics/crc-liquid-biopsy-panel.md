# Multi-Analyte Liquid Biopsy Panel for Early Colorectal Cancer Detection

## Objective
Develop a cost-optimized multi-analyte classifier combining cfDNA methylation patterns and protein biomarkers for early-stage colorectal cancer detection from blood samples.

## Background
Colorectal cancer (CRC) is the 3rd most common cancer globally (~1.9M cases/year) and 2nd leading cause of cancer death. Early-stage CRC (stages I-II) has >90% 5-year survival, but most cases are diagnosed at later stages. Colonoscopy is the gold standard for screening but has low compliance (~60% in the US, far lower globally) due to cost, access, invasiveness, and bowel preparation burden.

Blood-based screening could dramatically improve compliance. Current options:
- **Guardant Shield / GRAIL Galleri**: Multi-cancer early detection via cfDNA methylation — expensive ($949+), moderate sensitivity for early CRC
- **Cologuard (Exact Sciences)**: Stool DNA test — better compliance than colonoscopy but still requires stool collection
- **CEA**: Classic protein marker — poor sensitivity for early-stage CRC (<25% for stage I)

Key gap: No affordable blood test achieves high sensitivity for stage I-II CRC. Combining cfDNA methylation with protein biomarkers in a cost-optimized panel could fill this gap.

## Data Sources
- **TCGA-COAD and TCGA-READ** — Methylation (450K array), RNA-seq, clinical data for ~600 colon and rectal adenocarcinoma samples. Provides ground truth methylation signatures.
- **GEO: GSE149282** — cfDNA methylation profiles from CRC patients and healthy controls using targeted bisulfite sequencing.
- **GEO: GSE164191** — Circulating protein biomarker panels in CRC screening cohorts.
- **GEO: GSE48684** — Genome-wide DNA methylation in CRC tissue and adjacent normal mucosa (Illumina 450K).
- **ClinVar / COSMIC** — Known CRC driver mutations for cfDNA mutation panel design.
- **UK Biobank proteomics** — Large-scale plasma proteomic data with cancer outcomes (if accessible).

## Methodology
1. **Methylation signature discovery**: Analyze TCGA-COAD/READ methylation data to identify CpG sites differentially methylated between tumor and normal tissue. Filter for sites with large effect sizes (delta-beta > 0.3) and biological relevance (promoter CpG islands of known CRC genes).
2. **cfDNA validation**: Cross-reference tissue-derived signatures against cfDNA methylation data (GSE149282). Identify markers detectable in blood with sufficient signal-to-noise.
3. **Protein biomarker integration**: Analyze circulating protein data (GSE164191) for markers complementary to methylation — i.e., proteins that catch cases methylation misses. Candidate proteins: CEA, CA19-9, SEPT9, TIMP-1, M2-PK.
4. **Panel optimization**: Use LASSO/elastic net regularization to select minimal marker set maximizing sensitivity for stage I-II CRC. Cost-weight features (methylation assays vs. immunoassays) to optimize clinical impact per dollar.
5. **Classifier development**: Train ensemble classifier (logistic regression + gradient boosting) on combined methylation + protein features. Nested cross-validation for unbiased performance estimation.
6. **Stage-stratified evaluation**: Report sensitivity/specificity separately for stages I, II, III, IV. Primary endpoint: sensitivity for stages I-II at 95% specificity.
7. **Cost modeling**: Estimate per-test cost for the optimized panel. Compare cost-effectiveness against colonoscopy and existing blood tests using standard QALY frameworks.

## Expected Outputs
- Ranked list of cfDNA methylation markers most informative for early CRC detection
- Optimized multi-analyte panel (target: 5-15 markers) with cost estimate
- Stage-stratified classifier performance (sensitivity, specificity, AUC)
- Cost-effectiveness comparison against existing screening modalities
- Feature importance analysis showing contribution of methylation vs. protein markers

## Success Criteria
- Stage I-II CRC sensitivity >= 70% at specificity >= 95%
- Combined panel outperforms any single-modality panel (methylation-only or protein-only) by >= 5% AUC
- Estimated per-test cost < $200 for the optimized panel
- Validation performance within 5% of discovery performance in cross-validation

## Labels
biomarker, screening, cost-reduction, novel-finding, multi-omics
