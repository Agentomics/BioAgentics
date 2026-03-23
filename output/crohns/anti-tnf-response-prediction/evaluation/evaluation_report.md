# Anti-TNF Response Classifier Evaluation Report

## Pipeline: Within-Fold LOSO-CV (Leakage-Free)

## Best Model
- Model: xgboost
- Aggregate AUC: 0.711
- Balanced Accuracy: 0.620
- Sensitivity: 0.791
- Specificity: 0.450

## All Models
- elastic_net: AUC=0.600, BA=0.492
- random_forest: AUC=0.701, BA=0.657
- xgboost: AUC=0.711, BA=0.620

## Per-Study Performance

### elastic_net
- GSE16879: AUC=0.668 (n=37, R=20, NR=17)
- GSE12251: AUC=0.826 (n=23, R=12, NR=11)
- GSE73661: AUC=0.633 (n=23, R=8, NR=15)

### random_forest
- GSE16879: AUC=0.668 (n=37, R=20, NR=17)
- GSE12251: AUC=0.841 (n=23, R=12, NR=11)
- GSE73661: AUC=0.500 (n=23, R=8, NR=15)

### xgboost
- GSE16879: AUC=0.628 (n=37, R=20, NR=17)
- GSE12251: AUC=0.833 (n=23, R=12, NR=11)
- GSE73661: AUC=0.500 (n=23, R=8, NR=15)

## Benchmark Comparison
- adalimumab_ML_clinical: AUC=0.935
- Porto_5gene_mucosal: AUC=0.880
- TabNet_multimodal: AUC=0.858
- Ours (xgboost): AUC=0.711 <<<
- Ours (random_forest): AUC=0.701 <<<
- Ours (elastic_net): AUC=0.600 <<<
- EPIC_CD_blood_methylation: AUC=0.250

## Success Criteria
- LOSO AUC > 0.75: FAIL (0.711)