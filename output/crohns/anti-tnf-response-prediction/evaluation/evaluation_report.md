# Anti-TNF Response Classifier Evaluation Report

## Pipeline: Within-Fold LOSO-CV (Leakage-Free)

## Best Model
- Model: ensemble
- Aggregate AUC: 0.742
- Balanced Accuracy: 0.671
- Sensitivity: 0.442
- Specificity: 0.900

## All Models
- elastic_net: AUC=0.722, BA=0.668
- random_forest: AUC=0.701, BA=0.642
- xgboost: AUC=0.711, BA=0.611
- ensemble: AUC=0.742, BA=0.671

## Per-Study Performance

### elastic_net
- GSE16879: AUC=0.632 (n=37, R=20, NR=17)
- GSE12251: AUC=0.871 (n=23, R=12, NR=11)
- GSE73661: AUC=0.758 (n=23, R=8, NR=15)

### random_forest
- GSE16879: AUC=0.668 (n=37, R=20, NR=17)
- GSE12251: AUC=0.841 (n=23, R=12, NR=11)
- GSE73661: AUC=0.500 (n=23, R=8, NR=15)

### xgboost
- GSE16879: AUC=0.628 (n=37, R=20, NR=17)
- GSE12251: AUC=0.833 (n=23, R=12, NR=11)
- GSE73661: AUC=0.500 (n=23, R=8, NR=15)

### ensemble
- GSE12251: AUC=0.848 (n=23, R=12, NR=11)
- GSE16879: AUC=0.665 (n=37, R=20, NR=17)
- GSE73661: AUC=0.758 (n=23, R=8, NR=15)

## Benchmark Comparison
- adalimumab_ML_clinical: AUC=0.935
- Porto_5gene_mucosal: AUC=0.880
- TabNet_multimodal: AUC=0.858
- Ours (ensemble): AUC=0.742 <<<
- Ours (elastic_net): AUC=0.722 <<<
- Ours (xgboost): AUC=0.711 <<<
- Ours (random_forest): AUC=0.701 <<<
- EPIC_CD_blood_methylation: AUC=0.250

## Success Criteria
- LOSO AUC > 0.75: FAIL (0.742)