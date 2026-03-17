# PLAN: NSCLC Therapeutic Target Discovery via DepMap Dependency Mapping

## Objective

Identify actionable therapeutic targets and synthetic lethal pairs in non-small cell lung cancer (NSCLC) by mapping DepMap cancer dependencies to TCGA patient molecular profiles, stratified by oncogenic driver and immune context.

## Background

NSCLC remains the leading cause of cancer death worldwide. Despite 7 FDA approvals in 2025 alone (c-MET, TROP-2, EGFR exon20, ROS1, HER2), most patients lack targeted options. KRAS-mutant NSCLC — the largest molecular subgroup (~30%) — subdivides into clinically distinct subtypes: KP (KRAS+TP53, ~40-50%, immunotherapy-responsive), KL (KRAS+STK11, ~20-30%, immunotherapy-resistant), and KOnly. Each subtype has different vulnerabilities, but systematic computational mapping of these dependencies to patient-level data is incomplete.

The DepMap project (Broad Institute) provides genome-scale CRISPR knockout dependency scores across 1000+ cancer cell lines, plus PRISM drug response data for ~1500 compounds. Meyers et al. (Nature Cancer, Aug 2024) demonstrated that expression-only elastic-net models trained on DepMap can be transposed to predict dependencies in 9,596 TCGA patients — the "translational cancer dependency map" (TCGADEPMAP). We will reproduce and extend this approach with DepMap 25Q3 data, focusing specifically on NSCLC with molecular subtype stratification.

Recent advances in synthetic lethality (SL) prediction — SLAYER, CILANTRO-SL (conformal prediction), PARIS, and experimentally validated SL pairs from combinatorial CRISPR screens — provide both computational methods and ground-truth benchmarks. KEAP1-targeted and STK11-targeted SL pairs are particularly relevant for NSCLC KL patients who lack immunotherapy options.

**Gap we fill:** Existing TCGADEPMAP is pan-cancer and does not incorporate: (1) NSCLC molecular subtype stratification, (2) SL prediction integration, (3) PRISM drug response mapping to predicted dependencies, or (4) single-cell validation via the LUCA atlas (1.28M cells).

## Data Sources

| Dataset | Source | Access | Size/Scope |
|---------|--------|--------|------------|
| DepMap 25Q3 CRISPR scores | depmap.org | Open, no auth | 1000+ cell lines, genome-scale |
| DepMap 25Q3 gene expression | depmap.org | Open | Matched to CRISPR lines |
| DepMap 25Q3 mutations/CNV | depmap.org | Open | Matched to CRISPR lines |
| PRISM drug response | depmap.org | Open | ~1500 compounds, ~900 lines |
| TCGA LUAD/LUSC (GDC) | portal.gdc.cancer.gov | Open-access tier | RNA-seq, mutations, CN, clinical |
| SL benchmark: isogenic CRISPR | Mendeley Data | CC-BY-4.0 | KEAP1/STK11 SL pairs (doi:10.17632/k6wm46g4tw.1) |
| SL benchmark: combinatorial CRISPR | Genome Biology 2025 | Open | 117 validated SL pairs, lung/pancreatic/melanoma |
| LUCA scRNA-seq atlas | Zenodo | CC-BY-4.0 | 1.28M cells, 44 cell types, 318 patients |
| MLOmics TCGA benchmark | Hugging Face/Figshare | Open | Pre-processed LUAD/LUSC multi-omics |

## Methodology

### Phase 1: Data Acquisition & Preprocessing
1. Download DepMap 25Q3 datasets (CRISPR scores, expression, mutations, CNV, PRISM drug response)
2. Download TCGA LUAD and LUSC data from GDC (RNA-seq counts, somatic mutations, copy number, clinical metadata)
3. Download SL benchmark datasets
4. Harmonize gene identifiers across datasets (Ensembl → HUGO symbols)
5. Filter DepMap to NSCLC cell lines; annotate with KRAS/TP53/STK11/KEAP1 mutation status
6. Classify TCGA NSCLC patients into molecular subtypes (KP, KL, KOnly, KRAS-WT) using mutation profiles

### Phase 2: Dependency Prediction Models
7. Reproduce TCGADEPMAP approach: train elastic-net models predicting gene dependency from expression features using DepMap NSCLC cell lines
8. Apply models to predict patient-level dependency scores for all TCGA LUAD/LUSC patients
9. Identify subtype-specific dependencies: genes selectively essential in KP vs KL vs KOnly vs KRAS-WT
10. Validate predictions against known NSCLC dependencies (KRAS, EGFR, ALK, ROS1, MET) as positive controls

### Phase 3: Synthetic Lethality Integration
11. Run SL prediction on NSCLC driver genes (KRAS, STK11, KEAP1, TP53) using available methods (SLAYER framework)
12. Cross-reference predicted SL pairs with DepMap dependency scores for validation
13. Benchmark against experimentally validated SL pairs from isogenic and combinatorial CRISPR screens
14. Prioritize SL pairs that are: (a) predicted by multiple methods, (b) supported by DepMap dependency data, (c) druggable

### Phase 4: Drug Response Mapping
15. Map PRISM drug response data to predicted patient dependencies
16. For each molecular subtype, identify compounds with selective sensitivity profiles
17. Cross-reference with FDA-approved drugs and clinical trial compounds for translational relevance
18. Identify combination therapy candidates from SL pairs where both partners have available compounds

### Phase 5: Validation & Visualization
19. Validate key findings against LUCA scRNA-seq atlas — check if predicted target genes show expected expression patterns in tumor vs. TME cells
20. Perform survival analysis: do patients with high predicted dependency on identified targets have different outcomes?
21. Generate publication-quality visualizations: dependency heatmaps, subtype stratification plots, SL network graphs, drug-target mapping Sankey diagrams

## Expected Outputs

1. **NSCLC Dependency Atlas:** Patient-level predicted dependency scores for all TCGA LUAD/LUSC patients, stratified by molecular subtype
2. **Subtype-specific Target List:** Ranked therapeutic targets for each NSCLC molecular subtype (KP, KL, KOnly) with confidence scores
3. **SL Pair Predictions:** Prioritized synthetic lethal pairs for NSCLC driver genes, benchmarked against experimental data
4. **Drug Mapping:** Compounds matched to predicted dependencies and SL pairs, with clinical status annotations
5. **Reproducible Pipeline:** End-to-end analysis pipeline in `src/bioagentics/` with documented inputs/outputs
6. **Analysis Report:** Comprehensive findings document with visualizations

## Success Criteria

1. **Positive control recovery:** Pipeline correctly identifies known NSCLC dependencies (KRAS in KRAS-mutant, EGFR in EGFR-mutant lines) with AUC > 0.75
2. **Subtype differentiation:** Statistically significant differences in predicted dependency profiles between KP, KL, and KOnly subtypes (FDR < 0.05)
3. **SL benchmark performance:** Predicted SL pairs recover ≥30% of experimentally validated pairs from benchmark sets
4. **Novel candidates:** At least 3 target genes or SL pairs not previously reported as NSCLC-specific in literature
5. **Survival association:** At least one identified target shows significant survival association in TCGA clinical data (log-rank p < 0.05)
6. **Drug actionability:** ≥50% of top-ranked targets or SL partners have existing compounds in PRISM or clinical development

## Labels

`genomic, drug-screening, drug-candidate, resistance, high-priority, novel-finding`

## Key References

- Meyers et al. "Building a translational cancer dependency map for The Cancer Genome Atlas" (Nature Cancer, Aug 2024)
- Sannigrahi et al. 2024, Mol Oncol: PAK2 in HNSCC via DepMap
- CILANTRO-SL (bioRxiv Feb 2026): Foundation model SL prediction with conformal prediction
- SLAYER: Synthetic lethality analysis framework
- Genome Biology 2025: 117 experimentally validated SL pairs
- LUCA Atlas (Salcher et al., Cancer Cell 2022): 1.28M cell NSCLC scRNA-seq
- Flexynesis (Nature Communications 2025): Multi-omics integration toolkit
