# KEAP1/NRF2 Pan-Cancer Dependency Atlas

## Objective

Map synthetic lethal dependencies specific to KEAP1-loss and NRF2-GOF cancers across all DepMap tumor types to identify novel therapeutic targets for oxidative-stress-reprogrammed tumors resistant to standard chemotherapy and immunotherapy.

## Background

KEAP1 is the primary negative regulator of NRF2 (NFE2L2), the master transcription factor for oxidative stress response. KEAP1 loss-of-function or NRF2 gain-of-function mutations constitutively activate the NRF2 pathway, creating tumors with enhanced antioxidant defenses, altered metabolism, and treatment resistance.

**Mutation frequencies:** KEAP1 is mutated in ~23% of lung adenocarcinoma, ~30% of lung squamous cell carcinoma, ~12% of head and neck squamous, ~8% of bladder, and ~5% of esophageal cancers. NFE2L2 GOF mutations occur in ~15% of lung squamous and ~5% of other types. Together, KEAP1/NRF2 pathway alterations affect >50,000 new US cancer patients per year.

**Unmet clinical need:** KEAP1-mutant NSCLC patients have significantly worse outcomes on immune checkpoint inhibitors (ORR ~28% vs ~40-50% in KEAP1-WT, per KRYSTAL-7/CodeBreaK 200 subgroup data) and on chemotherapy (via enhanced drug efflux and glutathione-mediated detoxification). No KEAP1-directed therapies exist. STK11 co-mutations further compound the problem.

**Gap we fill:** Individual NRF2 target gene studies and ferroptosis resistance work exist, but no comprehensive pan-cancer atlas of KEAP1/NRF2-specific synthetic lethal dependencies has been constructed from DepMap data. Our portfolio already touches KEAP1 peripherally (nsclc-depmap-targets identified KEAP1 co-mutation effects; pancancer-ferroptosis-atlas covers NRF2/KEAP1 stratified ferroptosis defense), but neither provides a dedicated KEAP1-centric SL analysis.

**Key hypothesis:** KEAP1-loss creates distinct metabolic and redox dependencies (beyond ferroptosis resistance) that differ from other oncogenic drivers and are targetable with existing or emerging compounds.

## Data Sources

- **DepMap 24Q4:** CRISPR dependency scores (Chronos), gene expression (Expression_24Q4), mutation calls (OmicsSomaticMutations), copy number (OmicsCNGene). ~50-60 KEAP1-mutant cell lines, ~30 NFE2L2-GOF lines across multiple lineages.
- **CCLE/DepMap drug sensitivity:** PRISM Repurposing and CTD² datasets for drug response correlation.
- **TCGA:** Mutation frequencies, co-mutation patterns, clinical outcomes by KEAP1/NRF2 status. Pan-cancer cohort for mutation co-occurrence analysis (STK11, KRAS, TP53, NFE2L2).
- **ClinicalTrials.gov:** Active trials enrolling KEAP1/NRF2-stratified patients; NRF2 pathway-targeting agents.

## Methodology

### Phase 1: KEAP1/NRF2 Cohort Definition and Pan-Cancer SL Screen
1. Define KEAP1-loss cohort: damaging mutations (nonsense, frameshift, splice-site) + deep deletions from DepMap.
2. Define NRF2-GOF cohort: known activating mutations in NFE2L2 (DLG/ETGE motif hotspots).
3. Define combined cohort (KEAP1-loss OR NRF2-GOF) as the primary analysis group.
4. For each gene in DepMap CRISPR (~18,000 genes), compute Cohen's d effect size comparing KEAP1/NRF2-altered vs wild-type, both pan-cancer and per lineage. Apply Benjamini-Hochberg FDR correction (q < 0.1).
5. Filter hits: |d| > 0.5, minimum 3 mutant lines per lineage for tissue-specific calls.
6. Output: ranked SL gene list with effect sizes, p-values, lineage specificity.

### Phase 2: Pathway and Functional Annotation
1. Map SL hits to pathways (KEGG, Reactome, MSigDB Hallmarks).
2. Cluster by functional categories: metabolism, DNA damage repair, proteostasis, signaling, epigenetic.
3. Cross-reference with NRF2 transcriptional targets (ARE-containing genes) to distinguish direct pathway members from genuine SL partners.
4. Annotate druggability using DGIdb and manually curated compound lists.
5. Evaluate KEAP1-specific vs NRF2-GOF-specific dependencies (are they the same?).

### Phase 3: Co-Mutation Stratification and Clinical Mapping
1. Stratify KEAP1-mutant lines by STK11 co-mutation (KL subtype), KRAS co-mutation, and TP53 co-mutation.
2. Test whether SL dependencies differ by co-mutation context (e.g., KEAP1+STK11 vs KEAP1-only).
3. Cross-reference top SL gene targets with compounds in clinical trials.
4. Map to clinical data: compare TCGA KEAP1-mutant patient outcomes stratified by expression of top SL targets.
5. Produce clinical prioritization table: target → compound → trial → patient population.

### Phase 4: Drug Sensitivity Correlation
1. Correlate KEAP1/NRF2 status with PRISM drug sensitivity across ~4,000 compounds.
2. Identify compounds with selective activity in KEAP1-mutant cells.
3. Cross-reference drug targets with SL gene hits from Phase 1 for orthogonal validation.
4. Flag FDA-approved drugs with repurposing potential.

## Expected Outputs

1. **Pan-cancer KEAP1/NRF2 SL gene atlas** — ranked table of genes with selective essentiality in KEAP1/NRF2-altered cancers, with effect sizes and tissue specificity.
2. **Pathway enrichment analysis** — functional categories of KEAP1-specific vulnerabilities.
3. **Co-mutation interaction map** — how STK11, KRAS, TP53 co-mutations modify KEAP1 dependencies.
4. **Drug repurposing candidates** — existing compounds matching SL targets.
5. **Clinical prioritization table** — target-compound-trial mapping for actionable findings.
6. **Visualizations** — volcano plots, heatmaps, co-mutation interaction diagrams.

## Success Criteria

1. Identify ≥10 high-confidence SL genes (|d| > 0.8, FDR < 0.05) specific to KEAP1/NRF2-altered cancers.
2. At least 3 SL targets with existing drugs or compounds in clinical development.
3. Demonstrate that KEAP1-loss dependencies differ meaningfully from generic tumor dependencies (i.e., hits are not simply common essential genes).
4. Show co-mutation context matters (KEAP1+STK11 vs KEAP1-only produce distinct dependency profiles).
5. Cross-validate ≥3 top hits with PRISM drug sensitivity data.

## Labels

genomic, novel-finding, drug-candidate, high-priority
