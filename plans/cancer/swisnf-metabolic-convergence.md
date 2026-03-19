# SWI/SNF Complex Metabolic Convergence Initiative

**Initiative:** swisnf-metabolic-convergence
**Division:** cancer
**Type:** Catalyst — Wildcard Ideation
**Source Atlases:** arid1a-pancancer-sl-atlas, smarca4-pancancer-sl-atlas, pancancer-ferroptosis-atlas (in analysis)
**Labels:** catalyst, novel-finding, high-priority

---

## Objective

**Hypothesis:** ARID1A-mutant and SMARCA4-mutant tumors converge on druggable metabolic vulnerabilities — specifically cholesterol biosynthesis (HMGCR) and mitochondrial oxidative phosphorylation (ADCK5, OXPHOS complex genes) — that are mechanistically linked to SWI/SNF complex disruption rather than being tissue-specific artifacts. If confirmed, FDA-approved metabolic drugs (statins, OXPHOS inhibitors) could be immediately repurposed for >200,000 SWI/SNF-mutant cancer patients/year.

This is falsifiable: if HMGCR dependency in ARID1A-mutant breast/uterus/lung does NOT extend to SMARCA4-mutant lines, and if mitochondrial vulnerabilities in SMARCA4-ovarian do NOT extend to ARID1A-mutant lines, then the metabolic signal is tissue-specific — not SWI/SNF-driven.

## Rationale

Two independently conducted atlases — one for ARID1A and one for SMARCA4 — both discovered metabolic vulnerabilities as their most novel findings. But each atlas treated these as secondary observations because the primary question was about paralog dependencies (ARID1B for ARID1A, SMARCA2 for SMARCA4). The metabolic angle was never pursued.

**ARID1A atlas findings:**
- HMGCR (HMG-CoA reductase, statin target) is SL in 3 cancer types: Breast (d=-1.68), Uterus (d=-0.78), Lung (d=-0.75)
- Literature validation: Simvastatin selectively kills ARID1A-mutant cells via pyroptosis (Cancer Cell 2023, Zhang lab)
- ADCK5 (mitochondrial kinase) is SL in 4 cancer types: Biliary (d=-1.96), Skin (d=-1.35), Pancreas (d=-0.92), Uterus (d=-0.68)
- Both HMGCR and ADCK5 are metabolic genes

**SMARCA4 atlas findings:**
- 7 of 10 FDR-significant novel SL genes in ovarian cancer are mitochondrial: MTERF4, MICOS13, DMAC2, MRPS35, COX6C, WARS2, HIGD2A
- TIMM22 (mitochondrial translocase) is the sole lung-specific novel hit
- This mitochondrial cluster was the most surprising finding in the SMARCA4 atlas

**The bridge:** ADCK5 (mitochondrial kinase) appears in the ARID1A atlas, and the mitochondrial vulnerability dominates the SMARCA4 atlas. If SWI/SNF loss systematically rewires metabolic gene expression — perhaps by altering chromatin accessibility at metabolic gene promoters — then both HMGCR and OXPHOS dependencies are consequences of the same upstream cause.

**Why this is unconventional:** Current SWI/SNF therapeutics focus EXCLUSIVELY on:
1. Paralog dependencies (ARID1B degraders, SMARCA2 inhibitors)
2. Epigenetic antagonism (EZH2 inhibitors)

No clinical program targets the metabolic axis. If metabolic convergence is real, it opens an entirely new therapeutic strategy using cheap, well-characterized drugs.

## Data Sources

- **DepMap 25Q3 CRISPR:** Gene effect scores for all metabolic genes (KEGG metabolic pathways: ~1,800 genes covering glycolysis, TCA cycle, OXPHOS, cholesterol biosynthesis, fatty acid metabolism, amino acid metabolism)
- **Existing atlas data:** ARID1A and SMARCA4 cell line classifications already computed
- **TCGA expression:** Metabolic gene expression changes in SWI/SNF-mutant vs WT tumors
- **Ferroptosis atlas data** (in analysis): Lipid peroxidation pathway dependencies — could overlap with HMGCR (cholesterol feeds into lipid metabolism)

## Methodology

### Phase 1: Metabolic Gene Dependency Screen
Filter DepMap CRISPR data to KEGG metabolic pathway genes (~1,800). Compare dependency scores between:
- ARID1A-mutant vs WT (across all 11 qualifying cancer types from the atlas)
- SMARCA4-mutant vs WT (across 4 qualifying cancer types)
- Combined SWI/SNF-mutant (ARID1A OR SMARCA4) vs WT

For each metabolic gene, compute Cohen's d, p-value, and FDR correction. Identify metabolic gene clusters with consistent SWI/SNF-selective dependency.

### Phase 2: Cross-Atlas Validation
Test whether:
- HMGCR dependency (found in ARID1A atlas) also exists in SMARCA4-mutant lines
- Mitochondrial gene dependencies (found in SMARCA4 atlas) also exist in ARID1A-mutant lines
- Any metabolic genes are SL in BOTH ARID1A-mutant AND SMARCA4-mutant cells

### Phase 3: Pathway Enrichment
Run GSEA or hypergeometric enrichment on the SWI/SNF-selective metabolic dependencies. Are specific pathways over-represented? (e.g., mevalonate pathway, electron transport chain, one-carbon metabolism)

### Phase 4: Expression Basis
Using TCGA, test whether the metabolic dependency genes are transcriptionally deregulated in SWI/SNF-mutant tumors. If SWI/SNF loss alters chromatin accessibility at metabolic gene promoters, we'd expect expression changes to precede dependency.

### Phase 5: Drug Repurposing Analysis
For confirmed metabolic dependencies, identify FDA-approved drugs targeting those pathways:
- Statins (HMGCR) — immediately available
- Metformin (Complex I inhibitor) — if OXPHOS dependency confirmed
- IACS-010759 (OXPHOS inhibitor) — in clinical trials for other indications
- Fenofibrate (lipid metabolism) — if fatty acid dependencies emerge

## Success Criteria

1. **Convergence:** ≥3 metabolic genes show SL in BOTH ARID1A-mutant AND SMARCA4-mutant cells (nominal p<0.05, |d|>0.3)
2. **Pathway coherence:** Convergent genes belong to ≤2 metabolic pathways (not scattered across all metabolism)
3. **Expression basis:** ≥50% of convergent dependency genes show altered expression in TCGA SWI/SNF-mutant tumors (FDR<0.05)
4. **Drug availability:** At least 1 FDA-approved drug targets the convergent pathway

## Risk Assessment

- **Primary risk:** HMGCR (cholesterol) and OXPHOS (mitochondria) may be unrelated metabolic axes with no shared mechanism. The ARID1A and SMARCA4 metabolic findings could be tissue-specific rather than SWI/SNF-driven.
- **Secondary risk:** SWI/SNF complex has many subunits. Testing only ARID1A and SMARCA4 may miss the full picture — but these are the only two with published atlases.
- **What we learn if it fails:** If no metabolic convergence exists, this definitively rules out metabolic repurposing as a general SWI/SNF strategy and redirects efforts back to paralog/epigenetic approaches. Still valuable as a negative result.
