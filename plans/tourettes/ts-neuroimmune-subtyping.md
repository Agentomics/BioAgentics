# TS Neuroimmune Subtyping

## Objective
Characterize the immunogenomic architecture of Tourette syndrome to identify an immune-mediated disease subtype, pinpoint specific immune pathways contributing to tic pathology, and nominate immunomodulatory treatment candidates.

## Background
Multiple converging lines of evidence implicate immune dysregulation in TS:
1. **PANDAS/PANS autoimmune subgroup** — post-streptococcal molecular mimicry with autoantibodies targeting basal ganglia neurons
2. **Wang et al. 2025** — first snRNA-seq of TS caudate showed microglial activation with immune response pathway upregulation (journal #447)
3. **Elevated pro-inflammatory cytokines** — TNF-α, IL-12 documented in TS patients
4. **Th17-mediated BBB disruption** — leading to microglial activation in basal ganglia
5. **Frontiers in Immunology (Mar 2026)** — comprehensive review synthesizing three core immune pathophysiology pathways and identifying IL-1RN genetic variations as a susceptibility factor (journal #490)
6. **Sun & Bai three-pathway framework (Frontiers Immunol Mar 2026; doi:10.3389/fimmu.2026.1708940)** — organizes TS immune mechanisms into three testable pathways: (1) post-infectious autoimmunity/PANDAS, (2) Th17/microglial neuroinflammation, (3) gut-brain axis dysregulation. Identifies biomarkers: CSF oligoclonal bands (20-38% of TS patients), MCP-1/CCL2, PET TSPO (microglial activation imaging), IL-1RN variants. FMT and B-cell therapies as emerging approaches [task #933]

Despite this evidence, no study has systematically characterized the **genetic architecture** of immune-mediated TS or computationally determined which immune pathways are causally linked to tic pathology through genomic data. This initiative fills that gap using GWAS-based immunogenomic methods.

## Data Sources
- **TSAICG GWAS summary statistics** — PGC portal (being acquired via task #235)
- **Autoimmune disease GWAS summary stats** — GWAS Catalog: rheumatoid arthritis, SLE, multiple sclerosis, type 1 diabetes, IBD, celiac disease, psoriasis
- **Wang et al. 2025 microglia DEGs** — snRNA-seq (being acquired via task #237)
- **ImmGen / DICE** — immune cell expression reference atlases
- **Immune genomic loci** — IL-1RN, HLA region, TNF, IL12A/B, IL17A, TGFB1
- **EMTICS cohort** — European Multicentre Tics in Children Study (check for public data release)

## Methodology

### Phase 1 — Genetic Correlation Screening
LDSC cross-trait genetic correlations between TS GWAS and 15+ autoimmune/inflammatory disease GWAS. Identify which immune conditions share significant genetic risk with TS. The TSAICG paper tested 112 genetic correlations but focused on psychiatric disorders — autoimmune correlations were not systematically tested.

### Phase 1b — Lymphocytic GWAS Enrichment Replication [NEW]
Prior TS GWAS (Translational Psychiatry 2020) found significant lymphocytic gene set enrichment via set-based association — the strongest existing genomic evidence for immune involvement in TS. **Replicate this finding using the larger 2024 TSAICG GWAS (9,619 cases)** as foundational validation before proceeding with cell-type-specific analyses. If lymphocytic enrichment replicates, it confirms immune pathways contribute at the genetic level, not just as secondary inflammation [task #331].

### Phase 2 — Immune Cell-Type Enrichment
Apply MAGMA cell-type analysis and seismic (Nat Comms 2025, journal #485) to TS GWAS using immune cell expression references (ImmGen, DICE). **Prioritize Th17 and NK cell gene sets** based on meta-analytic evidence of low CD4+ T cells, high NK cells, and high IL-17 in TS patients. Test enrichment in specific immune cell populations: microglia, **Th17 cells [high priority]**, B cells, **NK cells [high priority]**, monocytes/macrophages. Compare with brain cell-type enrichment (already known: 5 brain cell types from TSAICG) [task #328].

### Phase 3 — Mendelian Randomization
Two-sample MR for causal effects of circulating immune biomarkers on TS risk:
- Exposures: CRP, IL-6, TNF-α, lymphocyte/monocyte/neutrophil counts, IgG levels, **NK cell count, IL-17, CD4+ count** [NEW — meta-analytic evidence, task #328]
- Outcome: TS GWAS summary stats
- Methods: IVW, weighted median, MR-Egger, CAUSE
- Bidirectional MR to test reverse causation (TS → immune dysregulation)
- **CCL2/CCR2 axis**: CSF MCP-1/CCL2 elevated in TS children — links peripheral immune activation to microglial activation. Include CCL2 as MR exposure and CCR2 as druggable target candidate [task #328]

### Phase 3b — Gut-Brain Axis Pathway Analysis [NEW task #933]
The Sun & Bai framework identifies gut-brain axis dysregulation as the third immune pathway in TS, complementing PANDAS autoimmunity (Phase 1) and Th17/microglial neuroinflammation (Phase 2). This pathway involves gut microbiome-mediated immune priming, vagal nerve signaling, and microbial metabolite (SCFA) effects on BBB integrity and microglial activation.
- Test LDSC genetic correlation between TS GWAS and IBD/IBS GWAS (complement Phase 1 autoimmune screen)
- Test enrichment of gut-associated immune gene sets (Peyer's patch, intestinal ILC3, mucosal Treg) in TS GWAS via MAGMA
- Evaluate MR of gut microbiome composition (using available GWAS for gut microbial taxa) on TS risk
- Cross-reference with PANS microbiome literature for overlapping dysbiosis signatures

### Phase 4 — Microglia Transcriptomic Integration
Cross-reference TS GWAS hits with Wang et al. microglia DEGs and published activated microglia gene signatures. Identify GWAS-supported immune genes expressed in disease-relevant microglia states. Test enrichment of TS risk genes in microglia activation modules.

### Phase 5 — Immune Subtype Characterization & Therapeutic Nomination
Synthesize phases 1-4 to define an immune-genomic TS subtype profile. Cross-reference with ts-drug-repurposing-network to identify immunomodulatory candidates:
- Anti-TNF (adalimumab) — FDA-approved for RA, IBD
- Anti-IL-12/23 (ustekinumab) — FDA-approved for psoriasis, IBD
- Anti-CD20 (rituximab) — FDA-approved for RA, MS
- JAK inhibitors (tofacitinib) — FDA-approved for RA
- **FMT (fecal microbiota transplant)** — emerging approach targeting gut-brain axis; if Phase 3b gut-immune pathway enrichment confirmed, FMT becomes a mechanistically grounded therapeutic option [task #933]
- **PET TSPO imaging** as clinical biomarker endpoint — 20-38% of TS patients show CSF oligoclonal bands; TSPO PET provides non-invasive microglial activation measurement for patient stratification into immune-mediated subtype [task #933]

## Expected Outputs
- Genetic correlation matrix: TS vs. 15+ autoimmune diseases
- Immune cell-type enrichment results (MAGMA + seismic)
- MR estimates for immune biomarker → TS causal effects
- Immune gene overlap with TS GWAS and microglia DEGs
- Immune-genomic TS subtype profile
- Ranked immunomodulatory treatment candidates for immune-mediated TS subset

## Success Criteria
- ≥3 autoimmune diseases show significant (FDR < 0.05) genetic correlation with TS
- ≥1 immune cell type shows significant GWAS enrichment beyond brain cell types
- ≥1 immune biomarker shows significant causal effect via MR (p < 0.05 across ≥2 MR methods)
- Results converge on a coherent immune pathway connecting genetics → microglia → tic pathology

## Labels
multi-omics, comorbidity, biomarker, novel-finding, high-priority
