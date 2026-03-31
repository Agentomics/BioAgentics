# SWI/SNF Complex Metabolic Convergence: ARID1A and SMARCA4 Mutant Tumors Share Druggable OXPHOS Vulnerability

**Project:** swisnf-metabolic-convergence
**Division:** Cancer
**Date:** March 31, 2026
**Data Sources:** DepMap 25Q3 (CRISPR, mutations, copy number), KEGG metabolic pathways, TCGA (LUAD, LUSC), PRISM 24Q2
**Pipeline:** `src/cancer/swisnf_metabolic_convergence/01–06_*.py`
**Validation Status:** Analysis complete; independent analyst methodology review performed. Awaiting formal validation.

---

## Executive Summary

This study tested whether ARID1A-mutant and SMARCA4-mutant cancers — which harbor loss-of-function mutations in different subunits of the SWI/SNF chromatin remodeling complex — converge on shared druggable metabolic vulnerabilities. Using DepMap 25Q3 CRISPR knockout data across 2,119 cancer cell lines and 1,590 KEGG metabolic pathway genes, we identified 104 metabolic genes with synthetic lethal (SL) signal in both ARID1A-mutant and SMARCA4-mutant contexts.

**The central finding is that mitochondrial oxidative phosphorylation (OXPHOS) is the dominant convergent vulnerability.** Of the 104 convergent genes, 34 encode OXPHOS components spanning all five electron transport chain (ETC) complexes (4.33-fold enrichment over background, FDR = 2.56 x 10^-13). This OXPHOS convergence was independently confirmed by pathway enrichment in lipoic acid metabolism (5.63-fold, FDR = 0.0045), a mitochondrial cofactor pathway. PRISM drug sensitivity data validated this finding pharmacologically: SMARCA4-mutant cell lines are significantly more sensitive to IACS-010759, a selective Complex I inhibitor in Phase I clinical trials (Cohen's d = -0.36, p = 0.007).

Two initially hypothesized convergent axes were **falsified**: HMGCR/statin dependency (ARID1A-specific, no SMARCA4 signal) and glutathione/eprenetapopt vulnerability (ARID1A-specific, 0/5 GSH genes convergent). TCGA expression analysis in lung cancer showed minimal transcriptional changes (4/104 genes at FDR < 0.05), suggesting the OXPHOS dependency operates at a functional rather than transcriptional level.

These results propose a new therapeutic strategy for SWI/SNF-mutant cancers — OXPHOS inhibition — distinct from the paralog dependency (SMARCA2 inhibitors, ARID1B degraders) and epigenetic antagonism (EZH2 inhibitors) approaches currently in clinical development. Metformin, an FDA-approved Complex I inhibitor, targets 18 of the 34 convergent OXPHOS genes.

---

## Important Caveats

- **The 104-gene convergent set is inflated by ~40%.** The analyst's methodology review estimated ~40 false positives from the use of nominal p-values across multiple cancer types. Zero genes pass FDR < 0.05 in both ARID1A and SMARCA4 contexts simultaneously. The OXPHOS pathway-level conclusion is robust to this inflation (remains significant after correction), but individual gene claims should be treated as hypothesis-generating.
- **SMARCA4 convergence is driven by ovarian cancer.** 64% of convergent genes have their best SMARCA4 signal in Ovary/Fallopian Tube (n = 8 mutant lines). The small sample size (n = 8) limits statistical power and generalizability.
- **TCGA expression validation was limited to lung cancer.** The strongest dependency signals were in ovarian and uterine cancers, which were not tested in TCGA due to data availability during this analysis.
- **IACS-010759 PRISM result may be confounded by tissue.** SMARCA4 mutations concentrate in lung cancer; the PRISM analysis does not control for cancer type. A stratified analysis is needed to confirm the effect is SWI/SNF-driven rather than lung-specific.
- **Metformin clinical translation is uncertain.** Achievable patient plasma concentrations (~10-40 uM) are 100-1,000x below typical in vitro effective concentrations for OXPHOS inhibition.
- **This project has not undergone formal validation scientist review.** All analysis was performed computationally and reviewed by the analyst for methodology, but independent wet-lab or external dataset validation has not been performed.

---

## Background

The SWI/SNF (BAF) chromatin remodeling complex is the most frequently mutated chromatin regulator in human cancer, with loss-of-function mutations occurring in ~20% of all malignancies. ARID1A (a DNA-binding subunit) and SMARCA4/BRG1 (a catalytic ATPase subunit) are the two most commonly mutated SWI/SNF genes, each altered in 5-10% of cancers depending on tumor type.

Current therapeutic strategies for SWI/SNF-mutant cancers focus exclusively on two axes:
1. **Paralog synthetic lethality:** ARID1A-mutant cells depend on ARID1B; SMARCA4-mutant cells depend on SMARCA2. Clinical programs include FHD-909/LY4050784 (SMARCA2 ATPase inhibitor, Phase I) and ARID1B degraders (preclinical).
2. **Epigenetic antagonism:** SWI/SNF-deficient cells become dependent on PRC2 activity, creating vulnerability to EZH2 inhibitors (tazemetostat, approved for INI1-negative tumors).

Two independently conducted pan-cancer synthetic lethality atlases — one for ARID1A and one for SMARCA4 — both discovered metabolic vulnerabilities as unexpected findings:
- The **ARID1A atlas** identified HMGCR (HMG-CoA reductase, the statin target) as SL in breast (d = -1.68), uterus (d = -0.78), and lung (d = -0.75), with literature validation from simvastatin-induced pyroptosis in ARID1A-mutant cells (Cancer Cell 2023). It also identified ADCK5, a mitochondrial kinase, as SL in 4 cancer types.
- The **SMARCA4 atlas** found that 7 of 10 FDR-significant novel SL genes in ovarian cancer are mitochondrial (MTERF4, MICOS13, DMAC2, MRPS35, COX6C, WARS2, HIGD2A), the most striking finding of that atlas.

Neither atlas pursued the metabolic angle as a primary question. This project was designed to bridge the two atlases by systematically testing whether the independently observed metabolic vulnerabilities converge on shared druggable pathways — which would indicate a fundamental metabolic rewiring driven by SWI/SNF loss rather than tissue-specific artifacts.

---

## Methodology

### Phase 1a: SWI/SNF Cell Line Classification

All 2,119 DepMap 25Q3 cell lines were classified for ARID1A and SMARCA4 disruption status using:
- **Loss-of-function mutations:** LikelyLoF = True OR VEP Impact = HIGH (from OmicsSomaticMutations)
- **Homozygous deletions:** log2 copy number < -1.0 (from OmicsCNGeneLog2)

Result: 173 ARID1A-disrupted lines, 88 SMARCA4-disrupted lines, 245 with any SWI/SNF disruption. Cancer types qualifying for analysis required >= 5 mutant and >= 10 wild-type lines: 11 types for ARID1A, 4 for SMARCA4 (Ovary/Fallopian Tube, Bowel, Lung, Lymphoid), 14 for combined.

### Phase 1b: Metabolic Gene Dependency Screen

1,590 metabolic genes from 73 KEGG pathways (covering glycolysis, TCA cycle, OXPHOS, cholesterol biosynthesis, fatty acid metabolism, amino acid metabolism, and others) were filtered from DepMap CRISPRGeneEffect scores.

For each metabolic gene in each qualifying cancer type, Mann-Whitney U tests compared gene effect scores between mutant and wild-type lines, with Cohen's d for effect size and Benjamini-Hochberg FDR correction per cancer type.

### Phase 2: Cross-Atlas Convergence

A gene was classified as "convergent" if it showed SL signal (nominal p < 0.05, |d| > 0.3) in at least one cancer type in the ARID1A screen AND at least one cancer type in the SMARCA4 screen.

Targeted validation tested specific hypotheses: HMGCR (from ARID1A atlas) in SMARCA4 contexts; mitochondrial genes (from SMARCA4 atlas) in ARID1A contexts. An addendum tested glutathione pathway genes (GCLC, GCLM, GSR, GPX4, GSS, SLC7A11) and ROCK1/ROCK2 based on literature reports.

### Phase 3: Pathway Enrichment

Hypergeometric tests assessed whether convergent genes cluster in specific KEGG pathways, using the 1,590 metabolic genes as background. BH-FDR correction across 81 testable pathways. Ranked enrichment analysis compared pathway-level dependency shifts in ARID1A-mutant and SMARCA4-mutant lines separately.

### Phase 4: TCGA Expression Validation

Differential expression of convergent genes between SWI/SNF-mutant and wild-type tumors was tested in TCGA LUAD (n = 41 mutant, 477 WT) and LUSC (n = 27 mutant, 474 WT) cohorts. Mann-Whitney U tests with BH-FDR correction. Spearman correlation tested whether dependency scores correlated with expression changes.

### Phase 5: Drug Repurposing

Convergent metabolic gene targets were mapped to FDA-approved drugs and clinical compounds. PRISM 24Q2 drug sensitivity data validated selected drugs (IACS-010759, atorvastatin, pitavastatin, rosuvastatin) in SWI/SNF-mutant vs. wild-type cell lines.

---

## Results

### Metabolic Gene Convergence Identifies 104 Shared Vulnerability Genes

The metabolic dependency screen tested 1,590 genes across 11 ARID1A and 4 SMARCA4 cancer types. At nominal significance:
- **ARID1A screen:** 502 genes (32%) showed SL signal in at least one cancer type
- **SMARCA4 screen:** 233 genes (15%) showed SL signal in at least one cancer type
- **Convergent overlap:** 104 genes showed SL in both contexts

The 104 convergent genes are dominated by OXPHOS components (34 genes, 33%), followed by other metabolism (13), central carbon metabolism (11), glycan biosynthesis (10), amino acid metabolism (9), cholesterol/lipid (7), lipid metabolism (7), nucleotide metabolism (5), redox/cofactor (4), fatty acid metabolism (2), and one-carbon metabolism (2).

The top convergent genes by combined effect size:

| Gene | ARID1A Best Cancer Type | ARID1A d | SMARCA4 Best Cancer Type | SMARCA4 d | Category |
|------|------------------------|----------|--------------------------|-----------|----------|
| SRD5A3 | Uterus | -1.20 | Ovary/Fallopian Tube | -2.19 | Cholesterol/Lipid |
| MAN1A2 | Ovary/Fallopian Tube | -0.79 | Ovary/Fallopian Tube | -2.46 | Glycan biosynthesis |
| IDH3B | Biliary Tract | -1.60 | Lymphoid | -1.61 | Central carbon |
| PLD3 | Lymphoid | -1.50 | Lymphoid | -1.66 | Lipid metabolism |
| NDUFB9 | Ovary/Fallopian Tube | -1.26 | Ovary/Fallopian Tube | -1.88 | OXPHOS |
| NDUFB3 | Ovary/Fallopian Tube | -0.86 | Ovary/Fallopian Tube | -2.16 | OXPHOS |
| COX6C | Bowel | -0.57 | Ovary/Fallopian Tube | -2.39 | OXPHOS |
| LARGE2 | Skin | -1.29 | Lymphoid | -1.59 | Glycan biosynthesis |
| PRDX6 | Skin | -2.18 | Lung | -0.63 | Redox/cofactor |
| NDUFA4 | Ovary/Fallopian Tube | -1.27 | Ovary/Fallopian Tube | -1.51 | OXPHOS |

**Methodological caveat:** The analyst estimated ~40 of 104 genes may be false positives based on the expected overlap rate under the null when using nominal p-values across multiple cancer types (2.6x enrichment over null expectation of ~40 chance overlaps). Zero genes pass FDR < 0.05 in both contexts simultaneously. The convergent set should be treated as a discovery set with ~60% estimated true positive rate.

### HMGCR/Statin Convergence Is Falsified

HMGCR, the strongest ARID1A metabolic dependency (d = -1.68 in breast), showed no SL signal in any SMARCA4 context (d ranged from -0.10 to +0.14, all p > 0.7). The cholesterol biosynthesis convergence hypothesis is definitively rejected. Statin-based therapy is ARID1A-specific, not a general SWI/SNF strategy.

### Glutathione/Eprenetapopt Convergence Is Falsified

Despite literature reports of eprenetapopt sensitivity in SMARCA4/SMARCB1/PBRM1-deficient cells, DepMap CRISPR analysis found:
- 5 core GSH genes (GCLC, GCLM, GSR, GPX4, GSS): 4 ARID1A-specific SL hits, 0 SMARCA4 hits, 0 convergent
- SLC7A11 (cystine transporter): 0 SL hits in 14 tests across both contexts
- Glutathione metabolism enrichment: fold = 0.27x, FDR = 1.0

GSH vulnerability is not convergent in DepMap. The literature-reported eprenetapopt sensitivity may act through p53-dependent mechanisms or combinatorial redox stress not captured by single-gene CRISPR knockout.

### OXPHOS Is the Dominant Convergent Pathway

Hypergeometric pathway enrichment identified two significant pathways at FDR < 0.05:

| Pathway | Convergent/Background | Fold Enrichment | P-value | FDR |
|---------|-----------------------|-----------------|---------|-----|
| Oxidative phosphorylation | 34/120 | 4.33x | 3.16 x 10^-15 | 2.56 x 10^-13 |
| Lipoic acid metabolism | 7/19 | 5.63x | 1.10 x 10^-4 | 4.45 x 10^-3 |

The 34 convergent OXPHOS genes span all five ETC complexes:
- **Complex I (NADH dehydrogenase):** NDUFA2, NDUFA4, NDUFA4L2, NDUFA5, NDUFA8, NDUFA9, NDUFA11, NDUFA13, NDUFB3, NDUFB4, NDUFB7, NDUFB8, NDUFB9, NDUFC1, NDUFC2, NDUFS2, NDUFS3, NDUFS7, NDUFV2 (19 genes)
- **Complex II (SDH):** SDHB, SDHC, SDHD (3 genes)
- **Complex III (Cytochrome bc1):** CYC1, UQCR10, UQCRC2, UQCRQ (4 genes)
- **Complex IV (Cytochrome c oxidase):** COX5A, COX6A1, COX6B1, COX6C, COX7B (5 genes)
- **Complex V (ATP synthase):** ATP5F1A, ATP6V1G3 (2 genes)
- **Other:** PPA1 (1 gene)

This enrichment is robust to the estimated ~40% false positive rate in the convergent gene set. Even assuming 40 random false positives, only ~3 would be expected in OXPHOS by chance (40 x 120/1590 = 3), leaving ~31 true OXPHOS positives — still highly significant.

Lipoic acid metabolism (7 convergent genes: BCKDHA, DLD, GCSH, LIAS, LIPT1, LIPT2, OGDH) reinforces the mitochondrial theme, as lipoic acid is a cofactor for pyruvate dehydrogenase and alpha-ketoglutarate dehydrogenase, both mitochondrial enzyme complexes.

**Ranked analysis revealed an important asymmetry:** SMARCA4-mutant lines show extremely strong pathway-level OXPHOS dependency shift (FDR = 2.43 x 10^-17, median d_in = -0.905 vs. d_out = -0.339), while ARID1A-mutant lines do not reach FDR significance for OXPHOS independently. The OXPHOS vulnerability is primarily driven by SMARCA4-mutant cells (especially ovarian), with ARID1A contributing through gene-level convergence overlap.

### TCGA Expression Analysis: Dependency Is Not Transcriptionally Driven

In TCGA lung cancer (the only cohort tested), only 4 of 104 convergent genes showed differential expression at FDR < 0.05:

| Gene | log2 Fold Change | FDR | Direction |
|------|-------------------|-----|-----------|
| UGDH | +0.63 | 0.050 | Upregulated |
| SEPHS2 | +0.57 | 0.010 | Upregulated |
| COX5A | +0.36 | 0.010 | Upregulated |
| COX7B | +0.31 | 0.027 | Upregulated |

This 3.8% rate fails the pre-specified success criterion of >= 50%. The dependency-expression correlation was effectively zero (Spearman r = -0.055, p = 0.577).

**Interpretation:** CRISPR dependency measures functional essentiality — a gene becomes lethal to lose only when SWI/SNF is disrupted, regardless of whether its expression has changed. The OXPHOS dependency likely reflects a metabolic state change (e.g., increased reliance on mitochondrial respiration due to chromatin-mediated metabolic rewiring) rather than transcriptional deregulation of individual OXPHOS genes. This analysis is also limited by testing only lung TCGA, when the strongest dependency signals are in ovarian and uterine cancer.

### Drug Repurposing: IACS-010759 Validated, Statins Falsified

| Drug | Target | Approval Status | Convergent Targets | PRISM Validation |
|------|--------|----------------|--------------------|-----------------|
| **Metformin** | Complex I/OXPHOS | FDA-approved (diabetes) | 18 | No PRISM data |
| **IACS-010759** | Complex I | Phase I (AML/solid tumors) | 18 | **SMARCA4: d=-0.36, p=0.007** |
| **Lonidamine** | Complex II/SDH | EU-approved (some countries) | 3 | No PRISM data |
| Phenformin | Complex I | Withdrawn (research only) | 5 | Not tested |
| Atorvastatin | HMGCR | FDA-approved | 0 | d=0.007, p=0.87 (no signal) |
| Pitavastatin | HMGCR | FDA-approved | 0 | d=0.014, p=0.59 (no signal) |
| Rosuvastatin | HMGCR | FDA-approved | 0 | d=0.035, p=0.83 (no signal) |
| Eprenetapopt | GSH depletion | FDA-approved (MDS) | 0 | Not in PRISM |

IACS-010759, a potent selective Complex I inhibitor, showed statistically significant selective sensitivity in SMARCA4-mutant lines (d = -0.36, p = 0.007, n = 61 mutant, 728 WT). This effect was not significant in ARID1A-mutant lines (d = -0.02, p = 0.56) or combined SWI/SNF lines (d = -0.17, p = 0.02, marginal and driven by SMARCA4). All three statins showed zero SWI/SNF-selective effect, confirming statin therapy is not a convergent strategy.

### ROCK1/ROCK2: SMARCA4-Specific, Not Convergent

ROCK1 and ROCK2 (tested per literature reports of ROCK+OXPHOS synergy) showed SL signal only in SMARCA4-mutant ovarian cancer (ROCK1: d = -0.68, p = 0.036; ROCK2: d = -0.71, p = 0.048) with no significant ARID1A signal. ROCK+OXPHOS combination may be viable for SMARCA4-specific tumors but lacks convergent support.

---

## Discussion

### A Third Therapeutic Axis for SWI/SNF-Mutant Cancer

This study identifies mitochondrial OXPHOS as a previously unrecognized convergent vulnerability in SWI/SNF-mutant cancers, distinct from the two established therapeutic strategies (paralog dependency and epigenetic antagonism). The finding that 34 OXPHOS genes across all five ETC complexes show synthetic lethal signal in both ARID1A-mutant and SMARCA4-mutant contexts — two different subunits of the same complex — suggests that SWI/SNF loss fundamentally alters cellular metabolic dependencies.

The most parsimonious mechanistic explanation is that SWI/SNF chromatin remodeling regulates metabolic gene expression at the level of chromatin accessibility. When SWI/SNF function is lost, cells may shift metabolic programs toward increased OXPHOS reliance, making them selectively vulnerable to mitochondrial disruption. The near-zero correlation between dependency and expression in lung TCGA suggests this may operate through post-transcriptional or activity-level mechanisms, or may require tissue-specific contexts (ovarian, uterine) not yet tested in expression data.

### Clinical Implications

The OXPHOS convergence finding opens a potential drug repurposing opportunity:

1. **Metformin** (FDA-approved, ~$4/month) inhibits 18 of 34 convergent OXPHOS genes via Complex I inhibition. However, achievable plasma concentrations may be insufficient for anti-cancer effects without formulation innovations.
2. **IACS-010759** (Phase I trial) showed the strongest pharmacological validation: SMARCA4-mutant lines are significantly more sensitive (d = -0.36, p = 0.007 in PRISM). This drug is already in clinical development for AML and solid tumors.
3. **Lonidamine** (EU-approved in some countries) targets 3 convergent SDH/Complex II genes.

Importantly, these metabolic therapies could potentially be combined with paralog-targeting agents (e.g., FHD-909/LY4050784 SMARCA2 inhibitor, now in Phase I) or EZH2 inhibitors for multi-axis attack on SWI/SNF-mutant tumors.

### The Falsification of HMGCR and Glutathione Axes

The clear negative results for HMGCR/statins and glutathione/eprenetapopt are scientifically valuable. HMGCR dependency, despite being the strongest single-gene ARID1A SL hit (d = -1.68 in breast), is definitively ARID1A-specific and tissue-restricted. This rules out statins as a general SWI/SNF repurposing strategy and clarifies that the ARID1A atlas HMGCR finding reflects ARID1A-specific biology in hormone-responsive cancers, not a general SWI/SNF metabolic rewiring.

Similarly, the GSH/eprenetapopt negative result (0 convergent GSH genes despite literature reports of SWI/SNF-deficient cell sensitivity) highlights a divergence between pharmacological sensitivity and CRISPR genetic dependency. Eprenetapopt's complex pharmacology (p53 reactivation, redox stress beyond GSH depletion) may explain sensitivity not captured by single-gene knockout.

### External Validation: SWI/SNF Therapeutic Relevance

Recent external data support the broader thesis that SWI/SNF alterations create druggable vulnerabilities. FHD-286, a SMARCA4/SMARCA2 ATPase inhibitor, sensitizes KRAS-mutant NSCLC to pan-RAS and KRAS G12D inhibitors by dampening EMT-mediated resistance (bioRxiv 2026.02.27.708377v1). This FHD-286 + KRAS inhibitor synergy in organoid and PDX models demonstrates that SWI/SNF inhibition has therapeutic utility beyond paralog dependency. While this represents an oncogene-interaction axis distinct from the metabolic convergence studied here, it reinforces that SWI/SNF function intersects with multiple druggable pathways.

Additionally, Pai et al. (J Proteome Res, March 2026, PMID: 41885501) demonstrated that SMARCA4 knockdown in medulloblastoma triggers metabolic reprogramming toward lipid biosynthesis, with upregulation of FABP5, CYP27A1, and SCP2 — proteomics-level validation that SMARCA4 loss rewires cellular metabolism. LY4050784 (FHD-909), the first selective SMARCA2 ATPase inhibitor, entered Phase I trials for SMARCA4-altered solid tumors (AACR 2026, CT109), providing clinical validation of the SMARCA4-SMARCA2 synthetic lethal paradigm that underlies our convergence hypothesis.

### Tissue Specificity and the Ovarian Cancer Question

A critical nuance is that the convergent OXPHOS signal is heavily driven by ovarian cancer. 64% of convergent genes have their best SMARCA4 signal in Ovary/Fallopian Tube, and 45 of 56 same-tissue convergent pairs are ovarian. SMARCA4-mutant ovarian cancer (including SCCOHT) may represent a uniquely metabolically rewired tumor subtype. Whether OXPHOS convergence extends to other SWI/SNF-mutant lineages (e.g., lung, uterine, bladder) at larger sample sizes remains an open question.

---

## Limitations

1. **Statistical inflation of the convergent gene set.** The 104-gene set uses nominal p-values across multiple cancer types, inflating the count by ~40%. The pathway-level OXPHOS conclusion withstands this, but individual gene calls are preliminary.
2. **Small SMARCA4 sample sizes.** Only 4 cancer types qualify for SMARCA4 analysis, with the key ovarian cohort having only 8 mutant lines. Effect size estimates may be inflated.
3. **Ovarian cancer dominance.** The convergence finding may not generalize to non-ovarian SWI/SNF-mutant cancers.
4. **TCGA limited to lung.** Expression validation needs expansion to UCEC (uterine), OV (ovarian), and STAD (stomach) cohorts where SWI/SNF mutations are more prevalent and dependency signals are strongest.
5. **IACS-010759 tissue confounding.** SMARCA4 mutations concentrate in lung cancer; the PRISM sensitivity result requires cancer-type-stratified reanalysis to separate SWI/SNF-driven from tissue-driven effects.
6. **No wet-lab validation.** All findings are computational. Functional validation of OXPHOS inhibitor sensitivity in isogenic SWI/SNF-mutant cell line pairs has not been performed.
7. **Metformin pharmacokinetics.** Clinical OXPHOS inhibition may require concentrations far above those achievable with standard metformin dosing.
8. **Two-subunit analysis only.** Testing only ARID1A and SMARCA4 misses other SWI/SNF subunits (SMARCB1, PBRM1, ARID2, ARID1B). The convergence may be broader than measured.

---

## Next Steps

1. **Expand TCGA expression analysis** to ovarian (OV), uterine (UCEC), and stomach (STAD) cohorts, using pathway-level OXPHOS gene set scores rather than individual gene FDR.
2. **Cancer-type-stratified PRISM analysis** for IACS-010759 to separate SWI/SNF-driven from tissue-driven drug sensitivity.
3. **Cross-reference new proteomics data** (Pai et al. FABP5, CYP27A1, SCP2) against SMARCA4 DepMap dependencies to test the lipid metabolism axis computationally.
4. **Formal validation scientist review** of methodology, statistical rigor, and reproducibility.
5. **Wet-lab validation:** Test IACS-010759 and metformin sensitivity in isogenic ARID1A-knockout and SMARCA4-knockout cell line pairs.
6. **Extend to other SWI/SNF subunits** (SMARCB1, PBRM1) to determine if OXPHOS convergence is a pan-SWI/SNF phenomenon.
7. **Investigate mechanism:** Use ATAC-seq or ChIP-seq data in SWI/SNF-mutant lines to test whether OXPHOS gene promoters show altered chromatin accessibility.

---

## References

- DepMap 25Q3: CRISPRGeneEffect, OmicsSomaticMutations, OmicsCNGeneLog2, OmicsExpressionProteinCodingGenesTPMLogp1 (Broad Institute)
- PRISM Repurposing 24Q2 (Broad Institute)
- TCGA Pan-Cancer Atlas: LUAD (n = 518), LUSC (n = 501)
- KEGG REST API: 73 metabolic pathways, 1,590 metabolic genes (Kanehisa et al.)
- Zhang et al. "Simvastatin selectively kills ARID1A-deficient cancer cells via pyroptosis." Cancer Cell, 2023.
- ARID1A pan-cancer SL atlas (this project, `arid1a-pancancer-sl-atlas`)
- SMARCA4 pan-cancer SL atlas (this project, `smarca4-pancancer-sl-atlas`)
- FHD-286 + KRAS inhibitor combo: bioRxiv 2026.02.27.708377v1
- Pai et al. "Quantitative DIA-MS Uncovers Functional Impact of SMARCA4 Knockdown in Group 3 Medulloblastoma." J Proteome Res, March 2026. PMID: 41885501.
- LY4050784 (FHD-909) Phase I: AACR 2026 abstracts CT109, 3779.
- Eprenetapopt in SWI/SNF-deficient cells: Sci Rep, December 2024.
- ROCK + OXPHOS synergy in SMARCA4-mutant cells: bioRxiv, 2024.
- Foghorn Therapeutics pipeline updates, Q4 2025 / 2026 strategic outlook.
- IACS-010759 Phase I: Molina et al., Nature Medicine, 2018.
- Hypergeometric test: scipy.stats.hypergeom. BH-FDR correction: statsmodels.
