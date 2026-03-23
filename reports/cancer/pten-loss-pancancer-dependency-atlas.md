# PTEN-Loss Pan-Cancer Dependency Atlas

**Project:** pten-loss-pancancer-dependency-atlas
**Division:** cancer
**Date:** 2026-03-23
**Data Sources:** DepMap 25Q3 (CRISPRGeneEffect, PortalOmicsCNGeneLog2, OmicsSomaticMutations, OmicsExpressionTPMLogp1), PRISM 24Q2, TCGA PanCancer Atlas, ACS 2024 incidence
**Pipeline:** `src/cancer/pten_loss_pancancer_dependency_atlas/01–05*.py`
**Validation Status:** APPROVED — all 3 validation tasks PASSED (journal #1485, #1487, #1488, #1489). ICMT addendum reviewed (journal #1929): not confirmed in DepMap, likely underpowered.

---

## Executive Summary

PTEN-null cancer cells undergo a p110α→p110β isoform switch that creates a targetable PI3Kβ dependency. This atlas mapped gene dependencies and drug sensitivities in 209 PTEN-lost versus 1,885 PTEN-intact cell lines across 9 qualifying cancer types in DepMap 25Q3, with convergent validation from PRISM 24Q2 drug sensitivity data.

PIK3CB (encoding p110β) is robustly PTEN-selective in breast cancer (Cohen's d = −2.23, FDR = 3.6 × 10⁻⁴) and melanoma (d = −1.50, FDR = 2.2 × 10⁻³), while PIK3CA (p110α) shows the expected inverse pattern — PTEN-lost cells are *less* dependent on PIK3CA (breast d = +0.95, FDR = 9.1 × 10⁻³). This isoform switch is mirrored pharmacologically: afuresertib (AKT inhibitor) shows strong PTEN-selective sensitivity in uterus (d = −1.97) and ovary (d = −1.68), while inavolisib (PI3Kα inhibitor) shows inverse selectivity with PTEN-lost cells *less* sensitive (pan-cancer d = +0.44, FDR = 1.6 × 10⁻⁴). All five ROBUST CRISPR hits passed leave-one-out stability testing.

The genome-wide screen (18,435 genes × 9 cancer types) yielded zero novel non-PI3K dependencies at FDR < 0.05 — a power limitation with 5–34 PTEN-lost lines per type, not evidence of absence. An estimated ~177,000 US patients per year have PTEN-lost tumors. Fourteen cancer types — including melanoma (~12,000 patients), colorectal (~10,700), and bladder (~8,300) — lack PTEN-targeted clinical trials, representing major unmet therapeutic needs.

---

## Important Caveats

1. **Genome-wide screen is underpowered.** With 5–34 PTEN-lost lines per cancer type, BH correction over 18,435 genes requires extreme raw p-values. Zero genes passed genome-wide significance. The PIK3CB positive control — which has d = −2.23 in breast — achieves only FDR = 0.10 in the genome-wide context (n_lost = 10). The null result for novel non-PI3K dependencies reflects statistical power, not biological absence.

2. **Key drugs missing from PRISM.** Capivasertib (FDA-approved AKT inhibitor), ipatasertib, alpelisib, AZD8186, GSK2636771, and GT220 (PI3Kβ-selective agents) are absent from PRISM 24Q2. The most clinically relevant drugs for PTEN-lost cancers could not be tested pharmacologically.

3. **Prostate cancer is critically underrepresented.** Only 3 PTEN-lost prostate lines exist in DepMap, versus ~57,660 estimated US patients/year (20% PTEN-loss prevalence). Prostate did not qualify for powered analysis despite being a top clinical priority.

4. **Combination candidates have unresolved identifiers.** The 49 non-PI3K PTEN-selective drugs from genome-wide PRISM screening are identified only by Broad Institute compound IDs (BRD-K format). Without annotation to specific drugs and mechanisms, these hits are hypothesis-generating but not immediately actionable.

5. **PTEN loss classification combines heterogeneous mechanisms.** Deep deletion and truncating mutation are grouped as "PTEN-lost," but these may have different functional consequences. Missense variants are excluded as VUS, potentially missing partial-loss-of-function phenotypes.

---

## Background

PTEN (phosphatase and tensin homolog) is the third most commonly altered tumor suppressor in human cancer, with loss-of-function alterations in approximately 8–10% of all cancers and much higher frequencies in specific types: endometrial (~65%), glioblastoma (~36%), prostate (~20%), and melanoma (~12%). PTEN functions as a lipid phosphatase that dephosphorylates PIP3 to PIP2, directly opposing PI3K signaling. Loss of PTEN constitutively activates PI3K/AKT/mTOR, driving cell survival, growth, and proliferation.

A critical but underappreciated consequence of PTEN loss is the PI3K isoform switch. In PTEN-intact cells, the predominant PI3K catalytic subunit is p110α (PIK3CA), which is activated by receptor tyrosine kinases. When PTEN is lost, constitutive PIP3 accumulation shifts dependency to p110β (PIK3CB), which maintains basal PI3K signaling through distinct regulatory mechanisms. This isoform switch has been demonstrated in cell-based and mouse models but had not been systematically validated across cancer types at scale.

**Therapeutic landscape (March 2026):**

- **Capivasertib** (AstraZeneca): FDA-approved November 2023 for PIK3CA/AKT1/PTEN-altered HR+/HER2- breast cancer (CAPItello-291). Phase 3 CAPItello-281 results (ASCO GU 2026) showed rPFS improvement in PTEN-deficient metastatic hormone-sensitive prostate cancer (33.2 vs 25.7 months, HR 0.81, p = 0.034).
- **Ipatasertib** (Roche): AKT inhibitor, Phase 3 in prostate (IPATential150) with mixed results in the PTEN-loss subgroup.
- **Alpelisib** (Novartis): PI3Kα-selective, approved for PIK3CA-mutant breast cancer — does not directly address PTEN loss.
- **Inavolisib** (Genentech): PI3Kα-selective mutant degrader, approved October 2024 for PIK3CA-mutant breast cancer. Our data shows PTEN-lost cells are *less* sensitive, consistent with the isoform switch.
- **GT220** (PI3Kβ-selective): Phase 1/preclinical, specifically designed for PTEN-null tumors — the first isoform-selective agent targeting the node most robustly dependent on PTEN loss.
- **Key gap:** No PI3Kβ-selective inhibitor is approved. AKT inhibitors show promise but have modest single-agent efficacy. Combination strategies and cancer-type-specific prioritization are urgently needed.

**Cross-project context:** This atlas complements the PIK3CA allele-specific dependency atlas. Both address PI3K pathway dependencies but through opposite genetic mechanisms: PIK3CA mutation constitutively activates p110α (creating PIK3CA oncogene addiction), while PTEN loss constitutively activates p110β (creating PIK3CB dependency). Together, they define the full PI3K pathway vulnerability landscape.

---

## Methodology

### Data Sources

- **DepMap 25Q3 CRISPRGeneEffect** — genome-wide CRISPR dependency scores for 2,119 cell lines
- **DepMap 25Q3 PortalOmicsCNGeneLog2** — gene-level copy number (log2 ratio, 1.0 = diploid) for PTEN and RB1 deletion calling
- **DepMap 25Q3 OmicsSomaticMutations** — PTEN, PIK3CA, TP53, RB1 somatic mutation calls with VEP impact and LoF annotations
- **DepMap 25Q3 OmicsExpressionTPMLogp1** — PTEN expression (log2 TPM+1) for classifier cross-validation
- **PRISM 24Q2** — drug sensitivity (Repurposing_Public_24Q2_Extended_Primary_Data_Matrix) and compound metadata
- **TCGA PanCancer Atlas** — PTEN alteration frequencies across 19 cancer types
- **ACS 2024** — US cancer incidence estimates for patient population calculations

### Phase 1: PTEN Status Classification (`01_pten_classifier.py`)

All 2,119 DepMap 25Q3 cell lines were classified by PTEN status using a two-criterion approach:

- **Deep deletion:** Copy number log2 ratio ≤ 0.3 on the PortalOmicsCNGeneLog2 scale (where 1.0 = diploid)
- **Truncating mutation:** Nonsense, frameshift, or splice-site mutations annotated as LikelyLoF = TRUE
- **PTEN-lost:** Deep deletion OR truncating mutation OR both
- **Excluded:** Lines with missense-only PTEN variants (classified as variants of uncertain significance)

Co-alterations were annotated: PIK3CA hotspot mutations (E542K, E545K/A/G/Q, H1047R/L/Y, C420R, N345K, R88Q), TP53 mutations (HIGH or MODERATE VEP impact), and RB1 loss (LoF mutation or CN log2 ≤ 0.3). PTEN expression (log2 TPM+1) was compared between groups via Mann-Whitney U test. Cancer types with ≥5 PTEN-lost and ≥10 PTEN-intact lines with CRISPR data qualified for analysis.

**Result:** 209 PTEN-lost (34 deletion-only, 175 mutation-only, 0 both), 1,885 intact, 25 excluded (missense VUS). Expression validation: median PTEN-lost 4.40 vs intact 5.00 (p = 9.55 × 10⁻¹³); 54.6% of PTEN-lost lines below the intact Q25, with 36 discordant cases (lost with high expression). Nine of 34 cancer types qualified.

### Phase 2: PI3K/AKT/mTOR Pathway Effect Sizes (`02_pi3k_akt_effect_sizes.py`)

For each of 9 qualifying cancer types, dependency scores for 7 pathway genes (PIK3CB, PIK3CA, AKT1, AKT2, MTOR, RICTOR, RPTOR) were compared between PTEN-lost and PTEN-intact groups:

- **Effect size:** Cohen's d with pooled standard deviation
- **Hypothesis test:** Mann-Whitney U (two-sided, non-parametric)
- **Confidence intervals:** Bootstrap 95% CI (1,000 iterations, seed = 42)
- **Multiple testing:** Benjamini-Hochberg FDR across all 63 tests
- **Classification:** ROBUST (FDR < 0.05, |d| > 0.5); MARGINAL (FDR < 0.10); NOT_SIGNIFICANT (otherwise)
- **Robustness:** Leave-one-out analysis for all ROBUST hits; PIK3CA co-mutation stratification

### Phase 3: Genome-Wide Dependency Screen (`03_genomewide_screen.py`)

All ~18,435 genes were tested per qualifying cancer type using Mann-Whitney U with Cohen's d effect sizes. Significance thresholds: FDR < 0.05 and |d| > 0.3. Pathway enrichment (Fisher's exact test) covered PI3K/AKT/mTOR, insulin/IGF signaling, DNA damage response, chromatin remodeling, protein prenylation, and NF-κB signaling. Priority targets from literature (ICMT, CHD1, PNKP, SMARCA4, BCL2L1, ATM, TTK, RAD51) were specifically tracked against a synthetic lethality benchmark set.

### Phase 4: PRISM Drug Sensitivity (`04_prism_drug_sensitivity.py`)

Seven PI3K pathway drugs available in PRISM 24Q2 were tested for PTEN-selective sensitivity: GSK2110183/afuresertib (AKT), triciribine-phosphate (AKT), GDC-0077/inavolisib (PI3Kα), wortmannin (pan-PI3K), PI3Kd-IN-2 (PI3Kδ), GDC-0084 (PI3K/mTOR dual, brain-penetrant), and CC-223 (mTOR catalytic). Genome-wide PRISM screening tested all compounds for PTEN selectivity. CRISPR-PRISM concordance was assessed via Spearman correlation between gene dependency and drug sensitivity across cell lines. Combination candidates were identified as non-PI3K drugs with d < −0.3 and FDR < 0.05.

### Phase 5: TCGA Clinical Integration (`05_tcga_integration.py`)

PTEN alteration frequency per cancer type was derived from TCGA PanCancer Atlas data. Addressable patient populations were estimated as TCGA PTEN-loss prevalence × ACS 2024 US cancer incidence. Priority ranking combined dependency effect size (|Cohen's d|), population size (log-transformed), and druggability. Active clinical trials targeting PI3K/AKT/mTOR in PTEN-selected populations were mapped to identify trial gaps.

---

## Results

### PTEN Loss Landscape (Phase 1)

Of 2,119 DepMap cell lines, 209 (9.9%) were classified as PTEN-lost. The dominant mechanism was truncating mutation (175, 83.7%) rather than deep deletion (34, 16.3%), with no lines showing both. Among PTEN-lost lines, TP53 co-mutation was common (68.9%), followed by RB1 loss (21.1%) and PIK3CA hotspot co-mutation (7.2%).

Nine cancer types met the power threshold (≥5 PTEN-lost, ≥10 PTEN-intact with CRISPR data):

| Cancer Type | PTEN-Lost | PTEN-Intact | Loss Frequency | CRISPR (Lost/Intact) |
|---|---|---|---|---|
| Uterus | 23 | 20 | 53.5% | 19/14 |
| CNS/Brain | 47 | 75 | 38.5% | 34/50 |
| Skin | 22 | 128 | 14.7% | 14/61 |
| Breast | 13 | 82 | 13.7% | 10/42 |
| Bowel | 10 | 86 | 10.4% | 8/53 |
| Ovary/Fallopian Tube | 7 | 67 | 9.5% | 6/52 |
| Lymphoid | 23 | 233 | 9.0% | 13/78 |
| Lung | 21 | 237 | 8.1% | 8/117 |
| Esophagus/Stomach | 7 | 97 | 6.7% | 5/64 |

Expression cross-validation confirmed the classifier: PTEN-lost lines had significantly lower PTEN mRNA (p = 9.55 × 10⁻¹³). However, 36 PTEN-lost lines retained high expression, possibly reflecting incomplete loss-of-function or regulatory complexity.

### PI3Kβ Isoform Switch — Core Discovery (Phase 2)

Across 63 tests (9 cancer types × 7 pathway genes), 5 ROBUST hits were identified (FDR < 0.05, |d| > 0.5). All 5 passed leave-one-out robustness testing.

**PIK3CB (p110β) — Positive control PASSED:**

| Cancer Type | Cohen's d | 95% CI | FDR | n_lost | n_intact | LOO Range |
|---|---|---|---|---|---|---|
| Breast | −2.229 | [−3.359, −1.570] | 3.56 × 10⁻⁴ | 10 | 42 | [−2.49, −2.09] |
| Skin | −1.497 | [−2.462, −0.741] | 2.20 × 10⁻³ | 14 | 61 | [−1.72, −1.36] |
| Lung | −1.304 | — | 1.78 × 10⁻¹ | 8 | 117 | — |
| Esoph/Stomach | −1.090 | — | 5.33 × 10⁻¹ | 5 | 64 | — |
| CNS/Brain | −0.323 | — | 3.23 × 10⁻¹ | 34 | 50 | — |

PTEN-lost breast cancer cells show a massive PIK3CB dependency (d = −2.23), meaning the average PTEN-lost line is >2 standard deviations more dependent on PIK3CB than PTEN-intact lines. Skin/melanoma shows a similarly strong effect (d = −1.50). Lung and esophagus/stomach show suggestive effects that did not survive FDR correction, likely due to small PTEN-lost sample sizes (n = 8 and n = 5).

**PIK3CA (p110α) — Inverse selectivity confirmed:**

| Cancer Type | Cohen's d | 95% CI | FDR | Classification |
|---|---|---|---|---|
| Breast | +0.945 | [+0.420, +1.546] | 9.09 × 10⁻³ | ROBUST |
| Lymphoid | +0.715 | [+0.355, +1.112] | 2.00 × 10⁻² | ROBUST |

PTEN-lost cells are *less* dependent on PIK3CA, consistent with the p110α→p110β switch. When PTEN is lost and p110β becomes the dominant kinase, knocking out p110α has reduced impact.

**AKT2 — Downstream validation:**

Lymphoid cancer showed ROBUST AKT2 dependency in PTEN-lost cells (d = −1.07, FDR = 9.1 × 10⁻³). AKT1, despite being the primary downstream effector of PI3K, did not reach significance in any cancer type — possibly because AKT1 is commonly essential regardless of PTEN status.

### Genome-Wide Screen (Phase 3)

The genome-wide screen of 18,435 genes across 9 cancer types yielded zero PTEN-selective dependencies at FDR < 0.05. This is consistent with the statistical power analysis: with the largest PTEN-lost cohort at 34 lines (CNS/Brain), BH correction over 18,435 tests demands extreme raw p-values that small sample sizes cannot generate.

Priority targets from the literature showed suggestive but non-significant effects:

| Gene | Best Cancer Type | Cohen's d | Raw p | FDR | Known Biology |
|---|---|---|---|---|---|
| ICMT | Bowel | −0.70 | — | NS | SL with PTEN loss (PMID 41507994) |
| TTK | CNS/Brain | −0.56 | 0.02 | 0.48 | Spindle checkpoint |
| CHD1 | — | — | — | NS | Chromatin remodeling |
| ATM | — | — | — | NS | DNA damage response |

Validation review of ICMT (journal #1929) confirmed it is present across all 9 cancer types in DepMap but reaches significance in none, likely reflecting power constraints rather than biological absence.

### Drug Sensitivity — PRISM Concordance (Phase 4)

**Afuresertib (AKT inhibitor) — Top PTEN-selective drug:**

| Cancer Type | Cohen's d | FDR | n_lost | n_intact |
|---|---|---|---|---|
| Uterus | −1.970 | 2.39 × 10⁻³ | 15 | 8 |
| Ovary | −1.679 | 2.14 × 10⁻² | 6 | 31 |
| Skin | −0.992 | 9.68 × 10⁻² | 13 | 30 |
| Pan-cancer | −0.734 | 1.08 × 10⁻⁵ | 85 | 465 |
| Breast | −0.449 | 7.76 × 10⁻¹ | 7 | 19 |
| CNS/Brain | −0.412 | 4.85 × 10⁻¹ | 22 | 18 |

Afuresertib shows robust PTEN-selective sensitivity, with the strongest effects in uterine and ovarian cancers. The pan-cancer signal (d = −0.73, FDR = 1.1 × 10⁻⁵) confirms a consistent directionality across cancer types.

**Inavolisib (PI3Kα inhibitor) — Inverse selectivity:**

Inavolisib shows a significant pan-cancer *inverse* effect (d = +0.44, FDR = 1.6 × 10⁻⁴): PTEN-lost cells are less sensitive to PI3Kα-selective inhibition. This mirrors the CRISPR finding of reduced PIK3CA dependency in PTEN-lost cells, providing pharmacological confirmation of the isoform switch. Clinically, this suggests inavolisib and alpelisib are not optimal for PTEN-lost cancers.

**CRISPR-PRISM concordance:**

| Drug | Gene Target | Spearman r | p-value |
|---|---|---|---|
| GDC-0077 (inavolisib) | PIK3CA | 0.335 | 3.58 × 10⁻²⁰ |
| GSK2110183 (afuresertib) | AKT1 | 0.230 | 6.17 × 10⁻⁷ |
| GDC-0084 | MTOR | 0.133 | 3.74 × 10⁻⁴ |
| CC-223 | MTOR | 0.139 | 2.00 × 10⁻⁴ |
| Triciribine | AKT1 | −0.003 | 0.946 |
| Wortmannin | PIK3CA | 0.055 | 0.139 |

Strong concordance for inavolisib-PIK3CA (r = 0.34) and afuresertib-AKT1 (r = 0.23) validates that drug sensitivity reflects on-target gene dependency. Triciribine and wortmannin show no concordance, suggesting off-target or non-specific mechanisms at PRISM concentrations.

**Genome-wide PRISM screen:** 110 significant drugs at FDR < 0.05, of which 51 showed PTEN-selective sensitivity (d < −0.3). The top non-PI3K combination candidates (BRD-K99023089, d = −0.93; BRD-K25325018, d = −0.92; BRD-K71480163, d = −0.84) have unresolved compound identities.

### Clinical Translation (Phase 5)

**Estimated US PTEN-loss patient population: ~177,000 per year**

| Cancer Type | PTEN Loss % | Est. US Patients/yr | Dominant Mechanism | TP53 Co-mut |
|---|---|---|---|---|
| Endometrial (UCEC) | 65% | 44,122 | Truncation (52%) | 40% |
| Prostate (PRAD) | 20% | 57,660 | Deletion (15%) | 35% |
| Breast (BRCA) | 6% | 18,643 | Mixed | 60% |
| Melanoma (SKCM) | 12% | 12,076 | Mixed | 20% |
| Colorectal (COADREAD) | 7% | 10,696 | Mixed | 55% |
| Bladder (BLCA) | 10% | 8,319 | Mixed | 50% |
| GBM (GBM) | 36% | 5,040 | Deletion (25%) | 30% |
| Lung Adeno (LUAD) | 6% | 3,526 | Mixed | 50% |
| Lung Squamous (LUSC) | 5% | 2,938 | Mixed | 75% |
| Kidney (KIRC) | 3% | 2,454 | — | — |
| Stomach (STAD) | 7% | 1,882 | Mixed | 45% |
| Head/Neck (HNSC) | 3% | 1,753 | — | — |
| Uterine Sarcoma (UCS) | 35% | 1,750 | Truncation (25%) | — |
| Liver (LIHC) | 3% | 1,236 | — | — |
| DLBC Lymphoma | 5% | 1,210 | — | — |
| Ovarian (OV) | 6% | 1,180 | Mixed | 85% |
| Esophageal (ESCA) | 5% | 1,118 | — | — |
| Pancreas (PAAD) | 1.5% | 996 | — | — |
| Thyroid (THCA) | 1% | 440 | — | — |

**Priority ranking (PIK3CB dependency × population):**

| Rank | Cancer Type | PIK3CB d | Est. Patients/yr | Trial Status |
|---|---|---|---|---|
| 1 | Breast | −2.229 (ROBUST) | 18,643 | Active (capivasertib, ipatasertib) |
| 2 | Melanoma | −1.497 (ROBUST) | 12,076 | **No trials** |
| 3 | Lung Adeno | −1.304 (NS) | 3,526 | **No trials** |
| 4 | Lung Squamous | −1.304 (NS) | 2,938 | **No trials** |
| 5 | Stomach | −1.090 (NS) | 1,882 | **No trials** |

**Active PTEN-selected trials (5 cancer types covered):**

| Cancer Type | Drug(s) | Phase | Trial |
|---|---|---|---|
| Breast | Capivasertib + fulvestrant | FDA-approved | CAPItello-291 |
| Breast | Ipatasertib + paclitaxel | Phase 3 | IPATential150-like |
| Prostate | Capivasertib + abiraterone | Phase 3 | CAPItello-281 |
| Prostate | Ipatasertib + abiraterone | Phase 3 | IPATential150 |
| Endometrial | Alpelisib + fulvestrant | Phase 2 | EPIK-E3 |
| GBM | GDC-0084 (paxalisib) | Phase 2 | GBM AGILE |

**Trial gaps — 14 cancer types with no PTEN-targeted trials:**

The largest unmet needs by patient volume are melanoma (~12,000/yr with PTEN loss, plus ROBUST PIK3CB dependency), colorectal (~10,700/yr), and bladder (~8,300/yr). Melanoma stands out as the most compelling gap: high PTEN-loss prevalence (12%), strong PIK3CB dependency (d = −1.50, ROBUST), afuresertib sensitivity trending (d = −0.99), and zero active PTEN-selected trials.

---

## Discussion

### The PI3Kβ Isoform Switch as a Therapeutic Principle

The central finding of this atlas is convergent CRISPR and pharmacological evidence for the p110α→p110β isoform switch in PTEN-null cancers. This is not a new biological concept — Wee et al. (2008) and Ni et al. (2012) demonstrated PI3Kβ dependency in PTEN-null models — but our analysis provides the first pan-cancer, large-scale validation using unbiased functional genomic data. The effect is massive in breast cancer (d = −2.23) and melanoma (d = −1.50), and the inverse PIK3CA finding provides an elegant internal control: the same genetic event (PTEN loss) simultaneously increases dependency on one PI3K isoform and decreases dependency on another.

The pharmacological mirror is equally compelling. Afuresertib (AKT inhibitor, downstream of both PI3K isoforms) shows PTEN selectivity, while inavolisib (PI3Kα-selective) shows inverse selectivity. This has immediate clinical implications: PI3Kα-selective agents like inavolisib and alpelisib, while effective for PIK3CA-mutant cancers, are not optimal — and may be counterproductive — for PTEN-lost tumors. The rational therapeutic strategy is PI3Kβ-selective inhibition (AZD8186, GSK2636771, GT220) or downstream AKT inhibition (capivasertib, afuresertib).

### Cancer-Type Heterogeneity

Not all PTEN-lost cancers show PIK3CB dependency equally. Breast and melanoma show ROBUST effects; CNS/brain does not (d = −0.32, NS). Uterine cancer, despite having the highest PTEN-loss frequency (53.5%), shows paradoxical lack of PIK3CB dependency (d = +0.38, NS) — yet afuresertib shows strong PTEN-selective sensitivity in uterus (d = −1.97). This dissociation between CRISPR-based dependency and drug sensitivity may reflect the fact that AKT inhibition captures the combined output of all PI3K isoforms, not just p110β. In uterine cancer, the exceptionally high PIK3CA co-mutation rate (50%) may create a "double-hit" where both p110α and p110β contribute to PI3K signaling, making AKT (the common downstream node) the better target.

### Implications for the CAPItello-281 Result

The CAPItello-281 trial (ASCO GU 2026) demonstrated capivasertib + abiraterone improves rPFS in PTEN-deficient mHSPC (33.2 vs 25.7 months, HR 0.81, p = 0.034). Our atlas data supports this result mechanistically: AKT inhibition is expected to work in PTEN-lost prostate cancer because it captures PI3K output regardless of which isoform is dominant. However, prostate is critically underrepresented in DepMap (3 PTEN-lost lines), so we could not directly validate PIK3CB dependency in prostate.

### ICMT and Novel Targets

A recent publication (PMID 41507994) identified ICMT as a synthetic lethal target in PTEN-deficient triple-negative breast cancer using combinatorial CRISPR screening. Our validation review (journal #1929) found ICMT present in DepMap across all 9 cancer types, with directionally consistent effects in 4 types (bowel d = −0.70, lymphoid d = −0.48, breast d = −0.38, ovary d = −0.30), but no cohort reached FDR significance. The ICMT inhibitor cysmethynil is not in PRISM 24Q2. Given that Phase 3 found zero genome-wide significant hits across all 18,435 genes, this null result for ICMT reflects general underpowering rather than ICMT-specific failure. ICMT remains a plausible combination target warranting dedicated follow-up with larger sample sizes or focused experimental validation.

---

## Limitations

1. **Statistical power is the dominant limitation.** The largest PTEN-lost cohort (CNS/Brain, n = 34) is below the ~64 per group estimated for 80% power at a medium effect size. Bowel (n = 8), ovary (n = 6), and esophagus/stomach (n = 5) are severely underpowered. This prevents discovery of novel, non-PI3K dependencies that may exist with moderate effect sizes.

2. **DepMap cell lines do not perfectly represent clinical tumors.** Thyroid is heavily overrepresented (DepMap 16% vs TCGA 1% PTEN loss), breast is moderately overrepresented (13.5% vs 6%). Prostate, the second-largest PTEN-loss patient population, is essentially absent.

3. **Drug screen coverage is incomplete.** The most clinically relevant agents — capivasertib, PI3Kβ-selective inhibitors (AZD8186, GSK2636771, GT220), and combination regimens — are not in PRISM 24Q2. Afuresertib is the only testable AKT inhibitor with on-target concordance.

4. **CRISPR knockout ≠ pharmacological inhibition.** Complete gene loss (CRISPR) is more severe than partial enzymatic inhibition (drugs). Effect sizes from CRISPR may overestimate clinical drug responses.

5. **Co-alteration confounding.** PTEN-lost lines often co-harbor TP53 mutations (69%) and RB1 loss (21%). While the leave-one-out analysis demonstrates that no single cell line drives the ROBUST hits, systematic confounding by co-alterations was not fully deconvolved.

---

## Next Steps

1. **PI3Kβ-selective inhibitor trials in melanoma.** Melanoma is the highest-priority trial gap: ROBUST PIK3CB dependency (d = −1.50), ~12,000 US patients/year with PTEN loss, afuresertib sensitivity trending (d = −0.99), and zero active PTEN-targeted trials. GT220 or AZD8186 basket trials should include a melanoma cohort.

2. **Expanded DepMap representation.** Prostate cancer (3 PTEN-lost lines vs ~57,660 patients/year) and bladder cancer (not qualifying, ~8,300 patients/year) need more PTEN-lost cell lines to power dependency analyses in these clinically important cancer types.

3. **ICMT and combination target validation.** Focused CRISPR screens in isogenic PTEN-knockout models could test ICMT and other suggestive targets with adequate power. The 49 unresolved PRISM combination candidates need compound identity annotation.

4. **Integration with CAPItello-281 biomarker data.** As prostate-specific data emerges from CAPItello-281 and IPATential150, cross-referencing with DepMap dependency patterns could identify predictive biomarkers beyond PTEN status alone.

5. **PIK3CA co-mutation stratification.** Endometrial cancer's high PIK3CA co-mutation rate (50%) may define a subset requiring dual PI3Kα + AKT inhibition. Dedicated analysis of double-hit (PTEN loss + PIK3CA mutation) versus single-hit vulnerabilities is warranted.

---

## References

- Wee S, et al. PTEN-deficient cancers depend on PIK3CB. *Proc Natl Acad Sci*. 2008.
- Ni J, et al. Functional characterization of an isoform-selective inhibitor of PI3K-p110β as a potential anticancer agent. *Cancer Discov*. 2012.
- Juric D, et al. Capivasertib plus fulvestrant in PIK3CA/AKT1/PTEN-altered HR+/HER2- breast cancer (CAPItello-291). *NEJM*. 2023.
- CAPItello-281 Phase 3 results. Capivasertib + abiraterone in PTEN-deficient mHSPC. ASCO GU 2026.
- Sweeney CJ, et al. Ipatasertib plus abiraterone and prednisolone in mCRPC (IPATential150). *Lancet Oncol*. 2021.
- Andre F, et al. Alpelisib for PIK3CA-mutated, HR+/HER2- breast cancer (SOLAR-1). *NEJM*. 2019.
- Turner S, et al. Inavolisib plus fulvestrant/palbociclib in PIK3CA-mutant breast cancer (INAVO120). *NEJM*. 2026.
- ICMT synthetic lethality in PTEN-deficient TNBC. *Exp Hematol Oncol*. 2025. PMID 41507994.
- DepMap 25Q3 Release. Broad Institute. 2025.
- PRISM Repurposing 24Q2. Broad Institute. 2024.
- TCGA PanCancer Atlas. Cell. 2018.
- American Cancer Society. Cancer Facts & Figures 2024.
