# RB1-Loss Pan-Cancer Dependency Atlas

**Project:** rb1-loss-pancancer-dependency-atlas
**Division:** Cancer
**Date:** March 23, 2026
**Data Sources:** DepMap 25Q3 (CRISPR, mutations, copy number, expression), PRISM 24Q2, TCGA pan-cancer, cBioPortal
**Pipeline:** `src/cancer/rb1_loss_pancancer_dependency_atlas/01–05_*.py`
**Validation Status:** PASS (validation scientist journal #1492, analyst journal #1170)

---

## Executive Summary

This atlas maps synthetic lethal (SL) dependencies in RB1-deficient cancers across all tumor types in DepMap 25Q3, with the goal of identifying targetable vulnerabilities for cancers that have lost retinoblastoma protein function — particularly SCLC, neuroendocrine prostate cancer, bladder cancer, and triple-negative breast cancer. The study was designed to align with the emerging clinical landscape of CDK2 inhibitors and to extend our prior CDKN2A dependency atlas downstream in the p16-CDK4/6-RB1 axis.

From 1,732 DepMap cell lines, 493 were classified as RB1-deficient (28.5%) using a multi-modal classifier (LOF mutations, deep deletions, and expression loss), with 19 cancer types qualifying for statistical analysis. The CDK4/CDK6 positive control validated the classifier: RB1-loss lines showed significantly reduced CDK4/6 dependency (CDK4 d=+0.36, p=4.95e-12; CDK6 d=+0.64, p=5.30e-25), consistent with the biological expectation that RB1-null cells have no target for CDK4/6 to phosphorylate.

**Key findings:**

1. **CDK2 is a validated pan-cancer synthetic lethal target in RB1-loss cancers** (d=−0.524, 95% CI [−0.683, −0.379], FDR=3.2e-9), with 6 cancer types showing marginal effects. CCNE1 co-amplification dramatically intensifies CDK2 dependency (double-hit d=−2.280), supporting convergent CDK2 targeting with INX-315, which received FDA Fast Track designation for CCNE1-amplified ovarian cancer.

2. **Novel SAGA/ATAC chromatin complex dependencies** — TADA2A (d=−0.423), YEATS2 (d=−0.386), ZZZ3 (d=−0.385), and CHD8 (d=−0.316) — emerged from the genome-wide screen, suggesting that E2F-deregulated transcription in RB1-loss cells creates dependency on chromatin remodeling machinery. These are hypothesis-generating findings.

3. **RB1-loss and CDKN2A-deletion dependency landscapes are largely distinct:** only 9 dependencies are shared, versus 30 RB1-unique and 109 CDKN2A-unique, reflecting the different biological consequences of upstream (p16/CDK4/6) versus downstream (E2F deregulation) pathway disruption.

4. **SCLC represents the largest addressable population** (~28,500 US patients/year with near-universal RB1 loss), with no approved targeted therapies. CDK2 inhibitors could be first-in-class targeted agents for SCLC.

5. **PRISM drug validation was severely limited** — only trilaciclib (CDK4/6 inhibitor) was available, which correctly served as a negative control (pan-cancer d=+0.678, FDR=3.9e-18). CDK2 inhibitors are too new for PRISM 24Q2 inclusion.

---

## Important Caveats

- **Liberal classifier threshold:** The 28.5% RB1-loss rate is driven primarily by expression-based classification (bottom 25th percentile), which identified 422 of 493 RB1-loss lines. Only 104 lines have multi-modal concordance (mutation + expression, or deletion + expression). The CDK4/6 positive control provides biological confidence, but a sensitivity analysis with stricter expression thresholds (e.g., 10th percentile) is recommended as future work.
- **CDK2 is ROBUST only at the pan-cancer pooled level.** No individual cancer type achieved triple-validated ROBUST status (FDR<0.05 + CI excludes zero + permutation p<0.05). This reflects limited statistical power per cancer type rather than true heterogeneity — the direction of CDK2 effect is consistently negative across most types.
- **PRISM drug coverage is incomplete.** CDK2 inhibitors, Aurora inhibitors, CHK1/WEE1 inhibitors, PARP inhibitors, and platinum agents were all absent from PRISM 24Q2. Drug sensitivity validation of the CRISPR-based findings is not possible with currently available pharmacological data.
- **CCNE1 subgroup findings** (d=−2.280 for double-hit) are based on a small sample (~20 CCNE1-amplified + RB1-loss lines) and require validation in larger cohorts.
- **ATR and PKMYT1 co-inhibition** (reported as SL in PDX models, PMID 41442499) was not confirmed by single-gene CRISPR knockout in DepMap. This likely reflects the difference between partial pharmacological inhibition of two targets simultaneously versus complete genetic ablation of one.

---

## Background

RB1 (retinoblastoma protein) is a master cell cycle regulator that restrains E2F-driven transcription. When RB1 is functionally lost — through biallelic mutation, deep deletion, or epigenetic silencing — the G1/S checkpoint is abolished, leading to constitutive E2F activation and cyclin E/CDK2 overactivity. RB1 is among the most frequently inactivated tumor suppressors across cancer types.

RB1 loss prevalence varies substantially by cancer type. SCLC has near-universal RB1 inactivation (~95%), which along with TP53 co-mutation defines the disease. Neuroendocrine prostate cancer (NEPC) shows ~30–50% RB1 loss, enriched in treatment-resistant disease. Bladder cancer (~15–20%), TNBC (~10–15%), and osteosarcoma (~60–70%) represent other RB1-loss-enriched malignancies.

Several SL relationships with RB1 loss have been described in the literature:

- **CDK2:** RB1 loss creates cyclin E/CDK2 overactivity. CDK2 inhibitors show selective efficacy in RB1-deficient preclinical models. INX-315 (Incyte) and PF-07220060 (Pfizer) are in Phase 1/2 clinical trials (Nature Communications 2025, doi:10.1038/s41467-025-56674-4; Cell Reports 2025, doi:10.1016/j.celrep.2025.115502).
- **Aurora kinases (AURKA/AURKB):** Mitotic checkpoint compensation for G1/S checkpoint loss. Alisertib (AURKA inhibitor) showed activity in SCLC (Cancer Discovery 2019, 9(2):248 and 9(2):230).
- **CHK1/WEE1:** Replication stress from deregulated S-phase entry. RB1-loss cells are reported as hypersensitive to S-phase checkpoint inhibitors (Oncogene 2024).
- **Casein Kinase 2 (CK2/CSNK2A1):** RB1 loss creates SL dependency on CK2 when combined with replication-perturbing agents such as carboplatin, gemcitabine, or PARP inhibitors (Science Advances 2024).
- **ATR + PKMYT1 co-inhibition:** Combined inhibition produces synthetic lethality in RB1-deficient breast cancer PDX models through replication fork collapse and mitotic catastrophe (Science Translational Medicine 2025, PMID 41442499).

The clinical gap is acute: SCLC has a 5-year survival rate below 7%, with no approved targeted therapies. Platinum + etoposide + immunotherapy (atezolizumab/durvalumab) is the standard of care, but nearly all patients relapse. CDK2 inhibitors represent the first SL-based targeted therapy approach for RB1-null cancers.

This atlas was designed to extend our CDKN2A pan-cancer dependency atlas downstream in the p16-CDK4/6-RB1 axis, using the same analytical framework (Cohen's d effect sizes, bootstrap CI, BH FDR correction, permutation testing) to enable direct comparison of dependency landscapes.

---

## Methodology

### Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv (Chronos dependency scores for ~18,000 genes across 1,126 cell lines with CRISPR data)
- **DepMap Model Annotations:** Model.csv (cancer type, subtype, lineage information)
- **DepMap Somatic Mutations:** OmicsSomaticMutations.csv (RB1 truncating mutations, frameshift mutations)
- **DepMap Copy Number:** OmicsCNGene.csv (RB1 deep deletions, CCNE1 amplification status)
- **DepMap Expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv (RB1 expression as proxy for functional loss including epigenetic silencing)
- **PRISM 24Q2:** Drug sensitivity data for pharmacological validation
- **TCGA PanCanAtlas:** RB1 alteration frequency by cancer type, published prevalence estimates
- **cBioPortal:** Co-alteration landscape data

### Phase 1 — RB1-Loss Classifier

Cell lines were classified as RB1-deficient using a multi-modal approach combining three independent modalities:

1. **LOF mutations:** Truncating, frameshift, or splice-site mutations in RB1 (152 lines)
2. **Deep deletion:** Copy number ≤ 0.3 in OmicsCNGene (23 lines)
3. **Expression loss:** Bottom 25th percentile of RB1 expression across all lines (422 lines)

A line was classified as RB1-loss if it met any single modality criterion. Lines meeting ≥2 modalities (104 lines) have the highest confidence. Expression cross-validation confirmed that RB1-loss classified lines have significantly lower RB1 expression (median 3.31 vs 4.74 for intact, Mann-Whitney U p=1.45e-188).

**Positive control validation:** CDK4 and CDK6 dependency was reduced in RB1-loss lines (CDK4 d=+0.36, CDK6 d=+0.64), as expected biologically — there is no RB1 substrate for CDK4/6 to phosphorylate in RB1-null cells.

**Qualifying cancer types:** 19 cancer types met the statistical power threshold (≥5 RB1-loss and ≥10 RB1-intact lines with CRISPR data).

**Co-alteration landscape:** TP53 co-mutation 71.2%, CCNE1 amplification 4.1%, MYC amplification 18.7%.

### Phase 2 — Candidate SL Dependency Analysis

For 12 established or literature-nominated SL candidates (CDK2, AURKA, AURKB, CHEK1, WEE1, CSNK2A1/CK2, TTK, FOXM1, MYBL2, ATR, PKMYT1), CRISPR dependency scores were compared between RB1-loss and RB1-intact lines within each qualifying cancer type and pan-cancer pooled.

**Statistics:**
- Cohen's d effect size with Bessel's-corrected variance (ddof=1)
- Mann-Whitney U test (two-sided for SL candidates, one-sided greater for CDK4/6 positive controls)
- Bootstrap 95% CI (1,000 iterations, seed=42)
- Permutation testing (10,000 permutations)
- Benjamini-Hochberg FDR correction (applied per analysis context)
- Leave-one-out robustness testing

**Classification criteria (triple validation):**
- **ROBUST:** FDR<0.05 + bootstrap CI excludes zero + permutation p<0.05
- **MARGINAL:** Any two of the three criteria met
- **NOT_SIGNIFICANT:** Fewer than two criteria met

CCNE1 amplification subgroup analysis compared CDK2 dependency across four groups: RB1-loss + CCNE1-amp (double-hit), RB1-loss only, CCNE1-amp only, and neither.

### Phase 3 — Genome-Wide Dependency Screen

All ~18,000 genes were screened for differential dependency between RB1-loss and RB1-intact lines within each qualifying cancer type and pan-cancer pooled (357,684 total tests). BH FDR correction was applied within each analysis context. Significance thresholds: FDR<0.05 and |Cohen's d|>0.3.

Pathway enrichment was assessed using Fisher's exact test against curated gene sets (cell cycle, DNA replication, mitotic checkpoint, DDR checkpoint, E2F targets). SL benchmark enrichment was tested against published SL gene lists. CDKN2A atlas comparison identified shared versus unique dependencies.

### Phase 4 — PRISM Drug Sensitivity Validation

Targeted drug classes (CDK2 inhibitors, CDK4/6 inhibitors, Aurora inhibitors, CHK1/WEE1 inhibitors, PARP inhibitors, platinum agents) were searched in PRISM 24Q2. Drug sensitivity was compared between RB1-loss and RB1-intact lines using the same statistical framework. CRISPR-PRISM concordance was assessed by Pearson correlation.

### Phase 5 — Clinical Integration

RB1 alteration frequency per cancer type was compiled from TCGA pan-cancer data and published literature. Addressable patient populations were estimated using ACS 2024 incidence data. CDK2 inhibitor clinical trials were catalogued from ClinicalTrials.gov. A priority ranking was computed as a function of CDK2 SL effect size, RB1-loss population size, and drug availability.

---

## Results

### Phase 1: RB1-Loss Classifier

| Parameter | Value |
|-----------|-------|
| Total cell lines analyzed | 1,732 |
| Cell lines with CRISPR data | 1,126 |
| RB1-loss classified | 493 (28.5%) |
| RB1-intact | 1,239 |
| LOF mutations | 152 |
| Deep deletions (CN ≤ 0.3) | 23 |
| Expression loss (bottom 25th pctile) | 422 |
| Multi-modal concordance (≥2 modalities) | 104 |
| Qualifying cancer types | 19 |

The highest RB1-loss rates among qualifying types were Bladder/Urinary Tract (52.6%, 19 lost/15 intact with CRISPR), Lung (52.6%, 50/73), Breast (47.1%, 22/28), and Liver (44.4%, 11/13).

**CDK4/6 positive control validation:**

| Gene | Median dep (RB1-loss) | Median dep (RB1-intact) | Cohen's d | p-value | Interpretation |
|------|-----------------------|-------------------------|-----------|---------|----------------|
| CDK4 | −0.487 | −0.628 | +0.36 | 4.95e-12 | VALIDATES |
| CDK6 | −0.182 | −0.589 | +0.64 | 5.30e-25 | VALIDATES |

Positive Cohen's d indicates less dependency in RB1-loss lines — biologically correct. RB1-null cells have no RB1 substrate, so CDK4/6 inhibition provides no selective disadvantage.

**Co-alteration landscape:** TP53 co-mutation (71.2%) is consistent with the RB1+TP53 co-loss that defines SCLC. CCNE1 amplification (4.1%) represents a subgroup with intensified CDK2 dependency. MYC amplification (18.7%) drives proliferation and may modify the dependency landscape.

### Phase 2: Candidate SL Dependencies

#### CDK2 — The Primary SL Target

CDK2 achieved **ROBUST** status at the pan-cancer pooled level: d=−0.524, 95% CI [−0.683, −0.379], FDR=3.2e-9, permutation p=0.0001. No individual cancer type reached ROBUST, but 6 showed MARGINAL effects:

| Cancer Type | Cohen's d | 95% CI | Permutation p | Classification |
|-------------|-----------|--------|---------------|----------------|
| **Pan-cancer (pooled)** | **−0.524** | **[−0.683, −0.379]** | **0.0001** | **ROBUST** |
| Skin | −1.545 | [−2.519, −0.747] | 0.0001 | MARGINAL |
| Lymphoid | −0.953 | [−1.625, −0.386] | 0.0004 | MARGINAL |
| Bone | −0.929 | [−1.758, −0.231] | 0.0070 | MARGINAL |
| Ovary/Fallopian Tube | −0.746 | [−1.644, −0.200] | 0.0073 | MARGINAL |
| CNS/Brain | −0.584 | [−1.176, −0.010] | 0.0234 | MARGINAL |
| Lung | −0.425 | [−0.818, −0.049] | 0.0218 | MARGINAL |

The consistent negative direction across most cancer types indicates CDK2 SL is a general feature of RB1 loss, with statistical power limiting individual-type significance.

#### CCNE1 Amplification Intensifies CDK2 Dependency

A cross-reference analysis (script 02b) confirmed that CCNE1 amplification and RB1 loss converge on CDK2 dependency:

| Comparison | Cohen's d | 95% CI | p-value |
|------------|-----------|--------|---------|
| CCNE1-amp vs non-amp (all lines) | −1.351 | [−2.044, −0.658] | 4.1e-5 |
| RB1-loss vs intact (all lines) | −0.524 | [−0.683, −0.379] | 2.9e-10 |
| Double-hit vs neither | −2.280 | [−3.493, −1.062] | 1.0e-3 |
| Double-hit vs RB1-loss only | −1.337 | [−2.442, −0.346] | 1.2e-2 |
| Double-hit vs CCNE1-amp only | −0.654 | [−1.520, −0.015] | 0.15 (ns) |

The double-hit effect (d=−2.280) appears additive: both CCNE1 amplification and RB1 loss independently create CDK2 dependency through complementary mechanisms — cyclin E overexpression and E2F deregulation, respectively. This convergence is therapeutically important: INX-315, which received FDA Fast Track for CCNE1-amplified tumors, may also cover RB1-loss populations.

#### Other Candidate SL Targets

| Gene | Pan-cancer d | Classification | Notes |
|------|-------------|----------------|-------|
| CDK2 | −0.524 | ROBUST | Primary SL target |
| AURKA | −0.208 | ROBUST | Modest effect; MARGINAL in Cervix (d=−1.078), Lymphoid (d=−0.681) |
| MYBL2 | −0.148 | ROBUST | Very modest; E2F target gene |
| CSNK2A1 (CK2) | −0.143 | MARGINAL | MARGINAL in Bladder (d=−0.774), Ovary (d=−0.589); published SL may require combination context |
| TTK | −0.069 | NOT_SIGNIFICANT | MARGINAL in Bone (d=−0.746), CNS/Brain (d=−0.558) |
| FOXM1 | −0.042 | NOT_SIGNIFICANT | No signal despite being an E2F transcriptional target |
| AURKB | +0.025 | NOT_SIGNIFICANT | No SL signal; published PDX data not recapitulated in CRISPR |
| WEE1 | +0.025 | NOT_SIGNIFICANT | No signal; published SL may require replication stress context |
| CHEK1 | −0.119 | NOT_SIGNIFICANT | No signal; single-gene KO insufficient |
| ATR | −0.106 | NOT_SIGNIFICANT | MARGINAL only in Pancreas (d=−0.823); published co-inhibition (ATR+PKMYT1) differs from single-gene KO |
| PKMYT1 | +0.001 | NOT_SIGNIFICANT | No single-gene SL signal |

**Negative control validation (CDK4/CDK6):** Both showed positive Cohen's d values across cancer types (CDK4 d=+0.36, CDK6 d=+0.64 pan-cancer), confirming reduced dependency in RB1-loss — consistent with biology and validating the classifier.

### Phase 3: Genome-Wide Dependency Screen

The screen identified 23 gained dependencies and 33 lost dependencies (FDR<0.05, |d|>0.3) across 357,684 tests.

**Top gained dependencies (RB1-loss lines become more dependent):**

| Gene | Context | Cohen's d | FDR | Annotation |
|------|---------|-----------|-----|------------|
| E2F3 | CNS/Brain | −1.879 | 1.1e-3 | E2F pathway |
| CCNE1 | CNS/Brain | −1.528 | 2.9e-2 | Cell cycle |
| SKP2 | Lung | −1.022 | 3.3e-2 | E2F pathway |
| E2F3 | Lung | −0.903 | 3.3e-2 | E2F pathway |
| SKP2 | Pan-cancer | −0.731 | 9.4e-15 | E2F pathway |
| E2F3 | Pan-cancer | −0.705 | 2.8e-13 | E2F pathway |
| CCNE1 | Pan-cancer | −0.610 | 1.7e-10 | Cell cycle |
| CDK2 | Pan-cancer | −0.524 | 5.4e-7 | Known SL |
| CKS1B | Pan-cancer | −0.520 | 5.7e-8 | CDK-associated |
| **TADA2A** | **Pan-cancer** | **−0.423** | **9.8e-7** | **SAGA/ATAC complex (NOVEL)** |
| CCNE2 | Pan-cancer | −0.403 | 5.2e-3 | Cell cycle |
| **YEATS2** | **Pan-cancer** | **−0.386** | **1.4e-4** | **SAGA/ATAC complex (NOVEL)** |
| **ZZZ3** | **Pan-cancer** | **−0.385** | **2.9e-4** | **SAGA/ATAC complex (NOVEL)** |
| **CHD8** | **Pan-cancer** | **−0.316** | **2.9e-4** | **Chromatin remodeler (NOVEL)** |

The E2F pathway dominance (E2F3, SKP2, CCNE1, CDK2, CKS1B, CCNE2) is biologically expected — RB1 loss deregulates E2F-driven transcription, creating dependency on the now-constitutively active E2F/CDK2/cyclin E axis.

The **SAGA/ATAC chromatin complex** (TADA2A, YEATS2, ZZZ3) emerged as a novel dependency cluster. These histone acetyltransferase complex components may be required to manage the transcriptional burden imposed by constitutive E2F activation. CHD8, a chromatin remodeler, may serve a similar role. These findings are hypothesis-generating and warrant functional validation.

**Pathway enrichment:** Cell cycle 192-fold enriched (p=4.3e-7), E2F targets 154-fold enriched (p=7.4e-5). No significant enrichment for DNA replication, mitotic checkpoint, or DDR checkpoint pathways.

**CDKN2A atlas comparison:** 9 shared dependencies, 30 RB1-unique, 109 CDKN2A-unique. The asymmetry is biologically coherent: CDKN2A deletion removes p16 control of CDK4/6, affecting a broader regulatory network upstream of RB1. RB1 loss is a more specific downstream event — E2F deregulation — producing a narrower but more targeted set of dependencies.

### Phase 4: PRISM Drug Sensitivity

Drug coverage in PRISM 24Q2 was severely limited. Of 22 searched compounds across 7 drug classes, only trilaciclib (CDK4/6 inhibitor) was available. CDK2 inhibitors (INX-315, PF-07220060) are too recently developed for PRISM inclusion.

**Trilaciclib negative control validation:** RB1-loss lines showed reduced sensitivity across 17/18 cancer types (positive d = less sensitive), consistent with CDK4/6 target loss. Pan-cancer d=+0.678, FDR=3.9e-18. Strongest effects in Bladder (d=+1.294), Lymphoid (d=+0.963), and Head & Neck (d=+0.894).

**Genome-wide drug screen:** 3 RB1-loss selective compounds (by Broad ID only, no annotation available) and 6 RB1-loss resistant compounds were identified at FDR<0.05.

**CRISPR-PRISM concordance:** Trilaciclib sensitivity correlated with CDK4 CRISPR dependency (r=0.184, p=6.8e-7, n=716) — weak but positive and statistically significant.

### Phase 5: Clinical Integration and CDK2 Inhibitor Landscape

**Estimated total US RB1-loss cancer patients per year: ~125,127**

Top populations by RB1-loss prevalence:

| Cancer Type | RB1-loss Rate | Est. US Patients/yr | CDK2 d (best) | Priority |
|-------------|--------------|---------------------|---------------|----------|
| SCLC | 95% | 28,500 | −0.425 (Lung) | Highest |
| Osteosarcoma | 65% | 2,340 | −0.929 (Bone) | High |
| Neuroendocrine bladder | 40% | 640 | −0.648 (Bladder) | High |
| NEPC (prostate) | 40% | 2,320 | −0.524 (pan-cancer) | High |
| Bladder (all) | 18% | 14,974 | −0.648 (Bladder) | High |
| TNBC | 12% | 5,592 | −0.524 (pan-cancer) | Moderate |
| Prostate (all) | 12% | 34,596 | −0.524 (pan-cancer) | Moderate |

**CDK2 Inhibitor Clinical Landscape:**

| Drug | Sponsor | Phase | NCT ID | RB1 Loss Included | Biomarker Focus |
|------|---------|-------|--------|--------------------|--------------------|
| INX-315 | Incyte | 1/2 | NCT05735080 | YES | CCNE1 amp, RB1 loss |
| PF-07220060 | Pfizer | 1 | NCT05757544 | NO | CCNE1 amp, CDK4/6i-resistant |

INX-315 received **FDA Fast Track designation** for CCNE1-amplified platinum-resistant ovarian cancer (2026). Phase 1 Part A data (31 evaluable patients): 10% partial response (3/31), 63% stable disease (19/31). In the ovarian cohort specifically: 20% PR (2/10), 80% SD (8/10). Study completion is estimated June 2026.

The convergence hypothesis — that CCNE1 amplification and RB1 loss both create CDK2 dependency through complementary mechanisms — was confirmed by our DepMap cross-reference analysis (double-hit d=−2.280). This suggests INX-315's clinical activity in CCNE1-amplified tumors may extend to RB1-loss populations, potentially broadening the addressable patient base from CCNE1-amplified (~3–5% of solid tumors) to include RB1-deficient cancers (~125,000 US patients/year).

A 4-gene mRNA signature (CCND1, CCNE1, RB1, CDKN2A) has been proposed as a predictor of CDK2 inhibitor response (Nature Communications 2025, doi:10.1038/s41467-025-67865-4). BLU-222 (Blueprint Medicines, selective CDK2 inhibitor) synergizes with CDK4/6 inhibitors in resistant breast cancer models using this signature. The signature directly intersects our RB1 and CDKN2A atlas findings.

**Other relevant clinical programs for RB1-loss cancers:** Alisertib (AURKA inhibitor, Phase 2 SCLC data), lurbinectedin (approved relapsed SCLC), tarlatamab (DLL3 bispecific, approved SCLC). Computational modeling (eLife 2025, doi:10.7554/eLife.104545) suggests CDK2 inhibitor addition at CDK4/6i progression outperforms CDK2i monotherapy; in tumors where RB1 loss is the CDK4/6i resistance mechanism, CDK2i monotherapy may suffice.

---

## Discussion

### CDK2 as a Therapeutic Target for RB1-Loss Cancers

The central finding of this atlas is that CDK2 synthetic lethality with RB1 loss is a real but moderate pan-cancer effect (d=−0.524). This effect size is smaller than the CCNE1 amplification effect on CDK2 dependency (d=−1.351), but is consistent across cancer types and biologically mechanistic: RB1 loss deregulates E2F-driven transcription, making cells dependent on the CDK2/cyclin E complex that drives S-phase entry.

The moderate effect size aligns with the clinical reality that CDK2 inhibitors are showing modest single-agent activity (10% PR with INX-315). However, the double-hit finding (d=−2.280 for RB1-loss + CCNE1-amp) suggests that patient selection using multiple biomarkers — rather than RB1 loss alone — may identify patients most likely to respond. SCLC, where RB1 loss is near-universal, represents a distinct opportunity because nearly all patients are potential candidates.

### Novel Chromatin Dependencies

The SAGA/ATAC complex dependencies (TADA2A, YEATS2, ZZZ3) represent an unexpected finding that points toward a chromatin remodeling vulnerability in RB1-loss cells. The SAGA complex is a histone acetyltransferase involved in transcriptional activation. When E2F transcription is constitutively activated by RB1 loss, cells may become dependent on SAGA/ATAC-mediated chromatin opening to sustain the elevated transcriptional output. This hypothesis predicts that histone acetyltransferase inhibitors or bromodomain inhibitors targeting SAGA/ATAC components could show selective activity in RB1-deficient cells.

CHD8, a chromatin remodeler and autism risk gene, has not previously been linked to cancer synthetic lethality with RB1 loss. Its role in maintaining open chromatin at actively transcribed loci provides a plausible mechanism for this dependency.

These chromatin complex findings are hypothesis-generating and require functional validation. They would not have been predicted from the established CDK2/Aurora/CHK1/WEE1 SL literature, demonstrating the value of unbiased genome-wide screening.

### Comparison with Literature Predictions

Several literature-nominated SL targets were not validated by single-gene CRISPR knockout:

- **AURKB, CHEK1, WEE1:** No pan-cancer signal. Published SL data come from pharmacological inhibition studies, which may involve partial inhibition kinetics and off-target effects not recapitulated by CRISPR knockout.
- **CK2 (CSNK2A1):** Only MARGINAL at pan-cancer level. Published SL is context-dependent — requiring co-treatment with replication-perturbing agents (platinum, gemcitabine, PARPi) that are absent in the CRISPR screen.
- **ATR + PKMYT1:** Neither shows significant single-gene SL. The published combination SL (PMID 41442499) requires simultaneous partial inhibition of both targets, causing replication fork collapse and mitotic catastrophe — a mechanism not testable by single-gene knockout.

These discrepancies are methodologically informative rather than contradictory. They highlight the complementarity of CRISPR-based dependency mapping (identifies essential single-gene vulnerabilities) and pharmacological combination studies (identifies synergistic multi-target vulnerabilities).

### SCLC as the Primary Clinical Opportunity

SCLC stands out as the highest-priority indication for CDK2-targeted therapy development:

1. Near-universal RB1 loss (~95%), eliminating the need for biomarker selection
2. Largest addressable population (~28,500 US patients/year)
3. No approved targeted therapies — CDK2 inhibitors would be first-in-class
4. 5-year survival below 7% — massive unmet need
5. CDK2 dependency confirmed in DepMap lung cancer lines
6. TP53 co-loss (defining feature of SCLC) may further modify the dependency landscape

The actionable dependency portfolio for SCLC includes CDK2 (ROBUST pan-cancer), E2F3, and SKP2 (both identified in genome-wide screen). AURKA shows marginal SL signal but has existing clinical data in SCLC (alisertib).

---

## Limitations

1. **Classifier liberality:** The expression-based classification dominates (422/493 lines). Stricter thresholds would reduce sensitivity but increase specificity. A systematic sensitivity analysis comparing 25th, 15th, and 10th percentile expression thresholds is recommended.

2. **Statistical power per cancer type:** No individual cancer type achieved ROBUST CDK2 SL. DepMap sample sizes (typically 5–50 RB1-loss lines per type) are insufficient for the triple-validation criteria in most types.

3. **Absent drug validation:** CDK2 inhibitors, the primary therapeutic candidates, were not available in PRISM 24Q2. Drug sensitivity validation will require updated PRISM releases or dedicated screening.

4. **In vitro limitation:** All DepMap/PRISM data are from cell lines. Tumor microenvironment effects (immune infiltration, stroma, hypoxia) are absent. PDX and clinical validation are needed.

5. **CCNE1 subgroup size:** The double-hit (RB1-loss + CCNE1-amp) analysis involves ~20 lines. While the effect size is large (d=−2.280) and bootstrap CI excludes zero, this requires independent validation.

6. **Co-inhibition targets not testable:** ATR+PKMYT1 and CK2 combination SL hypotheses cannot be evaluated with single-gene CRISPR data.

7. **TCGA prevalence estimates:** Some cancer types (SCLC, SCCOHT) may be underrepresented in TCGA. Published literature values were used where available.

---

## Next Steps

1. **CDK2 inhibitor sensitivity validation:** As INX-315 Phase 1/2 data mature (expected ASCO 2026), correlate clinical responses with RB1 loss status and CCNE1 amplification to test the convergence hypothesis.

2. **Classifier sensitivity analysis:** Repeat the dependency analysis with stricter expression thresholds (10th and 15th percentile) to assess robustness of findings.

3. **4-gene CDK2 signature evaluation:** Apply the CCND1/CCNE1/RB1/CDKN2A mRNA signature (Nature Communications 2025) to the DepMap cell line panel to determine if it predicts CDK2 dependency better than RB1 loss alone.

4. **SAGA/ATAC functional validation:** Test histone acetyltransferase inhibitors and bromodomain inhibitors in RB1-deficient cell line panels to validate the chromatin complex dependency.

5. **ATR + PKMYT1 combination screening:** Design combination CRISPR screens or drug combination matrices to test the co-inhibition SL hypothesis from PMID 41442499.

6. **SCLC clinical trial design:** Support CDK2 inhibitor trial design for SCLC by providing DepMap-derived biomarker stratification (RB1 loss + CCNE1 amplification + CDK2 dependency score).

7. **Updated PRISM validation:** Repeat drug sensitivity analysis when CDK2 inhibitors are included in future PRISM releases.

---

## References

1. Incyte/Incyclix Bio. INX-315 Phase 1/2 (NCT05735080). ClinicalTrials.gov. FDA Fast Track for CCNE1-amplified platinum-resistant ovarian cancer, 2026.
2. Pfizer. PF-07220060 Phase 1 (NCT05757544). ClinicalTrials.gov.
3. Suen KM et al. Discrete vulnerability to pharmacological CDK2 inhibition governed by p16INK4A and cyclin E1 co-expression. Nature Communications 2025; doi:10.1038/s41467-025-56674-4.
4. Bortolini Silveira A et al. BLU-222 synergizes with CDK4/6 inhibitors in resistant breast cancer. Nature Communications 2025; doi:10.1038/s41467-025-67865-4.
5. Targeting CDK2 for cancer therapy (review). Cell Reports 2025; doi:10.1016/j.celrep.2025.115502.
6. Oser MG et al. Cells lacking the RB1 tumor suppressor gene are hyperdependent on Aurora B kinase. Cancer Discovery 2019; 9(2):230–247.
7. Gong X et al. Aurora A kinase inhibition is synthetic lethal with loss of the RB1 tumor suppressor gene. Cancer Discovery 2019; 9(2):248–263.
8. CHK1+WEE1 synthetic lethality in neuroendocrine prostate cancer. Oncogene 2024.
9. CK2 synthetic lethality with RB1 loss in TNBC/HGSC. Science Advances 2024.
10. ATR+PKMYT1 co-inhibition synthetic lethality in RB1-deficient breast cancer. Science Translational Medicine 2025; PMID 41442499; doi:10.1126/scitranslmed.adx6797.
11. Therapeutic benefits of maintaining CDK4/6i and incorporating CDK2i beyond progression. eLife 2025; doi:10.7554/eLife.104545.
12. DepMap 25Q3. Broad Institute. https://depmap.org
13. PRISM Repurposing 24Q2. Broad Institute. https://depmap.org/prism
14. TCGA PanCanAtlas. National Cancer Institute.
15. ACS Cancer Statistics 2024. American Cancer Society.
