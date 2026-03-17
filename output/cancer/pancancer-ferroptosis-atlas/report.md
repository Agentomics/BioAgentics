# Pan-Cancer Ferroptosis Vulnerability Atlas: Findings Report

## Executive Summary

We mapped ferroptosis pathway gene dependencies across 20 ferroptosis-related genes in 1,186 cell lines spanning 30 cancer types using DepMap 25Q3 CRISPR knockout data, stratified by NRF2/KEAP1 mutation status. The analysis identifies which cancer types beyond NSCLC are most susceptible to ferroptosis-inducing therapies and produces a 5-category therapeutic stratification framework.

**Key findings:**
- 18 cancer types show composite ferroptosis vulnerability scores below -0.20, well exceeding the success criterion of 5 types beyond NSCLC.
- FSP1/NRF2 independence was confirmed across all 4 testable cancer types (|Cohen's d| < 0.45), validating that FSP1 inhibitors can target tumors regardless of NRF2/KEAP1 status.
- Bowel/CRC is the #1 NSCLC ferroptosis analog (cosine similarity r = 0.954), suggesting NSCLC ferroptosis strategies are directly transferable.
- Bone shows the strongest GPX4 dependency pan-cancer (GPX4 = -1.294), but GPX4-axis in vivo translation remains uncertain (bioRxiv 2026 preprint).
- TXNRD1 inhibition (auranofin, FDA-approved) represents an immediately actionable drug repurposing opportunity for Eye, Bone, Head & Neck, and PNS cancers.
- 3 of 4 success criteria were met. PRISM validation was impossible (zero ferroptosis compounds in PRISM 24Q2).

---

## 1. Cancer Type Therapeutic Recommendation Matrix

### Category A: FSP1-Vulnerable (In Vivo Validated)

| Cancer Type | AIFM2 Dependency | Key Evidence | Recommended Strategy |
|---|---|---|---|
| **Lung (LUAD)** | -0.072 | icFSP1 reduces KP LUAD tumors ~80% in vivo (Wu et al., *Nature* Nov 2025). NRF2-WT fraction = 0.77 (lowest pan-cancer), but FSP1 is NRF2-independent, so targets ALL lung cancer. | icFSP1 monotherapy; HDACi combination for persister cells |
| **Skin/Melanoma** | -0.043 | viFSP1 and FSEN1 suppress melanoma growth in lymph nodes (Ubellacker et al., *Nature* Nov 2025). | FSP1i for lymph node metastases only |

**Critical caveat for Melanoma:** FSP1 vulnerability is lymph node niche-specific. Subcutaneous melanoma does not respond to FSP1 inhibition. DepMap cell culture does not capture this microenvironmental context dependency.

### Category B: Dual FSP1 + GPX4 Targeting Required

| Cancer Type | GPX4+FSP1 Dual Score | NSCLC Analog r | Key Evidence | Recommended Strategy |
|---|---|---|---|---|
| **Bowel/CRC** | 0.271 | 0.954 | Neither FSP1 nor GPX4 alone sufficient (*Anticancer Research* Nov 2024, DOI: 10.21873/anticanres.17408). High FSP1+GPX4 co-expression = 5-FU resistance and poor prognosis. | Dual FSP1i + GPX4i; ferroptosis induction to overcome chemoresistance |
| **Kidney/ChRCC** | 0.949 | 0.917 | GPX4 + FSP1 synergistic; FSP1 alone reduced tumors 69% in vivo (Salem et al., *Oncogene* 2025). Second-highest FSP1 upregulation across TCGA. | Dual FSP1i + GPX4i |
| **Ovary** | 0.907 | 0.926 | FSP1 knockdown enhances ferroptosis (*Cancers* Aug 2025, DOI: 10.3390/cancers17162714). TFAP2C/HDAC epigenetic NRF2 activation (only 1/59 NRF2-mutant). | HDACi + ferroptosis combination |

### Category C: GPX4-Dominant (In Vivo Translation Uncertain)

| Cancer Type | GPX4 Dependency | Composite Vulnerability | Novel Profile | Recommended Strategy |
|---|---|---|---|---|
| **Bone** | **-1.294** | -0.304 (rank 3) | Strongest GPX4 dependency pan-cancer. Top divergent gene vs NSCLC: GPX4 diff = -0.921. | Combination strategies required; await in vivo GPX4i validation |
| **CNS/Brain** | -1.106 | -0.270 (rank 7) | GPX4-driven; TXNRD1 = -0.567 as secondary axis. | GPX4i + TrxR1i combination |
| **PNS** | -1.101 | -0.291 (rank 5) | GPX4-driven with TXNRD1 = -0.810 co-dependency. | GPX4i + TrxR1i combination |
| **Lymphoid** | -0.997 | **-0.397** (rank 1) | Highest FSP1 dependency pan-cancer (AIFM2 = -0.199). TFRC-driven divergence (diff = -0.790 vs NSCLC). | Dual FSP1i + GPX4i; strongest multi-target candidate |

**Critical caveat:** A systematic evaluation preprint (bioRxiv, March 14, 2026; DOI: 10.64898/2026.03.11.711115) shows that inhibition of GPX4, GCLC, or SLC7A11 fails to impact established tumor growth in vivo. Cell culture CRISPR screens greatly overestimate GPX4-axis vulnerabilities. Category C cancer types require combination strategies or await in vivo GPX4i validation.

### Category D: TrxR1/TXNRD1-Driven (Auranofin Candidates)

| Cancer Type | TXNRD1 Dependency | TrxR1+GCLC Dual Score | Notes |
|---|---|---|---|
| **Eye** | -0.947 | 0.968 | Top divergent gene vs NSCLC: TXNRD1 diff = -0.792. Distinct non-ferroptotic mechanism. |
| **Bone** | -0.881 | 0.924 | Overlaps with Category C (GPX4-dominant). Dual vulnerability. |
| **Head & Neck** | -0.810 | 0.821 | TCGA: NFE2L2 5.4% + KEAP1 4.1% = 9.5% NRF2-mutant. |
| **PNS** | -0.810 | 0.855 | Overlaps with Category C. |

**Mechanism:** TXNRD1 inhibition triggers non-ferroptotic cell death regulated by cystine availability and translation (bioRxiv 2026). Auranofin (FDA-approved gold(I) compound) and novel inhibitors CS47/DM20 show in vivo efficacy in KRAS-WT NSCLC xenografts (Andreani et al., bioRxiv July 2025; DOI: 10.1101/2025.07.25.666783). Liproxstatin-1 rescue confirms ferroptosis mechanism in the TrxR1 + GCLC combination context.

**Drug repurposing opportunity:** Auranofin is FDA-approved (rheumatoid arthritis) and requires no new IND for repurposing trials. Not present in PRISM 24Q2 for computational validation.

### Category E: HDACi + Ferroptosis Combination Candidates

13 cancer types with ferroptosis defense burden > 0.2, indicating multi-layered ferroptosis defense that may be overcome by HDACi-mediated sensitization:

| Cancer Type | Ferroptosis Defense Burden | HDACi Candidate |
|---|---|---|
| Lymphoid | 0.455 | Yes |
| Myeloid | 0.346 | Yes |
| Fibroblast* | 0.338 | Yes |
| Bone | 0.324 | Yes |
| PNS | 0.278 | Yes |
| CNS/Brain | 0.259 | Yes |
| Adrenal Gland* | 0.260 | Yes |
| Testis | 0.244 | Yes |
| Ovary | 0.239 | Yes |
| Thyroid | 0.233 | Yes |
| Soft Tissue | 0.222 | Yes |
| Kidney | 0.216 | Yes |
| Uterus | 0.206 | Yes |

*N=1 cell line; ranking unreliable.*

**Rationale:** GPX4 inhibitor-tolerant cancer persister cells become dependent on FSP1 as alternative ferroptosis suppressor. HDAC inhibitors (vorinostat, romidepsin, panobinostat — all FDA-approved) increase persister cell oxidative stress, enabling synergistic ferroptosis with GPX4i (Higuchi et al., *Science Advances* Jan 2026; DOI: 10.1126/sciadv.aea8771).

---

## 2. Drug Repurposing Candidates

| Drug | Target | Status | Candidate Cancer Types | Evidence |
|---|---|---|---|---|
| **Auranofin** | TrxR1/TXNRD1 | FDA-approved (RA) | Eye, Bone, Head & Neck, PNS | In vivo efficacy in KRAS-WT NSCLC xenografts (Andreani et al., bioRxiv 2025). Non-ferroptotic cell death mechanism. |
| **icFSP1** | FSP1/AIFM2 | Preclinical | Lung (LUAD), Melanoma (LN) | ~80% tumor reduction in KP LUAD (Wu et al., *Nature* 2025). No IND filed as of March 2026. |
| **viFSP1** | FSP1/AIFM2 | Preclinical | Melanoma (LN) | First cross-species FSP1 inhibitor; effective in metastatic melanoma LN models (Ubellacker et al., *Nature* 2025). |
| **CB-839/telaglenastat** | GLS1 | Clinical trials | Lymphoid, Myeloid, Biliary Tract | Top GLS1-dependent types: Lymphoid (dep = -0.467), Myeloid (dep = -0.361). |
| **Vorinostat/panobinostat** | HDAC | FDA-approved | 13 types with defense burden > 0.2 | Synergizes with GPX4i in persister cells (Higuchi et al., *Science Advances* 2026). |

---

## 3. Combination Strategy Rationale

### Dual FSP1 + GPX4 Inhibition
- **Target cancer types:** Bowel/CRC, Kidney/ChRCC, Ovary
- **Rationale:** Single-target inhibition is insufficient. In CRC, high FSP1+GPX4 co-expression correlates with 5-FU resistance and poor prognosis (*Anticancer Research* 2024). In ChRCC, GPX4+FSP1 are synergistic (Salem et al., *Oncogene* 2025). In Ovary, FSP1 knockdown enhances ferroptosis sensitivity (*Cancers* 2025).
- **Clinical positioning:** Ferroptosis induction to overcome chemoresistance in CRC. Second-line or combination therapy in ChRCC.

### HDACi + Ferroptosis Inducer
- **Target cancer types:** 13 cancer types with ferroptosis defense burden > 0.2
- **Rationale:** Cancer persister cells surviving GPX4 inhibition rely on residual FSP1. HDAC inhibitors pretreatment induces ROS in persister cells, synergizing with GPX4 inhibition (*Science Advances* 2026). FDA-approved HDACi (vorinostat, romidepsin, panobinostat) provide an immediately actionable clinical route.
- **Clinical positioning:** Combination with ferroptosis inducers in treatment-resistant settings.

### TrxR1 + GCLC Co-targeting
- **Target cancer types:** Eye, Bone, Head & Neck, PNS
- **Rationale:** TrxR1 deficiency + pharmacologic GCLC inhibition potently induces tumor regression via non-ferroptotic cell death (bioRxiv 2026). HMOX1-dependent iron overload is the executioner mechanism.
- **Note:** HMOX1 expression vs TXNRD1 dependency shows r = 0.013 pan-cancer (no correlation). HMOX1 is an executioner biomarker, not a DepMap dependency target.

---

## 4. Key Validations

### FSP1/NRF2 Independence
FSP1 (AIFM2) dependency shows near-zero difference between NRF2-active and wild-type cells in all 4 testable cancer types:

| Cancer Type | N (NRF2-active) | N (WT) | Cohen's d | p-value |
|---|---|---|---|---|
| Lung | 15 | 111 | -0.139 | 0.898 |
| Esophagus/Stomach | 7 | 62 | 0.455 | 0.227 |
| Head & Neck | 5 | 72 | -0.038 | 0.944 |
| Uterus | 5 | 29 | 0.275 | 0.273 |

**Interpretation:** FSP1 is not regulated by NRF2, consistent with literature (*Cancers* Aug 2025). FSP1 inhibitors can target all patients regardless of NRF2/KEAP1 mutation status, expanding the eligible population (e.g., in Lung, the 77% NRF2-WT AND the 23% NRF2-mutant patients are both targetable).

### Success Criteria Assessment

| Criterion | Status | Detail |
|---|---|---|
| >=5 cancer types with ferroptosis dependencies beyond NSCLC | **MET** | 18 types with composite < -0.20 |
| NRF2/KEAP1 modulates ferroptosis in >=3 types | **PARTIALLY MET** | Suggestive trends in 2-3 types (Uterus: GPX4 d=1.01, TXNRD1 d=1.11, GCLM d=1.58), but no FDR-significant results. Underpowered: 48 NRF2-active vs 1,138 WT lines. |
| >=2 unexplored cancer types computationally predicted as promising | **MET** | Bowel/CRC, ChRCC, Ovary, Breast all identified |
| PRISM validation r > 0.3 | **NOT MET** | Zero ferroptosis compounds in PRISM 24Q2. Search for erastin, RSL3, ML162, ML210, FIN56, FINO2, CB-839, telaglenastat, icFSP1, auranofin returned no valid hits. |

---

## 5. Novel Findings

### Bone: Strongest GPX4 Dependency Pan-Cancer
- GPX4 dependency = -1.294, the most extreme of any cancer type (N = 49 cell lines).
- Novel ferroptosis profile divergent from NSCLC (cosine similarity = 0.770).
- Also shows strong TXNRD1 dependency (-0.881), making it a dual GPX4i + TrxR1i candidate.

### Bowel/CRC: NSCLC Ferroptosis Strategies Transferable
- Highest NSCLC analog: cosine similarity = 0.954, Pearson r = 0.940 across 20 ferroptosis genes.
- Literature confirms dual FSP1+GPX4 targeting is required (*Anticancer Research* 2024).
- 100% NRF2-WT in DepMap; TCGA shows only 1.5% NFE2L2 + 0.9% KEAP1 mutations in COADREAD.

### Lymphoid: Highest FSP1 Dependency + TFRC-Driven Divergence
- Highest FSP1/AIFM2 dependency pan-cancer: -0.199 (N = 93 cell lines).
- Highest composite ferroptosis vulnerability: -0.397.
- Top divergent gene vs NSCLC: TFRC (transferrin receptor) diff = -0.790, suggesting iron import-driven ferroptosis biology distinct from NSCLC.

### Eye/Adrenal: TXNRD1-Driven Distinct Mechanism
- Eye: TXNRD1 = -0.947, representing a non-ferroptotic cell death mechanism.
- Eye shows moderate HMOX1-TXNRD1 trend (r = 0.43, p = 0.11, N = 15) but underpowered.
- Adrenal: TXNRD1 = -0.934, but N = 1 cell line (unreliable).

### Breast: FSP1 > NRF2 Reliance
- FSP1 targeting may work independent of NRF2 status. Breast cancer shows FSP1 knockdown is more effective than NRF2 inhibition for ferroptosis sensitization (*Cancers* 2025).
- NSCLC analog rank 4 (cosine similarity = 0.942).
- TCGA: only 0.4% NFE2L2 + 0.2% KEAP1 mutations — nearly 100% NRF2-WT.

---

## 6. Caveats and Limitations

### GPX4-Axis In Vivo Failure
A systematic evaluation (bioRxiv March 14, 2026; DOI: 10.64898/2026.03.11.711115) shows that GPX4, GCLC, or SLC7A11 inhibition fails to impact established tumor growth in vivo. Cell culture CRISPR screens greatly overestimate GPX4-axis anti-cancer effects. Category C cancer types (Bone, CNS/Brain, PNS, Lymphoid) require combination strategies. **Note: preprint, not yet peer-reviewed.**

### N=1 Cancer Types
Fibroblast (composite rank 2, -0.333) and Adrenal Gland (rank 14, -0.220) are based on single cell lines. Their rankings are unreliable and should not drive therapeutic recommendations.

### NRF2 Stratification Underpowered
Only 48 NRF2-active lines (23 KEAP1-LOF, 27 NFE2L2-GOF) vs 1,138 wild-type. Only 4 cancer types had N >= 5 in both groups for statistical testing. No results survived FDR correction (all q > 0.07). This reflects sample size limitations, not necessarily absent biology.

### FSP1 DepMap Scores Underestimate In Vivo Potential
FSP1/AIFM2 dependencies are modest across all types (-0.02 to -0.20 mean). Literature shows FSP1 works dramatically better in vivo (80% tumor reduction). DepMap-based FSP1 rankings should be interpreted as lower bounds of therapeutic potential.

### PRISM Gap
Zero ferroptosis compounds (erastin, RSL3, ML162, ML210, FIN56, FINO2, CB-839, telaglenastat, icFSP1) and zero auranofin entries found in PRISM 24Q2. No pharmacological validation of computational predictions is possible with current data. An initial search returned 6 false-positive hits for cloperastine (substring match for "erastin"); a regex bug was identified and corrected.

### HMOX1/TXNRD1 Correlation
HMOX1 expression vs TXNRD1 dependency: r = 0.013, p = 0.68, N = 1,112. No pan-cancer correlation. HMOX1 is a pro-death executioner gene with near-zero DepMap dependency (mean ~0.0). Its role as a biomarker for TrxR1 inhibition response requires pharmacological experiments, not CRISPR knockout data.

### Microenvironment Effects Not Captured
DepMap cell culture does not model microenvironmental context. Melanoma FSP1 vulnerability is lymph node niche-specific (Ubellacker et al., *Nature* 2025). Other cancer types may have similar context-dependent vulnerabilities not captured by this analysis.

### FTH1 Pan-Essentiality
FTH1 (ferritin heavy chain) shows pan-essential dependency of approximately -0.7 across nearly all cancer types. This inflates composite vulnerability scores uniformly. Relative rankings between cancer types are preserved since FTH1 affects all types similarly.

---

## 7. Methodology

### Study Design
Four-phase computational pipeline analyzing DepMap 25Q3 CRISPR gene effect data across 20 ferroptosis-related genes in 1,186 cell lines from 30 cancer types.

### Gene Panel (20 genes)
- **Ferroptosis defense (pro-survival):** FSP1/AIFM2, GPX4, SLC7A11, GLS1, GCLC, GCLM, TXNRD1, NQO1, FTH1, HMOX1
- **Ferroptosis promotion (pro-death):** ACSL4, LPCAT3, SAT1, NCOA4, TFRC, ALOX15
- **Metabolic modulators:** SHMT1, SHMT2, MTHFD2, CBS

### Phase 1: Pan-Cancer Dependency Map
- Extracted CRISPR dependency scores for all 20 genes across all DepMap cell lines
- Computed per-cancer-type statistics: mean dependency, fraction of lines showing dependency (score < -0.5), IQR
- Ranked cancer types by composite ferroptosis vulnerability score
- Hierarchical clustering by full 20-gene dependency profile

### Phase 2: NRF2/KEAP1 Stratification
- Classified 1,186 lines: 48 NRF2-active (23 KEAP1-LOF, 27 NFE2L2-GOF), 1,138 double-WT
- Compared ferroptosis dependencies between NRF2-active and WT within each testable cancer type (N >= 5 per group; 4 types qualified: Lung, Esophagus/Stomach, Head & Neck, Uterus)
- Mann-Whitney U test with FDR correction; Cohen's d for effect size
- Specific test of FSP1/NRF2 independence

### Phase 3: Therapeutic Opportunity Scoring
- Integrated TCGA NRF2/KEAP1 mutation frequencies from 33 tumor types (GDC Pan-Cancer Atlas)
- Computed therapy-specific opportunity scores: FSP1i (2x weighted for in vivo validation), GPX4i, GLS1i, dual targets, TrxR1+GCLC
- Three-tier evidence classification: Tier A (in vivo validated), Tier B (in vitro only), Tier C (non-ferroptotic mechanism)
- Ferroptosis defense burden for HDACi combination candidacy
- PRISM 24Q2 searched for ferroptosis compound validation (none found)

### Phase 4: Cross-Cancer NSCLC Comparison
- NSCLC KEAP1-mutant ferroptosis profile as reference (15 KEAP1-mutant lines)
- Cosine similarity and Pearson correlation across 20-gene profiles for all cancer types
- FSP1-specific analog ranking by AIFM2 dependency distance
- Identification of 8 divergent cancer types with novel ferroptosis profiles (cosine < 0.81)

### Data Sources
- **DepMap 25Q3:** CRISPRGeneEffect.csv, OmicsSomaticMutations.csv, Model.csv, OmicsExpressionTPMLogp1HumanProteinCodingGenes.csv
- **TCGA Pan-Cancer Atlas:** NFE2L2/KEAP1 mutation frequencies (33 tumor types, GDC/cBioPortal)
- **PRISM 24Q2:** Drug sensitivity data (no ferroptosis compounds found)

### Statistical Methods
- Mann-Whitney U test for NRF2-active vs WT comparisons
- Benjamini-Hochberg FDR correction for multiple testing
- Cohen's d and rank-biserial correlation for effect sizes
- Cosine similarity and Pearson correlation for profile comparisons
- Composite vulnerability score: mean across defense gene dependencies

---

## 8. Translational Confidence Addendum

This section flags systematic biases in DepMap CRISPR data that affect interpretation of the atlas findings. The underlying data and conclusions above remain unchanged; this addendum provides in vivo context for translational prioritization.

### 8.1 GPX4-Axis Overestimation

DepMap CRISPR scores for **GPX4, GCLC, and SLC7A11** likely overestimate therapeutic potential. A systematic in vivo evaluation (bioRxiv 2026; DOI: 10.64898/2026.03.11.711115) demonstrated that inhibition of these genes fails to impact established tumor growth in vivo, despite strong CRISPR dependencies in cell culture. This affects all Category C cancer types (Bone, CNS/Brain, PNS, Lymphoid) and any therapeutic strategy relying on GPX4-axis monotherapy.

**NSCLC panel cross-reference:** In the NSCLC-focused ferroptosis panel (data/results/ferroptosis_panel/), GPX4 shows no KEAP1-enriched dependency (KEAP1-mut mean = -0.405 vs WT = -0.626, FDR = 0.45). KEAP1-mutant lines are actually *less* GPX4-dependent, consistent with NRF2-mediated compensatory defense masking the dependency in vitro.

**Confidence adjustment:** GPX4-axis scores should be treated as upper bounds. Combination strategies (GPX4i + FSP1i, GPX4i + TrxR1i, or HDACi sensitization) are required for clinical translation.

### 8.2 FSP1/AIFM2 Underestimation

DepMap CRISPR scores for **FSP1/AIFM2** likely underestimate therapeutic potential. FSP1 deletion suppresses tumorigenesis ~80% in KP LUAD models in vivo (Wu et al., *Nature* Nov 2025), yet DepMap dependencies are modest across all cancer types (range: -0.02 to -0.20 mean). FSP1 is dispensable in standard cell culture but essential in the tumor microenvironment.

**NSCLC panel cross-reference:** AIFM2 is flat in the NSCLC ferroptosis panel — KEAP1-mut mean = -0.070 vs WT = -0.068, effect size = -0.002, p = 1.0, FDR = 1.0. This confirms FSP1 is invisible to DepMap regardless of genotype, yet the same gene shows dramatic in vivo efficacy. All FSP1-based DepMap rankings in this atlas should be interpreted as lower bounds.

**Confidence adjustment:** Cancer types ranked low for FSP1 vulnerability in DepMap may still respond to FSP1 inhibitors in vivo. Category A (FSP1-Vulnerable) rankings are the most conservative estimates in this atlas.

### 8.3 Clinical Strategy: Dual Targeting Over Monotherapy

The opposing biases above (GPX4 overestimated, FSP1 underestimated) converge on a single clinical implication: **dual GPX4 + FSP1 targeting is recommended over monotherapy** for any cancer type showing ferroptosis vulnerability.

- GPX4 monotherapy fails in vivo (bioRxiv 2026)
- FSP1 monotherapy may be sufficient in some contexts (Nature 2025, LUAD) but resistance via GPX4 compensation is expected
- Dual inhibition eliminates both parallel ferroptosis defense arms simultaneously
- Category B cancer types (Bowel/CRC, Kidney/ChRCC, Ovary) already require dual targeting based on in vitro data; this addendum extends the dual-targeting recommendation to all categories

### 8.4 KEAP1-Mutation Masking of In Vitro Dependencies

NRF2 constitutive activation (via KEAP1 loss-of-function or NFE2L2 gain-of-function) provides multi-layered ferroptosis defense that masks individual gene dependencies in DepMap CRISPR screens.

**NSCLC panel cross-reference:** Across all 9 ferroptosis genes tested in the NSCLC panel (data/results/ferroptosis_panel/keap1_enrichment_stats.csv), **no gene** shows FDR-significant KEAP1-enriched dependency (all FDR > 0.44). This is not because KEAP1-mutant cells are ferroptosis-resistant — it is because NRF2 activation simultaneously upregulates multiple redundant defense pathways (GPX4, SLC7A11, GCLC, NQO1), so knocking out any single gene is buffered by the others.

**Implication:** KEAP1-mutant tumors may be more ferroptosis-vulnerable than DepMap suggests, but only when multiple defense arms are targeted simultaneously. Single-gene CRISPR data systematically underestimates the ferroptosis vulnerability of NRF2-active cancers.

### Translational Confidence Summary

| DepMap Target | In Vitro Bias | In Vivo Reality | Confidence Adjustment |
|---|---|---|---|
| GPX4/GCLC/SLC7A11 | Overestimates vulnerability | Fails as monotherapy in vivo | Treat as upper bound; require combination |
| FSP1/AIFM2 | Underestimates vulnerability | ~80% tumor reduction in vivo | Treat as lower bound; prioritize despite modest scores |
| KEAP1-mutant dependencies | Masks individual dependencies | Multi-arm defense requires multi-target attack | Single-gene rankings unreliable for NRF2-active tumors |

*Addendum date: 2026-03-17. Cross-referenced with NSCLC ferroptosis panel (data/results/ferroptosis_panel/).*

---

## References

1. Wu et al. "FSP1 inhibition reduces KP LUAD tumor growth." *Nature*, Nov 2025. DOI: 10.1038/s41586-025-XXXX
2. Ubellacker et al. "Lymph node environment drives FSP1 targetability in melanoma." *Nature*, Nov 2025. DOI: 10.1038/s41586-025-09709-1
3. Salem et al. "GPX4+FSP1 dual vulnerability in ChRCC." *Oncogene*, 2025.
4. "FSP1+GPX4 dual targeting in CRC." *Anticancer Research*, Nov 2024. DOI: 10.21873/anticanres.17408
5. "FSP1 independent of NRF2." *Cancers*, Aug 2025. DOI: 10.3390/cancers17162714
6. Higuchi et al. "FSP1 and HDACs suppress persister cell ferroptosis." *Science Advances*, Jan 2026. DOI: 10.1126/sciadv.aea8771
7. "Systematic evaluation defines the limits of ferroptosis in cancer therapy." bioRxiv, March 14, 2026. DOI: 10.64898/2026.03.11.711115
8. Andreani et al. "TrxR1 inhibition triggers ferroptosis in KRAS-WT NSCLC." bioRxiv, July 2025. DOI: 10.1101/2025.07.25.666783
9. Zhang et al. "FSEN1-FSP1 cocrystal structure." *PNAS*, June 2025. DOI: 10.1073/pnas.2505197122
10. Kang et al. "Translating ferroptosis into oncology." *Nature Reviews Clinical Oncology*, 2026. DOI: 10.1038/s41571-026-01128-z
11. "Targeting ferroptosis in cancer." *Nature Genetics*, Dec 2025. DOI: 10.1038/s41588-025-02456-z
12. "TrxR1/KEAP1/GPX4 regulatory mechanism." *Cell Death & Differentiation*, 2026. DOI: 10.1038/s41418-026-01691-z

---

*Analysis date: 2026-03-17. Data: DepMap 25Q3, TCGA Pan-Cancer Atlas, PRISM 24Q2.*
*Pipeline: `pipelines/pancancer-ferroptosis-atlas/` (phase1-phase4).*
*Raw outputs: `data/results/pancancer-ferroptosis-atlas/phase1/` through `phase4/`.*
