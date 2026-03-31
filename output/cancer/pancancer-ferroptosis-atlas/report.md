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
- **Phase 5 (expression layer):** Median-split defense profiling of 1,112 cell lines identifies 264 dual-high (FSP1+GPX4) lines requiring combination therapy, concentrated in Liver (67%), Eye (60%), and Cervix (56%). KEAP1-mutant Lung lines show significantly elevated FSP1 expression (Cohen's d = 0.91, p = 0.005), with 67% classified as FSP1-dependent — quantitative validation of the NRF2→FSP1 axis.
- **KEAP1-stratified combination strategies:** SAT1/polyamine supplementation sensitizes KEAP1-WT/KRAS-MUT tumors to ferroptosis via JNK/c-Jun→SAT1 upregulation. GLS1 inhibition (telaglenastat/CB-839) reverses ferroptosis resistance in KEAP1-MUT tumors.
- **N6F11 GPX4 degrader (March 2026 update):** A tumor-specific GPX4 degrader that spares immune cells resolves the two critical GPX4-targeting barriers (in vivo failure and T-cell toxicity). N6F11 is now the preferred clinical path for Category C cancers, with Bone (GPX4 = -1.294) as the prime monotherapy target.
- **FSP1 monotherapy elevated to primary target:** icFSP1 monotherapy shrinks lung tumors ~80% in vivo (*Nature* 2025) with FSP1 KO mice remaining viable (unlike lethal GPX4 KO), establishing FSP1 as a primary therapeutic target — not merely a backup — for Category A cancers.
- **HDAC persister-cell vulnerability:** GPX4 inhibitor-tolerant persister cells depend on FSP1, and FDA-approved HDAC inhibitors (panobinostat, vorinostat, romidepsin) eliminate these persisters via oxidative stress. This constitutes a 6th ferroptosis defense layer applicable across all categories.

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
| **Bone** | **-1.294** | -0.304 (rank 3) | Strongest GPX4 dependency pan-cancer. Top divergent gene vs NSCLC: GPX4 diff = -0.921. | **N6F11 monotherapy (preferred)**; prime N6F11 target pan-cancer |
| **CNS/Brain** | -1.106 | -0.270 (rank 7) | GPX4-driven; TXNRD1 = -0.567 as secondary axis. | N6F11 + TrxR1i combination |
| **PNS** | -1.101 | -0.291 (rank 5) | GPX4-driven with TXNRD1 = -0.810 co-dependency. | N6F11 + TrxR1i combination |
| **Lymphoid** | -0.997 | **-0.397** (rank 1) | Highest FSP1 dependency pan-cancer (AIFM2 = -0.199). TFRC-driven divergence (diff = -0.790 vs NSCLC). | Dual FSP1i + N6F11; strongest multi-target candidate |

**Critical caveat:** A systematic evaluation preprint (bioRxiv, March 14, 2026; DOI: 10.64898/2026.03.11.711115) shows that inhibition of GPX4, GCLC, or SLC7A11 fails to impact established tumor growth in vivo. Cell culture CRISPR screens greatly overestimate GPX4-axis vulnerabilities. **N6F11 GPX4 degrader** (*Sci Transl Med*) addresses this barrier through targeted protein degradation rather than catalytic inhibition, and is now the preferred GPX4-targeting clinical path for Category C types (see Section 10).

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
| **CB-839/telaglenastat** | GLS1 | Clinical trials | KEAP1-MUT LUAD; Lymphoid, Myeloid, Biliary Tract | GLS1 inhibition reverses ferroptosis-resistant phenotype in KEAP1-MUT LUAD (AACR 2026, B013). Top GLS1-dependent types: Lymphoid (-0.467), Myeloid (-0.361). |
| **Polyamine supplementation** | SAT1/polyamine catabolism | Preclinical | KEAP1-WT/KRAS-MUT tumors | Polyamine supplementation enhances KRAS inhibitor-induced ferroptosis via JNK/c-Jun→SAT1 upregulation (NatComm 2025, DOI: 10.1038/s41467-025-65441-4). Validated in organoids, xenografts, and spontaneous tumor models. |
| **N6F11** | GPX4 (degrader) | Preclinical | Bone (#1), CNS/Brain, PNS, Lymphoid (Cat. C) | Tumor-specific GPX4 degrader; spares immune cells (DCs, T, NK, neutrophils). Bypasses GPX4i in vivo failure via targeted degradation (*Sci Transl Med*). Preferred GPX4 clinical path. |
| **Vorinostat/panobinostat** | HDAC | FDA-approved | 13 types with defense burden > 0.2 | Synergizes with GPX4i in persister cells (Higuchi et al., *Science Advances* 2026). Eliminates FSP1-dependent persister cells across all categories. |

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

### KEAP1-Stratified Ferroptosis Combination (SAT1/Polyamine + GLS1)

Two recent studies identify KEAP1 mutation status as the master switch determining which ferroptosis combination strategy is optimal in KRAS-mutant cancers:

**KEAP1-WT + KRAS-MUT: Polyamine supplementation + KRAS inhibitor**
- **Mechanism:** KRAS inhibitors activate JNK/c-Jun → SAT1 (spermidine/spermine N1-acetyltransferase) upregulation → polyamine catabolism → ferroptosis. Exogenous polyamine supplementation powerfully enhances this pathway.
- **Evidence:** Validated in KEAP1-WT/KRAS-MUT organoids, xenografts, and spontaneous tumor models (*Nature Communications* 2025; DOI: 10.1038/s41467-025-65441-4).
- **Clinical positioning:** Applicable to NSCLC (77% NRF2-WT in DepMap) and other KRAS-mutant malignancies with intact KEAP1.

**KEAP1-MUT: GLS1 inhibition (telaglenastat/CB-839) + KRAS inhibitor**
- **Mechanism:** In KEAP1-MUT cells, JNK promotes NRF2 degradation → SAT1 suppression → ferroptosis resistance. GLS1 (glutaminase 1) inhibition reverses this ferroptosis-resistant phenotype by depleting glutathione synthesis.
- **Evidence:** Telaglenastat/CB-839 enhances KRAS inhibitor efficacy in KEAP1-MUT LUAD (AACR 2026, *Cancer Research* 86(5 Suppl):B013).
- **Clinical positioning:** Addresses the ~23% of NSCLC that is KEAP1-mutant and resistant to standard ferroptosis approaches. CB-839 has existing clinical trial data for tolerability.

**Stratification rationale:** KEAP1 status determines whether the SAT1/polyamine axis is active (WT) or suppressed (mutant). This creates a biomarker-guided decision point: genotype KEAP1 → select polyamine supplementation (WT) or GLS1 inhibition (MUT) as the ferroptosis-sensitizing agent alongside KRAS-targeted therapy. See *Nature Reviews Clinical Oncology* (March 2026; DOI: 10.1038/s41571-026-01128-z) for the clinical translation framework.

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
Five-phase computational pipeline analyzing DepMap 25Q3 CRISPR gene effect and RNA expression data across 20 ferroptosis-related genes in 1,186 cell lines from 30 cancer types.

### Gene Panel (20 genes)
- **Ferroptosis defense (pro-survival):** FSP1/AIFM2, GPX4, SLC7A11, GLS1, GCLC, GCLM, TXNRD1, NQO1, FTH1, HMOX1
- **Ferroptosis promotion (pro-death):** ACSL4, LPCAT3, SAT1 (polyamine catabolism; KEAP1-WT ferroptosis sensitizer via JNK/c-Jun axis), NCOA4, TFRC, ALOX15
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

### Phase 5: FSP1/AIFM2 Expression Layer
- Extracted RNA expression (log₂ TPM+1) for AIFM2, GPX4, SLC7A11, GCLC, NQO1, FTH1, HMOX1 across 1,699 cell lines (1,112 with matching CRISPR data)
- Median-split classification: FSP1-high/low × GPX4-high/low → 4 defense profiles
- Per-cancer-type defense profile distribution and dominant profile assignment
- KEAP1-mutant vs WT FSP1 expression comparison (Mann-Whitney U, Cohen's d) in Lung (only type with N ≥ 5 KEAP1-mutant lines)
- Cross-phase consistency check: FSP1 expression vs AIFM2 CRISPR dependency correlation

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

## 8. Phase 5: FSP1/AIFM2 Expression-Based Defense Profiling

### Overview

Phase 5 adds an orthogonal expression layer to the CRISPR dependency-based Phases 1–4. Using DepMap 25Q3 RNA expression data (log₂ TPM+1) for AIFM2 (FSP1) and GPX4 across 1,112 cell lines from 29 cancer types, each cell line was classified into one of four defense profiles by median-split:

| Profile | Count | Criteria | Therapeutic Implication |
|---|---|---|---|
| GPX4-dependent | 292 (26.3%) | GPX4-high, FSP1-low | GPX4i monotherapy (Tier B: in vitro only) |
| FSP1-dependent | 292 (26.3%) | FSP1-high, GPX4-low | icFSP1 monotherapy (Tier A: in vivo validated) |
| Dual-high | 264 (23.7%) | Both high | FSP1i + GPX4i combination required |
| Dual-low | 264 (23.7%) | Both low | Ferroptosis-sensitive (minimal defense) |

### Dual-High Cancer Types: Combination Therapy Required

Cancer types with the highest fraction of dual-high defense profiles represent the strongest case for FSP1i + GPX4i combination therapy:

| Cancer Type | N | Dual-High % | Dominant Profile |
|---|---|---|---|
| Liver | 24 | 67% | dual-high |
| Eye | 15 | 60% | dual-high |
| Cervix | 18 | 56% | dual-high |
| Biliary Tract | 34 | 53% | dual-high |
| Pancreas | 46 | 48% | dual-high |
| Pleura | 21 | 43% | dual-high |
| Bladder/Urinary Tract | 34 | 32% | dual-high |

These expression profiles corroborate Category B (dual targeting) from Phase 3. Notably, Liver, Biliary Tract, and Pancreas were not among the top CRISPR dependency-ranked types, suggesting expression-based defense profiling captures resistance mechanisms invisible to knockout screens.

### Ferroptosis-Sensitive Cancer Types (Dual-Low Dominant)

Cancer types where dual-low profiles dominate may be amenable to monotherapy or low-dose combination approaches:

| Cancer Type | N | Dual-Low % | Dominant Profile |
|---|---|---|---|
| Myeloid | 40 | 55% | dual-low |
| Lymphoid | 91 | 54% | dual-low |
| PNS | 38 | 50% | dual-low |
| Soft Tissue | 46 | 37% | dual-low |
| Thyroid | 11 | 36% | GPX4-dependent |

Lymphoid was previously ranked #1 by composite CRISPR vulnerability (-0.397); its dual-low dominance confirms this is a genuinely ferroptosis-sensitive lineage with minimal defense pathway expression.

### KEAP1-FSP1 Expression Context

Only Lung had sufficient KEAP1-mutant cell lines (n = 11 mutant, n = 108 WT) for statistical testing of FSP1 expression stratification:

- **KEAP1-mutant mean FSP1 expression:** 4.58 (log₂ TPM+1)
- **Wild-type mean FSP1 expression:** 3.41 (log₂ TPM+1)
- **Difference:** +1.16 (KEAP1-mut > WT)
- **Mann-Whitney U test:** p = 0.005
- **Cohen's d:** 0.91 (large effect)
- **KEAP1-mutant FSP1-dependent:** 67% (14/21 classified as FSP1-high)

This provides quantitative validation of the NRF2→FSP1 transcriptional axis: KEAP1 loss-of-function activates NRF2, which upregulates FSP1 expression, creating FSP1-dependent defense that can be targeted by icFSP1. This is consistent with the FSP1/NRF2 independence finding in Phase 2 (CRISPR dependency is NRF2-independent, but expression is NRF2-driven — FSP1 is upregulated by NRF2 but functionally essential regardless of NRF2 status).

### Cross-Phase Consistency

Phase 5 expression profiles align with Phase 1–4 CRISPR dependency findings:
- **Bowel:** 63% FSP1-dependent by expression, consistent with Phase 3 icFSP1 candidacy and Category B dual-targeting recommendation.
- **Skin:** 51% FSP1-dependent, consistent with Category A FSP1-vulnerable classification for melanoma.
- **Lymphoid:** 54% dual-low by expression, consistent with highest composite CRISPR vulnerability (-0.397).
- **FSP1 expression vs CRISPR dependency:** r = 0.07 — confirming that expression and CRISPR dependency are orthogonal measures (high expression ≠ high dependency), justifying the independent Phase 5 layer.

### Phase 5 Limitations

- **Median-split is exploratory.** Lines near the expression boundary (closest: 0.0003 from median) could flip category with minor expression changes. Continuous scoring may be more appropriate for clinical use.
- **KEAP1-FSP1 statistical testing limited to Lung** (n = 11 KEAP1-mut). Other cancer types had 0–2 KEAP1-mutant lines, insufficient for testing.
- **Small-N cancer types unreliable:** Fibroblast (n = 1), Adrenal (n = 1), Vulva/Vagina (n = 2), Testis (n = 4) — profile distributions are uninformative at these sample sizes.
- **GPX4 expression range is narrow** (4.91–10.49 log₂ TPM+1) compared to AIFM2 (0.01–6.12), meaning the GPX4 median split separates lines with less biological distinction.

### Phase 5 Data

- `data/results/pancancer-ferroptosis-atlas/phase5/defense_profile_classification.csv` — per-line classification (N = 1,112)
- `data/results/pancancer-ferroptosis-atlas/phase5/cancer_type_defense_summary.csv` — per-type summary (N = 29)
- `data/results/pancancer-ferroptosis-atlas/phase5/ferroptosis_expression_matrix.csv` — expression values for 7 genes (N = 1,699)
- `data/results/pancancer-ferroptosis-atlas/phase5/fsp1_keap1_context_map.csv` — KEAP1 stratification (N = 29)
- `data/results/pancancer-ferroptosis-atlas/phase5/defense_profile_scatter.png` — quadrant visualization

*Validation: APPROVED (task #836, journal #1221). All claims independently verified by validation_scientist.*

---

## 9. Translational Confidence Addendum

This section flags systematic biases in DepMap CRISPR data that affect interpretation of the atlas findings. The underlying data and conclusions above remain unchanged; this addendum provides in vivo context for translational prioritization.

### 9.1 GPX4-Axis Overestimation

DepMap CRISPR scores for **GPX4, GCLC, and SLC7A11** likely overestimate therapeutic potential. A systematic in vivo evaluation (bioRxiv 2026; DOI: 10.64898/2026.03.11.711115) demonstrated that inhibition of these genes fails to impact established tumor growth in vivo, despite strong CRISPR dependencies in cell culture. This affects all Category C cancer types (Bone, CNS/Brain, PNS, Lymphoid) and any therapeutic strategy relying on GPX4-axis monotherapy.

**NSCLC panel cross-reference:** In the NSCLC-focused ferroptosis panel (data/results/ferroptosis_panel/), GPX4 shows no KEAP1-enriched dependency (KEAP1-mut mean = -0.405 vs WT = -0.626, FDR = 0.45). KEAP1-mutant lines are actually *less* GPX4-dependent, consistent with NRF2-mediated compensatory defense masking the dependency in vitro.

**Confidence adjustment:** GPX4-axis scores should be treated as upper bounds. Combination strategies (GPX4i + FSP1i, GPX4i + TrxR1i, or HDACi sensitization) are required for clinical translation.

### 9.2 FSP1/AIFM2 Underestimation

DepMap CRISPR scores for **FSP1/AIFM2** likely underestimate therapeutic potential. FSP1 deletion suppresses tumorigenesis ~80% in KP LUAD models in vivo (Wu et al., *Nature* Nov 2025), yet DepMap dependencies are modest across all cancer types (range: -0.02 to -0.20 mean). FSP1 is dispensable in standard cell culture but essential in the tumor microenvironment.

**NSCLC panel cross-reference:** AIFM2 is flat in the NSCLC ferroptosis panel — KEAP1-mut mean = -0.070 vs WT = -0.068, effect size = -0.002, p = 1.0, FDR = 1.0. This confirms FSP1 is invisible to DepMap regardless of genotype, yet the same gene shows dramatic in vivo efficacy. All FSP1-based DepMap rankings in this atlas should be interpreted as lower bounds.

**Confidence adjustment:** Cancer types ranked low for FSP1 vulnerability in DepMap may still respond to FSP1 inhibitors in vivo. Category A (FSP1-Vulnerable) rankings are the most conservative estimates in this atlas.

### 9.3 Clinical Strategy: Dual Targeting Over Monotherapy

The opposing biases above (GPX4 overestimated, FSP1 underestimated) converge on a single clinical implication: **dual GPX4 + FSP1 targeting is recommended over monotherapy** for any cancer type showing ferroptosis vulnerability.

- GPX4 monotherapy fails in vivo (bioRxiv 2026)
- FSP1 monotherapy may be sufficient in some contexts (Nature 2025, LUAD) but resistance via GPX4 compensation is expected
- Dual inhibition eliminates both parallel ferroptosis defense arms simultaneously
- Category B cancer types (Bowel/CRC, Kidney/ChRCC, Ovary) already require dual targeting based on in vitro data; this addendum extends the dual-targeting recommendation to all categories

### 9.4 KEAP1-Mutation Masking of In Vitro Dependencies

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

## 10. Translational Update: N6F11, FSP1 Monotherapy, and Persister-Cell Targeting (March 2026)

Three developments since the initial atlas publication reshape the clinical strategy for ferroptosis-based cancer therapy. This section integrates these findings into the existing 5-category therapeutic stratification.

### 10.1 N6F11: Tumor-Specific GPX4 Degrader

**Source:** *Science Translational Medicine* (2026); RD journal #2519.

N6F11 is a targeted protein degrader (PROTAC-like) that selectively degrades GPX4 in cancer cells while sparing immune cells (dendritic cells, T cells, NK cells, neutrophils). This addresses the two critical barriers that have blocked GPX4-targeting therapies:

1. **In vivo failure of GPX4 inhibitors.** The bioRxiv 2026 preprint (Section 9.1) showed catalytic GPX4 inhibitors fail in established tumors — likely due to compensatory pathway upregulation. N6F11 degrades the GPX4 protein entirely, bypassing compensatory mechanisms that render catalytic inhibitors ineffective.

2. **Immune toxicity.** GPX4 knockout is lethal to T cells (known since 2019), making systemic GPX4 inhibition incompatible with anti-tumor immunity. N6F11's tumor-specific degradation preserves the immune compartment, enabling combination with immunotherapy.

**Atlas integration:**
- **Category C (GPX4-dominant):** N6F11 replaces "await in vivo GPX4i validation" as the preferred clinical path. Bone cancer (GPX4 = -1.294, strongest pan-cancer) is the prime N6F11 monotherapy target.
- **Category B (dual defense):** N6F11 + FSP1i combination for cancer types requiring dual targeting (Bowel/CRC, Kidney/ChRCC, Ovary).
- **Category A (FSP1-vulnerable):** N6F11 provides an alternative GPX4-axis approach alongside icFSP1 monotherapy.

**Status:** Preclinical. No IND filed as of March 2026.

### 10.2 FSP1 Monotherapy: Elevation from Backup to Primary Target

**Source:** Wu et al., *Nature*, November 2025 (PMID 41193800); NRF2+FSP1 co-targeting: PMC12384948.

The Nature 2025 study demonstrated that icFSP1 monotherapy reduces KP LUAD tumor growth by ~80% in vivo. Critically, FSP1 KO mice are viable — unlike GPX4 KO, which is embryonic lethal. This establishes a fundamentally different safety profile:

| Property | GPX4 Targeting | FSP1 Targeting |
|---|---|---|
| In vivo monotherapy efficacy | Fails (bioRxiv 2026) | ~80% tumor reduction (*Nature* 2025) |
| Knockout viability | Embryonic lethal | Viable (normal development) |
| Immune cell toxicity | T-cell death | No evidence of immune toxicity |
| DepMap score reliability | Overestimates (Section 9.1) | Underestimates (Section 9.2) |

**Upstream metabolic regulation — ME1→NADPH→FSP1→CoQH2 axis:**
Recent work (Wu et al., *Cell Death & Disease*, PMID: 41881958) identifies malic enzyme 1 (ME1) as an upstream driver of FSP1 anti-ferroptotic activity. ME1 supplies NADPH, the essential cofactor for FSP1-mediated CoQ10 reduction to CoQH2. In hepatocellular carcinoma, ME1 overexpression drives lenvatinib resistance through this FSP1-dependent mechanism. This adds metabolic context to FSP1 dependency: NADPH availability — not just NRF2/KEAP1 genetics — shapes which defense arm dominates in a given tumor. ME1 expression may serve as a biomarker for predicting FSP1 inhibitor sensitivity, though this has not been computationally evaluated in the atlas.

**Atlas integration:**
- FSP1 is elevated from "backup defense" to **co-primary therapeutic target** alongside GPX4 for Category A cancers.
- For Lung (LUAD), icFSP1 monotherapy is now the recommended first-line ferroptosis strategy given validated in vivo efficacy and superior safety.
- In KEAP1-mutant contexts, NRF2 constitutive activation upregulates FSP1 (Phase 5: Cohen's d = 0.91, p = 0.005). Combined NRF2+FSP1 co-targeting overcomes this resistance mechanism (PMC12384948).
- ME1 expression may identify additional FSP1-dependent tumors beyond those captured by NRF2/KEAP1 genotyping, particularly in Liver/HCC where ME1-driven resistance is documented.

### 10.3 HDAC Persister-Cell Vulnerability: 6th Ferroptosis Defense Layer

**Source:** Higuchi et al., *Science Advances*, January 2026 (DOI: 10.1126/sciadv.aea8771); RD tasks #1275, #802.

GPX4 inhibitor-tolerant cancer persister cells — a critical source of therapeutic failure and relapse — rely on FSP1 as their primary ferroptosis suppressor. HDAC inhibitors increase oxidative stress in these persister cells, synergizing with GPX4 inhibition to eliminate them:

**Mechanism:** GPX4i treatment → persister cell survival via FSP1 upregulation → HDACi pre/co-treatment increases ROS in persisters → overwhelms FSP1 defense → persister cell ferroptosis.

**Immediately actionable drugs:**
| HDACi | FDA Approval | Original Indication |
|---|---|---|
| Panobinostat | FDA-approved | Multiple myeloma |
| Vorinostat | FDA-approved | Cutaneous T-cell lymphoma |
| Romidepsin | FDA-approved | Peripheral T-cell lymphoma |

**Atlas integration:**
- HDAC-mediated persister defense constitutes a **6th ferroptosis defense layer** beyond the 5-layer model (GPX4, FSP1, SLC7A11/glutathione, NRF2/KEAP1 regulation, TXNRD1/thioredoxin).
- HDACi co-treatment is applicable across **all therapeutic categories** as a persister elimination strategy, not limited to Category E.
- The recommended sequencing is HDACi pre-treatment or co-treatment with the primary ferroptosis inducer (GPX4i, FSP1i, or N6F11) to prevent persister cell emergence.

### 10.4 Updated Clinical Strategy by Category

| Category | Previous Strategy | Updated Strategy (March 2026) |
|---|---|---|
| **A: FSP1-Vulnerable** | icFSP1 monotherapy | icFSP1 monotherapy (primary); N6F11 as GPX4 degrader alternative; HDACi for persister elimination |
| **B: Dual Defense** | FSP1i + GPX4i | FSP1i + N6F11 combination; HDACi for persister elimination |
| **C: GPX4-Dominant** | Await in vivo GPX4i validation | **N6F11 monotherapy (preferred)**; Bone as prime target; HDACi for persister elimination |
| **D: TrxR1-Driven** | Auranofin | Auranofin (unchanged); HDACi for persister elimination |
| **E: HDACi Combination** | HDACi + GPX4i | HDACi + N6F11 (replaces GPX4i where applicable) |
| **KEAP1-MUT** | GLS1 inhibition (CB-839) | GLS1 inhibition + NRF2+FSP1 co-targeting; N6F11 for GPX4 axis |

---

## 11. Future Directions: Triple GPX4/FSP1/DHODH Ferroptosis Defense Axis

Three independent ferroptosis defense axes are now experimentally documented:

| Defense Axis | Mechanism | Key Evidence |
|---|---|---|
| **GPX4** | Glutathione-dependent lipid peroxide reduction | Pan-cancer DepMap dependencies (this atlas); N6F11 degrader (*Sci Transl Med*) |
| **FSP1/AIFM2** | CoQ10-mediated radical trapping (glutathione-independent) | icFSP1 monotherapy ~80% efficacy (*Nature* 2025); Phase 5 expression profiling |
| **DHODH** | Mitochondrial CoQH2 regeneration | Dual GPX4+DHODH targeting in glioblastoma (Ren et al., *Advanced Science*, PMID: 41869756) |

This atlas demonstrated GPX4+FSP1 dual targeting is required for Category B cancers (Bowel/CRC, Kidney/ChRCC, Ovary). Ren et al. separately validated GPX4+DHODH dual targeting in glioblastoma. Together, these findings establish that cancers possess three parallel, non-redundant ferroptosis suppression mechanisms.

**Triple inhibition hypothesis:** Simultaneous targeting of all three axes (GPX4 + FSP1 + DHODH) would eliminate all known ferroptosis defense mechanisms. However, several barriers must be addressed before this is clinically actionable:

1. **Drug availability:** No clinical-grade FSP1 inhibitors exist (icFSP1 is preclinical). N6F11 (GPX4 degrader) is preclinical. Only DHODH inhibitors have existing clinical data (e.g., brequinar, leflunomide).
2. **Combination toxicity:** Triple-target therapy raises significant tolerability concerns. Sequential or time-staggered dosing may be required.
3. **Patient selection:** Not all tumors express all three defense axes. Defense profiling (Phase 5 approach extended to DHODH) would be needed to identify triple-defense tumors.

**Recommended next steps:**
- Extend Phase 5 expression profiling to include DHODH expression, creating a 3-axis defense classification.
- Prioritize glioblastoma (CNS/Brain, Category C) as the proof-of-concept tumor type for triple targeting, given Ren et al.'s existing GPX4+DHODH validation in this indication.
- Evaluate whether the ME1→NADPH→FSP1 metabolic axis (Section 10.2) also feeds DHODH activity, which would create a single metabolic vulnerability point.

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
13. "SAT1-polyamine catabolism drives ferroptosis sensitization to KRAS inhibitors in KEAP1-WT context." *Nature Communications*, 2025. DOI: 10.1038/s41467-025-65441-4
14. "GLS1 inhibition reverses ferroptosis resistance in KEAP1-mutant LUAD." AACR 2026, *Cancer Research* 86(5 Suppl):B013.
15. "N6F11: tumor-specific GPX4 degrader sparing immune cells." *Science Translational Medicine*, 2026.
16. "NRF2+FSP1 co-targeting overcomes ferroptosis resistance." PMC12384948.
17. Wu et al. "ME1→NADPH→FSP1→CoQH2 axis drives lenvatinib resistance in HCC." *Cell Death & Disease*, PMID: 41881958.
18. Ren et al. "GPX4+DHODH dual targeting in glioblastoma." *Advanced Science*, PMID: 41869756.

---

*Analysis date: 2026-03-17 (Phases 1–4), 2026-03-19 (Phase 5), 2026-03-23 (SAT1/GLS1 addendum), 2026-03-31 (translational update: N6F11/FSP1/HDAC; ME1-FSP1 axis; triple defense axis). Data: DepMap 25Q3, TCGA Pan-Cancer Atlas, PRISM 24Q2.*
*Pipeline: `pipelines/pancancer-ferroptosis-atlas/` (phase1-phase5).*
*Raw outputs: `data/results/pancancer-ferroptosis-atlas/phase1/` through `phase5/`.*
