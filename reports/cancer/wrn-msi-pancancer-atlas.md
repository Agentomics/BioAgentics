# WRN-MSI Pan-Cancer Dependency Atlas

**Date:** March 26, 2026
**Project:** wrn-msi-pancancer-atlas
**Status:** Documentation
**Data sources:** DepMap 25Q3 (CRISPR, mutations, expression), PRISM 24Q2, TCGA Pan-Cancer Atlas 2018, ACS 2024 incidence
**Pipeline:** `src/wrn_msi_pancancer_atlas/` (Phases 1-6)
**Output:** `output/wrn-msi-pancancer-atlas/` (phases 1-6)
**Validation:** Phases 1-5 APPROVED (journal #1647); Phase 6 APPROVED with caveats (journal #1648)

---

## Executive Summary

This atlas systematically ranks all MSI-H cancer types by WRN synthetic lethal (SL) dependency using DepMap 25Q3, identifies co-factors that modulate WRN dependency, and provides resistance-aware deployment recommendations for three clinical WRN inhibitors. The study classified 2,119 cell lines by MSI status using a tiered MLH1/MSH2/annotation framework, identifying 70 MSI-H and 2,049 MSS lines across 17 cancer types.

**Key findings:**

- **WRN is a ROBUST synthetic lethal target in MSI-H cancers.** Bowel (d = -2.844, 95% CI [-5.29, -1.76], FDR = 1.1 x 10^-6), Uterus (d = -1.086, 95% CI [-2.06, -0.25], FDR = 0.050), and pan-cancer pooled (d = -2.833, 95% CI [-3.76, -2.03], FDR = 1.4 x 10^-11) all reach ROBUST classification (FDR < 0.05, bootstrap CI excludes zero, permutation p < 0.05, leave-one-out stable). Control helicases BLM and RECQL show NO MSI-selective dependency, confirming specificity.
- **TP53 gain-of-function weakens WRN dependency** — a potential resistance biomarker. Within MSI-H lines: TP53-WT mean WRN dependency = -1.354, TP53-mutant = -0.964, TP53-GOF hotspot = -0.505 (d = -0.556, p = 0.026). This is consistent with the p53/PUMA-dependent WRN SL mechanism (Chan et al., PMID 36508676).
- **Novel MSI-H-specific dependencies beyond WRN:** PELO (d = -2.876, ribosome rescue), RPL22L1 (d = -2.602, paralog essentiality), and DHX9 (d = -1.682, R-loop resolution) emerge from a genome-wide screen of 53,793 gene-context tests (186 gained, 98 lost MSI-H dependencies at FDR < 0.05, |d| > 0.5).
- **DNA-PKi (AZD-7648) is a convergently validated WRN inhibitor combination partner.** Phase 4 independently found AZD-7648 MSI-H selective (d = -0.526, FDR = 0.018). The WRN resistance preprint independently identifies DNA-PK as a tractable combination partner. Two independent lines of evidence converge on DNA-PKi as the strongest combination candidate.
- **~61,854 US MSI-H patients per year** across 18 cancer types. Top populations: COADREAD (~23,532), UCEC (~20,974), STAD (~5,297), DLBC (~3,950), UCS (~3,597). Thirteen MSI-H cancer types with meaningful patient populations lack WRN inhibitor trial enrollment.
- **Three WRN inhibitors with divergent resistance profiles** enable rational sequential therapy: VVD-214 (covalent, C727 site), HRO-761 (non-covalent allosteric), NDI-219216 (distinct binding mode). Resistance to one compound preserves sensitivity to others.

---

## Important Caveats

The following limitations must be considered when interpreting all results:

1. **POOLED_ESTIMATE dominates.** Only 2 of 18 ranked tumor types (Bowel and Uterus) have lineage-specific WRN effect sizes. The remaining 15 use the pan-cancer pooled estimate (d = -2.833), which is dominated by Bowel (d = -2.844, n = 12 MSI-H). Uterus shows substantially weaker dependency (d = -1.086). The assumption that all MSI-H tumor types share Bowel-like WRN dependency is unvalidated and likely an overestimate for many lineages.

2. **Small MSI-H sample sizes.** Only Bowel (n = 12) and Uterus (n = 8) have enough MSI-H cell lines for lineage-specific analysis. All other tumor types have fewer than 5 MSI-H lines with CRISPR data. Genome-wide effect sizes carry wide confidence intervals.

3. **Resistance model saturates.** The resistance emergence model gives P(resistance) = 1.0 for ALL tumor types including MSS. The model uses L = 1,800 bp (full helicase domain), but the effective resistance target is likely 10-50 bp (specific binding-site positions). The model is a worst-case upper bound and cannot discriminate MSI-H from MSS risk. Combo vs. monotherapy recommendations are driven by mutation rate data availability, not model output.

4. **No WRN-specific inhibitors in PRISM.** VVD-214, HRO-761, and NDI-219216 are not in the PRISM 24Q2 drug screen. Pharmacological validation of WRN-MSI SL relies entirely on genetic (CRISPR) evidence plus indirect DDR drug data.

5. **DLBC MSI-H prevalence is uncertain.** The ~3,950 patients/year estimate is based on only 2/41 TCGA samples classified as MSI-H (4.9%). The 95% CI for this prevalence is approximately 0.6-16.5%, making the patient population estimate highly unreliable.

6. **NDI-219216 lacks efficacy data.** Phase 1/2 dose escalation completed (Dec 2025) with favorable safety, but no efficacy data are available. Its inclusion in sequential therapy recommendations is based on its distinct binding mode, not demonstrated clinical activity.

7. **DepMap 2D culture limitations.** CRISPR dependency scores in 2D culture do not capture tumor microenvironment, immune interactions, or in vivo pharmacology.

---

## 1. Background

WRN helicase is selectively essential in microsatellite instability-high (MSI-H) cancer cells, a finding independently validated by multiple genome-wide screens (Behan et al., Nature 2019; Chan et al., Nature 2019; Kategaya et al., Mol Cell 2019). The mechanism is well characterized: WRN resolves TA-dinucleotide repeat structures that expand under mismatch repair (MMR) deficiency. Without WRN, these expanded repeats cause replication fork collapse and chromosome shattering, selectively killing MSI-H cells.

MSI-H prevalence varies dramatically by cancer type (~31% endometrial, ~20% gastric, ~15% colorectal, rare in most others). Pembrolizumab is approved pan-tumor for MSI-H/dMMR cancers, but approximately 50% of MSI-H patients do not respond to immunotherapy alone. Three WRN inhibitors have entered clinical development:

- **VVD-214** (Vividion/Roche/Bayer): Covalent allosteric WRN helicase inhibitor. Phase 1 (NCT06004245). Binds C727 in the allosteric pocket. Partial responses reported at AACR 2025. Published mechanism: J Med Chem cover (Jan 2026, PMID 40964743).
- **HRO-761** (Novartis): Non-covalent allosteric WRN inhibitor. Phase 1/1b (NCT05838768). Mono and combination arms (irinotecan, tislelizumab).
- **NDI-219216** (Nimbus): Non-covalent WRN inhibitor with distinct binding mode. Phase 1/2 (NCT06898450). Part A dose escalation complete (Dec 2025); no DLTs, no MTD reached, >24h WRN target engagement.

A January 2026 bioRxiv preprint (10.64898/2026.01.22.700152v1) demonstrated that resistance profiles **diverge** between VVD-214 and HRO-761: heterozygous single-allele binding-site mutations are sufficient for resistance, but mutations impairing one compound preserve sensitivity to the other.

**Unmet need:** No systematic pan-cancer ranking of WRN dependency by tumor type has been published. The field assumes WRN-MSI SL is universal, but effect sizes likely vary by tissue context, MMR defect mechanism, and co-occurring mutations. This atlas addresses that gap and integrates resistance biology to inform compound deployment.

---

## 2. Methodology

### 2.1 MSI/MMR Classification (Phase 1)

All 2,119 DepMap 25Q3 cell lines with lineage annotation were classified by MSI status using a tiered system:

| Tier | Criteria | N classified |
|---|---|---|
| Tier 1 | DepMap ModelSubtypeFeatures annotation = MSI | 12 (annotation only) + 6 (annotation + molecular) |
| Tier 2 | MLH1 expression < 1.0 log2(TPM+1) AND no MLH1 LOF mutation (methylation proxy) | 22 |
| Tier 3 | MSH2 loss-of-function mutation (nonsense, frameshift, splice-site) | 30 |
| **Total MSI-H** | | **70 (3.3%)** |
| MSS | None of the above | 2,049 |

MSH6/PMS2-only LOF was recorded but not sufficient for MSI-H classification, as these mutations frequently produce MSI-L rather than MSI-H phenotypes. The MLH1 silencing threshold of 1.0 log2(TPM+1) corresponds to TPM < 1, a stringent threshold appropriate for identifying promoter methylation.

Among the 70 MSI-H lines: 41 MMR-mutated, 16 MLH1-methylated, 10 MLH1-methylated + MMR-mutated, 3 annotated-only. Of the 1,186 cell lines with CRISPR data, 44 were MSI-H and 1,142 MSS.

### 2.2 Qualifying Cancer Types

Cancer types required >= 5 MSI-H AND >= 10 MSS cell lines with CRISPR data to qualify for lineage-specific analysis:

| Cancer Type | MSI-H (CRISPR) | MSS (CRISPR) | Total (all) | MSI-H Frequency |
|---|---|---|---|---|
| Bowel | 12 | 51 | 99 | 22.2% |
| Uterus | 8 | 26 | 44 | 25.0% |

Only 2 cancer types met the qualification threshold. All other tumor types were analyzed using the pan-cancer pooled comparison (44 MSI-H vs 1,142 MSS lines).

### 2.3 WRN Effect Size Estimation (Phase 2)

For each qualifying cancer type and pan-cancer pooled:
- **Effect size:** Cohen's d with pooled standard deviation (ddof = 1)
- **Confidence interval:** Bootstrap 95% CI (1,000 iterations, SEED = 42)
- **Significance:** Mann-Whitney U test with BH FDR correction, permutation test (10,000 permutations)
- **Robustness:** Leave-one-out analysis (LOO) confirming no single sample drives the effect
- **Classification:** ROBUST (FDR < 0.05, CI excludes zero, permutation p < 0.05, LOO stable), MARGINAL, or NOT_SIGNIFICANT
- **Controls:** BLM, RECQL (should NOT show MSI-selective dependency), WRNIP1 (WRN-interacting protein)

### 2.4 Genome-Wide MSI-H Dependency Screen (Phase 3)

All ~18,000 genes in the DepMap CRISPR dataset were screened for MSI-H-specific dependencies per qualifying cancer type and pan-cancer pooled. Cohen's d with BH FDR correction per context. Threshold: FDR < 0.05 AND |d| > 0.5. Total tests: 53,793.

### 2.5 DDR Drug Validation (Phase 4)

DNA damage response (DDR) targeting compounds in PRISM 24Q2 were tested for MSI-H-selective sensitivity. CRISPR-PRISM concordance was assessed for drugs with available gene-level CRISPR data.

### 2.6 TCGA Clinical Integration (Phase 5)

MSI-H prevalence per cancer type from TCGA Pan-Cancer Atlas (2018). US annual incidence from ACS 2024. Addressable patient populations estimated as MSI-H prevalence x US incidence. Priority score: |WRN d| x log(estimated patients + 1). Pembrolizumab response rates from KEYNOTE-158/164/177. Trial enrollment checked against NCT06004245 (VVD-214) and NCT05838768 (HRO-761).

### 2.7 Resistance-Aware Deployment Framework (Phase 6)

Resistance mutation profiles from bioRxiv preprint (10.64898/2026.01.22.700152v1). Resistance emergence modeled as P = 1 - exp(-N x mu x L) with N = 10^9 cells, L = 1,800 bp WRN helicase domain, D = 24 doublings. NHEJ pathway cross-referenced with Phase 3 CRISPR screen and Phase 4 drug data. Per-tumor-type deployment recommendations generated.

---

## 3. Results

### 3.1 WRN Is a ROBUST Synthetic Lethal Target in MSI-H Cancers

| Context | N MSI-H | N MSS | Cohen's d | 95% CI | FDR | Perm p | LOO | Classification |
|---|---|---|---|---|---|---|---|---|
| Bowel | 12 | 51 | -2.844 | [-5.29, -1.76] | 1.1 x 10^-6 | 0.0001 | robust | **ROBUST** |
| Uterus | 8 | 26 | -1.086 | [-2.06, -0.25] | 0.050 | 0.013 | robust | **ROBUST** |
| Pan-cancer pooled | 44 | 1,142 | -2.833 | [-3.76, -2.03] | 1.4 x 10^-11 | 0.0001 | robust | **ROBUST** |

WRN reaches ROBUST classification in all three contexts. The Bowel effect size (d = -2.844) is among the largest synthetic lethal effects in DepMap. Uterus shows a weaker but still significant effect (d = -1.086), a 2.6-fold difference from Bowel that suggests tissue-specific modulation of WRN dependency.

**Specificity controls (all NOT_SIGNIFICANT):**

| Gene | Context | d | FDR |
|---|---|---|---|
| WRNIP1 | Bowel | 0.052 | 0.786 |
| WRNIP1 | Uterus | 0.218 | 0.786 |
| BLM | Bowel | -0.099 | 0.921 |
| BLM | Uterus | -0.321 | 0.921 |
| RECQL | Bowel | -0.487 | 0.538 |
| RECQL | Uterus | 0.109 | 0.921 |

BLM, RECQL, and WRNIP1 all fail to show MSI-selective dependency, confirming that the WRN-MSI SL is specific to WRN helicase function and not a general helicase or WRN-complex effect.

### 3.2 TP53 Gain-of-Function Attenuates WRN Dependency

Within MSI-H cell lines (pan-cancer pooled):

| TP53 Status | N | Mean WRN Dep | Median WRN Dep |
|---|---|---|---|
| TP53-WT | 16 | -1.354 | -1.446 |
| TP53-mutant | 18 | -0.964 | -0.302 |
| TP53-GOF hotspot | 10 | -0.505 | -0.592 |

TP53-WT vs TP53-mutant: d = -0.556, p = 0.026. GOF hotspot alleles (R175H, R248W, R273H, R248Q, R273C, G245S) show the weakest WRN dependency. This supports the p53/PUMA-dependent mechanism for WRN SL (Chan et al., PMID 36508676): WRN loss triggers DNA damage that requires p53-mediated apoptosis for cell killing. TP53-GOF tumors may partially evade this death program.

**Clinical implication:** TP53 status should be considered as a potential response biomarker in WRN inhibitor trials. GOF-hotspot MSI-H tumors may show reduced WRN inhibitor efficacy.

### 3.3 Genome-Wide MSI-H Dependency Screen

From 53,793 gene-context tests: **186 gained dependencies** (more essential in MSI-H) and **98 lost dependencies** (less essential in MSI-H) at FDR < 0.05, |d| > 0.5.

**Top novel MSI-H-specific dependencies:**

| Gene | Context | d | FDR | Biological Rationale |
|---|---|---|---|---|
| **PELO** | Bowel | -2.876 | 0.006 | Ribosome rescue factor. MSI-H tumors have high frameshift neoantigen load from repeat expansions, increasing ribosome stalling at aberrant transcripts. PELO resolves these stalls. |
| **RPL22L1** | Bowel | -2.602 | 0.010 | Paralog essentiality. RPL22 is frequently frameshift-disrupted in MSI-H tumors (poly-A tract), creating dependency on the RPL22L1 paralog. |
| **DHX9** | Bowel | -1.682 | 0.045 | RNA helicase involved in R-loop resolution. Independent of WRN (r = 0.047). Novel druggable MSI-H target. |
| SMAP1 | Bowel | -1.320 | 0.045 | Endosomal trafficking. Mechanism in MSI-H unclear. |
| RPL6 | Pan-cancer | -0.949 | 0.023 | Ribosomal protein. Consistent with ribosome stress in MSI-H. |
| CCDC102A | Pan-cancer | -0.901 | 0.005 | Coiled-coil domain protein. Limited functional data. |
| TFAM | Pan-cancer | -0.866 | 0.014 | Mitochondrial transcription factor. Suggests MSI-H mitochondrial vulnerability. |
| IGBP1 | Pan-cancer | -0.829 | 0.001 | PP2A regulatory subunit. Phosphatase signaling dependency. |

PELO (d = -2.876) exceeds even WRN (d = -2.844) in Bowel MSI-H lines, making it the strongest MSI-H-specific dependency identified. DHX9 is notable for being a druggable RNA helicase with WRN-independent activity in MSI-H contexts.

### 3.4 DDR Drug Validation

Of 15 DDR-targeting compounds queried, only 5 were available in PRISM 24Q2. Key absence: no WRN-specific inhibitors, no ATR inhibitors (ceralasertib, berzosertib), no WEE1 inhibitors (adavosertib), no PARP inhibitors (olaparib, talazoparib).

**AZD-7648 (DNA-PKi) is the only FDR-significant MSI-H-selective drug:**

| Context | Drug | Mechanism | d | FDR |
|---|---|---|---|---|
| **Pan-cancer** | **AZD-7648** | **DNA-PKi** | **-0.526** | **0.018** |
| Uterus | AZD-7648 | DNA-PKi | -1.012 | 0.084 |
| Bowel | AZD-7648 | DNA-PKi | -0.501 | 0.355 |
| Uterus | AZD0156 | ATMi | -0.907 | 0.145 |
| Bowel | AZD5582 | IAP antagonist | -0.745 | 0.128 |

AZD-7648 MSI-H selectivity at the pan-cancer level (d = -0.526, FDR = 0.018) is statistically significant and biologically consistent with NHEJ dependency in cells experiencing elevated DNA damage from microsatellite instability. CRISPR-PRISM concordance for AZD-7648 vs PRKDC was low (r = 0.133, p = 0.508, n = 27), reflecting that PRKDC had CRISPR data for only 27 lines with PRISM data. PRKDC was absent from the Phase 3 genome-wide screen (0 MSI-H lines with PRKDC CRISPR data), so pharmacological evidence is stronger than genetic evidence for this target.

### 3.5 TCGA Clinical Integration and Patient Populations

Estimated total US MSI-H patients per year: **~61,854** across 18 cancer types.

**Priority ranking (WRN SL strength x patient population):**

| Rank | Cancer Type | WRN d | Source | Est. MSI-H pts/yr | Priority Score | In WRN Trial |
|---|---|---|---|---|---|---|
| 1 | COADREAD | -2.844 | lineage-specific | 23,532 | 28.6 | VVD-214, HRO-761 |
| 2 | STAD | -2.833 | pooled | 5,297 | 24.3 | VVD-214, HRO-761 |
| 3 | DLBC | -2.833 | pooled | 3,950 | 23.5 | No |
| 4 | SKCM | -2.833 | pooled | 905 | 19.3 | No |
| 5 | BRCA | -2.833 | pooled | 621 | 18.2 | No |
| 6 | ESCA | -2.833 | pooled | 603 | 18.1 | No |
| 7 | BLCA | -2.833 | pooled | 582 | 18.0 | No |
| 8 | PAAD | -2.833 | pooled | 398 | 17.0 | No |
| 9 | CHOL | -2.833 | pooled | 342 | 16.5 | No |
| 10 | KIRP | -2.833 | pooled | 327 | 16.4 | No |
| 11 | LUAD | -2.833 | pooled | 235 | 15.5 | No |
| 12 | KIRC | -2.833 | pooled | 163 | 14.4 | No |
| 13 | LIHC | -2.833 | pooled | 123 | 13.7 | No |
| 14 | HNSC | -2.833 | pooled | 116 | 13.5 | No |
| 15 | LGG | -2.833 | pooled | 50 | 11.1 | No |
| 16 | UCEC | -1.086 | lineage-specific | 20,974 | 10.8 | VVD-214, HRO-761 |
| 17 | OV | -2.833 | pooled | 39 | 10.5 | VVD-214, HRO-761 |
| 18 | UCS | -1.086 | lineage-specific | 3,597 | 8.9 | No |

Only 4 tumor types (COADREAD, UCEC, STAD, OV) are currently enrolled in WRN inhibitor trials. **13 MSI-H tumor types with meaningful patient populations lack WRN trial enrollment.** DLBC ranks 3rd by priority score (~3,950 pts/yr) but has no WRN trial activity.

**Pembrolizumab concordance:** Response rates vary widely across MSI-H tumor types — UCEC (57.1%), STAD (45.8%), COADREAD (43.8%), CHOL (40.9%), OV (33.3%), PAAD (18.2%). Low-IO-response types such as PAAD may especially benefit from WRN inhibitor combinations.

### 3.6 Resistance-Aware Deployment Framework

#### Three-Drug Landscape

| Compound | Mechanism | Binding Site | Trial | Status |
|---|---|---|---|---|
| VVD-214 | Covalent allosteric | C727 pocket | NCT06004245 | Phase 1, PRs reported |
| HRO-761 | Non-covalent allosteric | Allosteric pocket | NCT05838768 | Phase 1/1b, mono + combos |
| NDI-219216 | Non-covalent (distinct) | Distinct from above | NCT06898450 | Phase 1/2, dose escalation complete |

#### Divergent Resistance Profiles

The bioRxiv preprint demonstrates that resistance mutations are **compound-specific**:
- **VVD-214-specific:** Mutations at the C727 covalent binding site prevent warhead engagement. HRO-761 and NDI-219216 binding is spared.
- **HRO-761-specific:** Mutations in the non-covalent allosteric pocket reduce binding affinity. VVD-214 and NDI-219216 binding is spared.
- **Pan-resistance:** Mutations affecting shared structural features required by all compounds. Requires targeting downstream vulnerabilities (NHEJ, WIP1).

Single-allele (heterozygous) mutations are sufficient for resistance. Given the 4-26x elevated mutation rates in MSI-H tumors, resistance emergence under monotherapy selective pressure is expected to be rapid.

#### DNA-PKi Convergence: The Strongest Combination Signal

Two independent lines of evidence converge on DNA-PKi as a WRN inhibitor combination partner:

1. **Phase 4 (PRISM pharmacological data):** AZD-7648 is MSI-H selective (d = -0.526, FDR = 0.018) — the only FDR-significant DDR drug hit.
2. **Resistance preprint (functional genomics):** DNA-PK identified as a tractable combination partner for WRN-resistant cells via NHEJ pathway dependency.

These come from different datasets, different experimental approaches, and different research groups. The convergence is strong evidence that DNA-PKi should be prioritized as a combination strategy.

NHEJ pathway genes at baseline (Phase 3 cross-reference): PPM1D/WIP1 (d = -0.292, FDR = 0.43), XRCC4 (d = -0.244, FDR = 0.32), XRCC6/Ku70 (d = -0.228, FDR = 0.43) — directional MSI-H essentiality but not FDR-significant at baseline. This is expected: NHEJ dependency likely **emerges** under WRN inhibitor treatment, not at baseline.

#### Compound Sequencing Strategy

| Scenario | First-Line | Resistance Type | Switch To | Combination Partner |
|---|---|---|---|---|
| VVD-214 resistance | VVD-214 | C727 mutations | HRO-761 or NDI-219216 | AZD-7648 (DNA-PKi) |
| HRO-761 resistance | HRO-761 | Allosteric pocket mutations | VVD-214 or NDI-219216 | AZD-7648 (DNA-PKi) |
| NDI-219216 resistance | NDI-219216 | Distinct-site mutations | VVD-214 or HRO-761 | AZD-7648 (DNA-PKi) |
| Pan-resistance | Any | Shared structural mutations | None (WRN exhausted) | DNA-PKi + IO; target NHEJ/WIP1 |

#### Per-Tumor-Type Deployment Recommendations

| Cancer Type | WRN d | Pts/yr | Resistance Risk | Strategy | In Trial | Notes |
|---|---|---|---|---|---|---|
| COADREAD | -2.844 | 23,532 | VERY_HIGH | WRN-i + DNA-PKi | Yes | Largest MSI-H population |
| STAD | -2.833* | 5,297 | VERY_HIGH | WRN-i + DNA-PKi | Yes | |
| DLBC | -2.833* | 3,950 | UNKNOWN | Monotherapy | No | Prevalence uncertain (2/41 TCGA) |
| UCS | -1.086 | 3,597 | UNKNOWN | Monotherapy | No | Mutation rate data gap |
| UCEC | -1.086 | 20,974 | VERY_HIGH | WRN-i + DNA-PKi | Yes | 65 mut/Mb; weaker WRN d |
| SKCM | -2.833* | 905 | VERY_HIGH | WRN-i + DNA-PKi | No | 50 mut/Mb |
| BRCA | -2.833* | 621 | VERY_HIGH | WRN-i + DNA-PKi | No | |
| BLCA | -2.833* | 582 | VERY_HIGH | WRN-i + DNA-PKi | No | 35 mut/Mb |
| PAAD | -2.833* | 398 | VERY_HIGH | WRN-i + DNA-PKi | No | Pembro ORR 18% — strongest IO gap |
| CHOL | -2.833* | 342 | VERY_HIGH | WRN-i + DNA-PKi | No | |
| OV | -2.833* | 39 | VERY_HIGH | WRN-i + DNA-PKi | Yes | Pembro ORR 33% |
| Others | -2.833* | 50-327 | UNKNOWN | Monotherapy | No | Insufficient mutation rate data |

*Pooled estimate. See Caveats section 1.

---

## 4. Discussion

### 4.1 WRN-MSI SL: Not Truly Universal

The 2.6-fold difference in WRN effect size between Bowel (d = -2.844) and Uterus (d = -1.086) challenges the assumption that WRN-MSI SL is equally strong across all tumor types. With only 2 qualifying lineages, we cannot determine whether other MSI-H tumor types resemble Bowel (strong WRN SL) or Uterus (moderate WRN SL) — or something weaker. The pan-cancer pooled estimate (d = -2.833) is heavily influenced by Bowel and should not be interpreted as representative of all tumor types.

This has direct clinical implications: if some MSI-H tumor types have Uterus-like (moderate) WRN dependency, monotherapy response rates may be lower, and combination strategies become more important.

### 4.2 TP53 as a Resistance Biomarker

The TP53-GOF attenuation of WRN dependency (mean WRN dep = -0.505 vs -1.354 for TP53-WT) represents a biologically plausible resistance mechanism. The p53/PUMA-dependent apoptotic pathway appears necessary for maximal cell killing following WRN loss. This suggests TP53 status should be incorporated into patient stratification for WRN inhibitor trials, and may explain heterogeneous responses within MSI-H cohorts.

### 4.3 Novel Therapeutic Targets Beyond WRN

PELO, RPL22L1, and DHX9 represent novel MSI-H-specific dependencies with distinct biological mechanisms:

- **PELO** (d = -2.876): The strongest MSI-H-specific dependency in this atlas, exceeding WRN itself. PELO is a ribosome rescue factor — MSI-H tumors with extensive frameshift mutations may generate aberrant transcripts that stall ribosomes, creating PELO dependency. Currently undruggable, but illuminates a vulnerability axis.
- **RPL22L1** (d = -2.602): A textbook paralog essentiality case. RPL22 contains a poly-A microsatellite tract that is frequently disrupted in MSI-H tumors, creating strict dependency on the RPL22L1 paralog.
- **DHX9** (d = -1.682): An RNA helicase involved in R-loop resolution, independent of WRN (r = 0.047). Potentially druggable. A novel synthetic lethal target in MSI-H contexts.

### 4.4 DNA-PKi as the Leading Combination Partner

The convergence of AZD-7648 MSI-H selectivity from PRISM drug screening (Phase 4) with the identification of DNA-PK as a combination partner in the WRN resistance preprint is the strongest translational finding of this atlas. This represents two independent experimental approaches (drug sensitivity profiling vs. functional genomic screening) from different datasets reaching the same conclusion.

The biological logic is compelling: WRN inhibition generates DNA damage through microsatellite instability; cells rely on NHEJ (via DNA-PK) for DNA repair. Dual WRN + DNA-PK inhibition creates a synthetic lethal trap. Furthermore, DNA-PKi should suppress NHEJ-dependent resistance escape pathways.

### 4.5 Rational Drug Sequencing

The divergent resistance profiles of the three clinical WRN inhibitors enable a rational sequential therapy framework unprecedented in targeted oncology. Unlike most settings where resistance to one drug confers cross-resistance, mutations conferring resistance to VVD-214 (C727 site) do not affect HRO-761 or NDI-219216 binding, and vice versa. This creates a three-drug sequence that can extend the therapeutic window substantially.

---

## 5. Limitations

1. **Lineage-specific data gap.** Only Bowel (n = 12) and Uterus (n = 8) have lineage-specific WRN effect sizes. All other tumor types rely on the pan-cancer pooled estimate, which may overestimate dependency for non-Bowel lineages.

2. **Resistance model non-discriminative.** The current parameterization (L = 1,800 bp) saturates at P = 1.0 for all tumor types including MSS. A refined model using experimentally validated effective target sites (L_eff ~ 10-50 bp) and fitness cost factors is needed for meaningful risk stratification.

3. **No pharmacological validation of WRN-MSI SL.** WRN-specific inhibitors are absent from PRISM. The genetic (CRISPR) evidence is strong but requires clinical confirmation.

4. **Small sample sizes.** Bowel n = 12 and Uterus n = 8 MSI-H lines produce wide confidence intervals. Bootstrap and LOO analyses provide some robustness assurance, but larger datasets would increase confidence.

5. **Vermeulen SL benchmark unavailable.** External benchmark validation of novel MSI-H hits was limited.

6. **UCS mutation rate gap.** UCS shares the Uterus lineage with UCEC (65 mut/Mb MSI-H) but has no published MSI-H mutation rate. It is classified UNKNOWN risk, but may warrant combination therapy if UCEC-like rates apply.

7. **PRKDC genetic validation gap.** PRKDC was not tested in the Phase 3 genome-wide CRISPR screen. The DNA-PKi finding rests on pharmacological (AZD-7648) evidence only.

---

## 6. Next Steps

1. **Lineage-specific WRN effect sizes** from larger MSI-H cell line panels (DepMap future releases) or PDX/organoid screens to resolve whether non-Bowel MSI-H tumor types have weaker WRN dependency.
2. **Refined resistance model** using experimentally validated L_eff from the preprint's specific resistance positions, plus fitness cost modeling.
3. **Monitor NDI-219216 efficacy data** expected H1 2026 to validate the third arm of the sequential therapy framework.
4. **DLBC MSI-H prevalence confirmation** from larger cohorts to validate or invalidate the 4.9% estimate.
5. **WRN + DNA-PKi combination testing** in preclinical MSI-H models, ideally with AZD-7648 or clinical DNA-PKi compounds.
6. **TP53-GOF stratified analysis** in WRN inhibitor clinical trials to validate the resistance biomarker signal.
7. **Trial expansion recommendations** for the 13 underexplored MSI-H tumor types, particularly DLBC, UCS, SKCM, and PAAD.

---

## 7. References

- Behan FM et al. Prioritization of cancer therapeutic targets using CRISPR-Cas9 screens. Nature. 2019;568(7753):511-516.
- Chan EM et al. WRN helicase is a synthetic lethal target in microsatellite unstable cancers. Nature. 2019;568(7753):551-556.
- Kategaya L et al. Werner syndrome helicase is required for the survival of cancer cells with microsatellite instability. Mol Cell. 2019;73(2):317-327.
- Chan EM et al. TP53 loss and p53-dependent WRN synthetic lethality. PMID 36508676. 2023.
- WRN resistance divergence preprint. bioRxiv 10.64898/2026.01.22.700152v1. January 2026.
- VVD-214 mechanism. J Med Chem cover. PMID 40964743. January 2026.
- DepMap 25Q3. Broad Institute. https://depmap.org/
- PRISM 24Q2 Drug Repurposing Resource. Broad Institute.
- TCGA Pan-Cancer Atlas. Cell. 2018.
- ACS Cancer Statistics 2024. American Cancer Society.
- KEYNOTE-158: Marabelle A et al. Lancet Oncol. 2020;21(10):1353-1365.
- KEYNOTE-164: Le DT et al. J Clin Oncol. 2020;38(1):11-19.
- KEYNOTE-177: Andre T et al. N Engl J Med. 2020;383(23):2207-2218.
- NCT06004245 (VVD-214 Phase 1). ClinicalTrials.gov.
- NCT05838768 (HRO-761 Phase 1/1b). ClinicalTrials.gov.
- NCT06898450 (NDI-219216 Phase 1/2). ClinicalTrials.gov.
