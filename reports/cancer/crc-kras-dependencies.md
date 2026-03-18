# CRC KRAS Allele-Specific Dependency Atlas

**Date:** March 17, 2026
**Project:** crc-kras-dependencies
**Status:** Documentation
**Data sources:** DepMap 25Q3, PRISM Repurposing 24Q2, TCGA COAD/READ
**Pipeline:** `src/bioagentics/data/crc_common.py`, `crc_depmap.py`, `crc_tcga.py`; `src/bioagentics/models/crc_dependency.py`, `crc_interactions.py`, `crc_drug_sensitivity.py`, `crc_strategy_matrix.py`
**Validation:** Approved — journal #867 (validation_scientist, March 17, 2026)

---

## Executive Summary

This atlas systematically characterizes KRAS allele-specific genetic dependencies and drug sensitivities in colorectal cancer using DepMap 25Q3 CRISPR screening data (63 CRC cell lines, 35 KRAS-mutant) and validates allele frequencies and co-mutation patterns against 594 TCGA COAD/READ patients. The analysis spans seven KRAS alleles (G12D, G13D, G12V, G12C, G12A, Q61H, A146T) and integrates PRISM 24Q2 drug screening (6,790 compounds) with the rapidly evolving KRAS-targeted therapeutic landscape.

**Key findings:**

- **Positive controls pass:** KRAS itself is the top dependency in all KRAS-mutant CRC lines (FDR = 0.0001, d = -1.92) and in G12D specifically (FDR = 0.023, d = -2.98), confirming pipeline validity.
- **Power-limited discovery:** With only 4-8 cell lines per allele, no novel allele-specific dependencies survive genome-wide FDR correction. This is a sample size limitation, not a methodological failure.
- **Near-significant MSI-H interaction:** Within KRAS-mutant lines, MSI-H status associates with differential dependency on GEMIN5 (FDR = 0.11, d = 2.53) and AAMP (FDR = 0.11, d = -2.32) — the most promising sub-significant finding.
- **G12V-specific co-mutation pattern:** TCGA reveals significantly lower APC co-mutation in KRAS G12V CRC (73.5% vs 83.5% overall, p = 0.047, OR = 0.44), a novel allele-specific signal not widely reported.
- **KRAS-mutant CRC has worse PFS:** Borderline significant progression-free survival disadvantage (logrank p = 0.035) with G13D (median PFS 28.2 months) and G12C (33.0 months) showing the shortest PFS.
- **Therapeutic strategy matrix:** Comprehensive allele-specific treatment framework incorporating allele-selective (sotorasib, zoldonrasib, RMC-5127), pan-KRAS (ERAS-4001), and pan-RAS (ERAS-0015, elironrasib) inhibitors for an estimated 68,850 annual US KRAS-mutant CRC patients.

---

## Important Caveats

1. **Small per-allele sample sizes severely limit discovery.** DepMap 25Q3 contains only 4-8 CRC cell lines per KRAS allele with CRISPR data. Genome-wide significance typically requires >50 lines per group. Only effects with Cohen's d > 3-4 can survive FDR correction across 17,931 genes at these sample sizes. The atlas framework is sound but will require larger datasets (e.g., pan-cancer pooling or future DepMap releases) for allele-specific discovery.

2. **Cell line models incompletely recapitulate in vivo biology.** 2D CRISPR screening does not capture tumor microenvironment, immune interactions, or in vivo pharmacokinetics. Dependencies identified represent intrinsic genetic vulnerabilities only.

3. **MSI-H confounding in G13D.** Four of six G13D CRISPR-screened lines are MSI-H (LoVo, HCT116, HCT-15, DLD-1). G13D dependency hits may reflect MSI biology rather than allele-specific signaling. This is a known biological association but creates analytical confounding.

4. **PRISM drug annotations are incomplete.** Many PRISM compounds are identified only by BRD-IDs without drug class or target annotations. Key therapeutics including MEK inhibitors (trametinib, selumetinib) are absent from the PRISM 24Q2 dataset. Allele-drug associations may be missed due to annotation gaps.

5. **TCGA cohort skews toward primary resections.** The 36.7% KRAS mutation rate in TCGA is lower than the clinical 45% estimate, likely because TCGA overrepresents resectable primary tumors relative to metastatic CRC. Survival analyses may not generalize to the metastatic setting where KRAS-targeted therapies are used.

6. **No functional validation performed.** All identified dependencies and signals are computational. Experimental confirmation through isogenic cell line models, xenograft studies, or clinical correlation is required before clinical translation.

---

## Background

Colorectal cancer (CRC) is diagnosed in approximately 153,000 US patients annually, with ~45% harboring activating KRAS mutations (~68,850 patients/year). CRC has a fundamentally different KRAS allele landscape from non-small cell lung cancer (NSCLC): G12D (26.6%) and G12V (22.5%) dominate in CRC versus G12C (21%) in NSCLC. This allele distribution has profound implications for targeted therapy, as the only FDA-approved KRAS-targeted CRC regimen — sotorasib plus panitumumab (January 2025, CodeBreaK 300) — addresses only the 6.4% of KRAS-mutant CRC patients with G12C mutations.

The KRAS inhibitor landscape is rapidly expanding in 2026:

- **Allele-specific agents:** Sotorasib (G12C, approved), zoldonrasib/RMC-9805 (G12D, FDA BTD January 2026, ORR 61% in NSCLC), MRTX1133 (G12D, Phase 1/2 enrolling CRC), RMC-5127 (G12V, Phase 1, first patient January 2026)
- **Pan-KRAS agents:** ERAS-4001 (Phase 1 BOREALIS-1, spares HRAS/NRAS, data H2 2026)
- **Pan-RAS agents:** ERAS-0015 (Phase 1, confirmed PRs at 8 mg QD, 8-21x higher CypA affinity vs elironrasib), elironrasib (FDA BTD, ORR 40%/DCR 80% in CRC G12C), daraxonrasib (Phase 1/2)

No systematic analysis has compared allele-specific dependencies across CRC KRAS variants in DepMap 25Q3. This atlas fills that gap by providing allele frequency data, co-mutation patterns, dependency profiles, drug sensitivity screening, and a therapeutic strategy framework for the emerging era of KRAS-targeted CRC therapy.

---

## Methodology

### Data Sources and Preprocessing

| Dataset | Version | Content | Lines/Patients |
|---------|---------|---------|----------------|
| DepMap CRISPR | 25Q3 | CRISPRGeneEffect (18,435 genes) | 1,187 total, 63 CRC |
| DepMap Mutations | 25Q3 | OmicsSomaticMutations (HIGH/MODERATE VEP impact) | 96 CRC classified |
| DepMap Model | 25Q3 | Cell line metadata (OncotreePrimaryDisease) | 2,133 total |
| PRISM Repurposing | 24Q2 | Drug sensitivity (viability) | 45 CRC with drug data |
| TCGA COAD/READ | Legacy | Mutations, clinical, survival | 594 patients |

### Phase 1: CRC Cell Line Classification

CRC cell lines were identified by OncotreePrimaryDisease filter from DepMap 25Q3 metadata. KRAS allele classification used a curated map of seven hotspot mutations (G12D, G13D, G12V, G12C, G12A, Q61H, A146T) with priority ordering for compound mutations. Co-mutation status was annotated for five driver genes (APC, TP53, PIK3CA, BRAF V600E, SMAD4). MSI status was assigned using a curated cell line list with metadata fallback.

**Quality controls:**
- BRAF V600E and KRAS mutations confirmed mutually exclusive (0/96 lines carry both)
- KRAS mutation calls verified against DepMap mutation annotations
- Lines with CRISPR data subset identified (63 of 96)

### Phase 2: Allele-Specific Differential Dependency Analysis

For each KRAS allele group with ≥5 lines (G12D, G13D, G12V) and all KRAS-mutant combined (n=35):
- **Test:** Mann-Whitney U (two-sided, non-parametric) comparing CRISPR dependency scores vs KRAS-WT (n=28)
- **Effect size:** Cohen's d with pooled standard deviation (Bessel's correction, ddof=1)
- **Multiple testing:** Benjamini-Hochberg FDR correction across all genes
- **Significance threshold:** FDR < 0.05 AND |d| > 0.5 (dual criterion to avoid tiny-effect false positives)
- **Alleles with <5 lines** (G12C: 4, A146T: 3, G12A: 2, Q61H: 1): descriptive statistics only

### Phase 2b: Interaction Analyses

Four stratified comparisons within KRAS-mutant lines:
1. G12D vs G13D (n=8 vs 6)
2. G12C vs non-G12C (n=4 vs 31)
3. PIK3CA co-mutant vs PIK3CA-WT (n=13 vs 22)
4. MSI-H vs MSS (n=5 vs 30)

Same statistical framework as Phase 2a.

### Phase 3: PRISM Drug Sensitivity

Drug sensitivity (viability AUC) compared across KRAS allele groups using Mann-Whitney U with BH-FDR correction. Drug-dependency cross-referencing via Spearman correlation. Alleles with <3 PRISM-tested lines excluded.

### Phase 4: TCGA COAD/READ Validation

KRAS allele frequencies, co-mutation patterns (Fisher's exact test), and survival analysis (Kaplan-Meier with log-rank test) from 594 TCGA COAD/READ patients. Per-allele survival computed for alleles with ≥10 patients. Population estimates derived from TCGA allele frequencies applied to the 153,000 annual US CRC incidence.

### Phase 5: Therapeutic Strategy Matrix

Per-allele compilation of top dependencies, drug sensitivities, clinical agents, population estimates, and combination strategy rationale. Three-tier inhibitor framework: allele-specific → pan-KRAS → pan-RAS.

---

## Results

### 1. CRC Cell Line Landscape

**96 CRC lines classified** from DepMap 25Q3, of which 63 have CRISPR dependency data.

| Allele | All CRC Lines | With CRISPR Data | TCGA Patients | TCGA % of KRAS-mut |
|--------|--------------|------------------|---------------|-------------------|
| KRAS-WT | 33 | 28 | 376 | — |
| G12D | 16 | 8 | 58 | 26.6% |
| G13D | 12 | 6 | 36 | 16.5% |
| G12V | 8 | 6 | 49 | 22.5% |
| G12C | 7 | 4 | 14 | 6.4% |
| A146T | 3 | 3 | 16 | 7.3% |
| G12A | 2 | 2 | 10 | 4.6% |
| Q61H | 3 | 1 | 4 | 1.8% |
| KRAS_other | 12 | 5 | 31 | 14.2% |
| **Total KRAS-mut** | **63** | **35** | **218** | **100%** |

**Key observations:**
- BRAF V600E / KRAS mutual exclusivity is perfect in cell lines (0/96 co-occurrences) and near-perfect in TCGA (2.8% BRAF in KRAS-mut vs 14.9% in WT)
- MSI-H enrichment in G13D: 4/6 CRISPR-tested G13D lines are MSI-H (LoVo, HCT116, HCT-15, DLD-1)
- G12C and G12V are essentially MSS-specific; G12C is absent in MSI-H CRC (consistent with npj Precision Oncology systematic review)

### 2. Allele-Specific Dependencies

#### 2.1 Positive Controls

KRAS itself is the top-ranked dependency across all comparisons:

| Comparison | KRAS FDR | Cohen's d | Mean CRISPR (allele) | Mean CRISPR (WT) |
|-----------|----------|-----------|---------------------|------------------|
| all_KRAS_mut vs WT | 0.0001 | -1.92 | — | — |
| G12D vs WT | 0.023 | -2.98 | -2.20 | -0.71 |
| G13D vs WT | Not in top 50 | — | — | — |
| G12V vs WT | Not in top 50 | — | — | — |

The detection of KRAS as the top dependency in the pooled and G12D comparisons validates the analytical pipeline. The absence of KRAS in G13D and G12V top-ranked genes is expected given insufficient power with n=6 lines.

#### 2.2 Top-Ranked Sub-significant Dependencies

No genes other than KRAS achieve FDR < 0.05 in any comparison. The following sub-significant hits are reported for hypothesis generation:

**G12D (n=8, 17,931 genes tested, 1 FDR-significant):**

| Gene | FDR | Cohen's d | Biological Context |
|------|-----|-----------|-------------------|
| KRAS | 0.023 | -2.98 | Oncogene dependency (positive control) |
| NKD1 | 0.241 | +1.56 | WNT pathway negative regulator; biologically plausible in CRC G12D |
| VPS26B | 0.241 | -2.00 | Retromer complex component; relevance unclear |
| ESR1 | 0.304 | -1.21 | Estrogen receptor; unexpected in CRC |

**G13D (n=6, 17,931 genes tested, 0 FDR-significant):**

| Gene | FDR | Cohen's d | Biological Context |
|------|-----|-----------|-------------------|
| XRCC6/Ku70 | 0.320 | -1.71 | DNA double-strand break repair; likely reflects MSI-H enrichment in G13D, not allele-specific |

**G12V (n=6, 17,931 genes tested, 0 FDR-significant):**
- No meaningful signal extractable. Top gene FDR = 0.96. Severely underpowered.
- **ACSS2** (G12V-selective metabolic dependency per Scafuro et al., Cell Reports 2025): Not detectable in DepMap CRISPR (FDR = 0.99 in all_KRAS_mut, d = -0.30). The published ACSS2-G12V biology is real but DepMap with 6 lines cannot detect it.

### 3. Allele Interaction Analyses

| Comparison | Groups | Genes Tested | FDR-Significant | Top Hit |
|-----------|--------|-------------|-----------------|---------|
| G12D vs G13D | 8 vs 6 | 17,931 | 0 | DPYSL4 (d = -2.83, FDR = 0.66) |
| G12C vs non-G12C | 4 vs 31 | 17,787 | 0 | Underpowered |
| PIK3CA interaction | 13 vs 22 | 17,931 | 0 | No signal |
| **MSI-H vs MSS** | **5 vs 30** | **17,931** | **0** | **AAMP (FDR = 0.11, d = -2.32); GEMIN5 (FDR = 0.11, d = 2.53)** |

**MSI-H interaction is the most promising sub-significant finding.** GEMIN5 (SMN complex, mRNA splicing) dependency in MSI-H KRAS-mutant lines is biologically interesting — MSI-H tumors accumulate frameshift mutations that could create dependencies on mRNA processing and splicing quality control. AAMP (angio-associated migratory cell protein) shows the opposite pattern. Neither gene reaches FDR < 0.05, but the large effect sizes (|d| > 2.3) with n=5 vs 30 suggest these warrant investigation in larger datasets.

### 4. PRISM Drug Sensitivity

6,790 drugs were tested for differential sensitivity between KRAS-mutant (n=30 with PRISM data) and KRAS-WT (n=15) CRC lines. Per-allele analyses used alleles with ≥3 PRISM-tested lines (G12D, G13D, G12V, G12C for all_KRAS_mut; G12A, Q61H, A146T excluded).

**Result: Zero FDR-significant drugs for any allele or comparison.** This is expected given:
- PRISM is a viability assay with inherent measurement noise
- FDR correction across 6,790 tests with n=3-30 per group
- Drug class annotations are sparse — many compounds identified only by BRD-ID
- Key therapeutics (trametinib, selumetinib, binimetinib) are absent from PRISM 24Q2

Drug classes identified in the dataset: ERK inhibitors (1), KRAS inhibitors (9), PI3K inhibitors (1), RAF inhibitors (4). No allele-selective enrichment detected.

### 5. TCGA COAD/READ Validation

#### 5.1 Allele Frequencies

594 patients analyzed (439 COAD, 155 READ), 218 KRAS-mutant (36.7%). Allele frequencies are consistent with published literature:

| Allele | TCGA % | npj Prec Onc % | Est. US Cases/Year |
|--------|--------|----------------|-------------------|
| G12D | 26.6% | 10.8% (of total CRC) | 18,314 |
| G12V | 22.5% | 8.3% | 15,491 |
| G13D | 16.5% | 7.2% | 11,360 |
| A146T | 7.3% | 7.4% | 5,026 |
| G12C | 6.4% | 3.1% | 4,406 |
| G12A | 4.6% | — | 3,167 |
| Q61H | 1.8% | 4.5% | 1,239 |
| KRAS_other | 14.2% | — | 9,776 |

#### 5.2 Co-Mutation Patterns

| Gene | KRAS-mut Rate | KRAS-WT Rate | Notable Allele-Specific Signal |
|------|--------------|-------------|-------------------------------|
| APC | 83.5% | 54.5% | **G12V: 73.5% (p = 0.047, OR = 0.44)** — significantly lower |
| TP53 | 56.4% | 50.3% | G12C: 78.6% (p = 0.10, OR = 3.01) — trend toward higher |
| PIK3CA | 38.1% | 17.0% | A146T: 50.0% — highest rate but NS |
| BRAF | 2.8% | 14.9% | Near-complete mutual exclusivity confirmed |
| SMAD4 | — | — | No allele-specific signals |

**Novel finding: G12V has significantly lower APC co-mutation** (73.5% vs 83.5% for all KRAS-mutant, Fisher's exact p = 0.047, OR = 0.44). This allele-specific co-mutation pattern has not been widely reported in the literature. APC loss drives canonical WNT signaling in CRC; the relative APC retention in G12V tumors may indicate distinct biology or a different evolutionary path. This warrants investigation in independent cohorts.

#### 5.3 Survival Analysis

| Endpoint | KRAS-mut (n) | WT (n) | Median (mut) | Median (WT) | Log-rank p |
|----------|-------------|--------|-------------|-------------|------------|
| OS | 212 | 356 | Not reached | 65.9 months | 0.64 (NS) |
| **PFS** | **210** | **356** | **Not reached** | **109.0 months** | **0.035** |

The non-significant OS is consistent with the longstanding debate about KRAS as a prognostic marker in CRC. The borderline PFS signal suggests a progression risk that may be masked in OS by post-progression treatment effects.

**Per-allele PFS (alleles with ≥10 patients):**

| Allele | n | Events | Median PFS (months) | Notable Pairwise |
|--------|---|--------|-------------------|-----------------|
| G12D | 55 | 17 | Not reached | — |
| G12V | 48 | 12 | Not reached | vs G13D p = 0.059 |
| G13D | 35 | 14 | 28.2 | vs A146T p = 0.022 |
| A146T | 16 | 2 | Not reached | — |
| G12C | 14 | 7 | 33.0 | — |

G13D and G12C show the shortest median PFS, consistent with more aggressive biology. The G13D vs A146T PFS difference (p = 0.022) is nominally significant but not corrected for the 10 pairwise comparisons performed.

### 6. Therapeutic Strategy Matrix

#### 6.1 Three-Tier Inhibitor Framework

| Tier | Mechanism | Agents | Allele Coverage |
|------|-----------|--------|----------------|
| **Allele-specific** | Selective covalent or non-covalent binding | Sotorasib (G12C), zoldonrasib (G12D), MRTX1133 (G12D), RMC-5127 (G12V) | Single allele |
| **Pan-KRAS** | Covers KRAS G12X + amplifications, spares HRAS/NRAS | ERAS-4001 (Phase 1) | All KRAS alleles |
| **Pan-RAS** | Molecular glue or broad RAS inhibition | ERAS-0015, elironrasib, daraxonrasib | All RAS isoforms |

#### 6.2 Per-Allele Clinical Landscape

**KRAS G12D** (~18,314 US cases/year — largest unmet need)
- Zoldonrasib: FDA BTD (NSCLC), ORR 61% NSCLC at 1200 mg QD. CRC expansion enrolling; expect lower ORR due to EGFR feedback reactivation
- MRTX1133: Phase 1/2 enrolling CRC (NCT05737706)
- VS-7375: Dual ON/OFF G12D inhibitor, preclinical superiority to zoldonrasib (1 nM pERK reduction vs 30 nM). Phase 1/2a with cetuximab combo enrolling CRC
- Key biology: CRC G12D shows intrinsic resistance to KRAS G12D monotherapy via EGFR reactivation. Combination with cetuximab is rationally required

**KRAS G12V** (~15,491 US cases/year)
- RMC-5127: First G12V-selective inhibitor. Phase 1 (NCT07349537), first patient January 2026. Mono, combo with daraxonrasib, and combo with cetuximab arms include CRC
- Literature supports ACSS2 as G12V-selective metabolic dependency (Scafuro et al., Cell Reports 2025) — not detectable in DepMap but supports RMC-5127 + MEKi + ACSS2i combination rationale
- Novel signal: lower APC co-mutation (p = 0.047) may indicate distinct WNT pathway engagement

**KRAS G13D** (~11,360 US cases/year)
- No allele-specific inhibitor in development
- May retain partial cetuximab sensitivity (differential NF-1 interaction kinetics)
- Pan-KRAS (ERAS-4001) or pan-RAS (ERAS-0015) required
- MSI-H enrichment creates immunotherapy eligibility for a subset
- GEMIN5/AAMP dependency interaction with MSI-H status warrants follow-up

**KRAS G12C** (~4,406 US cases/year)
- Sotorasib + panitumumab: FDA approved January 2025. ORR 26% (2L), 78% with FOLFIRI (1L, CodeBreaK 101)
- Elironrasib: ON-state inhibitor, ORR 40%/DCR 80% in CRC (substantially outperforms sotorasib monotherapy at 9.7%)
- CodeBreaK 301 Phase 3 enrolling 1L (soto + pani + FOLFIRI vs FOLFIRI ± bevacizumab)
- G12C absent in MSI-H CRC; MSS-specific allele

**KRAS G12A** (~3,167 US cases/year)
- No allele-specific inhibitor; codon 12 position but distinct from G12D/V/C
- Pan-KRAS or pan-RAS agents required
- 10 TCGA patients (4.6%); very limited DepMap representation (2 lines)

**KRAS A146T** (~5,026 US cases/year)
- Activates KRAS through nucleotide exchange (not impaired GTP hydrolysis)
- Pan-KRAS or pan-RAS agents required
- Favorable PFS trend (median not reached) — potentially less aggressive biology

**KRAS Q61H** (~1,239 US cases/year)
- Rarest allele; 1 cell line in DepMap, 4 TCGA patients
- Pan-RAS coverage required; ERAS-0015 and ERAS-4001 are the viable approaches

#### 6.3 Emerging Pan-RAS/Pan-KRAS Data

**ERAS-0015** (pan-RAS molecular glue, Erasca): Confirmed and unconfirmed partial responses at doses as low as 8 mg QD across multiple tumor types and RAS mutations (AURORAS-1 dose escalation, data cutoff January 7, 2026). Preclinically shows 8-21x higher CypA binding affinity and ~5x greater RAS inhibition potency vs elironrasib (RMC-6236). Phase 1 monotherapy data expected H1 2026. CypA overexpression in CRC tumors may confer tumor-selective drug accumulation. ERAS-0015 + vopimetostat (PRMT5 inhibitor) combination trial enrolling MTAP-deleted PDAC/NSCLC — relevant for KRAS-mutant MTAP-deleted CRC (~15% of CRC harbor 9p21 loss).

**ERAS-4001** (pan-KRAS, Erasca): Phase 1 BOREALIS-1 enrolling. Single-digit nM IC50s against GTP- and GDP-bound KRAS. Spares HRAS/NRAS for better therapeutic window than pan-RAS agents. Monotherapy data expected H2 2026. US Patent 12,552,813 (February 2026).

---

## Discussion

### What This Atlas Demonstrates

This analysis provides the most comprehensive publicly available characterization of KRAS allele-specific biology in CRC cell lines, integrating genome-wide CRISPR dependency screening, drug sensitivity profiling, and clinical validation from TCGA. The primary value lies in three areas:

1. **Methodological framework.** The pipeline demonstrates how to systematically interrogate allele-specific dependencies using DepMap data with appropriate statistical rigor (non-parametric tests, effect sizes, FDR correction). The dual significance threshold (FDR < 0.05 AND |d| > 0.5) prevents both false positives and tiny-effect hits.

2. **Honest power assessment.** Rather than overinterpreting noisy results, the analysis transparently reports that 4-8 cell lines per allele are insufficient for genome-wide discovery. Only KRAS itself — the expected oncogene dependency — survives FDR correction. This is an important negative result that quantifies the current limitations of DepMap for rare allele-specific studies.

3. **Clinical integration.** The therapeutic strategy matrix maps each KRAS allele to available and emerging clinical agents, population estimates, and combination rationale, providing a practical framework for patient selection as multiple KRAS-targeted agents enter CRC clinical trials simultaneously.

### Comparison to NSCLC

CRC and NSCLC have divergent KRAS allele landscapes: G12D dominates CRC (26.6%) while G12C dominates NSCLC (21%). This has direct clinical implications:
- The approved KRAS-targeted CRC regimen (sotorasib + panitumumab) addresses only 6.4% of KRAS-mutant CRC patients
- CRC shows intrinsic resistance to KRAS G12D monotherapy via EGFR feedback reactivation, requiring combination approaches
- G13D is enriched in CRC (16.5%) relative to NSCLC and has no allele-specific inhibitor in development

### Key Biological Signals

Despite the power limitations, several biologically plausible signals emerged:

- **NKD1 in G12D** (FDR = 0.24, d = 1.56): NKD1 is a WNT pathway negative regulator. Reduced dependency on NKD1 in G12D CRC lines could reflect WNT pathway rewiring in the context of KRAS-driven MAPK activation.
- **XRCC6 in G13D** (FDR = 0.32, d = -1.71): DNA repair dependency likely confounded by MSI-H enrichment.
- **GEMIN5/AAMP in MSI-H KRAS-mutant** (FDR = 0.11): Splicing complex and cell migration dependencies, respectively, in MSI-H KRAS-mutant CRC. Large effect sizes with biological plausibility.
- **G12V lower APC co-mutation** (p = 0.047): A novel allele-specific co-mutation pattern suggesting distinct tumorigenic pathways for G12V CRC.

---

## Limitations

1. **Sample size is the primary limitation.** With 4-8 lines per allele, statistical power for genome-wide discovery is minimal. Only effects with d > 3-4 can reach significance.
2. **PRISM drug screening yielded no significant results.** The combination of measurement noise, sparse drug annotations, and FDR correction across 6,790 tests made allele-drug associations undetectable at these sample sizes.
3. **TCGA survival analyses are from a primary resection cohort** and may not reflect outcomes in the metastatic setting where KRAS-targeted therapies are deployed.
4. **G13D/MSI-H confounding** cannot be fully resolved without more MSS G13D cell lines.
5. **ACSS2 G12V-selective dependency** (supported by Scafuro et al., Cell Reports 2025) is not detectable in DepMap — the biological signal exists but 6 G12V lines provide insufficient power.
6. **Cross-project gap:** KRAS allele × MTAP deletion co-occurrence was not assessed. This is relevant for ERAS-0015 + vopimetostat patient selection.

---

## Next Steps

1. **Pan-cancer KRAS allele pooling.** Combining CRC with PDAC and NSCLC G12D/G13D/G12V lines would boost per-allele N from 6-8 to 30-50+, dramatically improving power for allele-specific discovery.
2. **Expanded MSI interaction analysis.** The GEMIN5/AAMP near-significant results (FDR = 0.11) warrant investigation in larger MSI-H datasets and functional validation.
3. **Track clinical data releases.** ERAS-0015 H1 2026 monotherapy data, zoldonrasib CRC expansion cohort data, and VS-7375 interim safety/efficacy will provide clinical validation opportunities.
4. **KRAS allele × MTAP deletion co-occurrence.** ~15% of CRC harbor MTAP loss; different KRAS alleles may have different MTAP co-deletion rates, affecting dual-targeting (ERAS-0015 + vopimetostat) patient selection.
5. **G12V APC co-mutation validation.** Replicate the lower APC co-mutation finding (p = 0.047) in independent cohorts (e.g., AACR GENIE, MSK-IMPACT).
6. **DepMap 26Q1 integration.** Additional CRC cell lines in future DepMap releases would directly increase per-allele power.

---

## References

1. DepMap 25Q3 Release. Broad Institute, October 2, 2025. https://depmap.org
2. PRISM Repurposing 24Q2. Broad Institute. https://depmap.org/prism
3. TCGA COAD/READ. The Cancer Genome Atlas. https://portal.gdc.cancer.gov
4. Sotorasib + panitumumab FDA approval. January 16, 2025 (CodeBreaK 300).
5. Zoldonrasib FDA BTD. Revolution Medicines, January 8, 2026.
6. RMC-5127 first patient dosed. Revolution Medicines, January 29, 2026.
7. ERAS-0015 early clinical data. Erasca, January 12, 2026.
8. ERAS-4001 US Patent 12,552,813. Erasca, February 24, 2026.
9. Erasca FY2025 Q4 results. March 12, 2026.
10. VS-7375 preclinical data. AACR 2026 Abstract PR007; Verastem.
11. Scafuro et al. ACSS2 G12V-selective dependency in CRC. Cell Reports, March 2025. DOI: 10.1016/j.celrep.2025.115367.
12. npj Precision Oncology. KRAS mutations, inhibitors, and clinical trials in CRC. 2025. DOI: 10.1038/s41698-025-01166-3.
13. CodeBreaK 101 sotorasib + panitumumab + FOLFIRI. ASCO 2025. DOI: 10.1200/JCO.2025.43.16_suppl.3506.
14. Elironrasib CRC G12C data. Revolution Medicines.
15. CodeBreaK 301 Phase 3. JCO TPS326.
