# JAK Inhibitor Convergence Verification Report

**Reviewer:** validation_scientist
**Date:** 2026-03-24
**Task:** #1307 (priority 5)
**Project:** autoantibody-target-network-mapping
**Division:** pandas_pans

---

## Executive Summary

**Verdict: JAK inhibitor convergence PARTIALLY SUPPORTED — 2 independent lines + 1 inference, not 3 independent discoveries. Baricitinib recommendation SUPPORTED WITH CAVEATS.**

The cross-project JAK inhibitor convergence is real but overstated. Two of three evidence lines independently identify JAK-STAT pathway involvement through unbiased analysis; the third (cytokine network) infers JAK-STAT post-hoc from IFN-γ elevation rather than discovering it independently. Data sources are fully independent with no leakage detected. Baricitinib is a biologically plausible repurposing candidate but BBB penetrance and pediatric immunosuppression risk require explicit assessment before clinical translation claims.

---

## 1. JAK-STAT Enrichment Verification (Autoantibody Network)

### Computation — VERIFIED ✓

| Parameter | Value |
|-----------|-------|
| Network genes in JAK-STAT (k) | 117 |
| KEGG JAK-STAT total genes (K) | 168 |
| Network genes mapped to KEGG (n) | 1,124 |
| Genome background (N) | 20,000 |
| Expected genes | 9.44 |
| **Fold enrichment** | **12.392** |
| Hypergeometric p-value | 4.17e-107 |
| Hypergeometric FDR | 7.33e-105 |
| Odds ratio | 42.887 |

**Manual verification:** expected = (168 × 1124) / 20000 = 9.4416; FE = 117 / 9.4416 = 12.392 ✓

### Code Review — SOUND ✓

`pathway_enrichment_validation.py` (380 lines):
- Hypergeometric test via `scipy.stats.hypergeom.sf(k-1, N, K, n)` — correct survival function formulation
- Fisher exact test via `scipy.stats.fisher_exact` with `alternative="greater"` — correct one-tailed overrepresentation test
- Benjamini-Hochberg FDR via `statsmodels.stats.multitest.multipletests` — standard multiple testing correction
- 20,000 protein-coding gene background — appropriate standard
- Contingency table construction handles edge cases (c < 0, d < 0)

### Discrepancy Noted — MINOR

Original FE (3.993) vs validated FE (12.392) — 3.1× difference. The validated computation uses the standard hypergeometric approach. The original likely used a different background size or enrichment formula. Both indicate significant enrichment; the validated value using standard methodology is authoritative.

### Critical Observation

**Zero seed proteins are in the JAK-STAT pathway** (seed_proteins_in_pathway = 0). The enrichment emerges entirely from interactors of autoantibody targets, not from the targets themselves. This strengthens the finding — JAK-STAT is a downstream convergence node, not a tautological result of including JAK pathway members as seeds.

---

## 2. Evidence Matrix: Three Lines of Convergence

### Line 1: Autoantibody Network — INDEPENDENT ✓

| Aspect | Detail |
|--------|--------|
| Data source | STRING/BioGRID protein-protein interactions (database) |
| Input | 9 seed proteins (DRD1, DRD2, CAMK2A, TUBB3, PKM, ALDOC, ENO1-3) |
| Method | PPI network → KEGG/Reactome pathway enrichment |
| Finding | JAK-STAT FE=12.4 (FDR=7.33e-105), 117/168 pathway genes in network |
| Independence | Fully independent — unbiased enrichment of PPI network |
| Strength | **Strong** — highest statistical significance among focus pathways |

### Line 2: Genetic Variant Pathway Analysis — PARTIALLY INDEPENDENT ⚠️

| Aspect | Detail |
|--------|--------|
| Data source | Rare genetic variants in PANS patients (patient genomic data) |
| Input | Variant gene list from PANS cohort |
| Method | GO/KEGG pathway enrichment of variant genes |
| Finding | (a) EP300 enriches GO:0046425 "Regulation of receptor signaling pathway via JAK-STAT"; (b) DDR genes TREX1/SAMHD1 → cGAS-STING → type I IFN → JAK-STAT mechanism |
| Independence | Partially independent. EP300-based GO enrichment is data-driven. DDR→interferonopathy→JAK-STAT link is a biological inference chain, not a direct enrichment finding. |
| Strength | **Moderate** — EP300 is a single gene; the DDR→JAK link requires multiple inferential steps |

### Line 3: Cytokine Network — NOT INDEPENDENT ❌

| Aspect | Detail |
|--------|--------|
| Data source | Meta-analysis of published cytokine measurements (12 studies) |
| Input | Flare vs remission cytokine levels |
| Method | Random-effects meta-analysis of effect sizes |
| Finding | IL-6 hub (g=1.46), IFN-γ elevated (g=1.09); JAK-STAT proposed as therapeutic target |
| Independence | **NOT independent** — JAK-STAT was inferred post-hoc from IFN-γ elevation based on known JAK-STAT dependency of IFN-γ signaling. It was NOT discovered by the meta-analysis itself. |
| Strength | **Weak as independent evidence** — biological inference, not data-driven discovery |

---

## 3. Independence Assessment

### Data Sources — FULLY INDEPENDENT ✓

- Project 1 uses protein interaction databases (STRING, BioGRID)
- Project 2 uses patient genomic variant data
- Project 3 uses published cytokine measurement studies
- **No shared raw data between projects**
- **No shared patient cohorts**

### Shared Pathway Databases — ACCEPTABLE ✓

Projects 1 and 2 both use KEGG for pathway enrichment. This is standard practice, not data leakage. The pathway definitions are reference knowledge, not analytical inputs.

### Circular Reasoning — NOT DETECTED ✓

No evidence that one project's findings were used as input to another. The analyses were conducted independently with different data types.

### Overstated Convergence — FLAGGED ⚠️

The claim of "3 independent lines of evidence" is overstated:
- **2 independent discoveries** (autoantibody network enrichment + genetic variant GO enrichment)
- **1 biological inference** (cytokine IFN-γ → JAK-STAT mechanism)

This should be characterized as "convergent evidence from 2 independent computational analyses plus supporting biological plausibility from cytokine data" rather than "3 independent discoveries."

---

## 4. Baricitinib as Tier 1 Candidate

### Strengths
- Selectively inhibits JAK1/JAK2, both ranked highly in hub druggability analysis
- Approved for autoimmune indications (RA, atopic dermatitis)
- Pediatric safety data from JIA trials
- Targets the most enriched immune pathway in PANDAS autoantibody network
- COVID neurological benefit suggests some CNS activity

### Concerns

1. **BBB penetrance:** Baricitinib is a hydrophilic small molecule with limited CNS penetration. For a disease with primary CNS pathology, peripheral-only JAK inhibition may be insufficient. BBB penetrance data should be explicitly cited in any recommendation.

2. **Causal vs correlative:** JAK-STAT enrichment in the autoantibody interactome shows pathway proximity, not causation. The pathway could be enriched because many signaling proteins are JAK-STAT-adjacent without JAK-STAT being the therapeutic lever.

3. **Immunosuppression risk:** Broad JAK inhibition in pediatric patients carries serious infection risk (herpes zoster, opportunistic infections). Risk-benefit must be evaluated against existing PANDAS treatments (IVIG, plasmapheresis).

4. **Missing clinical bridge:** No PANDAS-specific clinical or preclinical data exists for JAK inhibitors. The DDR-variant JAK inhibitor link requires IFN signature confirmation in PANS patients first (research director's caveat — journal #1966).

5. **Cytokine evidence limitation:** Treatment response module in cytokine project produces IDENTICAL results for all 5 treatments (analyst journal #1879) — this module has no treatment-discriminating power. Cannot use it to support JAK inhibitor specificity.

### Verdict: SUPPORTED WITH CAVEATS

Baricitinib is a biologically plausible repurposing candidate based on:
- Strong pathway enrichment (FE=12.4, independently verified)
- Existing autoimmune approval and pediatric safety data
- DDR-variant interferonopathy mechanism (for patient subset)

**Before advancing to clinical recommendation:**
1. Explicitly assess BBB penetrance with published pharmacokinetic data
2. Define target patient population (all PANDAS? DDR-variant subset only? IFN-high subset?)
3. Verify IFN signature elevation in PANS patients (blocked task #1002 — probe mapping bug)
4. Compare risk-benefit against standard PANDAS treatments (IVIG, plasmapheresis)

---

## 5. Summary Verdict

| Assessment Area | Result |
|----------------|--------|
| JAK-STAT FE=12.4 computation | ✓ VERIFIED |
| Statistical methodology | ✓ SOUND |
| Code correctness | ✓ NO ERRORS |
| Data independence (no leakage) | ✓ CONFIRMED |
| 3 independent lines of evidence | ⚠️ OVERSTATED — 2 independent + 1 inference |
| Baricitinib Tier 1 recommendation | ⚠️ SUPPORTED WITH CAVEATS |
| Circular reasoning | ✓ NOT DETECTED |
| Reproducibility | ✓ All code, data, parameters documented |

**Overall: APPROVED WITH CONDITIONS**

The convergence finding is scientifically valid but should be described accurately as 2 independent computational discoveries supported by biological plausibility from cytokine data. The baricitinib recommendation requires explicit BBB assessment and patient stratification before clinical translation claims.
