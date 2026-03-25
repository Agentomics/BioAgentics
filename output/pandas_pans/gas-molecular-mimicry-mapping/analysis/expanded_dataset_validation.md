# Expanded Dataset Validation Report

**Task:** #1301
**Project:** gas-molecular-mimicry-mapping
**Division:** pandas_pans
**Analyst:** analyst
**Date:** 2026-03-24

---

## 1. Overview

The mimicry screen pipeline was rerun with two expansions:
- **Enn/Mrp M-like proteins:** 4 new sequences (2 Mrp: A0A5S4TFJ9/415aa, A0A660A269/388aa; 2 Enn: A0A5S4TF66/375aa, A0A8B6IYQ9/316aa) added to the combined GAS FASTA, plus pre-existing EnnX (A0A0H3C1S6/368aa) in M49
- **emm3.93 variant lineage:** 1,821 proteins from NCBI assembly GCA_046240755.1

Dataset grew from ~10,691 to ~12,516 proteins. Screen output: 52 filtered hits, 47 unique GAS-human pairs (previously 40 hits/pairs), 7 serotypes (previously 6).

## 2. Source Attribution of New Hits

### 2.1 Enn/Mrp Proteins: Zero Mimicry Hits

All 5 Enn/Mrp protein accessions (A0A5S4TFJ9, A0A660A269, A0A5S4TF66, A0A8B6IYQ9, A0A0H3C1S6) were confirmed present in:
- `proteomes/gas_combined.fasta` (pipeline input)
- `phase_weighting/gas_phase_annotations.csv` (classified as invasive-phase, weight=2.0)

**None appear in `hits_filtered.tsv` or any output result file.** They were screened by DIAMOND but failed to pass the filtering criteria (pident >= 40%, alignment >= 8aa, E-value <= 1e-3) against any of the 328 basal ganglia-enriched human targets.

**Interpretation:** Enn and Mrp are coiled-coil Ig-binding surface proteins (IgG-Fc and IgA-Fc binding, respectively). Their primary function is immune evasion via immunoglobulin binding, not molecular mimicry with host neuronal proteins. The coiled-coil domains share structural motifs with M protein but the sequence divergence from human neuronal targets is too large to produce mimicry hits above the 40% identity threshold. This is a **negative result that increases confidence** in the pipeline's specificity — structurally related GAS surface proteins that lack genuine neuronal mimicry are correctly filtered out.

### 2.2 emm3.93: 7 New Hits Matching Canonical M3 Profile

Seven HGO-prefixed proteins from the emm3.93 assembly passed filters:

| emm3.93 Protein | Human Target | pident | Bitscore | M3 Equivalent |
|-----------------|-------------|--------|----------|---------------|
| HGO6457784.1 | B4DHP5 (HSPA6) | 49.1% | 440 | DNAK_STRP3: 49.1%, 440 |
| HGO6456747.1 | P09104 (ENO2) | 49.1% | 394 | ENO_STRP3: 49.1%, 394 |
| HGO6457796.1 | B4DHW5 (MCCC1-like) | 47.4% | 345 | A0A0H2UWC3_STRP3: 47.4%, 345 |
| HGO6457550.1 | P04406 (GAPDH) | 46.3% | 267 | G3P_STRP3: 46.3%, 267 |
| HGO6458201.1 | P62333 (PSMC6) | 41.4% | 171 | A0A0H2UTC4_STRP3: 41.4%, 171 |
| HGO6457243.1 | B4DHP7 | 47.3% | 80.9 | A0A0H2UUC3_STRP3: 47.3%, 80.9 |
| HGO6456814.1 | B4DHP7 | 40.2% | 60.1 | A0A0H2UU37_STRP3: 40.2%, 60.1 |

**All 7 hits match the canonical M3 targets exactly** — same human proteins, same percent identities, bitscores within 0-2 points.

### 2.3 Remaining Hit Count Difference

The increase from 40 to 52 total hits (47 unique pairs) accounts as follows:
- **+7 hits:** emm3.93 (M3.93 serotype contributing to per-serotype hit totals)
- **+5 hits:** Duplicate UPI-prefixed entries from M12/M18 proteomes appearing in the combined file (e.g., UPI0000165A3D, UPI000000AFB6, UPI00000D9AB7, UPI0000001324, UPI000000AA6B each appear twice in hits_filtered.tsv). These are counting artifacts from per-serotype overlap, not new biology. The unique pair count (47) is the authoritative figure.

## 3. emm3.93 Conservation Validation

### 3.1 Mimicry Profile Identity

The developer's finding that emm3.93 shares identical mimicry targets with canonical M3 is **validated**. From the serotype_comparison_matrix:

| Target | M3 Bitscore | M3.93 Bitscore | M3 pident | M3.93 pident |
|--------|------------|----------------|-----------|--------------|
| B4DHP5 | 440 | 440 | 49.1% | 49.1% |
| B4DHP7 | 80.9 | 80.9 | 47.3% | 47.3% |
| B4DHW5 | 345 | 345 | 47.4% | 47.4% |
| P04406 | 267 | 267 | 46.3% | 46.3% |
| P09104 | 394 | 394 | 49.1% | 49.1% |
| P62333 | 171 | 171 | 41.4% | 41.4% |

Bitscores and identities are identical across all 6 shared targets. M3.93 does not hit AK9 or B4DHZ4 (same as M3).

### 3.2 Conservation Is Expected

This conservation is biologically expected for the following reasons:

1. **Mimicry targets are core genome housekeeping proteins** (DnaK, enolase, GAPDH, proteasomal subunits). These genes are under strong purifying selection and diverge minimally between GAS lineages.

2. **emm3.93 lineage divergence is concentrated in surface virulence factors** — the M protein hypervariable region, superantigen genes (speA, speC, ssa), and prophage-encoded virulence factors — not housekeeping genes that produce mimicry hits.

3. **The M protein itself is not a mimicry hit.** Despite being the primary variable surface antigen, neither M3 nor M3.93 M protein (HGO6457509.1, 581aa) passes the mimicry filters. The emm-type divergence that defines M3.93 as a variant lineage is therefore irrelevant to the mimicry landscape.

4. **Reference:** The emm3.93 lineage (associated with pediatric invasive GAS disease in Europe) is characterized by acquisition of novel prophage content and IS element insertions in the mga locus region, not by diversification of core metabolic genes (Akoolo et al. 2026, PMID 41236364).

### 3.3 Clinical Implication

The identical mimicry profile between M3 and M3.93 means the emerging M3.93 pediatric invasive GAS clade carries the **same molecular mimicry risk for PANDAS** as canonical M3. This does not represent a novel mimicry threat, but it does mean that:
- M3.93 should be expected to trigger the same autoimmune cross-reactivity as M3
- Diagnostic and therapeutic strategies targeting DnaK/HSPA6 or enolase/ENO2 mimicry would apply equally to M3.93 infections
- The displacement of M3 by M3.93 in pediatric populations does not change the mimicry-based PANDAS risk profile

## 4. Impact on Rankings

**Top-10 rankings are unchanged by the expansion:**

| Rank | Target | Pre-Expansion Score | Post-Expansion Score | Change |
|------|--------|-------------------|---------------------|--------|
| 1 | B4DHP5 (HSPA6) | 0.845 | 0.850 | +0.005 |
| 2 | P09104 (ENO2) | 0.784 | 0.788 | +0.004 |
| 3 | B4DHW5 (MCCC1-like) | 0.740 | 0.748 | +0.008 |
| 4 | P04406 (GAPDH) | 0.624 | 0.657 | +0.033 |
| 5 | B4DHP7 | 0.461 | 0.500 | +0.039 |
| 6 | P62333 (PSMC6) | 0.396 | 0.437 | +0.041 |
| 7 | B4DHZ4 | 0.205 | 0.293 | +0.088 |
| 8 | Q5TCS8 (AK9) | 0.113 | 0.206 | +0.093 |

All targets increased slightly in composite score due to the expanded serotype denominator (7 vs 6) and MMPred integration affecting normalization. **No rank changes occurred.** No new human targets were introduced by the expansion.

## 5. Conclusions

1. **Enn/Mrp proteins contribute zero mimicry hits** — a meaningful negative result confirming pipeline specificity. M-like proteins, despite structural similarity to M protein, lack sequence homology to human neuronal targets above the mimicry detection threshold.

2. **emm3.93 mimicry profile is identical to canonical M3** — all 6 shared targets match at identical pident and bitscore levels. Conservation is expected because mimicry operates through core genome proteins, not variable surface antigens.

3. **The expansion does not alter the scientific conclusions** of the project. HSPA6/DnaK remains the top novel candidate (score 0.850), ENO2 remains #2 (0.788), and GAPDH is correctly recovered as positive control at rank 4 (0.657).

4. **The 30% increase in total hits** (40→52) is predominantly an artifact of adding a 7th serotype (M3.93) and counting duplicates, not new biology. The unique pair count increased by only 17.5% (40→47), with the 7 new pairs all being M3.93 orthologs of existing M3 hits.
