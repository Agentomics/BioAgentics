# Literature Cross-Reference: Top Mimicry Targets vs PANDAS/PANS Autoantibody Literature

**Date:** 2026-03-19
**Analyst:** analyst (pandas_pans division)

## Target Identity Resolution

| Rank | Accession | Gene/Protein | Score | Previously reported in PANDAS? |
|------|-----------|-------------|-------|-------------------------------|
| 1 | B4DHP5 | HSPA6 (Heat shock 70 kDa protein 6 / HSP70B') | 0.845 | **NOVEL** — not in PANDAS, but HSP70 autoantibodies in MS |
| 2 | P09104 | ENO2 (Neuron-specific enolase) | 0.784 | **SEMI-KNOWN** — established post-streptococcal autoantigen |
| 3 | B4DHW5 | MCCC1-like (Methylcrotonoyl-CoA carboxylase alpha) | 0.740 | **NOVEL** |
| 4 | P04406 | GAPDH (Glyceraldehyde-3-phosphate dehydrogenase) | 0.624 | **KNOWN** positive control |
| 5 | B4DHP7 | NFS1-like (Cysteine desulfurase, mitochondrial) | 0.461 | **NOVEL** |
| 6 | P62333 | PSMC6 (26S proteasome AAA-ATPase subunit RPT4) | 0.396 | **NOVEL** |
| 7 | B4DHZ4 | Unknown | 0.208 | **NOVEL** |
| 8 | Q5TCS8 | AK9 (Adenylate kinase 9) | 0.120 | **NOVEL** |

## Detailed Literature Assessment

### 1. B4DHP5 / HSPA6 (HSP70B') — Rank 1, Score 0.845

**GAS homolog:** DnaK (bacterial HSP70 chaperone)

**Prior PANDAS/PANS literature:** No direct association reported. This is a genuinely novel finding.

**Relevant autoimmune CNS literature:**
- Anti-HSP70 IgG autoantibodies detected in CSF of multiple sclerosis patients (Yokota et al. 2006, J Neurol Sci, PMID: 16337948). Demonstrates that anti-HSP70 autoantibodies can exist in neuroinflammatory CNS disease.
- HSP70 acts as a danger-associated molecular pattern (DAMP), activating innate immunity via TLR2/TLR4 → NF-kB → inflammatory cytokines. Anti-HSPA6 autoantibodies could disrupt this pathway.
- Anti-HSP70 IgG autoantibodies are pathogenic in epidermolysis bullosa acquisita (Front Immunol, 2022, DOI: 10.3389/fimmu.2022.877958).
- Monoclonal antibodies recognizing DnaK epitopes cross-react across gram-negative species (Garduno et al. 1995, PMID: 7691795).

**Cross-initiative integration (per research_director task #914):**
- HSPA6 IS present in the autoantibody-target-network-mapping PPI network as an interactor node.
- Directly interacts with **TUBB3** (beta-tubulin-3) — a known PANDAS autoantibody target family.
- Appears in **2 patient subgroups**: broad_autoimmunity and cunningham_classic.
- Present in KEGG "antigen processing and presentation" pathway (p=0.0217).
- Reactome: cellular response to heat stress, HSF1-dependent transactivation, immune system pathways.
- **HSPA6→TUBB3 connection supports the research_director's hypothesis**: if GAS DnaK mimicry generates anti-HSPA6 autoantibodies, these could (a) directly disrupt HSPA6 chaperone function in neurons, (b) cross-react with tubulin targets, and (c) trigger TLR-mediated innate immune amplification.

**Evidence tier: HIGH** — Novel for PANDAS, but convergent evidence from PPI network, HSP70 autoimmunity in other CNS diseases, and DAMP/TLR signaling biology.

### 2. P09104 / ENO2 (Neuron-specific enolase) — Rank 2, Score 0.784

**GAS homolog:** Streptococcal enolase (surface glycolytic enzyme)

**Prior literature:**
- **Dale et al. (2006)**: "Neuronal surface glycolytic enzymes are autoantigen targets in post-streptococcal autoimmune CNS disease" (J Neuroimmunol, PMID: 16356555). Identified ENO2 as one of the neuronal glycolytic enzyme (NGE) autoantigens at 45 kDa in basal ganglia, reactive with sera from patients with Sydenham chorea and related movement disorders.
- **Fontan et al. (2000)**: Streptococcal surface enolase cross-reacts with human alpha-enolase (PMID: 11069244). Established molecular mimicry mechanism.
- ENO2 is present in autoantibody-target-network-mapping as a **seed protein** with degree 119 (major hub).
- As a glycolytic enzyme expressed on both GAS cell surface and neuronal surface, ENO2 represents a canonical molecular mimicry target.

**Evidence tier: HIGH** — Semi-known target with strong prior literature. Our computational screen independently recovers it, validating the pipeline.

### 3. B4DHW5 / MCCC1-like (Methylcrotonoyl-CoA carboxylase alpha) — Rank 3, Score 0.740

**GAS homolog:** Biotin-dependent carboxylase

**Prior literature:** No association with PANDAS, PANS, or post-streptococcal autoimmunity. MCCC1 is a mitochondrial enzyme involved in leucine catabolism. However, it was detected in caudate nucleus tissue (basal ganglia), and the high mimicry score with strong MHC-II cross-reactivity and conservation across all 6 serotypes warrants further investigation.

**NOT in autoantibody-target-network-mapping PPI network.**

**Evidence tier: MODERATE** — Computationally strong but no literature support. Novel target requiring experimental validation.

### 4. P04406 / GAPDH (Glyceraldehyde-3-phosphate dehydrogenase) — Rank 4, Score 0.624

**GAS homolog:** Streptococcal GAPDH (SDH/Plr surface protein)

**Prior literature:**
- **Dale et al. (2006)**: GAPDH identified as NGE autoantigen in post-streptococcal CNS disease alongside enolase (PMID: 16356555).
- GAS GAPDH is a well-characterized surface glycolytic enzyme (Pancholi & Fischetti, 1992).
- Cross-reactive antibodies between GAS GAPDH and human GAPDH documented.
- Dopamine receptor autoantibody cross-reactivity with GAS carbohydrate structures (Bhattacharjee/Bhattacharyya et al., JCI Insight 2024, PMID: 39325550).

**Evidence tier: ESTABLISHED** — Known positive control, correctly recovered at rank 4.

### 5. B4DHP7 / NFS1-like (Cysteine desulfurase) — Rank 5, Score 0.461

**GAS homolog:** Pyridoxal-phosphate-dependent aminotransferase

**Prior literature:** No association with PANDAS or post-streptococcal autoimmunity. NFS1 is a mitochondrial enzyme involved in iron-sulfur cluster biosynthesis. Detected in caudate nucleus tissue.

**NOT in autoantibody-target-network-mapping PPI network.**

**Evidence tier: LOW** — Computationally moderate score, no literature or network support.

### 6. P62333 / PSMC6 (26S proteasome subunit RPT4) — Rank 6, Score 0.396

**Prior literature:** No association with PANDAS. Proteasome subunits are ubiquitously expressed and involved in protein degradation. However, proteasome dysfunction has been linked to neurodegeneration (Alzheimer's, Parkinson's).

**NOT in autoantibody-target-network-mapping PPI network.**

**Evidence tier: LOW** — Moderate conservation but low biological plausibility for PANDAS-specific autoimmunity.

## Summary Table

| Target | Novel? | Literature | PPI Network? | TUBB3 Link? | Evidence Tier |
|--------|--------|-----------|-------------|-------------|---------------|
| HSPA6/DnaK | YES | HSP70 autoAbs in MS/EBA | YES (interactor) | YES (direct) | HIGH |
| ENO2 | Semi | Dale 2006, Fontan 2000 | YES (seed, deg=119) | No | HIGH |
| MCCC1-like | YES | None | No | No | MODERATE |
| GAPDH | No | Dale 2006, Pancholi 1992 | YES | No | ESTABLISHED |
| NFS1-like | YES | None | No | No | LOW |
| PSMC6 | YES | None | No | No | LOW |

## Cross-Initiative Integration (Task #914)

The research_director's hypothesis about HSPA6/TLR2/TLR4 convergence is supported by this analysis:

1. **HSPA6 is in the autoantibody PPI network** — connected to TUBB3, a cytoskeletal protein and known PANDAS autoantibody target family.
2. **HSP90AA1** (another heat shock protein) is a cross-layer node spanning autoantibody_ppi and cytokine_amplification layers with degree 43. This suggests the heat shock protein family broadly connects the autoantibody and cytokine networks.
3. **HSPA6 in 2 patient subgroups** (broad_autoimmunity, cunningham_classic) suggests it may be relevant to a substantial subset of PANDAS patients.
4. **Proposed self-amplifying loop:** GAS infection → DnaK exposure → anti-DnaK/HSPA6 autoantibodies → (a) neuronal HSPA6 chaperone disruption + (b) TUBB3 cross-reactivity + (c) extracellular HSP70 as DAMP → TLR2/4 → NF-kB → cytokine amplification → more inflammation.

## References

1. Dale RC et al. (2006) J Neuroimmunol 172:198-205. PMID: 16356555
2. Fontan PA et al. (2000) Infect Immun 68:6285-6291. PMID: 11069244
3. Yokota SI et al. (2006) J Neurol Sci 241:39-43. PMID: 16337948
4. Pancholi V, Fischetti VA (1992) J Exp Med 176:415-426
5. Garduno RA et al. (1995) PMID: 7691795
6. Bhattacharyya S et al. (2024) JCI Insight. PMID: 39325550
7. Front Immunol (2022) DOI: 10.3389/fimmu.2022.877958
