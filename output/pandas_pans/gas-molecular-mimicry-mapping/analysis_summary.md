# GAS Molecular Mimicry Mapping — Analysis Summary (Updated)

**Project:** gas-molecular-mimicry-mapping
**Division:** pandas_pans
**Analyst:** analyst
**Date:** 2026-03-24 (replaces 2026-03-19 version)
**Status:** Analysis complete, ready for research_writer handoff

---

## 1. Executive Summary

A systematic computational screen of the Group A Streptococcus (GAS) proteome (**7 serotypes:** M1, M3, M3.93, M5, M12, M18, M49; **~12,516 total proteins** including Enn/Mrp M-like protein supplement) against 328 human basal ganglia-enriched proteins identified **8 mimicry targets** passing sequence similarity filters (pident >= 40%, alignment >= 8aa, E-value <= 1e-3). These were scored using an **8-component composite system** integrating sequence identity, B-cell epitope overlap, MHC-II cross-reactivity (PSSM), MHC-II cross-reactivity (MMPred), serotype conservation, FasBCAX phase weighting, query coverage, and known target recovery.

**Top mimicry target candidates:**

| Rank | Target | GAS Homolog | Score | Evidence Tier | Key Finding |
|------|--------|------------|-------|---------------|-------------|
| 1 | **B4DHP5/HSPA6** (Heat shock 70 kDa protein 6) | DnaK (HSP70) | 0.850 | HIGH | Novel for PANDAS; 12 overlapping B-cell epitopes; invasive-phase protein; connected to TUBB3 in autoantibody PPI network |
| 2 | **ENO2** (Neuron-specific enolase) | Streptococcal enolase | 0.788 | HIGH | Semi-known autoantigen (Dale 2006); TM=0.66 structural mimicry; MMPred rank 0.01 (top 0.01% MHC-II binding); seed protein in autoantibody network (degree 119) |
| 3 | **B4DHW5/MCCC1-like** (Methylcrotonoyl-CoA carboxylase alpha) | Biotin-dependent carboxylase | 0.748 | MODERATE | Novel; strong epitope overlap (score 0.582) but no literature support |
| 4 | **GAPDH** (Positive control) | Streptococcal GAPDH | 0.657 | VALIDATED | Known PANDAS autoantigen; TM=0.64 structural mimicry; validates pipeline |
| 5 | **B4DHP7** | GAS ortholog | 0.500 | LOW | Conserved across 7 serotypes but no epitope overlap |
| 6 | **PSMC6** (Proteasomal ATPase) | GAS proteasomal subunit | 0.437 | LOW | Moderate MHC-II signal but no epitope overlap |
| 7 | **B4DHZ4** | GAS ortholog | 0.293 | MINIMAL | 1 serotype only (M12), short alignment (31aa) |
| 8 | **AK9** (Adenylate kinase 9) | GAS kinase | 0.206 | MINIMAL | 2 serotypes only (M18, M5), weak MHC-II signal |

## 2. HSPA6/DnaK Mimicry Hypothesis (Top Novel Finding)

The top-ranked novel finding is **HSPA6 (HSP70B') as a potential autoantibody target in PANDAS**, based on molecular mimicry with GAS DnaK.

### Evidence Chain

1. **Sequence mimicry:** 49.3% identity over 523 residues (80% query coverage), conserved across all **7** GAS serotypes
2. **B-cell epitope cross-reactivity:** 12 overlapping epitope pairs (best score: 0.606) — highest of any target
3. **MHC-II presentation:** Cross-reactive binding across all 7 tested HLA-DR alleles (PSSM max score: 6.0; MMPred rank: 0.28, top 0.28%)
4. **FasBCAX phase:** DnaK classified as invasive-phase (2.0x weight)
5. **PPI network integration:** HSPA6 connected to TUBB3 (known PANDAS autoantibody target) in the autoantibody-target-network-mapping PPI network; present in broad_autoimmunity and cunningham_classic patient subgroups
6. **Biological mechanism:** HSP70 as DAMP activates TLR2/TLR4 → NF-kB; anti-HSPA6 autoantibodies could disrupt chaperone function, cross-react with tubulin via HSPA6-TUBB3 interaction, and amplify cytokine signaling
7. **Precedent:** Anti-HSP70 IgG autoantibodies detected in MS CSF (Yokota 2006, PMID: 16337948)

**Structural note:** Low TM-scores (0.08-0.17) indicate mimicry operates at the **epitope/peptide level**, not whole-fold structural similarity.

## 3. New Evidence from Dataset Expansion

### 3.1 Enn/Mrp M-like Proteins — Negative Result

Four Enn/Mrp sequences (2 Mrp, 2 Enn; 316-415aa) were added to the GAS proteome based on Akoolo et al. 2026 (PMID: 41236364). **None produced mimicry hits.** Despite structural similarity to M protein (coiled-coil domains), Enn/Mrp proteins lack sequence homology to human neuronal targets above the 40% identity threshold. This is a meaningful negative result confirming pipeline specificity.

### 3.2 emm3.93 Variant Lineage — Conservation Validation

The emm3.93 variant lineage (1,821 proteins from NCBI assembly GCA_046240755.1) produced **7 hits targeting the same 6 human proteins as canonical M3 at identical pident and bitscore values.** Conservation is expected because mimicry operates through core genome housekeeping proteins, not variable surface antigens. The M protein itself is not a mimicry hit.

**Clinical implication:** The emerging M3.93 pediatric invasive GAS clade carries the **same molecular mimicry risk for PANDAS** as canonical M3. This clade does not present a novel mimicry threat.

### 3.3 MMPred MHC-II Integration — Corroboration

MMPred-style scoring (TEPITOPEpan 9-pocket profiles, 7 HLA-DR alleles, percentile-rank calibrated) was integrated as an 8th composite component (weight=0.10). Key findings:

- **ENO2 achieves rank 0.01** (top 0.01% of random background) — strongest MHC-II cross-reactive binding, consistent with its fold-level structural mimicry (TM=0.662)
- **99.4% binary cross-reactivity (327/329 combinations) is a methodological artifact** of long alignment regions. The percentile rank is the meaningful metric.
- **Rankings are completely insensitive** to MMPred weight variation (0.05-0.20) due to score compression
- **MMPred corroborates PSSM predictions** with 6/8 targets in same relative tier

### 3.4 ImmunoStruct Evaluation — Cancelled

ImmunoStruct (task #1184) was correctly cancelled: it targets MHC class I prediction, whereas PANDAS autoimmunity is driven by class II (MHC-II/HLA-DR) antigen presentation. The existing structural evidence (TM-align scores via AlphaFold) provides sufficient structural mimicry assessment.

### 3.5 EMoMiS Evaluation — Pending

EMoMiS (Epitope-based Molecular Mimicry Search) evaluation remains as a medium-priority future enhancement. It could provide independent validation of the B-cell epitope overlap findings.

## 4. Updated Network Topology

- **55 nodes** (8 human targets + 47 GAS proteins from 7 serotypes)
- **47 edges** (unique GAS-human mimicry pairs)
- Hub architecture: each human target connected to 1-14 GAS ortholog nodes
- DnaK/HSPA6 cluster: 6 GAS DnaK orthologs (one per canonical serotype + M3.93), consistent pident ~49%
- B4DHP7 hub: highest connectivity (14 edges) due to two distinct GAS protein families hitting the same target

## 5. Serotype Comparison (7 Serotypes)

### Conserved targets (7/7 serotypes):
B4DHP5/HSPA6, ENO2, B4DHW5/MCCC1-like, GAPDH, B4DHP7, PSMC6 — all housekeeping/core genome proteins

### Serotype-specific:
- **AK9:** 2/7 serotypes (M18, M5) — pandas_specific category, low score (0.206)
- **B4DHZ4:** 1/7 serotypes (M12 only) — pandas_specific, minimal score (0.293)

### Differential mimicry:
No targets exclusive to PANDAS-associated serotypes vs non-PANDAS M49. The mimicry landscape is largely shared, supporting the view that GAS pathogenicity differences arise from factors beyond molecular mimicry (superantigen expression, immune evasion, colonization).

## 6. Statistical Significance

| Metric | Observed | Background | p-value | Reliable? |
|--------|----------|-----------|---------|-----------|
| TM >= 0.5 | 5/13 (38.5%) | ~2% random | 3.6e-06 | Inflated (pre-selected) |
| B-cell epitope overlap | 3/8 targets (18/47 pairs) | Unknown | N/A | Qualitative |
| MHC-II PSSM cross-reactivity | 6/8 targets | ~2-5% random peptides | N/A | Expected at pident >= 40% |
| MMPred cross-reactivity | 8/8 targets (rank metric) | Calibrated vs 10,000 random | N/A | ENO2 rank 0.01 notable |
| Conservation 7/7 | 6/8 (75%) | ~60% core genome | 0.32 | Underpowered (n=8) |

**The 30% hit increase (40→52) does NOT affect false discovery rate.** Same 8 targets; all new hits are M3.93 orthologs. Target-level FDR: 2.4% (8/328).

**Strength is convergent multi-modal evidence, not individual p-values.**

## 7. Scoring System

### Current Weights (8 components):
| Component | Weight | Description |
|-----------|--------|-------------|
| epitope | 0.18 | B-cell epitope overlap score |
| mhc (PSSM) | 0.15 | MHC-II cross-reactivity (3 alleles) |
| mmpred | 0.10 | MMPred MHC-II scoring (7 alleles) |
| conservation | 0.14 | Serotype conservation fraction |
| identity | 0.14 | Percent identity |
| coverage | 0.09 | Query coverage |
| phase | 0.10 | FasBCAX invasive-phase weighting |
| known_target | 0.10 | Known PANDAS autoantigen bonus |

### Scoring Validation
- Top-3 ranking (B4DHP5, ENO2, B4DHW5) stable across all weight perturbations ±0.05 and 13/14 at ±0.10
- Only instability: removing phase weight entirely causes ENO2 to narrowly overtake HSPA6
- MMPred weight insensitive at 0.05-0.20 (normalized scores compressed to 0.984-0.9999 range)
- GAPDH positive control at rank 4 validates pipeline

## 8. Limitations

1. **Small target set (n=8):** 8 human targets passed DIAMOND filters. Relaxing thresholds would increase hits and false positives.
2. **Computational predictions only:** B-cell epitopes (~60-70% accuracy), MHC-II via PSSM and TEPITOPEpan approximations. No experimental validation.
3. **7 HLA-DR alleles tested:** PANDAS susceptibility may involve additional alleles (HLA-DQ, HLA-DP).
4. **No non-brain control proteome:** Screen targets basal ganglia-enriched proteins but doesn't confirm brain specificity vs other tissues.
5. **Structural analysis coverage:** Limited to pairs with AlphaFold models; 7/20 pairs skipped (UniParc IDs). CA-only TM-score alignment.
6. **Pre-selection bias:** All results conditional on DIAMOND screen at pident >= 40%, E-value <= 1e-3.
7. **MMPred normalization:** Rank-to-score transformation (1-rank/100) compresses scores; min-max normalization recommended for future iterations.
8. **Enn/Mrp negative result caveat:** Absence of hits does not prove absence of mimicry — these proteins may contribute through mechanisms below the detection threshold (e.g., conformational epitopes not captured by sequence alignment).

## 9. Recommended Follow-Up Experiments

1. **HSPA6 autoantibody testing:** Test PANDAS patient sera for anti-HSPA6 IgG using ELISA/Western blot with recombinant HSPA6.
2. **DnaK-HSPA6 cross-reactivity:** Immunize mice with recombinant GAS DnaK; test for anti-HSPA6 cross-reactive antibodies.
3. **Shared epitope mapping:** Synthesize predicted DnaK/HSPA6 shared peptides; test binding in PANDAS sera via peptide arrays.
4. **TLR signaling assay:** Test whether anti-HSPA6 antibodies modulate TLR2/TLR4 signaling in microglial cells.
5. **Broader serotype coverage:** Extend to additional emm types, particularly emm49.8 (emerging dominant in England 2025-26).
6. **ENO2 cohort validation:** Confirm enolase cross-reactivity in PANDAS-specific cohort (prior work was Sydenham chorea).

## 10. Cross-Initiative Connections

Per research_director task #914:

- **HSPA6 → autoantibody-target-network-mapping:** HSPA6 interacts with TUBB3 in PPI network. HSP90AA1 (related HSP) is cross-layer hub (degree 43).
- **HSPA6 → cytokine-network-flare-prediction:** HSP70 as DAMP activating TLR2/TLR4 connects molecular mimicry to cytokine amplification. Proposed self-amplifying loop: GAS DnaK → anti-HSPA6 autoantibodies → neuronal damage + TLR activation → cytokine cascade → neuroinflammation.
- **ENO2:** Seed protein (degree 119) in autoantibody network — central node in PANDAS pathophysiology.
- **HSPA6 TLR pathway:** Research director flagged HSPA6-TLR connection as potential cross-initiative link between molecular mimicry (this project) and innate immunity models (innate-immunity-deficiency-model project).

## 11. Data Files

| File | Location | Description |
|------|----------|-------------|
| ranked_targets.tsv | target_scoring/ | 8-component composite scores for all 8 targets |
| scoring_summary.txt | target_scoring/ | Summary with MMPred integration note |
| scoring_validation_report.md | target_scoring/ | Weight sensitivity analysis |
| literature_cross_reference.md | target_scoring/ | Literature cross-reference with PMIDs |
| statistical_assessment.md | target_scoring/ | Original statistical assessment (6 serotypes) |
| epitope_predictions.tsv | epitope_predictions/ | B-cell epitope overlap (47 pairs) |
| mhc_predictions.tsv | mhc_predictions/ | PSSM MHC-II predictions (3 alleles) |
| mmpred_predictions.tsv | mhc_predictions/ | MMPred MHC-II predictions (7 alleles) |
| mmpred_target_summary.tsv | mhc_predictions/ | Per-target MMPred summary |
| structural_mimicry.tsv | structural_mimicry/ | TM-score/RMSD results |
| conservation_scores.tsv | serotype_comparison/ | 7-serotype conservation data |
| serotype_comparison_matrix.tsv | serotype_comparison/ | Per-serotype per-target hit matrix |
| serotype_heatmap.png | serotype_comparison/ | Bitscore heatmap |
| mimicry_network.png | network_visualization/ | Network graph |
| network_nodes.tsv | network_visualization/ | 55 nodes |
| network_edges.tsv | network_visualization/ | 47 edges |
| expanded_dataset_validation.md | analysis/ | Enn/Mrp + emm3.93 validation report |
| mmpred_integration_assessment.md | analysis/ | MMPred evaluation and sensitivity analysis |
| statistical_robustness_update.md | analysis/ | Updated statistical assessment (7 serotypes) |

## References

1. Dale RC et al. (2006) Neuronal surface glycolytic enzymes as autoantigen targets in post-streptococcal autoimmune CNS disease. J Neuroimmunol 172:198-205. PMID: 16356555
2. Fontan PA et al. (2000) Antibodies to streptococcal surface enolase react with human alpha-enolase. Infect Immun 68:6285-6291. PMID: 10722630
3. Yokota SI et al. (2006) Autoantibodies against HSP70 family proteins detected in CSF from MS patients. J Neurol Sci 241:39-43. PMID: 16337948
4. Pancholi V, Fischetti VA (1992) A major surface protein on GAS is a glycolytic enzyme with GAPDH activity. J Exp Med 176:415-426
5. Bhattacharyya S et al. (2024) Dopamine receptor autoantibody signaling in infectious sequelae. JCI Insight. PMID: 39325550
6. Zhang Y, Skolnick J (2004) Scoring function for protein structure template quality. Proteins 57:702-710
7. Akoolo L et al. (2026) PCR-based enn/mrp typing on 1,419 GAS genomes. Microbiology Spectrum. DOI: 10.1128/spectrum.02047-25. PMID: 41236364
8. Garduno RA et al. (1995) Monoclonal antibody recognition of DnaK epitope. PMID: 7691795
9. Junet V et al. (2024) MMPred: MHC-II molecular mimicry prediction. Front Genet 15:1500684. PMID: 39722794
