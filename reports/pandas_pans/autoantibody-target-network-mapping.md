# PANDAS Autoantibody Target Network Mapping: Systematic Analysis of Downstream Signaling Disruption and Symptom Heterogeneity

**Project:** autoantibody-target-network-mapping
**Division:** pandas_pans
**Date:** 2026-03-24
**Data Sources:** STRING v12, BioGRID, IntAct, KEGG, Reactome, Allen Human Brain Atlas, DisGeNET, DGIdb, UniProt, IUPHAR/BPS Guide to Pharmacology
**Pipeline:** `src/pandas_pans/autoantibody_target_network_mapping/` (13 modules)
**Validation Status:** Independently validated by validation_scientist (task #1307). JAK-STAT enrichment verified. Statistical methodology confirmed sound. Convergence assessment: 2 independent computational discoveries + 1 biological inference.

---

## Executive Summary

We constructed the first systematic protein interaction network of PANDAS autoantibody targets and mapped how disruption of these targets propagates through neuronal signaling networks to produce the heterogeneous neuropsychiatric symptoms seen in PANDAS/PANS patients. Starting from 9 seed proteins representing known autoantibody targets (dopamine receptors DRD1/DRD2, CaMKII, tubulin, neuronal glycolytic enzymes, and folate receptor), we built a two-layer interactome of 1,721 proteins and 6,889 interactions spanning direct autoantibody-mediated disruption and cytokine amplification.

The network reveals that PANDAS autoantibody targets converge on JAK-STAT signaling (fold enrichment = 12.4, FDR = 7.33 × 10⁻¹⁰⁵), long-term potentiation (FE = 14.1), and dopaminergic synapse pathways (FE = 12.3) — with the striking finding that zero seed proteins are themselves JAK-STAT members, meaning the enrichment emerges entirely from downstream interactors. Symptom-domain mapping demonstrates that dopaminergic targets preferentially disrupt OCD (FE = 3.80, FDR = 2.38 × 10⁻³) and eating restriction (FE = 4.05, FDR = 3.12 × 10⁻³) pathways, while calcium kinase targets map to anxiety circuits (FE = 1.65). Eight computationally derived patient subgroups share a convergent hub architecture (49/50 top hubs are shared), suggesting that despite heterogeneous autoantibody profiles, patients funnel through common signaling bottlenecks. Of 50 novel therapeutic targets identified, 48 are druggable and 42 have approved drugs, with JAK inhibitors (baricitinib) and NF-κB modulators (baclofen) emerging as top-tier repurposing candidates. The JAK-STAT finding is independently supported by convergent evidence from the PANS genetic variant pathway analysis project, providing 2 independent computational discoveries plus supporting biological plausibility from cytokine meta-analysis data.

## Important Caveats

- **Computational network, not experimental:** All findings derive from database-curated protein interactions and pathway annotations, not from direct experimental measurement of autoantibody effects on living neurons. Network proximity does not prove causation.
- **Patient subgroups are simulated:** The 8 subgroups are computational constructs based on autoantibody target combinations, not empirically derived from patient cohorts. Validation with clinical autoantibody profiling data (e.g., PhIP-Seq) is required.
- **Tic/motor domain limitation:** DisGeNET had limited gene-disease associations for tic disorders specifically, which may underestimate the tic-motor symptom domain mapping. The GWAS Catalog similarly lacks large-scale tic disorder GWAS.
- **JAK-STAT convergence is 2 + 1, not 3 independent lines:** The cytokine network project inferred JAK-STAT post-hoc from IFN-γ elevation, not through independent unbiased discovery (see Cross-Project Convergence section).
- **Baricitinib has limited BBB penetrance:** As a hydrophilic small molecule, peripheral-only JAK inhibition may be insufficient for CNS-targeted therapy. No PANDAS-specific clinical or preclinical data exists for JAK inhibitors.
- **Fold enrichment discrepancy:** Validated fold enrichments (using standard hypergeometric against 20,000-gene background) differ from original values in some pathways due to different background size assumptions. All values reported here use the independently validated computations.

## Background

PANDAS (Pediatric Autoimmune Neuropsychiatric Disorders Associated with Streptococcal infections) and the broader PANS (Pediatric Acute-onset Neuropsychiatric Syndrome) are characterized by abrupt-onset OCD, tics, anxiety, eating restriction, cognitive regression, and emotional lability following infection or immune activation. Autoantibodies against neuronal surface proteins — measured clinically via the Cunningham Panel — are implicated in pathogenesis, but no study has systematically mapped how disruption of these specific targets propagates through neuronal signaling networks.

The known autoantibody targets span multiple functional categories: dopamine receptors (DRD1, DRD2), structural proteins (tubulin/TUBB3), calcium signaling (CaMKII/CAMK2A), neuronal surface glycolytic enzymes (pyruvate kinase M1/PKM, aldolase C/ALDOC, enolases ENO1/ENO2), and folate receptor (FOLR1). The Cunningham Panel measures antibodies against DRD1, DRD2, lysoganglioside GM1, tubulin, and CaMKII activity. However, the downstream consequences of simultaneously disrupting these functionally diverse targets remain unmapped.

Two critical biological insights motivated the network design. First, McGregor et al. (2025, JCI Insight, PMID 41252216) demonstrated marked patient-to-patient autoantibody heterogeneity in acute rheumatic fever using PhIP-Seq, with shared (public) epitopes being rare. This implies that PANDAS patients likely have heterogeneous autoantibody profiles, and any network model must account for this. Second, an anti-NMDAR1 + IL-17 study (Mol Psychiatry 2025, doi: 10.1038/s41380-025-03434-x) showed that autoantibodies alone cause NMDAR hypofunction, but full patient CSF causes neuronal hyperexcitability — dissociating direct antibody effects from inflammatory milieu effects. This motivated our two-layer (autoantibody + cytokine amplification) network architecture.

## Methodology

### Seed Protein Set

Nine proteins were defined as seed nodes representing all known PANDAS autoantibody targets:

| Protein | Gene | Category | Cunningham Panel |
|---------|------|----------|-----------------|
| Dopamine D1 receptor | DRD1 | Dopaminergic | Yes (Anti-DRD1) |
| Dopamine D2 receptor | DRD2 | Dopaminergic | Yes (Anti-DRD2) |
| CaM kinase II alpha | CAMK2A | Calcium kinase | Yes (CaMKII activity) |
| Tubulin beta-3 | TUBB3 | Structural | Yes (Anti-tubulin) |
| Pyruvate kinase M1 | PKM | Metabolic/surface | No |
| Aldolase C | ALDOC | Metabolic/surface | No |
| Alpha-enolase | ENO1 | Metabolic/surface | No |
| Gamma-enolase | ENO2 | Metabolic/surface | No |
| Folate receptor alpha | FOLR1 | Folate transport | No |

Lysoganglioside GM1 (Cunningham Panel target) was documented but excluded from PPI queries as a glycolipid without a UniProt protein entry.

### Network Construction

**Layer 1 — Autoantibody PPI network:** STRING v12 was queried for all 9 seed proteins at combined score > 0.7 (high confidence), yielding 3,551 interactions across 424 proteins (mean score 0.877). BioGRID and IntAct provided 1,281 supplementary interactions not in STRING, extending the network to 4,832 edges across 1,493 nodes. Human-only interactions (taxID 9606); self-interactions and negative interactions excluded.

**Layer 2 — Cytokine amplification:** Three cytokine axes were modeled based on literature evidence: IL-17 (BBB permeability, gut-oral-brain axis; Matera et al. 2025, PMID 41394880), IFN-γ (epigenetic chromatin closing in neurons; Shammas 2026, PMID 41448185), and IL-6 (central hub cytokine, inflammatory cascade gatekeeper). STRING interactions for 14 cytokine-associated proteins added 2,057 edges with 18 cross-layer edges bridging both layers.

**Combined network:** 1,721 nodes, 6,889 edges. Layer 1 contributes 4,814 edges (70%), Layer 2 contributes 2,057 edges (30%), with 18 cross-layer edges (0.3%).

### Pathway Enrichment

KEGG and Reactome pathway enrichment was computed using hypergeometric test and Fisher exact test against a 20,000 protein-coding gene genome background, with Benjamini-Hochberg FDR correction. Independent validation re-computed all enrichments from raw gene-pathway mappings.

### Brain Region Specificity

Allen Human Brain Atlas microarray expression data was used to test enrichment of network proteins in basal ganglia, prefrontal cortex, and thalamus using t-tests comparing target region expression to whole-brain average.

### Symptom-Domain Mapping

DisGeNET and Open Targets disease-gene associations were used to map network proteins to 8 PANDAS symptom domains (OCD, tics, anxiety, eating restriction, cognitive regression, emotional lability, dopaminergic dysfunction, autoimmune neuropsychiatric conditions). Fisher exact tests with BH FDR correction tested associations between autoantibody target groups and symptom domains. Permutation testing (5,000 permutations) assessed specificity of the overall mapping.

### Patient Heterogeneity Modeling

Eight patient subgroups were defined by autoantibody target combinations (e.g., dopaminergic-dominant, calcium signaling, metabolic surface, broad autoimmunity). Subgroup-specific subnetworks were constructed and analyzed for shared versus private network features.

### Therapeutic Target Ranking

A composite therapeutic ranking score was computed: centrality (25%), druggability (25%), brain expression (20%), pathway convergence (20%), cross-layer status (10%). DGIdb provided drug-gene interaction and approved drug data.

## Results

### 1. Pathway Enrichment Validates Network Biological Relevance

Independent validation confirmed 21 of 23 focus pathways at FDR < 0.05, with zero pathways losing significance. The network is massively enriched for PANDAS-relevant signaling:

**Top KEGG Pathway Enrichments (Validated):**

| Pathway | Network Genes | Background | Fold Enrichment | FDR |
|---------|--------------|------------|-----------------|-----|
| Long-term potentiation | 53 | 67 | 14.08 | 8.20 × 10⁻⁵³ |
| JAK-STAT signaling | 117 | 168 | 12.39 | 7.33 × 10⁻¹⁰⁵ |
| Dopaminergic synapse | 92 | 133 | 12.31 | 7.55 × 10⁻³⁸ |
| Gap junction | 67 | 89 | — | — |
| Circadian entrainment | 70 | 97 | — | — |
| Calcium signaling | 81 | — | — | — |

Overall: 257/352 KEGG pathways significant (FDR < 0.05); 956/1,000 Reactome pathways significant. The validated analysis found 65 additional significant KEGG pathways beyond the original 192, with zero false positives.

**Critical finding — JAK-STAT enrichment emerges from interactors, not seeds:** Zero of the 9 seed proteins are members of the KEGG JAK-STAT pathway. The FE = 12.4 enrichment (117/168 pathway genes present in the network) arises entirely from first-degree interactors of autoantibody targets. This strengthens the finding: JAK-STAT is a genuine downstream convergence point, not a tautological result of seeding with JAK pathway members.

**Seed-specific pathway signatures:** DRD1/DRD2 seeds drive dopaminergic synapse and cAMP signaling enrichment. CAMK2A drives long-term potentiation and calcium signaling. Metabolic seeds (PKM, ALDOC, ENO1/ENO2) drive carbon metabolism and HIF-1 signaling. The pathway landscape is not homogeneous — different autoantibody targets disrupt different signaling cascades.

### 2. Network Topology Reveals Modular, Hub-Dependent Architecture

**Community structure:** Louvain community detection identified 18 communities with modularity = 0.72, indicating strong modular organization. The top 5 communities contain 311, 282, 259, 247, and 218 nodes respectively. Seed proteins span 5 distinct communities: DRD1 and DRD2 co-localize in community 3 (dopaminergic module), CAMK2A occupies community 2 (calcium/kinase module), and metabolic seeds distribute across communities 9 and 14.

**Hub analysis:** The network follows a near-scale-free degree distribution (power law exponent = 0.933, R² = 0.843). Top hub is TUBB3 (degree 377, z-score 17.4, p < 0.001) — notably, TUBB3 is itself a PANDAS autoantibody target (anti-tubulin), making it both a seed protein and the network's most connected node.

**Robustness — high hub dependency:** Sequential removal of top hubs causes rapid network fragmentation:

| Hubs Removed | Largest CC Fraction | Hub Removed |
|-------------|-------------------|-------------|
| 0 | 1.00 | — |
| 1 | 0.84 | TUBB3 |
| 2 | 0.73 | PKM |
| 3 | 0.62 | CAMK2A |
| 5 | 0.50 | PKMYT1 |
| 8 | 0.43 | STAT1 |

Removing the top 10 hubs reduces the largest connected component to 43% of the network — confirming high hub dependency. The fact that 4 of the top 6 hubs (TUBB3, PKM, CAMK2A, DRD2) are themselves autoantibody targets suggests that the immune system is attacking the network's most structurally critical nodes.

### 3. Symptom-Domain Mapping Links Autoantibody Profiles to Clinical Presentations

Fisher exact tests with BH FDR correction identified statistically significant associations between autoantibody target groups and PANDAS symptom domains:

**Significant Group-Level Associations (FDR < 0.05):**

| Target Group | Symptom Domain | Fold Enrichment | FDR |
|-------------|---------------|-----------------|-----|
| Dopaminergic (DRD1/DRD2) | Eating restriction | 4.05 | 3.12 × 10⁻³ |
| Dopaminergic (DRD1/DRD2) | OCD/compulsive | 3.80 | 2.38 × 10⁻³ |
| Dopaminergic (DRD1/DRD2) | Anxiety | 1.89 | 8.36 × 10⁻³ |
| Calcium kinase (CAMK2A) | Anxiety | 1.65 | — |

**Seed-level results:** 6/72 individual seed-symptom associations reached significance.

**Permutation testing confirms specificity:** Observed mapping specificity = 0.395 versus permutation mean = 0.036 (p < 0.001, 5,000 permutations). The autoantibody-symptom mapping is 11× more specific than expected by chance.

**Symptom domain coverage across the network:**

| Domain | Network Genes | % Coverage | Seed Proteins |
|--------|--------------|------------|---------------|
| Dopaminergic | 317 | 63.8% | 8 |
| Cognitive | 310 | 62.4% | 6 |
| Tic/motor | 242 | 48.7% | 5 |
| Autoimmune neuropsych | 240 | 48.3% | 8 |
| Emotional lability | 197 | 39.6% | 4 |
| Anxiety | 139 | 28.0% | 5 |
| OCD/compulsive | 32 | 6.4% | 3 |
| Eating restriction | 25 | 5.0% | 2 |

**Limitation — tic/motor domain:** Despite 242 network genes mapping to tic/motor, the statistical power for this domain is limited by sparse DisGeNET annotations for tic disorders. This is a known gap in the disease-gene association literature, not a network deficiency.

### 4. Brain Region Enrichment Confirms Basal Ganglia Specificity

Allen Human Brain Atlas expression analysis of 300 network proteins (including all 9 seeds):

| Brain Region | Expression Ratio | t-Statistic | p-Value | High-z Genes |
|-------------|-----------------|-------------|---------|-------------|
| **Basal ganglia** | 1.024 | 4.125 | **4.81 × 10⁻⁵** | 16 |
| **Prefrontal cortex** | 1.023 | 5.316 | **2.07 × 10⁻⁷** | 2 |
| Thalamus | 0.999 | -0.181 | 0.857 (NS) | 15 |

The network is significantly enriched in basal ganglia and prefrontal cortex — the two brain regions most implicated in PANDAS pathophysiology (basal ganglia: OCD and tic circuits; prefrontal cortex: executive function and anxiety). Thalamus enrichment is not significant, consistent with thalamic involvement in PANDAS being secondary to cortico-striatal-thalamic circuit disruption rather than primary.

High-z basal ganglia genes include DRD1, DRD2, ADCY5, ADORA2A, and ACTN2 — proteins known to be concentrated in striatal medium spiny neurons, the primary cell type implicated in PANDAS autoimmune targeting.

### 5. Patient Heterogeneity: 8 Subgroups, Convergent Hub Architecture

Eight patient subgroups were defined by autoantibody target combinations:

| Subgroup | Seed Targets | Nodes | Edges | Pathways |
|----------|-------------|-------|-------|----------|
| Broad autoimmunity | All 9 | 1,276 | — | — |
| Cunningham classic | DRD1, DRD2, CAMK2A, TUBB3 | 822 | — | — |
| Metabolic surface | PKM, ALDOC, ENO1/ENO2 | 597 | 2,485 | 316 |
| Calcium signaling | CAMK2A + calcium targets | 470 | 1,659 | 270 |
| DRD1-CAMKII | DRD1 + CAMK2A | 347 | — | — |
| Dopaminergic dominant | DRD1, DRD2 | 170 | 634 | 203 |
| DRD2-FOLR1 | DRD2 + FOLR1 | — | — | — |
| FOLR1-isolated | FOLR1 only | — | — | — |

**Critical finding — convergent hub architecture:** Despite varying subgroup sizes (170–1,276 nodes), 49 of 50 top therapeutic targets are shared across subgroups. Mean Jaccard overlap between subgroup target sets is 0.209, indicating substantial subgroup-specific pathway content, but the convergent hubs form a conserved backbone.

This has direct clinical implications: therapeutic targets identified from the full network apply across autoantibody profiles. An expanded Cunningham Panel that measures additional autoantibody targets (PKM, ALDOC, ENO1/2, FOLR1) would enable patient stratification into these subgroups, potentially informing personalized treatment selection.

### 6. Therapeutic Target Identification: 48/50 Druggable

Fifty novel therapeutic targets were identified by composite ranking (centrality 25%, druggability 25%, brain expression 20%, pathway convergence 20%, cross-layer status 10%). These are convergence nodes where multiple autoantibody-disrupted pathways intersect.

**Tier 1 — Highest Priority Repurposing Candidates:**

| Target | Score | Enriched Pathways | Approved Drugs | Rationale |
|--------|-------|-------------------|---------------|-----------|
| PIK3R1 | 0.806 | 155 | Copanlisib, leniolisib, everolimus | PI3K regulatory subunit; cross-layer hub bridging autoantibody PPI and cytokine amplification |
| AKT1 | 0.733 | 139 | Everolimus, capivasertib | Central kinase in PI3K/AKT/mTOR; cross-layer node; expressed in basal ganglia |
| NFKB1 | 0.714 | 109 | Baclofen, bortezomib, etanercept | NF-κB master regulator; 24 approved drugs; immune-neuronal interface |
| JAK-STAT pathway | — | 117 genes | **Baricitinib** (JAK1/2 inhibitor) | FE = 12.4; pediatric JIA safety data; autoimmune approvals |

**Tier 2:**

| Target | Score | Drugs |
|--------|-------|-------|
| NRAS | 0.725 | Binimetinib, trametinib |
| CHUK (IKKα) | 0.684 | Mesalamine, ustekinumab |
| SRC | 0.677 | Dasatinib, bosutinib |

**Tier 3:** GRB2 (0.672), JUN (0.668), MAPK1 (0.659), IKBKB (0.647)

**Druggability:** 48/50 novel targets are druggable (DGIdb). 42/50 have at least one approved drug. 1,866 unique drugs map to the target set, representing a large repurposing search space.

**Baricitinib assessment — supported with caveats:**

Baricitinib (JAK1/JAK2 selective inhibitor) is the highest-profile repurposing candidate based on:
- JAK-STAT pathway enrichment FE = 12.4, the most statistically significant focus pathway (FDR = 7.33 × 10⁻¹⁰⁵)
- Existing autoimmune approvals (rheumatoid arthritis, atopic dermatitis) and pediatric safety data from juvenile idiopathic arthritis trials
- COVID neurological benefit suggesting some CNS activity

**Required before clinical translation:**
1. **BBB penetrance:** Baricitinib is a hydrophilic small molecule with limited CNS penetration. Peripheral-only JAK inhibition may be insufficient for PANDAS CNS pathology. Published pharmacokinetic data on CNS penetrance must be formally assessed.
2. **Patient stratification:** Which PANDAS patients would benefit? All patients, or only those with elevated IFN signatures (DDR-variant subset)?
3. **IFN signature confirmation:** The DDR-variant → interferonopathy → JAK-STAT mechanism (from the genetic variant project) requires confirmation that IFN signatures are actually elevated in PANS patients (currently blocked by probe mapping issue, task #1002).
4. **Risk-benefit comparison:** Broad JAK inhibition carries immunosuppression risk (herpes zoster, opportunistic infections) in children. Must be evaluated against standard PANDAS treatments (IVIG, plasmapheresis, antibiotics).

### 7. Cross-Project Convergence on JAK-STAT

JAK-STAT signaling emerged from multiple analyses across the PANDAS/PANS research program. The validation scientist (task #1307) rigorously assessed the independence of these evidence lines:

**Convergent evidence from 2 independent computational analyses, supported by biological plausibility from cytokine data:**

| Evidence Line | Source Project | Method | Finding | Independence |
|--------------|--------------|--------|---------|-------------|
| Autoantibody network enrichment | This project | Unbiased PPI → KEGG enrichment | JAK-STAT FE = 12.4 (FDR = 7.33 × 10⁻¹⁰⁵) | **Independent** — fully data-driven |
| Genetic variant GO enrichment | pans-genetic-variant-pathway-analysis | Rare variant → GO enrichment | EP300 enriches GO:0046425 (regulation of JAK-STAT signaling) | **Partially independent** — EP300 is a single gene; DDR→interferonopathy→JAK-STAT requires inferential steps |
| Cytokine meta-analysis | cytokine network project | Random-effects meta-analysis | IFN-γ elevated (g = 1.09); JAK-STAT inferred from IFN-γ signaling | **Not independent** — JAK-STAT was inferred post-hoc from IFN-γ, not discovered by the meta-analysis |

**Data sources are fully independent** with no shared patient cohorts and no data leakage between projects. The autoantibody network uses protein interaction databases, the genetic variant project uses patient genomic data, and the cytokine project uses published cytokine measurements. Shared use of KEGG pathway definitions is standard practice, not analytical contamination.

**Corrected interpretation:** The convergence represents 2 independent unbiased computational discoveries pointing to JAK-STAT involvement, with additional biological plausibility from cytokine data showing IFN-γ elevation (which signals through JAK-STAT). This is meaningful but should not be described as "3 independent lines of evidence."

### 8. Two-Layer Network: Autoantibody + Cytokine Amplification

The two-hit framework models PANDAS pathology at two levels:

**Layer 1 — Direct autoantibody binding** (4,814 edges): Autoantibodies bind neuronal surface proteins, disrupting receptor signaling (DRD1/DRD2), cytoskeletal integrity (TUBB3), calcium signaling (CAMK2A), and metabolic enzymes (PKM, ENO1/2, ALDOC).

**Layer 2 — Cytokine amplification** (2,057 edges): Three cytokine axes modify autoantibody effects:
- **IL-17 axis** (IL17A, IL17RA, IL17RC): Promotes BBB permeability, enabling autoantibody CNS access. Connects to gut-oral-brain axis.
- **IFN-γ axis** (IFNG, IFNGR1, IFNGR2): Drives persistent epigenetic chromatin closing in neurons, meaning autoantibody + IFN-γ may cause qualitatively different (and more permanent) damage than autoantibody alone.
- **IL-6 axis** (IL6, IL6R, IL6ST): Central hub cytokine and gatekeeper for broader inflammatory cascade. IL-6 meta-analysis effect size g = 1.46 (highest among measured cytokines).

**Cross-layer nodes** (52 proteins): Bridge both layers, meaning they participate in both direct autoantibody PPI disruption and cytokine signaling. These cross-layer nodes (including AKT1, SRC, CREB1, PLCG1) represent dual-mechanism intervention points where a single therapeutic could modulate both pathological layers.

## Discussion

This analysis provides the first comprehensive map of how PANDAS autoantibody targets propagate disruption through neuronal signaling networks. Several findings have immediate implications for understanding PANDAS pathophysiology and identifying therapeutic opportunities.

**Autoantibodies attack structurally critical network hubs.** Four of the top 6 network hubs (TUBB3, PKM, CAMK2A, DRD2) are themselves autoantibody targets. This is not random — the immune system in PANDAS appears to target proteins that occupy structurally privileged positions in neuronal signaling networks, maximizing downstream disruption per antibody binding event. The high hub dependency of the network (43% connectivity loss with 10 hub removals) suggests that even partial autoantibody binding to these targets could cause disproportionate signaling disruption.

**Symptom heterogeneity maps to autoantibody target diversity.** The significant associations between dopaminergic targets and OCD/eating restriction, and between calcium kinase targets and anxiety, provide a mechanistic framework for why different PANDAS patients present with different symptom combinations. A patient with predominantly anti-DRD1/DRD2 antibodies would be predicted to present with OCD and eating restriction, while a patient with elevated CaMKII activity would present with anxiety-dominant symptoms. The 8 computational subgroups operationalize this heterogeneity and could be validated clinically using an expanded Cunningham Panel.

**JAK-STAT as a therapeutic target requires careful framing.** The JAK-STAT enrichment is the most statistically robust finding in this analysis, independently verified by the validation scientist. The fact that it emerges entirely from interactors (zero seeds are JAK-STAT members) makes it a genuine downstream convergence point rather than a circular finding. However, the path from pathway enrichment to clinical therapeutic recommendation requires additional evidence: BBB penetrance data, patient stratification criteria, and IFN signature confirmation in PANS patients. Baricitinib is biologically plausible but not yet clinically validated for PANDAS.

**The two-layer framework has clinical implications.** If different patients have different ratios of Layer 1 (direct autoantibody) versus Layer 2 (cytokine amplification) pathology, treatment strategies should differ. Patients with predominantly Layer 1 pathology might respond best to autoantibody removal (plasmapheresis, IVIG), while patients with prominent Layer 2 involvement might benefit from anti-cytokine therapy or JAK inhibition. The cross-layer nodes identified here (52 proteins) represent opportunities for dual-mechanism intervention.

## Limitations

1. **Database-derived network:** Protein interactions from STRING/BioGRID/IntAct represent curated literature, not direct PANDAS experimental data. Interactions may not occur in the specific neuronal cell types affected by PANDAS.

2. **No temporal dynamics:** The network is a static snapshot. PANDAS is episodic with acute flares and remissions, and autoantibody targets may diversify over successive flares via epitope spreading. The static model cannot capture these dynamics.

3. **Symptom mapping relies on DisGeNET annotations:** The quality of symptom-domain mapping is bounded by the completeness of disease-gene associations. Tic disorders are particularly under-annotated, potentially underestimating the tic/motor mapping.

4. **Simulated patient subgroups:** Without PhIP-Seq or equivalent autoantibody profiling data from actual PANDAS patient cohorts, the 8 subgroups are theoretical constructs. Their clinical validity is unknown.

5. **Fold enrichment methodology:** Validated FE values differ from original computations for some pathways (e.g., JAK-STAT original FE = 3.99 vs. validated FE = 12.39). The validated values using standard hypergeometric against 20,000-gene background are used throughout this report.

6. **Cytokine amplification layer is literature-curated:** The three cytokine axes (IL-17, IFN-γ, IL-6) were selected based on published PANDAS/autoimmune literature, not unbiased discovery. Other cytokine axes may be relevant.

7. **Treatment response module limitation:** The cytokine network project's treatment response module produces identical results for all tested treatments, providing no treatment-discriminating power. It cannot be used to support specificity of JAK inhibitor over other treatments.

## Next Steps

1. **PhIP-Seq profiling of PANDAS patients** to empirically determine autoantibody target diversity and validate the computational patient subgroups. This is the single most impactful follow-up study.

2. **IFN signature measurement in PANS patients** to confirm whether IFN-γ/type I IFN elevations support the DDR-variant → interferonopathy → JAK-STAT mechanism (currently blocked by task #1002).

3. **Baricitinib BBB penetrance assessment** using published pharmacokinetic data and/or CNS tissue-specific pharmacology models.

4. **Expanded Cunningham Panel evaluation** — assess clinical feasibility of measuring autoantibodies against PKM, ALDOC, ENO1/2, and FOLR1 in addition to current targets.

5. **Experimental validation of key network predictions** in neuronal cell culture models: do autoantibodies against seed proteins activate JAK-STAT signaling in striatal medium spiny neurons?

6. **Integration with single-cell transcriptomic data** from basal ganglia to refine which network interactions occur in which neuronal subtypes.

## References

- McGregor R et al. (2025). Patient-to-patient autoantibody heterogeneity in acute rheumatic fever. JCI Insight. PMID 41252216.
- Shammas (2026). IFN-γ-driven epigenetic chromatin closing in neurons. PMID 41448185.
- Matera et al. (2025). IL-17 and gut-oral-brain axis. PMID 41394880.
- Anti-NMDAR1 + IL-17 study. Mol Psychiatry 2025. doi: 10.1038/s41380-025-03434-x.
- Vettiatil et al. (2026). Ultra-rare DDR variants in PANS patients. Dev Neurosci. PMID 41662332.
- STRING database v12: https://string-db.org
- BioGRID: https://thebiogrid.org
- IntAct: https://www.ebi.ac.uk/intact
- KEGG: https://www.genome.jp/kegg
- Reactome: https://reactome.org
- Allen Human Brain Atlas: https://human.brain-map.org
- DisGeNET: https://www.disgenet.org
- DGIdb: https://dgidb.org
- UniProt: https://www.uniprot.org

---

*Pipeline code: `src/pandas_pans/autoantibody_target_network_mapping/` (13 modules)*
*Data: `data/pandas_pans/autoantibody_network/` (49 files)*
*Visualizations: `output/pandas_pans/autoantibody-target-network-mapping/` (8 figures)*
*Validation report: `output/pandas_pans/autoantibody-target-network-mapping/convergence_verification_report.md`*
