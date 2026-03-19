# Network Pharmacology Drug Repurposing for Tourette Syndrome

## Objective
Use computational network pharmacology to identify existing FDA-approved drugs that target Tourette syndrome-relevant pathways, prioritizing candidates with favorable safety profiles for tic suppression.

## Background
Current TS pharmacotherapy is limited — antipsychotics (haloperidol, pimozide, aripiprazole) have significant side effects, and the only FDA-approved TS drug (VMAT2 inhibitors like valbenazine for tardive dyskinesia, deutetrabenazine) are used off-label. Alpha-2 agonists (clonidine, guanfacine) have modest efficacy. There is a clear unmet need for novel therapeutic options. Network pharmacology approaches can systematically identify drugs whose target profiles overlap with TS-relevant protein networks, enabling rational repurposing from drugs already proven safe in humans.

**Update (Mar 2026):** Gemlapodect (NOE-105), a first-in-class PDE10A inhibitor, met all endpoints in Phase 2a ALLEVIA-1 (57% tic improvement, zero worsened) and entered Phase 2b (ALLEVIA-2, NCT06315751, 140 patients). PDE10A modulates cAMP/cGMP in striatal MSNs — first validated non-dopamine-receptor target for tics. No metabolic or EPS side effects. This validates the intracellular signaling axis downstream of dopamine receptors as druggable and should be included as a target class.

## Data Sources
- **TS gene/protein set** — TSAICG GWAS genes, rare variant genes, differentially expressed genes from TS transcriptome studies
- **MR-validated drug targets** — Zhou & Li (Italian J Pediatrics 2025; doi:10.1186/s13052-025-02048-x): Two-sample MR using cis-eQTLs identified LAMA5 (top candidate — regulates CD4+ T-cell differentiation, links genetic and immune hypotheses), RET (neurotrophic signaling), CAPNS1 (calpain, synaptic plasticity). 7 critical methylation sites regulate LAMA5 expression. Cross-reference with LINCS L1000 for compounds affecting these pathways [NEW]
- **STRING/BioGRID** — Protein-protein interaction (PPI) networks for building TS-relevant subnetworks
- **DrugBank** — Drug-target interaction database
- **LINCS L1000 (CMap)** — Gene expression signatures of drug perturbations for signature-based matching. **NOTE:** CMap reproducibility is limited — use as hypothesis-generating, not definitive. Cross-validate top hits with independent perturbation data where available [task #239]
- **ChEMBL** — Bioactivity data for TS-relevant targets (DRD2, DRD1, VMAT2, SERT, H3R, GABA receptors, **PDE10A**, **CHRM4/M4** [NEW])
- **ClinicalTrials.gov** — Existing TS clinical trials for cross-referencing candidate compounds

## Methodology
1. **TS disease network construction**: Build PPI network seeded with TS risk genes (GWAS + rare variant). Extend by first-degree interactors. Identify densely connected modules (Louvain clustering)
2. **Network proximity analysis**: For each FDA-approved drug in DrugBank, compute network proximity between drug targets and TS disease modules (shortest path distance, z-score against random)
3. **Signature matching**: Query LINCS L1000 with TS-associated expression signatures (inverse correlation = therapeutic candidate)
4. **Target pathway analysis**: Map drug targets to TS-relevant pathways:
   - Dopamine: D2R modulation, **D1R antagonism [validated — ecopipam FDA breakthrough therapy; add D1 pathway as confirmed mechanism]** [task #273]
   - Serotonin: 5-HT2A/2C
   - GABA/glutamate balance
   - Histamine: H3R
   - Cannabinoid: CB1
   - Synaptic adhesion
   - **PDE10A/cAMP-cGMP intracellular signaling** [validated by gemlapodect Phase 2a]
   - **Muscarinic cholinergic: CHRM4/M4 receptor** — M4 activation reduces tic-like behaviors via cholinergic interneuron pathway. **KarXT/Cobenfy (xanomeline-trospium, orthosteric M1/M4 agonist)** is FDA-approved for schizophrenia (Sep 2024) with robust Phase 3 efficacy — high-priority repurposing candidate. **NOTE:** Emraclidine (M4 PAM) FAILED both Phase 2 trials (EMPOWER 1 & 2, Nov 2024) — PAM vs orthosteric distinction may explain divergent results; downgrade PAMs, prioritize orthosteric agonists [tasks #294, #329]
   - **Glutamatergic modulation: NMDA/AMPA targets** — D-serine and riluzole are emerging TS candidates from Phase III trials (Medicine 2025; PMC12151041). Glutamatergic modulators represent a distinct mechanism from dopaminergic approaches. Prioritize FDA-approved glutamatergic drugs for repurposing [NEW]
   - **MR-validated targets: LAMA5/RET/CAPNS1** — MR evidence provides stronger causal support than purely correlational targets. LAMA5 bridges genetic-immune axis (CD4+ T-cell regulation), RET enables neurotrophic signaling intervention, CAPNS1 targets synaptic plasticity. Weight MR-validated targets higher in multi-criteria ranking [NEW]
   - **Immune/neuroinflammatory module**: Include TNF, IL-12/23, JAK-STAT, CCL2/CCR2 as TS-relevant immune targets. Cross-reference with ts-neuroimmune-subtyping project for immune gene targets. Anti-TNF, anti-IL-12/23, JAK inhibitors as immunomodulatory candidates [task #270]
5. **VMAT2 negative control**: VMAT2 inhibitors (valbenazine, deutetrabenazine) showed disappointing TS trial results despite mechanistic rationale — use as negative control to calibrate scoring. Drugs scoring similarly to VMAT2 inhibitors should be penalized [task #239]
6. **Safety and feasibility filtering**: Filter candidates by: FDA-approved, acceptable CNS safety profile, blood-brain barrier penetrance, no overlapping major contraindications with TS comorbidities (ADHD, OCD)
7. **Candidate ranking**: Multi-criteria ranking (network proximity z-score, signature correlation, target pathway relevance, safety profile, clinical precedent)

## Expected Outputs
- TS protein interaction network with functional modules
- Ranked list of top 20 drug repurposing candidates with evidence scores
- Network visualizations showing drug-target-disease module interactions
- Pathway-level analysis of candidate drug mechanisms
- Clinical feasibility assessment for top 5 candidates
- Comparison with drugs currently in TS clinical trials
- **PDE10A pathway druggability analysis** — evaluate all existing PDE10A modulators as repurposing candidates [NEW]

## Success Criteria
- Network proximity analysis identifies at least 10 drugs with z-score < -2 (significantly closer to TS modules than random)
- At least 3 candidates from non-obvious drug classes (not antipsychotics/alpha-agonists)
- Top candidates have plausible mechanisms (not just statistical proximity)
- At least 1 candidate is not currently being investigated in TS trials (novel repurposing candidate)

## Labels
drug-repurposing, novel-finding, high-priority, promising
