# Network Pharmacology Drug Repurposing for Tourette Syndrome

## Objective
Use computational network pharmacology to identify existing FDA-approved drugs that target Tourette syndrome-relevant pathways, prioritizing candidates with favorable safety profiles for tic suppression.

## Background
Current TS pharmacotherapy is limited — antipsychotics (haloperidol, pimozide, aripiprazole) have significant side effects, and the only FDA-approved TS drug (VMAT2 inhibitors like valbenazine for tardive dyskinesia, deutetrabenazine) are used off-label. Alpha-2 agonists (clonidine, guanfacine) have modest efficacy. There is a clear unmet need for novel therapeutic options. Network pharmacology approaches can systematically identify drugs whose target profiles overlap with TS-relevant protein networks, enabling rational repurposing from drugs already proven safe in humans.

## Data Sources
- **TS gene/protein set** — TSAICG GWAS genes, rare variant genes, differentially expressed genes from TS transcriptome studies
- **STRING/BioGRID** — Protein-protein interaction (PPI) networks for building TS-relevant subnetworks
- **DrugBank** — Drug-target interaction database
- **LINCS L1000 (CMap)** — Gene expression signatures of drug perturbations for signature-based matching
- **ChEMBL** — Bioactivity data for TS-relevant targets (DRD2, VMAT2, SERT, H3R, GABA receptors)
- **ClinicalTrials.gov** — Existing TS clinical trials for cross-referencing candidate compounds

## Methodology
1. **TS disease network construction**: Build PPI network seeded with TS risk genes (GWAS + rare variant). Extend by first-degree interactors. Identify densely connected modules (Louvain clustering)
2. **Network proximity analysis**: For each FDA-approved drug in DrugBank, compute network proximity between drug targets and TS disease modules (shortest path distance, z-score against random)
3. **Signature matching**: Query LINCS L1000 with TS-associated expression signatures (inverse correlation = therapeutic candidate)
4. **Target pathway analysis**: Map drug targets to TS-relevant pathways: dopamine (D2R modulation), serotonin (5-HT2A/2C), GABA/glutamate balance, histamine (H3R), cannabinoid (CB1), and synaptic adhesion
5. **Safety and feasibility filtering**: Filter candidates by: FDA-approved, acceptable CNS safety profile, blood-brain barrier penetrance, no overlapping major contraindications with TS comorbidities (ADHD, OCD)
6. **Candidate ranking**: Multi-criteria ranking (network proximity z-score, signature correlation, target pathway relevance, safety profile, clinical precedent)

## Expected Outputs
- TS protein interaction network with functional modules
- Ranked list of top 20 drug repurposing candidates with evidence scores
- Network visualizations showing drug-target-disease module interactions
- Pathway-level analysis of candidate drug mechanisms
- Clinical feasibility assessment for top 5 candidates
- Comparison with drugs currently in TS clinical trials

## Success Criteria
- Network proximity analysis identifies at least 10 drugs with z-score < -2 (significantly closer to TS modules than random)
- At least 3 candidates from non-obvious drug classes (not antipsychotics/alpha-agonists)
- Top candidates have plausible mechanisms (not just statistical proximity)
- At least 1 candidate is not currently being investigated in TS trials (novel repurposing candidate)

## Labels
drug-repurposing, novel-finding, high-priority, promising
