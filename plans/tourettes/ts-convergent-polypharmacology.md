# Convergent Pathway Polypharmacology for Tourette Syndrome

## Objective

Screen existing FDA-approved drugs and 2-drug combinations for multi-pathway coverage against the 11 convergent pathways identified in TS, to nominate polypharmacological agents or rational drug combinations that address the multi-node nature of TS pathophysiology.

**Falsifiable hypothesis:** Polypharmacological agents or drug pairs that cover ≥3 convergent TS pathways will score higher in network proximity analysis than current standard-of-care single-target drugs, and commonly co-prescribed TS drug combinations will show higher pathway coverage than random drug pairs.

## Rationale

1. **Convergent pathway evidence (ts-rare-variant-convergence):** 11 pathways show significant convergence between rare and common variant genes. Top: neuronal system (p=2.1e-5), nervous system development (p=2.9e-5), chemical synaptic transmission (p=6.1e-5), dopaminergic synapse (p=2.2e-4), axonogenesis (p=3.7e-4). Bridge genes CREBBP, DLG4/PSD-95, NOTCH1, GRIN2B connect modular gene clusters.

2. **Single-target drug limitation:** The drug repurposing network found top single-drug candidates (aripiprazole 0.899, ecopipam 0.876) but even the best score < 0.9. Standard D2 blockers cause significant side effects partly because achieving therapeutic effect at one node requires high occupancy that disrupts others.

3. **Clinical reality:** Many TS patients are already on 2+ medications (antipsychotic + alpha-2 agonist is common). But combination selection is empirical, not mechanism-based. If we can explain WHY certain combinations work (pathway coverage), we can predict WHICH new combinations to try.

4. **New circuit data:** The 5-HT2A/PFC finding (pimavanserin) and D3/striosome finding (Celsr3) suggest at least 2 independent circuit nodes. Multi-target approaches may be inherently necessary.

## Data Sources

- 11 convergent pathways + gene lists from ts-rare-variant-convergence Phase 4
- PPI network (1,731 nodes, STRING v12) from ts-drug-repurposing-network
- ChEMBL bioactivity records (7,582 records, 5,215 compounds, 16 targets)
- 177 TS clinical trials (54 drug interventions) from ts-drug-repurposing-network
- LINCS L1000 drug signatures (when iLINCS API is fixed)
- DrugBank drug-target mappings (when XML acquired)
- MR-validated targets: LAMA5, RET, CAPNS1

## Methodology

### Phase 1: Drug-Pathway Mapping
- For each of the 11 convergent pathways, enumerate all FDA-approved drugs with known targets in that pathway (from ChEMBL + DrugBank)
- Compute per-drug pathway coverage vector: binary (yes/no hits each pathway) and weighted (network proximity score per pathway)
- Identify existing polypharmacological agents that hit ≥2 convergent pathways through multiple targets

### Phase 2: Combination Screening
- For all drug pairs from Phase 1 candidates (constrained to FDA-approved, BBB-penetrant, CNS-safe), compute union pathway coverage
- Rank pairs by: (a) number of convergent pathways covered, (b) sum of pathway proximity scores, (c) safety interaction score
- Filter for known drug-drug interaction safety using DDI databases
- Identify "coverage-optimal" pairs: minimum drug count for maximum pathway coverage (set cover approximation)

### Phase 3: Clinical Validation
- Cross-reference top-scoring combinations with existing clinical practice:
  - Are commonly co-prescribed TS combinations (aripiprazole+guanfacine, risperidone+clonidine) explained by complementary pathway coverage?
  - Do clinical trial response rates correlate with the combination's pathway coverage score?
- Test whether the 5-HT2A (pimavanserin) + D3 antagonist combination scores highest (dual-circuit hypothesis)

### Phase 4: Novel Combination Nomination
- From Phase 2 results, nominate 3-5 novel drug combinations not currently used in TS
- For each: generate mechanistic rationale, pathway coverage diagram, safety assessment, predicted clinical trial design
- Prioritize combinations involving at least one drug already used in TS (lower clinical barrier)

## Success Criteria

1. ≥5 polypharmacological agents covering ≥3 convergent pathways identified
2. Common TS drug combinations score significantly higher in pathway coverage than random pairs (permutation p < 0.01)
3. ≥2 novel drug combinations nominated with mechanistic rationale and acceptable safety profiles

## Risk Assessment

**Most likely failure mode:** Combinatorial explosion makes exhaustive pairwise analysis computationally expensive. Drug-target mapping may be incomplete for many convergent pathway genes.

**What we'd learn anyway:** Even partial results would reveal which convergent pathways are druggable and which are therapeutic gaps. The pathway coverage framework itself is a novel contribution to TS pharmacology.

**Mitigation:** Constrain search to drugs with ≥1 target in the convergent pathway gene lists (pre-filter reduces space). Use greedy set cover rather than exhaustive enumeration.

## Labels

catalyst, drug-repurposing, novel-finding, high-priority, promising
