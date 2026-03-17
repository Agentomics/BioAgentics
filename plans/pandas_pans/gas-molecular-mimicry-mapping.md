# GAS Molecular Mimicry Proteome-Wide Mapping

## Objective
Perform a systematic computational screen of the entire Group A Streptococcus proteome against human basal ganglia and neuronal proteins to identify novel molecular mimicry epitopes driving PANDAS autoimmunity.

## Background
The molecular mimicry hypothesis — that GAS surface proteins share structural similarity with human basal ganglia proteins, triggering cross-reactive autoantibodies — is the leading mechanistic model for PANDAS. Prior work identified specific mimicry targets (lysoganglioside GM1, dopamine D1/D2 receptors, tubulin, β-lysoganglioside) primarily through experimental approaches. However, no study has performed a systematic, proteome-wide computational screen using modern sequence and structural bioinformatics tools. Such a screen could identify novel autoantibody targets, explain phenotypic heterogeneity across PANDAS patients, and reveal why certain GAS strains are more pathogenic.

## Data Sources
- **UniProt:** Complete proteomes for multiple GAS serotypes (M1, M3, M5, M12, M18 — historically associated with rheumatic fever/PANDAS)
- **Human Protein Atlas:** Basal ganglia-enriched and brain-enriched proteins
- **Allen Brain Atlas:** Regional brain expression data for hit prioritization
- **IEDB (Immune Epitope Database):** Known B-cell and T-cell epitopes for GAS and human neuronal proteins
- **PDB/AlphaFold Database:** 3D structures for structural mimicry analysis
- **Literature:** Published autoantibody targets in PANDAS/PANS and rheumatic fever

## Methodology
1. **Proteome Assembly:** Download complete GAS proteomes for 5+ serotypes from UniProt. Curate human target set: basal ganglia-enriched proteins (caudate, putamen, globus pallidus) from Human Protein Atlas + known PANDAS autoantibody targets.
2. **Sequence Mimicry Screen:** Run all-vs-all BLAST/DIAMOND of GAS proteins against human neuronal proteome. Filter hits by E-value, alignment length (≥8 amino acids for epitope relevance), and identity (≥40% over aligned region).
3. **Epitope Prediction:** For top mimicry hits, predict B-cell epitopes (BepiPred-3.0) and MHC-II binding (NetMHCIIpan) on both GAS and human sequences. Identify shared/cross-reactive epitope regions.
4. **Structural Mimicry Analysis:** For GAS-human pairs with available structures (PDB/AlphaFold), perform structural alignment (TM-align/FATCAT). Identify surface-exposed mimicry regions relevant to antibody binding.
5. **Serotype Comparison:** Compare mimicry profiles across GAS serotypes. Identify mimicry epitopes enriched in PANDAS-associated serotypes vs. non-associated serotypes.
6. **Target Prioritization:** Rank mimicry targets by: (a) epitope prediction confidence, (b) brain region expression (basal ganglia enrichment), (c) surface accessibility, (d) conservation across pathogenic serotypes, (e) overlap with known autoantibody targets.

## Expected Outputs
- Comprehensive mimicry map: GAS protein → human neuronal protein → shared epitope regions
- Ranked list of novel autoantibody target candidates
- Serotype-specific mimicry profiles (explaining strain pathogenicity differences)
- Network visualization of GAS-human mimicry relationships
- Structural models of top mimicry pairs

## Success Criteria
- Recover known mimicry targets (dopamine receptors, tubulin, lysoganglioside-related proteins) as positive controls
- Identify ≥5 novel high-confidence mimicry targets not previously reported
- Serotype comparison reveals differential mimicry potential between PANDAS-associated and non-associated strains

## Labels
autoimmune, immunology, genomic, novel-finding, high-priority
