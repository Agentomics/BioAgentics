# PANDAS Autoantibody Target Network Mapping

## Objective
Build a systematic protein interaction network of known PANDAS autoantibody targets to map downstream signaling disruption and explain the heterogeneity of neuropsychiatric symptoms across patients.

## Background
PANDAS autoantibodies target multiple neuronal proteins: dopamine D1 and D2 receptors, lysoganglioside GM1, tubulin, CaMKII (calcium/calmodulin-dependent protein kinase II), and neuronal surface glycolytic enzymes (pyruvate kinase M1, aldolase C, enolases). The Cunningham Panel measures antibodies against several of these targets clinically. However, no study has systematically mapped how disruption of these targets propagates through neuronal signaling networks to produce the specific symptom domains seen in PANDAS/PANS (OCD, tics, anxiety, eating restriction, cognitive regression, emotional lability).

Understanding the downstream network effects could explain why patients with different autoantibody profiles present with different symptom combinations, and could identify convergent signaling nodes as therapeutic targets.

## Data Sources
- **Protein interactions:** STRING database (v12), BioGRID, IntAct
- **Pathway databases:** KEGG (dopaminergic synapse, calcium signaling, basal ganglia circuits), Reactome (GPCR signaling, CaM pathway)
- **Receptor/signaling data:** UniProt, IUPHAR/BPS Guide to Pharmacology (DRD1, DRD2 signaling cascades)
- **Brain expression atlas:** Allen Human Brain Atlas, Human Protein Atlas (brain region specificity of target proteins)
- **Cunningham Panel targets:** Anti-DRD1, Anti-DRD2, Anti-lysoganglioside GM1, Anti-tubulin, CaMKII activity
- **Disease-gene associations:** DisGeNET (OCD, tic disorders, anxiety disorder gene sets)

## Methodology
1. **Seed network construction:** Define seed node set from all known PANDAS autoantibody targets. Query STRING/BioGRID for first-degree interactors (confidence > 0.7). Build the PANDAS autoantibody interactome.
2. **Pathway mapping:** Map network nodes to KEGG/Reactome pathways. Identify which signaling cascades are disrupted when seed proteins are functionally blocked by autoantibodies.
3. **Brain region specificity:** Overlay Allen Brain Atlas expression data to determine which network components are enriched in basal ganglia, prefrontal cortex, and thalamus — the brain regions implicated in PANDAS.
4. **Patient heterogeneity modeling:** McGregor R et al. 2025, JCI Insight (PMID 41252216) demonstrated marked patient-to-patient autoantibody heterogeneity in ARF using PhIP-Seq — public (shared) epitopes between patients were rare. This finding is critical for PANDAS: the network model must NOT assume all patients share the same autoantibody targets. Build subgroup-aware analysis: cluster patients by autoantibody profile, construct subgroup-specific subnetworks, and identify both shared and private network disruption patterns. PPP1R12B (enriched autoantigen in ARF) should be checked against PANDAS target lists. Account for **epitope spreading** — autoantibody targets may diversify over successive flares, meaning the network model should support temporal expansion of target sets.
5. **Symptom domain mapping:** Cross-reference disrupted pathways with disease-gene associations for OCD (compulsive behavior), tic disorders (motor circuits), anxiety, and eating disorders from DisGeNET/GWAS Catalog. Test whether different autoantibody targets map to different symptom pathways. **Note:** Patient heterogeneity (step 4) may explain why different patients with PANDAS present with different symptom combinations.
6. **Cytokine amplification layer:** Anti-NMDAR1 + IL-17 study (Mol Psychiatry 2025, doi: 10.1038/s41380-025-03434-x) demonstrated that autoantibodies alone cause NMDAR hypofunction, but full patient CSF causes neuronal hyperexcitability — dissociating antibody effects from inflammatory milieu effects. The network model must incorporate a **cytokine amplification layer** alongside direct autoantibody signaling disruption. Three key cytokines modify autoantibody target effects: (a) **IL-17**: promotes BBB permeability enabling autoantibody CNS access, connects to gut-oral-brain axis (Matera et al. 2025, PMID 41394880); (b) **IFNγ**: drives persistent epigenetic chromatin closing in neurons (Shammas 2026, PMID 41448185), meaning autoantibody + IFNγ may cause qualitatively different damage than autoantibody alone; (c) **IL-6**: central hub cytokine (approved task #363), gatekeeper for broader inflammatory cascade. Model this as a **two-hit framework**: Layer 1 = direct autoantibody binding and downstream protein interaction disruption; Layer 2 = cytokine-mediated amplification/modification of autoantibody effects. Different patients may have different ratios of Layer 1 vs Layer 2 pathology, connecting to the heterogeneity-aware design in step 4.
7. **Convergence analysis:** Identify hub proteins or signaling nodes where multiple autoantibody-disrupted pathways converge, including nodes where the cytokine amplification layer (step 6) intersects with direct autoantibody signaling. These represent candidate therapeutic targets — convergence between autoantibody and cytokine layers is of particular interest as it implies dual-mechanism intervention opportunities.
7. **Visualization:** Generate network graphs with symptom-domain coloring, pathway annotations, brain-region expression overlays, and patient subgroup stratification.

**Future experimental validation:** PhIP-Seq (phage immunoprecipitation sequencing) is the recommended method for systematically profiling PANDAS autoantibodies in future wet-lab studies, complementing computational predictions made here.

## Expected Outputs
- PANDAS autoantibody target interactome (network graph + table)
- Pathway disruption map: which signaling cascades are affected by each autoantibody target
- Symptom-pathway mapping: predicted associations between autoantibody profiles and clinical presentations
- Hub/convergence node analysis: candidate therapeutic targets where multiple disrupted pathways intersect
- Brain region specificity overlay for network components

## Success Criteria
- Network contains at least 50 high-confidence interactors across all seed targets
- At least 2 distinct pathway clusters map to different symptom domains (e.g., dopaminergic → OCD/tics, calcium signaling → anxiety)
- Identification of convergent hub nodes not previously proposed as PANDAS therapeutic targets
- Brain expression data confirms network enrichment in basal ganglia

## Labels
autoimmune, immunology, clinical, novel-finding, high-priority
