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
4. **Symptom domain mapping:** Cross-reference disrupted pathways with disease-gene associations for OCD (compulsive behavior), tic disorders (motor circuits), anxiety, and eating disorders from DisGeNET/GWAS Catalog. Test whether different autoantibody targets map to different symptom pathways.
5. **Convergence analysis:** Identify hub proteins or signaling nodes where multiple autoantibody-disrupted pathways converge. These represent candidate therapeutic targets.
6. **Visualization:** Generate network graphs with symptom-domain coloring, pathway annotations, and brain-region expression overlays.

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
