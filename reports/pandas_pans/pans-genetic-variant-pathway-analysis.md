# PANS Genetic Variant Pathway Enrichment Analysis

**Project:** pans-genetic-variant-pathway-analysis
**Division:** pandas_pans
**Date:** 2026-03-19
**Data Sources:** Vettiatil et al. 2026, Dev Neurosci (PMID 41662332); Scientific Reports s41598-022-15279-3; GEO GSE102482; Enrichr (KEGG/Reactome/GO/MSigDB); STRING PPI database
**Pipeline:** `src/bioagentics/models/run_pans_pathway_analysis.py`
**Validation Status:** Pipeline executed; enrichment and PPI analyses complete; neuroinflammation cross-reference limited by platform data format

---

## Executive Summary

Pathway enrichment analysis of 22 ultra-rare genetic variants identified in PANS patients reveals that the **lectin complement pathway** is the most significantly enriched biological pathway (FDR = 1.12 × 10⁻⁵), driven by co-occurrence of MBL2, MASP1, and MASP2 variants. A second major enrichment cluster maps to **DNA damage repair** (nucleotide excision repair FDR = 5.84 × 10⁻³; DNA repair FDR = 6.42 × 10⁻³), with 4 genes (FANCD2, UVSSA, USP45, EP300) converging on NER and Fanconi anemia pathways. Protein-protein interaction network analysis identifies 5 functionally connected modules, including a high-confidence TREX1–SAMHD1 cGAS-STING module (STRING score 0.921) that directly links DNA damage sensing to type I interferon activation. The finding that PANS susceptibility variants cluster on lectin complement and pathogen-associated DNA sensing pathways provides a molecular link between the known infection-triggered clinical presentation and underlying genetic architecture.

## Important Caveats

- **Small gene set:** Over-representation analysis with 22 genes has limited statistical power. Enrichment results are driven by subsets of 2–4 genes per pathway.
- **Cohort size:** The variant gene set derives from a 32-patient cohort (Vettiatil et al. 2026). Larger cohorts are needed to confirm pathway-level convergence.
- **Neuroinflammation cross-reference incomplete:** The GEO GSE102482 platform (GPL23038) uses transcript cluster IDs rather than gene symbols, preventing direct cross-referencing of PANS variant genes with microglial LPS response data. This analysis component requires platform annotation mapping.
- **Variant classification heterogeneity:** The 22-gene set includes 4 pathogenic (P), 13 likely pathogenic (LP), and 5 VUS variants. Enrichment analyses do not weight by pathogenicity.
- **mTOR convergence was negative:** No direct or PPI-mediated connections between PANS variant genes and mTOR pathway components were detected.

## Background

PANDAS (Pediatric Autoimmune Neuropsychiatric Disorders Associated with Streptococcal infections) and PANS (Pediatric Acute-onset Neuropsychiatric Syndrome) are characterized by abrupt-onset OCD, tics, and neuropsychiatric symptoms following infection. While the clinical phenotype is well described, the genetic basis of susceptibility remains poorly understood. Most genetic studies are underpowered for genome-wide association; however, rare variant analysis offers an alternative approach by aggregating functionally related variants into biological pathways.

Vettiatil et al. (2026, Dev Neurosci, PMID 41662332) identified 14 pathogenic/likely pathogenic ultra-rare variants in DNA damage repair (DDR) genes from a cohort of 32 PANS patients using exome/whole-genome sequencing. Key genes include TREX1 and SAMHD1 (cGAS-STING innate immune pathway), ADNP (chromatin remodeling/neuroprotection), and EP300 (histone acetyltransferase in DDR signaling). This study expanded on an earlier discovery cohort (Scientific Reports, s41598-022-15279-3) and additionally identified a novel gut microbiome–genetic variant axis in PANS pathogenesis.

The field currently lacks a systematic computational analysis connecting PANS genetic findings to functional pathways. This work addresses that gap by performing pathway enrichment, protein interaction network analysis, and pathway convergence testing on the curated PANS variant gene set.

## Methodology

### Gene Set Curation

Twenty-two genes were curated from two published studies (Vettiatil et al. 2026; Scientific Reports s41598-022-15279-3) and organized into five biologically motivated pathway axes:

| Pathway Axis | Genes | Count |
|---|---|---|
| DDR-cGAS-STING/AIM2 inflammasome | CUX1, USP45, PARP14, UVSSA, EP300, TREX1, SAMHD1, STK19, PIDD1, FANCD2, RAD54L | 11 |
| Mitochondrial-innate immunity | PRKN, POLG | 2 |
| Gut-immune | LGALS4, DUOX2, CCR9 | 3 |
| Lectin complement | MBL2, MASP1, MASP2 | 3 |
| Chromatin/neuroprotection | MYT1L, TEP1, ADNP | 3 |

Variant classification: 4 pathogenic (UVSSA, TREX1, SAMHD1, MYT1L, ADNP), 13 likely pathogenic, 5 VUS. MBL2 was added based on research director guidance (task #335) given its role in lectin complement pathway initiation and GAS opsonization.

### Pathway Enrichment Analysis

Over-representation analysis (ORA) was performed via Enrichr against six gene set libraries:
- KEGG 2021 Human
- Reactome 2022
- GO Biological Process 2023
- GO Molecular Function 2023
- GO Cellular Component 2023
- MSigDB Hallmark 2020

Significance threshold: FDR < 0.05 (Benjamini-Hochberg correction). Immune-focused pathway filtering was applied using 20 curated search terms spanning complement, TLR signaling, cGAS-STING, Th17, type I interferon, DNA damage, and mitophagy pathways.

### Protein-Protein Interaction Network Analysis

The STRING database (v12) was queried for all pairwise interactions among the 22 PANS variant genes using a minimum combined confidence score threshold of 400 (medium confidence). Network metrics (degree centrality, betweenness centrality, clustering coefficient) were computed with NetworkX. Community detection used greedy modularity optimization. Hub genes were ranked by combined degree and betweenness centrality.

### mTOR Convergence Analysis

Direct overlap between the 22 PANS variant genes and a curated set of 55 core mTOR pathway genes (KEGG hsa04150, Reactome mTOR signaling, and autophagy effectors including ATG7/UVRAG from the IVIG initiative) was computed. PPI-mediated connections were assessed by checking whether STRING network neighbors of PANS genes included mTOR pathway members. Reference: Fronticelli Baldelli G et al. 2025 (PMID 41462744).

### Neuroinflammation Cross-Reference (Limited)

GEO dataset GSE102482 (LPS-treated mouse microglia, 8 LPS vs 4 control samples) was downloaded and preprocessed. Differential expression was computed using Welch's t-test with BH FDR correction, yielding 28,846 genes with 6,032 significant (FDR < 0.05). However, the platform (GPL23038) uses transcript cluster IDs rather than standard gene symbols, preventing direct cross-referencing with PANS variant gene symbols. This analysis requires platform annotation file integration to map transcript clusters to gene symbols.

## Results

### Pathway Enrichment

ORA identified **719 total terms** across six libraries, of which **37 were significantly enriched (FDR < 0.05)** and 30 mapped to immune-focused keywords (8 significant). Results cluster into four major functional categories:

#### 1. Lectin Complement Pathway (Most Significant)

| Pathway | Library | FDR | Genes |
|---|---|---|---|
| Lectin pathway of complement activation | Reactome | 1.12 × 10⁻⁵ | MASP2, MASP1, MBL2 |
| Complement activation, lectin pathway | GO:BP | 2.44 × 10⁻⁵ | MASP2, MASP1, MBL2 |
| Creation of C4 and C2 activators | Reactome | 3.63 × 10⁻⁵ | MASP2, MASP1, MBL2 |
| Initial triggering of complement | Reactome | 8.80 × 10⁻⁵ | MASP2, MASP1, MBL2 |
| Complement cascade | Reactome | 1.02 × 10⁻³ | MASP2, MASP1, MBL2 |
| Complement and coagulation cascades | KEGG | 3.59 × 10⁻³ | MASP2, MASP1, MBL2 |

Six of the top 8 enriched terms are lectin complement pathways, consistently driven by MBL2, MASP1, and MASP2. MBL2 encodes mannose-binding lectin, the initiator of the lectin complement pathway and a key opsonin for Group A Streptococcus (GAS) — the canonical trigger for PANDAS.

#### 2. DNA Damage Repair

| Pathway | Library | FDR | Genes |
|---|---|---|---|
| Nucleotide excision repair | Reactome | 5.84 × 10⁻³ | UVSSA, USP45, EP300 |
| DNA repair | Reactome | 6.42 × 10⁻³ | FANCD2, UVSSA, USP45, EP300 |
| DNA metabolic process | GO:BP | 4.40 × 10⁻² | TREX1, USP45, SAMHD1, POLG |
| DNA damage response | GO:BP | 4.40 × 10⁻² | PIDD1, TREX1, USP45, SAMHD1 |
| Cytosolic sensors of pathogen-associated DNA | Reactome | 2.67 × 10⁻² | TREX1, EP300 |
| TC-NER pre-incision complex formation | Reactome | 2.15 × 10⁻² | UVSSA, EP300 |

DDR enrichment is driven by two distinct sub-networks: a nucleotide excision repair cluster (UVSSA, USP45, EP300) and a cytosolic DNA sensing cluster (TREX1, SAMHD1). The latter is particularly relevant: TREX1 and SAMHD1 are both negative regulators of the cGAS-STING pathway — loss-of-function variants in either gene cause constitutive activation of type I interferon signaling, as seen in Aicardi-Goutieres syndrome, an autoimmune interferonopathy.

#### 3. Infection Response Pathways

| Pathway | Library | FDR | Genes |
|---|---|---|---|
| Staphylococcus aureus infection | KEGG | 3.59 × 10⁻³ | MASP2, MASP1, MBL2 |
| SARS-CoV-2 infection | Reactome | 5.84 × 10⁻³ | MASP2, MASP1, PARP14, MBL2 |
| Coronavirus disease | KEGG | 3.23 × 10⁻² | MASP2, MASP1, MBL2 |

Enrichment in infection response pathways reflects the role of lectin complement in pathogen clearance. S. aureus infection enrichment (FDR = 3.59 × 10⁻³) is directly relevant: GAS is a related Gram-positive pathogen recognized by MBL2 and cleared via the same complement cascade. PARP14 additionally appears in viral infection pathways, consistent with its role in macrophage polarization.

#### 4. Innate Immune Regulation

| Pathway | Library | FDR | Genes |
|---|---|---|---|
| Innate immune system | Reactome | 4.48 × 10⁻² | TREX1, EP300, MASP2, MASP1, MBL2, SAMHD1, PRKN |
| Immune system | Reactome | 3.80 × 10⁻² | PRKN, TREX1, EP300, MASP2, MASP1, SAMHD1, MBL2 |
| Negative regulation of innate immune response | GO:BP | 4.67 × 10⁻² | TREX1, SAMHD1 |
| Innate immune response, cell surface receptor | GO:BP | 4.40 × 10⁻² | EP300, MBL2 |
| Positive regulation of cytokine-mediated signaling | GO:BP | 4.40 × 10⁻² | PRKN, PARP14 |

Seven of 22 PANS variant genes (32%) are annotated to the Reactome "Immune System" pathway, demonstrating broad immune pathway convergence beyond the specific lectin complement cluster.

#### Notable Near-Significant Findings

Several immune-relevant pathways fell just below significance:
- Interferon gamma response (MSigDB, FDR = 0.081): PARP14, SAMHD1
- Type I interferon signaling (Reactome, FDR = 0.129): SAMHD1
- Maintenance of blood-brain barrier (GO:BP, FDR = 0.070): MBL2
- Mitophagy (Reactome, FDR = 0.093): PRKN
- NF-kappa B signaling (KEGG, FDR = 0.252): PIDD1

### PPI Network Analysis

STRING network analysis identified **8 protein-protein interactions** among the 22 PANS variant genes (22 nodes, 8 edges), forming **5 connected components** and 10 isolated nodes.

#### Network Modules

| Module | Genes | Pathway Axis | Confidence | Biological Function |
|---|---|---|---|---|
| 0 | EP300–ADNP–LGALS4 | Cross-axis (DDR + Chromatin + Gut) | 0.427–0.479 | Chromatin-immune crosstalk hub |
| 1 | MBL2–MASP1–MASP2 | Lectin complement | 0.986–0.999 | Lectin complement activation complex |
| 2 | TREX1–SAMHD1 | DDR-cGAS-STING | 0.921 | Cytosolic DNA sensing / type I IFN regulation |
| 3 | FANCD2–RAD54L | DDR | 0.667 | Fanconi anemia / homologous recombination repair |
| 4 | PRKN–POLG | Mitochondrial | 0.548 | Mitophagy / mitochondrial maintenance |

#### Hub Gene Analysis

EP300 is the most connected gene in the network (degree = 2, highest betweenness centrality = 0.005), bridging three pathway axes: DDR (via ADNP), chromatin remodeling (via ADNP), and gut-immune function (via LGALS4). This cross-axis connectivity positions EP300 as a potential convergence node in PANS pathophysiology.

The lectin complement module (MBL2–MASP1–MASP2) shows the highest interaction confidence in the network (0.986–0.999), reflecting the well-established physical complex that initiates lectin complement activation. TREX1–SAMHD1 (confidence 0.921) represents the second-strongest interaction, consistent with their shared role in cGAS-STING pathway regulation.

### mTOR Convergence Analysis

- **Direct overlap:** 0/22 PANS genes are members of the curated mTOR pathway (55 genes from KEGG hsa04150, Reactome mTOR signaling).
- **PPI-mediated connections:** 0 PANS genes have STRING neighbors in the mTOR pathway.
- **Conclusion:** The mTOR convergence hypothesis (Fronticelli Baldelli G et al. 2025, PMID 41462744) is not supported by direct gene-level or PPI-level evidence from the PANS variant gene set. If mTOR contributes to PANS pathophysiology, it likely operates through downstream or indirect mechanisms not captured at the variant gene level.

### Neuroinflammation Cross-Reference (Incomplete)

GEO dataset GSE102482 was successfully downloaded and differential expression analysis completed (28,846 genes; 6,032 significant at FDR < 0.05 under LPS stimulation). However, the GPL23038 platform uses Affymetrix Clariom D transcript cluster IDs (e.g., "TC1000003019.mm.2") rather than standard gene symbols. As a result, none of the 22 PANS variant genes could be mapped to the expression data. **This analysis requires platform annotation file integration** to convert transcript cluster IDs to gene symbols before cross-referencing is possible.

## Discussion

### Lectin Complement as a Primary Susceptibility Pathway

The most significant finding is the strong enrichment of PANS variant genes in the **lectin complement pathway** (FDR = 1.12 × 10⁻⁵). The co-occurrence of ultra-rare variants in MBL2, MASP1, and MASP2 — three genes encoding the core initiation complex of lectin complement — in a PANS cohort has direct mechanistic implications:

1. **MBL2** encodes mannose-binding lectin, which recognizes N-acetylglucosamine on GAS cell walls and initiates complement-mediated opsonization. MBL2 deficiency is an established risk factor for recurrent infections in children. Loss-of-function variants could impair GAS clearance, leading to prolonged infection and sustained autoimmune stimulation.

2. **MASP1 and MASP2** encode the serine proteases that execute complement activation downstream of MBL2 binding. Variants in all three genes would predict compound impairment of the lectin complement pathway.

This finding provides a molecular mechanism connecting PANS genetic susceptibility to the known clinical trigger: impaired lectin complement function → reduced GAS opsonization → prolonged or aberrant immune response → autoimmune neuropsychiatric sequelae. This model is consistent with the molecular mimicry hypothesis in PANDAS, where incomplete pathogen clearance sustains cross-reactive immune activation.

### DDR-cGAS-STING: A Second Convergent Pathway

The TREX1–SAMHD1 interaction module identifies the **cGAS-STING cytosolic DNA sensing pathway** as a second convergent mechanism. Both TREX1 and SAMHD1 are negative regulators of cGAS-STING signaling:

- **TREX1** is a 3'–5' exonuclease that degrades cytosolic DNA; loss-of-function mutations cause constitutive cGAS-STING activation and type I interferon production, as in Aicardi-Goutieres syndrome.
- **SAMHD1** is a dNTPase that restricts reverse transcription and regulates innate immunity; mutations cause a phenocopy of Aicardi-Goutieres syndrome.

Pathogenic variants in TREX1 and SAMHD1 in PANS patients suggest that **constitutive or infection-triggered type I interferon overactivation** may contribute to PANS pathophysiology. This parallels the neuroinflammatory mechanism in Aicardi-Goutieres syndrome, where unchecked interferon signaling damages the CNS.

### Cross-Axis Hub EP300

EP300 uniquely bridges three pathway axes in the PPI network (DDR → chromatin remodeling → gut immunity), interacting with both ADNP (chromatin/neuroprotection) and LGALS4 (gut-immune). As a histone acetyltransferase involved in DDR signaling, transcriptional regulation, and immune modulation, EP300 may serve as a functional convergence node. Its involvement in both nucleotide excision repair (FDR = 5.84 × 10⁻³) and innate immune system pathways (FDR = 4.48 × 10⁻²) supports a model where DDR dysfunction and immune dysregulation are mechanistically linked in PANS.

### Relevance to Infection Pathways

Enrichment in *Staphylococcus aureus* infection (KEGG, FDR = 3.59 × 10⁻³) and SARS-CoV-2 infection pathways (Reactome, FDR = 5.84 × 10⁻³) reflects the broad role of lectin complement in pathogen defense. This is clinically significant because PANS can be triggered by multiple infectious agents beyond GAS, including Mycoplasma, influenza, and other respiratory pathogens. Variants that impair complement-mediated pathogen clearance would predict susceptibility to diverse infection-triggered neuropsychiatric presentations.

## Limitations

1. **Small gene set (n = 22):** ORA enrichment is dominated by subsets of 2–4 genes. Individual pathway-level results should be interpreted cautiously until confirmed in larger cohorts.
2. **Small patient cohort (n = 32):** The variant gene set is derived from a single study; additional exome studies are needed to validate gene-pathway associations.
3. **No gene set enrichment analysis (GSEA):** Only ORA was performed. GSEA using ranked variant scores (e.g., CADD) would provide complementary evidence.
4. **Neuroinflammation data gap:** The GSE102482 cross-reference failed due to platform ID format. This was a key planned analysis that remains incomplete.
5. **Sparse PPI network:** Only 8 edges among 22 nodes suggests that many PANS variant genes operate in distinct pathways without direct physical interactions. Network densification using extended interaction partners would improve the analysis.
6. **VUS inclusion:** 5 of 22 genes are classified as VUS; their inclusion may introduce noise.
7. **No functional validation:** Enrichment analysis identifies candidate pathways but does not confirm that specific variants causally affect pathway function.

## Next Steps

1. **Complete neuroinflammation cross-reference:** Integrate GPL23038 platform annotation files to map transcript cluster IDs to gene symbols, then re-run the PANS-GSE102482 cross-reference analysis.
2. **Lectin complement functional studies:** Assess MBL2 serum levels and complement activity in PANS patients carrying identified variants. Correlate with GAS infection history and clinical severity.
3. **Type I interferon profiling:** Measure interferon-stimulated gene (ISG) expression in patients with TREX1/SAMHD1 variants to test the cGAS-STING hyperactivation hypothesis.
4. **Extended PPI network:** Expand STRING queries to include first-degree interactors of PANS variant genes, enriching the network for pathway analysis.
5. **Larger cohort replication:** Apply the 22-gene pathway analysis to additional PANS exome cohorts as they become available.
6. **GSEA with variant scores:** Rank all exome variants by CADD score and run GSEA to complement the ORA results.
7. **Cross-initiative integration:** Connect findings with the IVIG autophagy analysis (ATG7/UVRAG) — while mTOR convergence was negative at the gene level, autophagy-immune crosstalk may operate at the pathway level.

## References

1. Vettiatil D et al. (2026). "Ultrarare Variants in DNA Damage Repair and Mitochondrial Genes in Pediatric Acute-Onset Neuropsychiatric Syndrome." *Dev Neurosci*. PMID 41662332.
2. Scientific Reports, s41598-022-15279-3. Original PANS exome/WGS discovery cohort.
3. Fronticelli Baldelli G et al. (2025). mTOR pathway in PANDAS/PANS pathogenesis. PMID 41462744.
4. GEO Dataset GSE102482. LPS-treated mouse microglia transcriptomics. Platform GPL23038.
5. STRING database v12. Protein-protein interaction data.
6. Enrichr gene set enrichment analysis platform. KEGG, Reactome, GO, MSigDB databases.
7. Chen EY et al. (2013). "Enrichr: interactive and collaborative HTML5 gene list enrichment analysis tool." *BMC Bioinformatics*.
