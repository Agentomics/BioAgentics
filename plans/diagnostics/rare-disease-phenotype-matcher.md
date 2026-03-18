# AI-Assisted Rare Disease Phenotype-to-Diagnosis Matching

## Objective
Build a graph-based phenotype matching system that ranks candidate rare disease diagnoses from clinical phenotype profiles (HPO terms), reducing the diagnostic odyssey for rare disease patients.

## Background
Rare diseases collectively affect ~400M people worldwide, yet the average time to diagnosis is 5-7 years ("diagnostic odyssey"). Over 7,000 rare diseases are catalogued in OMIM, most with characteristic phenotype profiles described using the Human Phenotype Ontology (HPO, >17,000 terms). Existing phenotype-matching tools (Exomiser, Phenomizer, LIRICAL) use information-content similarity metrics but do not leverage modern graph ML techniques. Recent advances in graph neural networks and knowledge graph embeddings could substantially improve matching accuracy, especially for partial/noisy phenotype profiles typical of real clinical encounters. A computational tool that accurately ranks diagnoses from HPO profiles could be embedded in EHR systems to flag potential rare disease diagnoses during routine clinical documentation.

## Data Sources
- **HPO** (Human Phenotype Ontology): 17,000+ phenotype terms with ontological relationships (is-a, part-of). Monthly releases, CC-BY.
- **OMIM** (Online Mendelian Inheritance in Man): 7,000+ disorders with phenotype annotations. Genemap2 download.
- **Orphanet**: Disease-phenotype associations with frequency qualifiers (obligate, very frequent, frequent, occasional, excluded). Public XML exports.
- **HPO Annotations** (phenotype.hpoa): Curated disease-phenotype associations from literature, frequency-qualified.
- **LIRICAL benchmark cases**: Published set of 384 diagnosed rare disease cases with HPO profiles from Robinsons et al. 2020.
- **UDN (Undiagnosed Diseases Network) published cases**: ~50 published case summaries with phenotype lists and final diagnoses.
- **ClinVar/DECIPHER**: Variant-phenotype associations for gene-level matching.

## Methodology

### Phase 1 — Knowledge Graph Construction
- Parse HPO ontology (OBO format) into directed acyclic graph
- Map OMIM diseases to HPO term sets with frequency annotations from phenotype.hpoa
- Add Orphanet disease-phenotype edges (different provenance, frequency detail)
- Construct heterogeneous knowledge graph: Disease ↔ Phenotype ↔ Gene nodes
- Add gene-disease edges from OMIM genemap2

### Phase 2 — Embedding & Matching Models
- **Baseline — Information Content (IC)**: Resnik/Lin semantic similarity with MICA (most informative common ancestor). Reproduce Phenomizer-style ranking.
- **Node2Vec embeddings**: Train node2vec on HPO DAG + disease nodes. Rank by cosine similarity of average query phenotype embeddings to disease embeddings.
- **Graph Attention Network (GAT)**: Train GAT on knowledge graph for disease-phenotype link prediction. Score = aggregated link prediction probabilities.
- **Frequency-weighted IC**: Extend IC model with Orphanet frequency weights (obligate terms contribute more than occasional).
- **Ensemble**: Combine IC + embedding + GAT scores via learned weights.

### Phase 3 — Evaluation
- **Simulated patients**: For each OMIM disease, sample HPO profiles of varying completeness (20%, 40%, 60%, 80% of annotated terms) with noise (5-20% random HPO terms added)
- **Real cases**: LIRICAL benchmark (384 cases), UDN published cases
- **Metrics**: Top-1, Top-5, Top-10, Top-20 accuracy; Mean Reciprocal Rank (MRR); ranking of correct diagnosis
- **Ablations**: Effect of profile completeness, noise level, disease prevalence, phenotype specificity

### Phase 4 — Clinical Utility Analysis
- Stratify performance by disease group (metabolic, neurological, immunological, skeletal, etc.)
- Analyze failure modes: which diseases are hardest to diagnose computationally and why
- Assess phenotype-to-time relationship: how early in clinical course (fewer phenotypes) can system achieve top-10 accuracy
- Compare against Phenomizer, LIRICAL, and Phen2Gene baselines

## Expected Outputs
- Heterogeneous knowledge graph (HPO + OMIM + Orphanet + Gene)
- Multiple phenotype matching models (IC baseline, node2vec, GAT, ensemble)
- Benchmark results on LIRICAL cases and simulated patients
- Analysis of diagnostic accuracy vs phenotype completeness
- Disease-group stratified performance report

## Success Criteria
- Top-10 accuracy ≥70% on LIRICAL benchmark cases
- MRR ≥0.40 on simulated patients at 60% phenotype completeness
- Outperform IC baseline by ≥10pp on Top-10 accuracy with graph-based models
- Graceful degradation: Top-20 accuracy ≥50% at 20% phenotype completeness

## Labels
rare-disease, ai-diagnostic, accessibility, novel-finding, high-priority
