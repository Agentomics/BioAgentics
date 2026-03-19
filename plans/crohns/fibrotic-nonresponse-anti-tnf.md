# Fibrotic Non-Response to Anti-TNF Hypothesis

## Objective

**Hypothesis:** A significant subset of anti-TNF non-responders in CD have a pre-existing fibrosis gene signature, and this "fibrotic non-responder" subtype can be identified at baseline and redirected to anti-fibrotic therapies (TL1A inhibitors, harmine) instead of biologic escalation.

**Falsification criterion:** If the overlap between anti-TNF non-response top genes and fibrosis signatures is not significant (hypergeometric p > 0.05) AND fibrosis scores do not improve prediction AUC by >0.03, the hypothesis is refuted.

## Rationale

### Cross-Project Connection (Catalyst Reasoning)

This initiative bridges three existing projects:

1. **anti-tnf-response-prediction** — Key finding: the Porto 5-gene immune signature performs BELOW chance (AUC 0.39-0.47) in our data. Only 1/30 top genes (OSMR) overlaps published immune-focused signatures. Pathway analysis shows ECM remodeling (3 genes) and epithelial barrier integrity (4 genes) among top pathways. This suggests non-response is NOT purely an immune phenotype.

2. **cd-fibrosis-drug-repurposing** — Has 7 distinct fibrosis pathway signatures totaling 1,335 genes across 6 signature types: bulk (998), cell-type (84), transition (74), GLIS3/IL-11 (38), CTHRC1/YAP-TAZ (69), TL1A-DR3/Rho (72). Harmine validated as TWIST1 inhibitor with Phase I human safety data.

3. **cd-stricture-risk-prediction** — Building a stricture risk model using pericyte and inflammatory monocyte signatures. Stricturing is the clinical endpoint of fibrosis.

### Why This Is Unconventional

- Anti-TNF response prediction universally focuses on immune biomarkers (TNF signaling, Th1/Th17, innate immunity)
- The field assumes non-response is about immune pathway mismatch (wrong target) rather than tissue state (irreversible damage)
- No published anti-TNF response study has tested fibrosis gene scores as predictors
- The insight comes from a NEGATIVE finding (immune signatures fail) combined with a POSITIVE finding (ECM genes in top 30) — this cross-project pattern only emerges when you look at both projects together
- The Research Director would not propose this because fibrosis and biologics response are treated as separate clinical domains

### Biological Basis

- CD fibrosis is largely inflammation-independent once established. TGF-β-driven ECM deposition continues even after TNF blockade.
- Anti-TNF drugs target inflammation. If a patient's disease is driven by fibrosis rather than active inflammation, anti-TNF will fail — not because the drug doesn't work, but because the wrong process is being targeted.
- The anti-TNF top-30 gene list includes ECM genes and epithelial barrier genes, consistent with a fibrotic/structural damage component to non-response.
- Stricturing behavior in CD develops early and is associated with worse outcomes including surgery. Early identification of fibrosis-prone patients could redirect them to TL1A inhibitors (tulisokibart, duvakitug) which have demonstrated anti-fibrotic properties.
- Literature finding: harmine (TWIST1 inhibitor) validated anti-fibrotic in JCI with Phase I human safety (journal #1343).

## Data Sources

- **Anti-TNF response DE results:** `output/crohns/anti-tnf-response-prediction/differential_expression/de_results.csv` — 30-gene signature with pathway annotations
- **Fibrosis signatures:** `src/bioagentics/data/cd_fibrosis/` — 6 gene signature modules (bulk, celltype, transition, GLIS3/IL-11, CTHRC1/YAP-TAZ, TL1A-DR3/Rho)
- **RISK cohort (GSE57945):** Pediatric CD with Montreal classification phenotype annotations (B1 inflammatory, B2 stricturing, B3 penetrating)
- **GSE16879:** Arijs infliximab response cohort with pre/post treatment biopsies
- **MSigDB pathways:** Already curated for fibrosis project (KEGG, Reactome, Hallmark, GO BP)

## Methodology

### Phase 1: Signature Overlap Analysis
1. Extract the top 30 anti-TNF non-response genes (with direction of effect)
2. Load all 6 fibrosis signatures from cd-fibrosis-drug-repurposing
3. Compute pairwise overlap using hypergeometric test (background: all genes on array platform)
4. For overlapping genes, check direction consistency (upregulated in non-response AND upregulated in fibrosis)
5. Expand to top 100-200 DE genes (relaxed threshold) for enrichment sensitivity

### Phase 2: Fibrosis Score as Anti-TNF Predictor
1. Compute composite fibrosis scores (ssGSEA or mean z-score) for each patient in anti-TNF cohorts using fibrosis signatures
2. Test association between baseline fibrosis score and anti-TNF response (logistic regression, AUC)
3. Add fibrosis score as a feature to the existing anti-TNF prediction pipeline
4. Compare AUC with vs without fibrosis score (must be done WITHIN CV folds to avoid leakage — the anti-tnf project learned this the hard way)

### Phase 3: Clinical Phenotype Validation
1. In RISK cohort (GSE57945): test whether Montreal B2 (stricturing) patients have higher fibrosis scores AND worse anti-TNF outcomes (if treatment data available)
2. In GSE16879: correlate pre-treatment fibrosis score with post-treatment histological response
3. Identify the "fibrotic non-responder" cluster: non-responders with high fibrosis score AND low immune activation score

### Phase 4: Therapeutic Redirect Analysis
1. For identified fibrotic non-responders, compute CMAP connectivity scores against fibrosis-reversing compounds from cd-fibrosis-drug-repurposing
2. Test whether TL1A pathway signature (from TL1A-DR3/Rho module) is inversely correlated with the fibrotic non-response profile
3. If validated, propose a decision tree: baseline fibrosis score → if high → TL1A inhibitor instead of anti-TNF → if low → anti-TNF with standard biomarkers

## Success Criteria

1. **Primary:** Significant gene overlap (hypergeometric p < 0.01) between anti-TNF non-response genes and ≥1 fibrosis signature
2. **Secondary:** Fibrosis score adds >0.05 AUC to anti-TNF prediction model (within proper CV)
3. **Exploratory:** Montreal B2 phenotype associated with anti-TNF non-response (OR > 2.0, p < 0.05)
4. **Translational:** At least one fibrosis-reversing compound from CMAP screen shows inverse connectivity with fibrotic non-response profile

## Risk Assessment

**Most likely failure mode:** The 3 ECM genes in the 30-gene anti-TNF signature are incidental (selected by the classifier for technical reasons, not biological). The hypergeometric test will show no significant overlap, and fibrosis scores will not predict response.

**What we learn if it fails:** Confirms that anti-TNF non-response is primarily an immune phenomenon despite immune signatures performing poorly, suggesting the problem is immune heterogeneity (different immune mechanisms across patients) rather than a non-immune mechanism. This would redirect efforts toward single-cell immune profiling of non-responders.

**Estimated probability of success:** 30-40%

## Labels

catalyst, clinical, drug-repurposing, immunology, novel-finding, promising
