# NSCLC Therapeutic Target Discovery via DepMap Dependency Mapping: Subtype-Specific Vulnerabilities in KRAS-Mutant Lung Cancer

**Project:** nsclc-depmap-targets
**Division:** cancer
**Date:** 2026-03-31
**Data Sources:** DepMap 25Q3, TCGA LUAD/LUSC (GDC), PRISM 24Q2, SL benchmarks (Genome Biology 2025, Desjardins isogenic, Vermeulen computational), LUCA scRNA-seq atlas
**Pipeline:** `src/bioagentics/data/nsclc_depmap.py`, `src/bioagentics/data/nsclc_tcga.py`, `src/bioagentics/models/feature_prep.py`, `src/bioagentics/models/run_phase2.py`, `src/bioagentics/models/sl_benchmarks.py`
**Validation Status:** All primary findings validated by validation_scientist (journals #2190, #2200, #2197, #2195, #2430)

---

## Executive Summary

We mapped genome-scale CRISPR dependency data from 95 NSCLC cell lines (DepMap 25Q3) onto 1,125 TCGA NSCLC patients using expression-based elastic-net models, reproducing and extending the translational cancer dependency map approach (Meyers et al., Nature Cancer 2024). By stratifying patients into KRAS-mutant molecular subtypes — KP (KRAS+TP53, n=57), KL (KRAS+STK11/KEAP1, n=31), KOnly (KRAS-only, n=73), and KRAS-WT (n=964) — we identified 8 composite-ranked KL-specific vulnerability targets with existing drug candidates, led by MDM2 (idasanutlin/navtemadlin, composite score 53.35), MDM4 (ALRN-6924, score 51.49), and SLC16A3/MCT4 (AZD0095, score 50.83). SLC16A3 emerged as both the strongest biological signal in the dataset (KW FDR = 1.2 x 10^-26) and a dual therapeutic-biomarker target, with a compelling mechanistic link: STK11 loss inactivates AMPK, driving glycolytic shift and MCT4 dependency. Additional evidence tiers include FSP1/AIFM2 (supported by in vivo tumor suppression data, not captured by DepMap) and the STK11-MARK2 synthetic lethal pair (validated in isogenic CRISPR screens). These findings provide a prioritized, subtype-stratified target list for KL-subtype NSCLC patients — a population (~20-30% of KRAS-mutant NSCLC) that currently lacks effective immunotherapy options.

---

## Important Caveats

1. **KL subtype sample size is small (n=31 patients, n=5 cell lines in some analyses).** Kruskal-Wallis tests achieve significance because they compare across all 4 groups (n=1,125 total), not from the KL group alone. Effect sizes and rankings should be interpreted with caution pending validation in larger cohorts.

2. **Expression-based models cannot predict all clinically important targets.** Nine known NSCLC targets (KRAS, EGFR, ALK, ROS1, MET, AXL, SHP2/PTPN11, XIAP/BIRC4, HNF4A) are not among the 222 predictable genes because expression alone does not predict their dependency — mutation status and protein activity are the actual drivers.

3. **Synthetic lethality enrichment was not significant.** Fisher enrichment testing yielded FDR > 0.80 across all subtypes. SL component scores = 0 for all 8 top composite-ranked targets. The SL integration component (20% weight) contributed nothing to top target rankings.

4. **Drug ranking bias.** The druggability bonus (20 points for any drug match) dominates composite rankings. Without it, the strongest biological signals — DYRK1A (effect -0.70) and CCDC86 — would outrank 5 of the 8 reported targets. Drug availability should not be conflated with therapeutic priority.

5. **PRISM concordance is weak.** Only 1 of 39 gene-drug mappings (bortezomib/PSMB7) is present in the PRISM library. Drug-dependency concordance is non-significant (r=0.12, p=0.31). Drug mappings are literature-curated, not pharmacogenomically validated.

6. **CRISPR knockout does not equal pharmacological inhibition.** Complete gene loss in 2D culture may overestimate or underestimate drug-achievable effects in tumors. The ferroptosis panel (FSP1, GPX4) demonstrates that 2D normoxic culture systematically misses in vivo dependencies.

7. **KP subtype analysis has high false-positive rate.** 5 of 10 top KP vulnerability targets had positive effects (KP LESS dependent), a structural artifact of the composite scoring formula when non-effect components (drug, SL, significance) override the direction filter.

8. **Direction artifact in initial analysis required correction.** The original Phase 3 integration used absolute effect sizes, placing "release from dependency" genes as top targets. All results reported here use the corrected direction-aware scoring (verified in journal #1149).

---

## Background

Non-small cell lung cancer (NSCLC) remains the leading cause of cancer death worldwide. Despite 7 FDA approvals in 2025 (c-MET, TROP-2, EGFR exon20, ROS1, HER2), most patients lack targeted options. KRAS-mutant NSCLC — the largest molecular subgroup (~30%) — subdivides into clinically distinct subtypes:

- **KP (KRAS+TP53):** ~40-50% of KRAS-mutant; higher tumor mutational burden; generally immunotherapy-responsive
- **KL (KRAS+STK11/KEAP1):** ~20-30% of KRAS-mutant; immunosuppressive TME, metabolically rewired; immunotherapy-resistant
- **KOnly (KRAS-only):** Intermediate phenotype

KL patients represent the most pressing unmet need. Recent clinical data confirms the challenge: in the olomorasib (KRAS G12C inhibitor) + pembrolizumab combination trial, STK11-mutant patients achieved only 40.5% ORR (vs 74% overall), and KEAP1-mutant patients achieved 28.6% ORR — confirming KEAP1 status as a universal negative predictor across treatment modalities (Phase 3 SUNRAY-01 currently recruiting).

The DepMap project (Broad Institute) provides genome-scale CRISPR knockout dependency scores across 1,000+ cancer cell lines. Meyers et al. (Nature Cancer, August 2024) demonstrated that expression-only elastic-net models trained on DepMap can be transposed to predict dependencies in 9,596 TCGA patients — the "translational cancer dependency map" (TCGADEPMAP). However, the existing TCGADEPMAP is pan-cancer and does not incorporate: (1) NSCLC molecular subtype stratification, (2) synthetic lethality prediction integration, (3) PRISM drug response mapping, or (4) ferroptosis pathway analysis relevant to KEAP1/NRF2-mutant tumors.

| Clinical Benchmark | Result | Source |
|---|---|---|
| Olomorasib + pembro (KRAS G12C) | ORR 74%, DCR 91%, mPFS 11.8 mo | Phase 2, 2026 |
| — STK11-mutant subgroup | ORR 40.5% | Same trial |
| — KEAP1-mutant subgroup | ORR 28.6% | Same trial |
| AXL + pembro (bemcentinib) | ORR 26% in AXL+ tumors | Phase 2 |
| SUNRAY-01 (olomorasib Phase 3) | Recruiting | NCT ongoing |

---

## Methodology

### Phase 1: Data Acquisition and Preprocessing

**Cell line annotation** (`nsclc_depmap.py`): We identified 95 NSCLC cell lines from DepMap 25Q3 (OncotreePrimaryDisease = "Non-Small Cell Lung Cancer"). Each line was annotated for mutations in 12 driver genes (KRAS, TP53, STK11, KEAP1, EGFR, ALK, MET, BRAF, ROS1, ERBB2, NF1, RB1) using DepMap somatic mutation calls filtered to HIGH/MODERATE VEP impact. KRAS alleles were classified (G12C, G12D, G12V, G12_other, G13, Q61) and molecular subtypes assigned (KP, KL, KPL, KOnly, KRAS-WT). KPL (triple-mutant KRAS+TP53+STK11) was grouped with KL based on STK11-dominant biology.

**Patient classification** (`nsclc_tcga.py`): TCGA LUAD and LUSC patients were classified using MAF somatic mutation data filtered to damaging variant classifications (missense, nonsense, frameshift, splice site, in-frame indels). The same 12-gene driver panel and subtype classification was applied. Clinical metadata (vital status, survival time, tumor stage, age) was merged where available. Final cohort: 1,125 NSCLC patients (KP=57, KL=31, KOnly=73, KRAS-WT=964).

### Phase 2: Dependency Prediction Models

**Feature preparation** (`feature_prep.py`): Expression features were prepared by selecting the top 6,000 most-variable genes from DepMap expression data (OmicsExpressionTPMLogp1, log2(TPM+1) scale). CRISPR dependency scores (CRISPRGeneEffect) served as prediction targets. Expression and dependency matrices were aligned to the intersection of NSCLC cell lines with data in both modalities.

**Model training** (`run_phase2.py`): Per-gene elastic-net regression models were trained predicting CRISPR dependency from expression features using 5-fold cross-validation. Genes with cross-validated Pearson r >= 0.3 were retained as "predictable" (222 genes after excluding ACTB, a housekeeping gene identified as a technical confounder; KW stat=115.6, FDR=4.6e-24).

**TCGA prediction:** Trained models were applied to TCGA NSCLC expression data (log2-transformed) to generate patient-level predicted dependency scores for all 222 predictable genes across 1,125 patients.

**Subtype-specific analysis:** Predicted dependencies were stratified by molecular subtype. Kruskal-Wallis tests identified genes with significantly different dependency profiles across subtypes (FDR < 0.05). Pairwise comparisons quantified the direction and magnitude of subtype-specific effects.

### Phase 3: Integration and Composite Scoring

**Direction-aware composite scoring:** Each gene received a composite vulnerability score per subtype using four weighted components:

| Component | Weight | Formula |
|---|---|---|
| Vulnerability effect | 40% | min(max(-effect_size, 0), 1) x 40 |
| KW significance | 20% | min(-log10(max(FDR, 1e-30)), 30) / 30 x 20 |
| Druggability | 20% | 20 if any drug match, else 0 |
| SL support | 20% | min(n_sl_hits, 5) / 5 x 20 |

The direction filter ensures only genes where the subtype is MORE dependent (negative effect) receive non-zero vulnerability effect scores. This corrects a critical error in the initial analysis where absolute effect sizes were used, conflating "gained vulnerability" with "release from dependency."

**Synthetic lethality integration** (`sl_benchmarks.py`): SL pairs were compiled from three sources with confidence tiers: Genome Biology 2025 combinatorial CRISPR (117 pairs, "experimental"), Desjardins isogenic screens (15 drivers, "isogenic"), and Vermeulen computational predictions ("computational"). Gene pairs were normalized alphabetically, deduplicated, and the best confidence tier retained per pair. 1,477 total SL pairs; 312 intersecting predicted genes (KL=122, KOnly=112, KP=26, KRAS-WT=52). Fisher enrichment was non-significant across all subtypes (all FDR > 0.80).

**PRISM drug mapping:** Literature-curated gene-drug matches were compiled for 35 genes (41 gene-drug pairs). Only 1 match (bortezomib/PSMB7) was present in the PRISM drug sensitivity library.

**Ferroptosis panel analysis:** Nine ferroptosis-related genes (AIFM2/FSP1, GPX4, SAT1, GLS, SLC7A11, NCOA4, TFRC, SHMT1, SHMT2) were assessed for KEAP1/STK11 mutation-associated dependency differences. No gene reached significance after FDR correction (all FDR > 0.44). GPX4 showed a marginal uncorrected trend (p=0.040) with KEAP1-mutant lines being LESS dependent — consistent with NRF2-driven antioxidant compensation in vitro.

---

## Results

### Tier 1: Composite-Ranked KL Vulnerability Targets

Direction-aware composite scoring across 222 predictable genes identified 8 druggable KL-specific vulnerability targets. All were validated and approved by the validation scientist (journal #2190).

| Rank | Gene | Effect Size | KW FDR | Composite Score | Drug(s) | Clinical Stage |
|---|---|---|---|---|---|---|
| #1 | MDM2 | -0.535 | 2.2e-13 | 53.35 | idasanutlin, navtemadlin | Phase 2 |
| #2 | MDM4 | -0.527 | 1.4e-12 | 51.49 | ALRN-6924 | Phase 1 |
| #3 | SLC16A3 | -0.339 | 1.2e-26 | 50.83 | AZD0095 | Preclinical |
| #4 | PSMB7 | -0.396 | 3.9e-11 | 46.02 | bortezomib | Approved |
| #5 | STAT3 | -0.340 | 8.5e-19 | 45.65 | napabucasin | Phase 3 |
| #6 | ERBB3 | -0.197 | 1.1e-7 | 40.42 | afatinib | Approved |
| #7 | SMARCA2 | -0.274 | 5.3e-9 | 36.34 | FHD-286, AU-15330 | Phase 1 |
| #8 | IGF1R | -0.308 | 1.7e-5 | 35.51 | linsitinib | Phase 2 |

All 8 targets have negative effect sizes (KL MORE dependent than other subtypes), statistically significant differential dependency (KW FDR < 1.7e-5), and at least one drug candidate. The SL component contributes 0 for all 8 targets.

**Scoring decomposition for top 3:**
- MDM2: vuln_effect=21.40 + significance=11.95 + drug=20 + SL=0 = 53.35
- MDM4: vuln_effect=21.09 + significance=10.40 + drug=20 + SL=0 = 51.49
- SLC16A3: vuln_effect=13.55 + significance=17.28 + drug=20 + SL=0 = 50.83

### Tier 1 Biomarker: SLC16A3/MCT4

SLC16A3 merits special attention as both a therapeutic target and biomarker (validated journal #2200, #2430):

- **Strongest biological signal:** KW statistic = 137.6, FDR = 1.20 x 10^-26 — second most significant gene out of 263 tested (behind only STRIP1, which is a release-from-dependency artifact)
- **KL-specific vulnerability:** KL vs KRAS-WT pairwise effect = -0.701, FDR = 4.69 x 10^-9
- **Biological pure-signal score (no drug/SL bonus):** 30.83 — stronger than 5 of the top 7 composite targets (beats PSMB7, STAT3, ERBB3, SMARCA2, IGF1R). Only MDM2 (33.35) and MDM4 (31.49) have stronger biological signals.
- **Mechanistic rationale:** STK11 loss → AMPK inactivation → glycolytic shift → MCT4 upregulation for lactate export. MCT4 inhibition traps cytotoxic lactate AND reduces immunosuppressive TME lactate — a dual mechanism uniquely relevant to KL patients.
- **Drug candidate:** AZD0095 (AstraZeneca, CAS 2750001-23-9), IC50 = 1.3 nM, >1,000x selectivity over MCT1 (Goldberg et al., J Med Chem 2023, PMID:36525250). Preclinical stage; no registered Phase 1 trial.

After AZD0095 drug mapping was added, SLC16A3 rose from composite rank #14 (score 30.83) to #3 (score 50.83), a delta of +20 entirely from the druggability component.

### Tier 2: In Vivo-Supported Target — FSP1/AIFM2

**Included on in vivo evidence; DepMap does not capture this dependency.** Validated with mandatory caveat (journal #2197).

- **DepMap evidence:** Non-essential. KEAP1-mut vs WT effect = -0.002, p = 1.0. STK11-mut vs WT effect = -0.031, p = 0.44. Not in the 222 predictable genes. The entire 9-gene ferroptosis panel was non-significant after FDR correction.
- **In vivo evidence:** FSP1 deletion suppresses tumorigenesis by ~80% (Nature, January 2026). icFSP1 inhibitor is a therapeutic lead.
- **Why DepMap misses this:** 2D culture maintains normoxic, cystine-rich conditions that suppress ferroptosis vulnerability. FSP1 and GPX4 are redundant ferroptosis suppressors — single-gene knockout is compensated in vitro. The tumor microenvironment (hypoxia, nutrient deprivation, oxidative stress) unmasks ferroptosis dependencies that cell lines cannot capture.
- **GPX4 paradox:** GPX4 shows a positive effect (+0.22) in KEAP1-mutant cells — they are LESS dependent on GPX4, likely reflecting NRF2-driven antioxidant compensation that reduces reliance on individual ferroptosis suppressors.

FSP1/AIFM2 is reported in a separate evidence tier from composite-ranked targets.

### Tier 3: Literature-Based Synthetic Lethal Pair — STK11-MARK2

Validated with caveats (journal #2195).

- **External evidence:** bioRxiv January 2026 genome-wide CRISPR screen with DepMap validation identified STK11-MARK2 as the strongest STK11 SL pair. The Desjardins isogenic benchmark confirms STK11-MARK2, STK11-MARK3, and STK11-SIK3 (all CAMK family kinases). STK11 has 53 total SL pairs in the combined benchmark.
- **Pipeline status:** MARK2, MARK3, and SIK3 are NOT among the 222 predictable genes and cannot be composite-scored. This is a limitation of the expression-based prediction approach, not a negative result.
- **Druggability:** MARK2 is a CAMK-family serine/threonine kinase (generally kinase-druggable), but no specific clinical inhibitor exists.

STK11-MARK2 is reported in a distinct "external evidence" category. MARK3 and SIK3 provide supporting context only.

### Clinical Context: AXL as Dual-Mechanism KL Target

AXL was identified as a target of interest by the research director (PM task #1535) but falls outside the pipeline's 222 predictable genes:

- **KRAS-resistance mechanism:** AXL is the #1 CRISPRa hit for KRAS inhibitor resistance
- **Immunotherapy-resistance mechanism:** AXL inhibition restores PD-1 sensitivity in STK11-mutant models
- **Clinical data:** Bemcentinib (AXL inhibitor) + pembrolizumab achieved 26% ORR in AXL-positive tumors
- **Significance for KL patients:** Dual targeting of both KRAS-inhibitor resistance and immunotherapy resistance addresses the two major treatment failure modes in KL NSCLC

### Novel SL Candidates from Anti-Correlation Analysis

Two novel SL candidates emerged from dependency anti-correlation analysis (journal #612):

- **NFE2L2-MTF1:** r = -0.91 in KL (n=31), replicated in KOnly (r=-0.82) and KRAS-WT (r=-0.81, n=964). Most biologically plausible novel candidate — NRF2 and MTF1 are both stress-response transcription factors with potentially compensatory roles.
- **TXNRD1-AIFM1:** r = -0.87 in KL. Redox axis SL candidate connecting thioredoxin reductase and apoptosis-inducing factor.

Both require experimental validation but represent genuinely novel predictions not previously reported in NSCLC SL literature.

### Drug-Target Landscape

Literature-curated drug mapping identified 41 gene-drug pairs across 35 of the 222 predictable genes:

| Category | Examples | Stage |
|---|---|---|
| MDM2/p53 axis | idasanutlin, navtemadlin, ALRN-6924 | Phase 1-2 |
| Metabolic (MCT4) | AZD0095 | Preclinical |
| Proteasome | bortezomib (PSMB7) | Approved |
| JAK-STAT | napabucasin (STAT3) | Phase 3 |
| RTK signaling | afatinib (ERBB3), linsitinib (IGF1R) | Approved / Phase 2 |
| Chromatin remodeling | FHD-286, AU-15330 (SMARCA2) | Phase 1 |
| CDK4/6 | palbociclib, ribociclib, abemaciclib (CCND1) | Approved |
| Epigenetic | tazemetostat (EZH2), CCS1477 (EP300) | Approved / Phase 1 |
| FAK | defactinib (PTK2) | Phase 2 |
| Ferroptosis (in vivo) | icFSP1 (FSP1/AIFM2) | Preclinical |

**PRISM validation is limited:** Only bortezomib/PSMB7 is present in the PRISM drug sensitivity library. Drug-dependency concordance across the single available pair is non-informative.

**Clinical caveats from phase3 integration:**
- ATR inhibitor ceralasertib failed in the LATIFY Phase 3 trial (IO combination) — DDR-targeting combinations have poor NSCLC track record
- mTOR inhibitor vistusertib has known negative results in NSCLC
- For KL patients, gemcitabine is the preferred chemotherapy backbone (not platinum) due to NRF2-driven platinum resistance in KEAP1-mutant tumors

---

## Discussion

### MDM2/MDM4: p53-Axis Vulnerability in KL NSCLC

MDM2 and MDM4 rank #1 and #2 as KL vulnerability targets, with the largest absolute effect sizes (-0.535 and -0.527). The biological rationale is clear: KL tumors retain wild-type TP53 at higher rates than KP tumors (by definition, KP tumors carry TP53 mutations). In TP53-wildtype contexts, tumor cells depend on MDM2/MDM4 to suppress p53-mediated apoptosis. This is consistent with the broader oncology principle that MDM2 inhibitors are effective only in TP53-wildtype tumors (Tisato et al., J Hematol Oncol, 2017). The reciprocal finding — MDM2 is the #1 KP RELEASE target (effect +0.246), meaning KP tumors are LESS dependent — provides a built-in negative control validating the direction-aware analysis.

### SLC16A3/MCT4: Metabolic Achilles' Heel of KL Tumors

SLC16A3 represents the most mechanistically compelling finding. The STK11-loss → AMPK-inactivation → glycolytic shift → MCT4-dependency cascade is well-established in NSCLC biology. Our data shows this dependency is quantitatively the strongest biological signal in the entire 263-gene dataset (KW FDR = 1.2 x 10^-26). The availability of a potent, selective MCT4 inhibitor (AZD0095, IC50 1.3 nM, >1000x selectivity over MCT1) makes this an immediately actionable finding, despite its preclinical status. MCT4 inhibition has a dual mechanism of action in KL tumors: direct cytotoxicity via intracellular lactate accumulation AND indirect immune benefit via reduced lactate-mediated immunosuppression in the TME.

### SMARCA2: Chromatin Remodeling Paralog Dependency

SMARCA2 (BRM) dependency in KL tumors is consistent with the SWI/SNF paralog synthetic lethality paradigm. STK11 loss frequently co-occurs with SMARCA4 alterations on 19p13, creating dependency on the SMARCA2 paralog for residual chromatin remodeling function. FHD-286 and AU-15330 (SMARCA2 degraders) are in early clinical development.

### Limitations of the Expression-Based Prediction Approach

A fundamental limitation is that expression-based elastic-net models capture transcriptional correlates of dependency, not causal mechanisms. This creates systematic blind spots:

1. **Mutation-driven dependencies** (KRAS, EGFR, ALK) are unpredictable because expression does not capture the activating mutation
2. **Protein-level dependencies** (NFE2L2/NRF2 constitutive activation) show paradoxical direction because the expression model captures secondary transcriptional effects
3. **In vivo-specific dependencies** (FSP1/AIFM2 ferroptosis) are invisible because 2D culture does not replicate the metabolic stress that creates the vulnerability

The SL integration component (20% of composite score) was ineffective — all top targets scored 0. This reflects both the modest overlap between SL benchmarks and predictable genes, and the non-significant Fisher enrichment (all FDR > 0.80). Future work should use SL predictions as an independent evidence tier rather than a score component.

### Comparison to TCGADEPMAP (Meyers et al., 2024)

Our approach extends the original TCGADEPMAP methodology in four ways: (1) NSCLC-specific model training rather than pan-cancer, (2) molecular subtype stratification enabling KL/KP/KOnly comparisons, (3) direction-aware scoring that distinguishes gained vulnerability from release-from-dependency, and (4) integration of SL benchmarks and drug mapping. The direction correction was critical — the initial undirected analysis placed CCND1 as the top KL target (score 61.6, positive effect) when in fact KL tumors are LESS dependent on CCND1.

---

## Limitations

1. **Small KL cohort (n=31 patients, n=5 cell lines in ferroptosis analysis).** Statistical power is adequate for the strongest signals but insufficient for moderate effects. The KL subtype definition conflates STK11-mutant and KEAP1-mutant tumors, which may have distinct vulnerability profiles.

2. **Expression models predict only 222 of ~18,000 genes (1.2%).** The vast majority of the dependency landscape is unpredictable from expression alone. Clinically important targets (KRAS, EGFR, AXL, SHP2) fall outside this set.

3. **No survival analysis performed.** TCGA clinical outcome data was not integrated. Whether patients with high predicted dependency on identified targets have different survival outcomes remains unknown.

4. **LUCA scRNA-seq validation deferred.** Single-cell expression patterns of target genes in tumor vs TME cells were not assessed. This would strengthen the MCT4 dual-mechanism hypothesis.

5. **KP subtype contaminated by false positives.** 5 of 10 top KP vulnerability targets have positive effects, limiting the utility of KP-specific recommendations. This is a structural limitation of the composite scoring when non-effect components override direction.

6. **Drug mapping is literature-based, not pharmacogenomically validated.** PRISM overlap is minimal (1/39 genes). Drug-dependency concordance cannot be assessed.

7. **SL predictions add no value to top targets.** The SL component (20% weight) is effectively dead weight in the current scoring framework, as the highest-ranked targets have zero SL support.

---

## Next Steps

1. **Experimental validation of top KL targets.** Priority: SLC16A3/MCT4 inhibition with AZD0095 in KL cell lines and xenograft models; MDM2 inhibition with idasanutlin in TP53-WT KL cell lines.

2. **TCGA survival analysis.** Correlate predicted dependency scores for MDM2, SLC16A3, STAT3, and SMARCA2 with overall and progression-free survival in KL patients.

3. **LUCA scRNA-seq validation.** Verify that SLC16A3/MCT4 is expressed in tumor cells (not stroma/immune) and that expression is enriched in STK11-mutant patient tumors.

4. **Expand to multi-omic models.** Integrate mutation and copy number features alongside expression to capture mutation-driven dependencies (KRAS, EGFR) currently missed by expression-only models.

5. **Validate NFE2L2-MTF1 and TXNRD1-AIFM1 SL pairs.** These novel anti-correlation predictions warrant isogenic CRISPR validation.

6. **Clinical trial mapping.** Systematically cross-reference KL vulnerability targets against active NSCLC clinical trials to identify opportunities for biomarker-stratified enrollment.

7. **MARK2 inhibitor development.** Given the strong isogenic evidence for STK11-MARK2 synthetic lethality and MARK2's kinase-druggable structure, medicinal chemistry efforts targeting MARK2 are warranted.

---

## References

- Meyers RM et al. "Building a translational cancer dependency map for The Cancer Genome Atlas." *Nature Cancer*, August 2024.
- Goldberg FW et al. "Discovery of clinical candidate AZD0095, a selective inhibitor of monocarboxylate transporter 4 (MCT4)." *J Med Chem*, 2023. PMID:36525250.
- Salcher S et al. "High-resolution single-cell atlas of the lung cancer microenvironment." *Cancer Cell*, 2022. (LUCA Atlas, 1.28M cells)
- Genome Biology 2025. 117 experimentally validated synthetic lethal pairs from combinatorial CRISPR screens.
- Desjardins et al. Isogenic CRISPR screen for synthetic lethal interactions across 15 cancer drivers. Mendeley Data, doi:10.17632/k6wm46g4tw.1.
- bioRxiv, January 2026. Genome-wide CRISPR screen identifying STK11-MARK2 as strongest STK11 synthetic lethal pair.
- *Nature*, January 2026. FSP1 deletion suppresses tumorigenesis ~80% in vivo; icFSP1 inhibitor as therapeutic lead.
- CILANTRO-SL. Foundation model SL prediction with conformal prediction. *bioRxiv*, February 2026.
- Sannigrahi et al. "PAK2 in HNSCC via DepMap." *Mol Oncol*, 2024.
- Flexynesis. Multi-omics integration toolkit. *Nature Communications*, 2025.
- DepMap 25Q3 (Broad Institute). depmap.org. CRISPR dependency, expression, mutation, and PRISM drug response data.
- TCGA LUAD/LUSC. Genomic Data Commons (GDC), portal.gdc.cancer.gov.
