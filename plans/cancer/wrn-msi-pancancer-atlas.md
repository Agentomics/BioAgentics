# WRN-MSI Pan-Cancer Dependency Atlas

## Objective

Rank all MSI-H cancer types by WRN synthetic lethal dependency in DepMap 25Q3, to predict which tumor types will respond best to WRN inhibitors (VVD-214, HRO-761, NDI-219216) and identify co-factors beyond MSI status that modulate WRN dependency. Incorporate resistance-aware deployment framework based on divergent resistance profiles between compounds.

## Background

WRN helicase is selectively essential in microsatellite instability-high (MSI-H) cancer cells, validated by multiple independent screens (Behan et al., Nature 2019; Chan et al., Nature 2019; Kategaya et al., Mol Cell 2019). The mechanism is well-characterized: WRN resolves TA-dinucleotide repeat structures that expand under mismatch repair (MMR) deficiency. Without WRN, these expanded repeats cause replication fork collapse and chromosome shattering, selectively killing MSI-H cells.

MSI-H prevalence varies dramatically by cancer type (~25% endometrial, ~20% gastric, ~15% CRC, rare in most others), but **no systematic pan-cancer ranking of WRN dependency by tumor type has been published.** The field assumes WRN-MSI SL is universal, but effect sizes likely vary by tissue context, MMR defect mechanism (MLH1 methylation vs MMR gene mutation), and co-occurring mutations.

**Clinical context (updated March 2026):**
- **Pembrolizumab** is approved pan-tumor for MSI-H/dMMR, but ~50% of MSI-H patients don't respond to IO alone.
- **Three WRN inhibitors now in clinic:**
  - **VVD-214** (Vividion/Roche/Bayer): Covalent allosteric WRN helicase inhibitor. Phase 1. JMedChem cover publication (Jan 2026, PMID 40964743) validates mechanism.
  - **HRO-761** (Novartis): Non-covalent allosteric WRN inhibitor. Phase 1.
  - **NDI-219216** (Nimbus): Non-covalent WRN inhibitor (distinct binding mode). Phase 1/2 Part A dose escalation COMPLETE (Dec 2025). Favorable safety — no DLTs, no MTD reached. >24h WRN target engagement at multiple dose levels. Initial efficacy data expected H1 2026.
- **Resistance framework (bioRxiv Jan 2026, 10.64898/2026.01.22.700152v1):** Resistance profiles DIVERGE between HRO-761 and VVD-214 — heterozygous single-allele binding-site mutations sufficient for resistance. Specific mutations impair one compound but preserve sensitivity to the other. MSI-H tumors have high mutation rates, making resistance emergence likely. Drug switching and combination with NHEJ/WIP1 targets proposed as mitigation strategies.
- **Unmet need:** Which MSI-H tumor types should WRN inhibitor trials prioritize? Which compound(s) per tumor type? What resistance-aware sequencing strategy?

**Cross-project links:**
- CRC-KRAS project: MSI-H enriched in CRC (~15%), MSI-H CRC may have distinct KRAS dependencies.
- ARID1A project: WRN flagged as potential "universal SL" in analyst review (task #276, item 3) — this project will test that claim rigorously.

## Data Sources

- **DepMap 25Q3 CRISPR:** CRISPRGeneEffect.csv — WRN, WRNIP1, BLM, RECQL dependency scores across ~1,100 cell lines
- **DepMap model annotations:** Model.csv — cancer type, subtype, MSI status annotations
- **DepMap mutations:** OmicsSomaticMutations.csv — MLH1, MSH2, MSH6, PMS2 mutation status
- **DepMap expression:** OmicsExpressionProteinCodingGenesTPMLogp1.csv — MLH1 expression (proxy for methylation)
- **PRISM 24Q2:** Drug sensitivity for DDR compounds (ATR inhibitors, WEE1 inhibitors, PARP inhibitors)
- **TCGA:** MSI-H prevalence by cancer type, survival data, MMR gene status (cBioPortal pan-cancer)

## Methodology

### Phase 1 — MSI/MMR Classifier
- Classify all DepMap cell lines by MSI status using model annotations
- Cross-validate with MMR gene status: MLH1 low expression (methylation proxy), MSH2/MSH6/PMS2 truncating mutations
- Report qualifying cancer types (≥5 MSI-H + ≥10 MSS lines with CRISPR data)
- Annotate MLH1-methylated vs MMR-mutated subtypes (biologically distinct)
- Power analysis: expected qualifying types include CRC, Endometrial, Gastric, Ovarian

### Phase 2 — WRN SL Effect Sizes by Cancer Type
- For each qualifying cancer type: compare WRN CRISPR dependency in MSI-H vs MSS
- Statistics: Cohen's d effect size, Mann-Whitney U test, bootstrap 95% CI (1000 iterations, seed=42)
- BH FDR correction across cancer types
- Control helicases: BLM, RECQL (should NOT show MSI-selective dependency)
- Leave-one-out robustness + permutation testing (10,000 permutations)
- Classify: ROBUST (FDR<0.05, CI excludes zero, permutation p<0.05), MARGINAL, NOT SIGNIFICANT
- **Key question:** Is WRN-MSI SL truly universal, or are some cancer types more vulnerable?

### Phase 3 — Genome-Wide MSI-H Dependency Screen
- Beyond WRN: screen all ~18K genes for MSI-H-specific dependencies per qualifying cancer type
- Identify novel MSI-H-specific SL partners (DNA repair, replication stress, immune signaling pathways)
- Fisher enrichment against SL benchmark datasets (Vermeulen, Desjardins)
- Pathway enrichment analysis (KEGG, Reactome) of MSI-H-specific dependency genes
- Anti-correlated dependency pairs (genes where MSI-H shows GAINED dependency)

### Phase 4 — PRISM Drug Sensitivity Validation
- Map DDR-targeting compounds in PRISM: ATR inhibitors (ceralasertib, berzosertib), WEE1 inhibitors (adavosertib), PARP inhibitors (olaparib, talazoparib)
- Test for MSI-H-selective drug sensitivity per cancer type
- CRISPR-PRISM concordance: do gene dependencies predict drug sensitivity?
- Note: WRN-specific inhibitors (VVD-130037) unlikely to be in PRISM yet

### Phase 5 — TCGA Clinical Integration & Three-Drug Landscape
- MSI-H prevalence per cancer type from TCGA pan-cancer data
- Estimate addressable patient populations (MSI-H patients per US cancer type per year)
- Survival analysis: MSI-H vs MSS by cancer type (known prognostic in CRC, less clear elsewhere)
- Priority ranking: WRN SL strength × MSI-H population size
- Cross-reference with pembrolizumab MSI-H response rates where available
- Identify underexplored MSI-H tumor types with strong WRN dependency but limited clinical attention
- **Three-drug clinical landscape annotation:** VVD-214 (covalent), HRO-761 (non-covalent allosteric), NDI-219216 (non-covalent, distinct binding mode) — map trial enrollment criteria, tumor type focus, and emerging efficacy signals per compound
- **Track NDI-219216 efficacy data** expected H1 2026 — first readout for this compound

### Phase 6 — Resistance-Aware Deployment Framework
- Annotate known resistance mutations per compound from bioRxiv preprint (HRO-761 vs VVD-214 divergent profiles)
- Map predicted resistance mutation sites onto WRN protein structure — classify by compound-specific vs pan-resistance
- Model resistance emergence probability: MSI-H tumors have elevated mutation rates (10-100x MSS), making single-allele binding-site resistance likely under selective pressure
- Propose compound sequencing strategy: first-line drug selection → predicted resistance mutation → switching to compound with non-overlapping binding site
- Identify combination vulnerabilities: NHEJ pathway factors, WIP1 phosphatase as synthetic vulnerabilities in WRN-resistant cells
- Per-tumor-type recommendation: optimal first-line compound and switching strategy based on tumor mutation rate and co-dependency profiles from Phases 2-3

## Expected Outputs

1. Pan-cancer ranking of WRN dependency effect size by MSI-H tumor type
2. Classification of cancer types as ROBUST / MARGINAL / NOT SIGNIFICANT for WRN-MSI SL
3. Novel MSI-H-specific gene dependencies beyond WRN (genome-wide screen)
4. DDR drug sensitivity validation by MSI status and cancer type
5. Priority matrix combining SL strength with addressable patient population
6. Population estimates for WRN inhibitor-eligible MSI-H patients per cancer type
7. Identification of co-factors (MLH1 methylation vs MMR mutation, co-occurring drivers) that modulate WRN dependency
8. Three-drug clinical landscape annotation per tumor type (VVD-214 vs HRO-761 vs NDI-219216 trial alignment)
9. Resistance-aware compound deployment model: first-line selection, switching strategy, and combination partner rationale per tumor type

## Success Criteria

- **WRN positive control:** Significant SL effect (FDR<0.05) in ≥3 MSI-H cancer types
- **Specificity control:** BLM/RECQL do NOT show MSI-selective dependency
- **Novel findings:** ≥1 novel MSI-H-specific dependency beyond WRN/WRNIP1 validated by SL benchmarks
- **Clinical concordance:** DepMap WRN rankings directionally align with known MSI-H tumor biology
- **Actionable prediction:** Identify ≥1 underexplored MSI-H tumor type for WRN inhibitor trial prioritization
- **Resistance framework:** Compound-specific resistance mutation map with switching recommendations for ≥2 compounds
- **Three-drug coverage:** All three clinical WRN inhibitors annotated with trial status, mechanism, and tumor type alignment

## Labels

genomic, drug-screening, drug-candidate, novel-finding, clinical
