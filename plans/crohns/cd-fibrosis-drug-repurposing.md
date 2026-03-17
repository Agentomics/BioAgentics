# Computational Drug Repurposing for Intestinal Fibrosis in Crohn's Disease

## Objective
Identify candidate compounds that reverse intestinal fibrosis gene signatures in Crohn's disease through computational drug repurposing, addressing the unmet need for anti-fibrotic therapies in IBD.

## Background
No approved anti-fibrotic treatment exists for Crohn's disease. Stricturing disease affects 30-50% of patients within 10 years of diagnosis, often requiring surgical resection with high recurrence rates. Recent advances have mapped the fibrosis landscape at unprecedented resolution: inflammatory fibroblast spatial programs (Kong et al., Nat Genet 2025), creeping fat mechanosensitive fibroblasts driving fibrosis via YAP/TAZ (Cell 2025), the inflammation-to-fibrosis transition mediated by CD38/PECAM1 (J Crohn's Colitis 2025), and protein-level validated markers (CTHRC1, POSTN, TNC, CPA3 — JCI Insight 2026). Obefazimod (miR-124 enhancer) showed preclinical anti-fibrotic activity in human intestinal fibroblasts (ECCO 2026), and anti-TL1A antibodies (duvakitug, tulisokibart) target dual inflammation/fibrosis pathways. These converging discoveries provide disease-specific fibrosis gene signatures suitable for computational drug screening — a strategy that has not been systematically applied to intestinal fibrosis.

## Data Sources
- **CMAP / L1000 (Connectivity Map)**: ~1.3M gene expression profiles from ~5,000 compounds across cell lines. Query fibrosis signatures for compound reversal scores. Available via clue.io API.
- **DrugBank**: Drug target annotations, mechanism of action, and safety/approval data for filtering candidates.
- **Kong et al. spatial transcriptomics (SCP2959)**: Inflammatory fibroblast spatial module gene signatures as disease-specific query input.
- **GSE275144**: CTHRC1+ fibroblast scRNA-seq reference — extract fibroblast-specific fibrosis gene program (distinct from bulk tissue signatures).
- **MSigDB**: Curated fibrosis, ECM, EMT, and TGFβ gene sets for pathway-level queries.
- **GEO stricture datasets**: GSE16879 (mucosal expression with phenotype), RISK cohort (GSE57945) progressors vs non-progressors for deriving CD-specific fibrosis signatures.
- **ChEMBL**: Bioactivity data for known anti-fibrotic compounds as positive controls.

## Methodology
1. **Fibrosis signature derivation**: Construct three complementary CD fibrosis gene signatures:
   - *Bulk tissue signature*: Differentially expressed genes between stricturing (B2) and inflammatory (B1) CD from GEO datasets (limma/DESeq2). Filter for fibrosis-relevant pathways (TGFβ, ECM, Wnt, collagen).
   - *Cell-type-resolved signature*: Extract fibroblast-specific gene program from Kong et al. spatial data and CTHRC1+ fibroblast scRNA-seq (GSE275144). This avoids dilution by non-fibroblast cell types — a key limitation of prior CMAP drug repurposing studies. **Key druggable gene targets** from fibrostenotic transcriptomic analysis (IBD Journal 2025, doi:10.1093/ibd/izae295): HDAC1 (HDAC inhibitors well-characterized in L1000), GREM1 (BMP antagonist), SERPINE1/PAI-1 (druggable serine protease inhibitor), plus LY96, AKAP11, SRM, EHD2, FGF2. **TWIST1+FAP+ fibroblasts** (JCI 2024, doi:10.1172/JCI179472): FAP+ fibroblasts are the highest ECM-producing subtype in fibrotic CD intestine; TWIST1 is the master TF induced by CXCL9+ macrophages via IL-1beta/TGF-beta. TWIST1 inhibition ameliorated fibrosis in mice — prioritize as druggable target.
   - *Transition signature*: Genes marking the inflammation-to-fibrosis transition (CD38/PECAM1 axis, YAP/TAZ mechanotransduction targets). This captures the therapeutic window where intervention could prevent irreversible fibrosis.

2. **CMAP/L1000 + iLINCS connectivity scoring**: Query each signature against CMAP using the clue.io API. Additionally use **iLINCS** (integrative LINCS, http://ilincs.org) as a complementary query platform — the JCC 2025 (jjaf137) CD fibrosis drug repurposing study demonstrated this approach with TRRUST/miRWalk/DGIdb for TFs/miRNAs/drugs targeting fibrotic signatures. Cross-reference hits between CMAP and iLINCS for convergent candidates. Rank compounds by negative connectivity score (signature reversal). Prioritize hits from fibroblast-relevant cell lines (IMR-90, WI-38 lung fibroblasts as surrogates — no intestinal fibroblast CMAP data exists). Filter for compounds with negative enrichment in all three signature types (convergent anti-fibrotic signal).

3. **Network pharmacology validation**: For top 50 CMAP hits:
   - Map compound targets to fibrosis PPI network (STRING database)
   - Assess target overlap with known fibrosis pathways (TGFβ, Wnt, YAP/TAZ, TL1A/DR3)
   - Cross-reference with anti-fibrotic activity in other organs (liver, lung, kidney — shared pathways exist)
   - Score for druggability (known safety profile, oral bioavailability, existing indications)

4. **Positive control validation**: Verify that compounds with known anti-fibrotic activity in CD preclinical models score highly:
   - Obefazimod (miR-124 enhancer) — preclinical anti-fibrotic in human intestinal fibroblasts (ECCO 2026)
   - Anti-TL1A pathway (TNFSF15 antagonism) — dual anti-inflammatory/anti-fibrotic
   - Pirfenidone/nintedanib — approved for IPF, tested in IBD preclinical models
   - CD38 inhibitors — reduced fibrosis in mouse CD model (J Crohn's Colitis 2025)
   - JAK inhibitors (JCC 2025, jjaf087) — upadacitinib superior to tofacitinib for fibrosis modulation via fibroblast JAK-STAT3 pathway. Use JAK inhibitor L1000 signatures as benchmark drug class.
   If positive controls do not rank in top 10%, re-evaluate signature specificity.

5. **Candidate prioritization**: Rank final candidates by composite score:
   - CMAP reversal strength (all three signatures)
   - Network proximity to fibrosis targets
   - Safety profile (approved/Phase 3 > Phase 2 > preclinical)
   - Mechanistic novelty (novel targets > known pathway agents)
   - Oral bioavailability and gut-relevant pharmacokinetics

6. **In silico validation**: For top 5-10 candidates, predict expression effects on key fibrosis markers (CTHRC1, POSTN, SERPINE1, GREM1, TNC, CD38) using CMAP dose-response data where available.

## Expected Outputs
- Ranked list of 10-20 candidate compounds with predicted anti-fibrotic activity in CD
- Network visualization showing compound-target-pathway relationships
- Fibrosis gene signature sets (bulk, cell-type-resolved, transition) as reusable resources
- Positive control validation report confirming signature relevance
- Mechanistic classification of candidates by target pathway (TGFβ, Wnt, YAP/TAZ, epigenetic, novel)

## Success Criteria
- At least 3 of 4 positive controls (obefazimod, anti-TL1A, pirfenidone, CD38 inhibitors) rank in top 20% of CMAP hits
- At least 5 candidates with approved/Phase 3 safety data and novel mechanism for CD fibrosis
- Cell-type-resolved signature produces higher-quality hits (lower false positive rate) than bulk tissue signature
- At least one candidate targets a pathway not previously associated with intestinal fibrosis (genuine novel finding)

## Labels
drug-repurposing, novel-finding, clinical, high-priority
