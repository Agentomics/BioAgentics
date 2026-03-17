"""Phase 3 Step 5: Integration and top target reporting.

Combine Phase 3 results (SL enrichment, PRISM mapping, ferroptosis panel)
into composite target scores and identify top actionable targets per subtype.
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path


DATA_DIR = Path("data/results")
OUT_DIR = DATA_DIR / "phase3_integration"

SUBTYPES = ["KL", "KP", "KOnly"]


def load_phase2_data():
    """Load Phase 2 subtype dependency results."""
    kw = pd.read_csv(DATA_DIR / "subtype_dependencies" / "kruskal_wallis_results.csv")
    pairwise = pd.read_csv(DATA_DIR / "subtype_dependencies" / "pairwise_comparisons.csv")
    sig_genes = pd.read_csv(
        DATA_DIR / "subtype_dependencies" / "significant_genes.txt", header=None
    )[0].tolist()
    predictable = pd.read_csv(DATA_DIR / "predictable_genes.txt", header=None)[0].tolist()
    return kw, pairwise, sig_genes, predictable


def load_phase3_data():
    """Load Phase 3 step outputs."""
    sl_hits = pd.read_csv(DATA_DIR / "sl_enrichment" / "subtype_sl_hits.csv")
    sl_enrich = pd.read_csv(DATA_DIR / "sl_enrichment" / "enrichment_stats.csv")
    anticorr = pd.read_csv(DATA_DIR / "sl_enrichment" / "anticorrelated_pairs.csv")
    gene_drugs = pd.read_csv(DATA_DIR / "prism_mapping" / "gene_drug_matches.csv")
    ferro_deps = pd.read_csv(DATA_DIR / "ferroptosis_panel" / "ferroptosis_gene_dependencies.csv")
    ferro_enrich = pd.read_csv(DATA_DIR / "ferroptosis_panel" / "keap1_enrichment_stats.csv")
    return sl_hits, sl_enrich, anticorr, gene_drugs, ferro_deps, ferro_enrich


def compute_subtype_effect(pairwise, gene, subtype):
    """Get mean effect size for a gene in a given subtype (how much more dependent)."""
    gene_pw = pairwise[pairwise["gene"] == gene]
    if gene_pw.empty:
        return 0.0

    effects = []
    # When subtype is group_a and has higher dependency
    wins_a = gene_pw[
        (gene_pw["group_a"] == subtype) & (gene_pw["direction"] == "a_higher")
    ]
    effects.extend(wins_a["effect_size"].tolist())

    # When subtype is group_b and has higher dependency
    wins_b = gene_pw[
        (gene_pw["group_b"] == subtype) & (gene_pw["direction"] == "b_higher")
    ]
    effects.extend(wins_b["effect_size"].tolist())

    return np.mean(effects) if effects else 0.0


def composite_scoring(kw, pairwise, sig_genes, predictable, sl_hits, gene_drugs):
    """Build composite target scores."""
    # SL support count per gene per subtype
    sl_count = {}
    if not sl_hits.empty:
        for _, row in sl_hits.iterrows():
            key = (row["gene"], row["subtype"])
            sl_count[key] = sl_count.get(key, 0) + 1

    # Druggability: genes with known drugs
    druggable = set()
    drug_info = {}
    if not gene_drugs.empty:
        has_drug = gene_drugs[gene_drugs["drug"].notna() & (gene_drugs["drug"] != "")]
        for _, row in has_drug.iterrows():
            druggable.add(row["gene"])
            if row["gene"] not in drug_info:
                drug_info[row["gene"]] = []
            drug_name = str(row["drug"])
            if drug_name and drug_name != "nan":
                drug_info[row["gene"]].append(drug_name)

    rows = []
    for gene in sig_genes:
        kw_row = kw[kw["gene"] == gene]
        kw_stat = kw_row["kw_stat"].iloc[0] if not kw_row.empty else 0
        kw_fdr = kw_row["kw_fdr"].iloc[0] if not kw_row.empty else 1

        for subtype in SUBTYPES:
            effect = compute_subtype_effect(pairwise, gene, subtype)
            n_sl = sl_count.get((gene, subtype), 0)
            is_druggable = gene in druggable

            # Composite score: weighted combination
            # Effect size (0-1 range) * 40 + SL support * 20 + druggability * 20 + KW significance * 20
            score_effect = min(abs(effect), 1.0) * 40
            score_sl = min(n_sl, 5) / 5 * 20  # cap at 5 SL hits
            score_drug = 20 if is_druggable else 0
            score_sig = min(-np.log10(max(kw_fdr, 1e-30)), 30) / 30 * 20

            composite = score_effect + score_sl + score_drug + score_sig

            rows.append({
                "gene": gene,
                "subtype": subtype,
                "effect_size": effect,
                "kw_stat": kw_stat,
                "kw_fdr": kw_fdr,
                "n_sl_hits": n_sl,
                "is_druggable": is_druggable,
                "drugs": "; ".join(drug_info.get(gene, [])),
                "score_effect": round(score_effect, 2),
                "score_sl": round(score_sl, 2),
                "score_drug": round(score_drug, 2),
                "score_significance": round(score_sig, 2),
                "composite_score": round(composite, 2),
            })

    return pd.DataFrame(rows).sort_values(["subtype", "composite_score"], ascending=[True, False])


def top_targets_per_subtype(scores_df, n=20):
    """Extract top N targets per subtype."""
    tops = []
    for subtype in SUBTYPES:
        sub = scores_df[scores_df["subtype"] == subtype].head(n).copy()
        sub["rank"] = range(1, len(sub) + 1)
        tops.append(sub)
    return pd.concat(tops, ignore_index=True)


def build_summary(scores_df, sl_hits, sl_enrich, anticorr, gene_drugs, ferro_enrich):
    """Build machine-readable Phase 3 summary."""
    summary = {
        "phase": 3,
        "steps_completed": ["SL_enrichment", "PRISM_mapping", "ferroptosis_panel", "integration"],
        "methodology_limitation": (
            "Control genes (KRAS, EGFR, ALK, ROS1, MET) are not among the 263 predictable "
            "genes. This is an expected limitation: oncogene addiction is mutation-driven, "
            "not expression-predictable. Expression-based dependency prediction captures "
            "non-oncogene dependencies and synthetic lethal vulnerabilities, which are the "
            "primary therapeutic targets for subtype-stratified treatment."
        ),
        "clinical_caveats": {
            "ATR_ceralasertib": (
                "LATIFY Phase 3 failure: DDR targets may not translate clinically in IO "
                "combinations. Ceralasertib + durvalumab did not improve OS in NSCLC."
            ),
            "mTOR_vistusertib": "Known negative: vistusertib failed in NSCLC trials.",
            "KL_chemo_backbone": (
                "Gemcitabine (not platinum) is preferred chemo backbone for KL patients "
                "due to NRF2-driven platinum resistance in KEAP1-mutant tumors."
            ),
        },
        "sl_enrichment": {
            "total_hits": int(len(sl_hits)),
            "hits_per_subtype": {
                s: int(n) for s, n in sl_hits.groupby("subtype").size().items()
            } if not sl_hits.empty else {},
            "fisher_enrichment": sl_enrich[["subtype", "fisher_p", "fisher_fdr"]].to_dict("records"),
            "anticorrelated_pairs": int(len(anticorr)),
        },
        "prism_mapping": {
            "total_gene_drug_mappings": int(len(gene_drugs)),
            "genes_with_drugs": int(gene_drugs[gene_drugs["drug"] != ""]["gene"].nunique()) if not gene_drugs.empty else 0,
            "in_prism": int(gene_drugs["in_prism"].sum()) if not gene_drugs.empty else 0,
        },
        "ferroptosis_panel": {
            "genes_tested": 9,
            "significant_keap1_enrichment": int(
                len(ferro_enrich[ferro_enrich["fdr"] < 0.05]) if "fdr" in ferro_enrich.columns else 0
            ),
            "fsp1_aifm2_flag": (
                "Priority target: Nature Jan 2026 showed FSP1 deletion suppresses "
                "tumorigenesis ~80% in vivo. icFSP1 inhibitor available. DepMap may "
                "underestimate importance (not essential in vitro)."
            ),
        },
        "top_targets_per_subtype": {},
    }

    for subtype in SUBTYPES:
        top = scores_df[scores_df["subtype"] == subtype].head(10)
        summary["top_targets_per_subtype"][subtype] = top["gene"].tolist()

    return summary


def main():
    print("Loading Phase 2 data...")
    kw, pairwise, sig_genes, predictable = load_phase2_data()

    print("Loading Phase 3 data...")
    sl_hits, sl_enrich, anticorr, gene_drugs, ferro_deps, ferro_enrich = load_phase3_data()

    print(f"\nPhase 2: {len(sig_genes)} significant genes, {len(predictable)} predictable")
    print(f"Phase 3 SL: {len(sl_hits)} hits, {len(anticorr)} anti-correlated pairs")
    print(f"Phase 3 PRISM: {len(gene_drugs)} gene-drug mappings")
    print(f"Phase 3 Ferroptosis: {len(ferro_deps)} measurements")

    print("\nComputing composite scores...")
    scores_df = composite_scoring(kw, pairwise, sig_genes, predictable, sl_hits, gene_drugs)
    print(f"  Total scored: {len(scores_df)} (gene x subtype)")

    print("\nTop targets per subtype:")
    top_df = top_targets_per_subtype(scores_df, n=20)
    for subtype in SUBTYPES:
        sub = top_df[top_df["subtype"] == subtype]
        print(f"\n  {subtype} (top 10):")
        for _, r in sub.head(10).iterrows():
            drug_str = f" [{r['drugs']}]" if r['drugs'] else ""
            sl_str = f" SL={r['n_sl_hits']}" if r['n_sl_hits'] > 0 else ""
            print(f"    {r['rank']:2d}. {r['gene']:12s} score={r['composite_score']:5.1f} "
                  f"effect={r['effect_size']:.3f}{sl_str}{drug_str}")

    print("\nBuilding summary...")
    summary = build_summary(scores_df, sl_hits, sl_enrich, anticorr, gene_drugs, ferro_enrich)

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUT_DIR / "composite_target_scores.csv", index=False)
    top_df.to_csv(OUT_DIR / "top_targets_per_subtype.csv", index=False)
    with open(OUT_DIR / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
