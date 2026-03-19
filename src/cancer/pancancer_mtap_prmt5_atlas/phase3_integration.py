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

# Housekeeping gene artifact — exclude from all results (RD accepted, journal #471)
EXCLUDE_GENES = {"ACTB"}


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
    """Get signed mean effect size for a gene in a given subtype.

    Positive = subtype LESS dependent (higher DepMap scores).
    Negative = subtype MORE dependent (lower DepMap scores, more essential).

    Uses ALL pairwise comparisons involving the subtype (excluding allele_kw).
    When subtype is group_a, effect_size is already from its perspective.
    When subtype is group_b, effect_size is negated to flip perspective.
    """
    gene_pw = pairwise[
        (pairwise["gene"] == gene) & (pairwise["direction"] != "allele_kw")
    ]
    if gene_pw.empty:
        return 0.0

    effects = []
    # When subtype is group_a: use effect_size directly
    as_a = gene_pw[gene_pw["group_a"] == subtype]
    effects.extend(as_a["effect_size"].tolist())

    # When subtype is group_b: negate to get subtype's perspective
    as_b = gene_pw[gene_pw["group_b"] == subtype]
    effects.extend((-as_b["effect_size"]).tolist())

    return float(np.mean(effects)) if effects else 0.0


def composite_scoring(kw, pairwise, sig_genes, predictable, sl_hits, gene_drugs):
    """Build composite target scores with separate vulnerability and release rankings.

    Vulnerability ranking: genes where subtype is MORE dependent (negative effect).
    Release ranking: genes where subtype is LESS dependent (positive effect).
    """
    # Exclude artifact genes
    sig_genes = [g for g in sig_genes if g not in EXCLUDE_GENES]

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

            # Shared components
            score_sl = min(n_sl, 5) / 5 * 20  # cap at 5 SL hits
            score_drug = 20 if is_druggable else 0
            score_sig = min(-np.log10(max(kw_fdr, 1e-30)), 30) / 30 * 20

            # Vulnerability: negative effect (MORE dependent) contributes positively
            vuln_effect = max(-effect, 0.0)
            score_vuln_effect = min(vuln_effect, 1.0) * 40
            vulnerability_score = score_vuln_effect + score_sl + score_drug + score_sig

            # Release: positive effect (LESS dependent) contributes positively
            release_effect = max(effect, 0.0)
            score_release_effect = min(release_effect, 1.0) * 40
            release_score = score_release_effect + score_sl + score_drug + score_sig

            rows.append({
                "gene": gene,
                "subtype": subtype,
                "effect_size": round(effect, 6),
                "kw_stat": kw_stat,
                "kw_fdr": kw_fdr,
                "n_sl_hits": n_sl,
                "is_druggable": is_druggable,
                "drugs": "; ".join(drug_info.get(gene, [])),
                "score_vuln_effect": round(score_vuln_effect, 2),
                "score_release_effect": round(score_release_effect, 2),
                "score_sl": round(score_sl, 2),
                "score_drug": round(score_drug, 2),
                "score_significance": round(score_sig, 2),
                "vulnerability_score": round(vulnerability_score, 2),
                "release_score": round(release_score, 2),
            })

    return pd.DataFrame(rows).sort_values(
        ["subtype", "vulnerability_score"], ascending=[True, False]
    )


def top_targets_per_subtype(scores_df, n=20, filter_direction=True):
    """Extract top N vulnerability and release targets per subtype.

    When filter_direction=True (default), vulnerability rankings only include
    genes with negative effect_size (genuinely MORE dependent) and release
    rankings only include genes with positive effect_size (genuinely LESS
    dependent). This prevents high druggability/significance/SL scores from
    pushing wrong-direction genes into the top rankings.
    """
    tops = []
    for ranking_type, score_col in [
        ("vulnerability", "vulnerability_score"),
        ("release", "release_score"),
    ]:
        for subtype in SUBTYPES:
            sub = scores_df[scores_df["subtype"] == subtype]
            if filter_direction:
                if ranking_type == "vulnerability":
                    sub = sub[sub["effect_size"] < 0]
                else:
                    sub = sub[sub["effect_size"] > 0]
            sub = (
                sub.sort_values(score_col, ascending=False)
                .head(n)
                .copy()
            )
            sub["rank"] = range(1, len(sub) + 1)
            sub["ranking_type"] = ranking_type
            tops.append(sub)
    return pd.concat(tops, ignore_index=True)


def build_summary(scores_df, sl_hits, sl_enrich, anticorr, gene_drugs, ferro_enrich):
    """Build machine-readable Phase 3 summary."""
    summary = {
        "phase": 3,
        "steps_completed": ["SL_enrichment", "PRISM_mapping", "ferroptosis_panel", "integration"],
        "scoring_methodology": {
            "description": (
                "Direction-aware composite scoring. Vulnerability rankings use negative "
                "effect sizes (subtype MORE dependent). Release rankings use positive "
                "effect sizes (subtype LESS dependent). Effect sizes are signed means "
                "across all pairwise subtype comparisons."
            ),
            "weights": "effect_size 40% + SL_support 20% + druggability 20% + KW_significance 20%",
            "excluded_genes": list(EXCLUDE_GENES),
        },
        "methodology_limitation": (
            "Control genes (KRAS, EGFR, ALK, ROS1, MET) are not among the 263 predictable "
            "genes. This is an expected limitation: oncogene addiction is mutation-driven, "
            "not expression-predictable. Expression-based dependency prediction captures "
            "non-oncogene dependencies and synthetic lethal vulnerabilities, which are the "
            "primary therapeutic targets for subtype-stratified treatment."
        ),
        "mdm2_mdm4_note": (
            "MDM2/MDM4 subtype assignment differs between Phase 2 pairwise (KL vs KRAS-WT) "
            "and composite per-subtype metric. Phase 2 pairwise shows MDM2 as KL vulnerability "
            "(effect -0.61 KL vs KRAS-WT). Composite metric averages across ALL pairwise "
            "comparisons including KP, where MDM2 shows strong effects due to TP53-mutant "
            "biology. The per-subtype metric captures different information than the pairwise "
            "KL-vs-WT comparison. Both are valid but answer different questions."
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
            "hits_per_subtype": (
                {s: int(n) for s, n in sl_hits.groupby("subtype").size().items()}
                if not sl_hits.empty
                else {}
            ),
            "fisher_enrichment": sl_enrich[["subtype", "fisher_p", "fisher_fdr"]].to_dict("records"),
            "anticorrelated_pairs": int(len(anticorr)),
        },
        "prism_mapping": {
            "total_gene_drug_mappings": int(len(gene_drugs)),
            "genes_with_drugs": (
                int(gene_drugs[gene_drugs["drug"] != ""]["gene"].nunique())
                if not gene_drugs.empty
                else 0
            ),
            "in_prism": int(gene_drugs["in_prism"].sum()) if not gene_drugs.empty else 0,
        },
        "ferroptosis_panel": {
            "genes_tested": 9,
            "significant_keap1_enrichment": int(
                len(ferro_enrich[ferro_enrich["fdr"] < 0.05])
                if "fdr" in ferro_enrich.columns
                else 0
            ),
            "fsp1_aifm2_flag": (
                "Priority target: Nature Jan 2026 showed FSP1 deletion suppresses "
                "tumorigenesis ~80% in vivo. icFSP1 inhibitor available. DepMap may "
                "underestimate importance (not essential in vitro)."
            ),
        },
        "top_vulnerability_targets": {},
        "top_release_targets": {},
    }

    for subtype in SUBTYPES:
        sub = scores_df[scores_df["subtype"] == subtype]
        vuln = (
            sub[sub["effect_size"] < 0]
            .sort_values("vulnerability_score", ascending=False)
            .head(10)
        )
        summary["top_vulnerability_targets"][subtype] = vuln["gene"].tolist()

        release = (
            sub[sub["effect_size"] > 0]
            .sort_values("release_score", ascending=False)
            .head(10)
        )
        summary["top_release_targets"][subtype] = release["gene"].tolist()

    return summary


def main():
    print("Loading Phase 2 data...")
    kw, pairwise, sig_genes, predictable = load_phase2_data()

    print("Loading Phase 3 data...")
    sl_hits, sl_enrich, anticorr, gene_drugs, ferro_deps, ferro_enrich = load_phase3_data()

    n_excluded = len([g for g in sig_genes if g in EXCLUDE_GENES])
    print(f"\nPhase 2: {len(sig_genes)} significant genes ({n_excluded} excluded), "
          f"{len(predictable)} predictable")
    print(f"Phase 3 SL: {len(sl_hits)} hits, {len(anticorr)} anti-correlated pairs")
    print(f"Phase 3 PRISM: {len(gene_drugs)} gene-drug mappings")
    print(f"Phase 3 Ferroptosis: {len(ferro_deps)} measurements")

    print("\nComputing direction-aware composite scores...")
    scores_df = composite_scoring(kw, pairwise, sig_genes, predictable, sl_hits, gene_drugs)
    print(f"  Total scored: {len(scores_df)} (gene x subtype)")

    top_df = top_targets_per_subtype(scores_df, n=20, filter_direction=True)
    top_unfiltered_df = top_targets_per_subtype(scores_df, n=20, filter_direction=False)

    for ranking_type in ["vulnerability", "release"]:
        score_col = f"{ranking_type}_score"
        print(f"\nTop {ranking_type} targets per subtype (direction-filtered):")
        for subtype in SUBTYPES:
            sub = top_df[
                (top_df["subtype"] == subtype) & (top_df["ranking_type"] == ranking_type)
            ]
            print(f"\n  {subtype} (top 10):")
            for _, r in sub.head(10).iterrows():
                drug_str = f" [{r['drugs']}]" if r['drugs'] else ""
                sl_str = f" SL={r['n_sl_hits']}" if r['n_sl_hits'] > 0 else ""
                print(f"    {r['rank']:2d}. {r['gene']:12s} score={r[score_col]:5.1f} "
                      f"effect={r['effect_size']:.3f}{sl_str}{drug_str}")

    print("\nBuilding summary...")
    summary = build_summary(scores_df, sl_hits, sl_enrich, anticorr, gene_drugs, ferro_enrich)

    # Save outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scores_df.to_csv(OUT_DIR / "composite_target_scores.csv", index=False)
    top_df.to_csv(OUT_DIR / "top_targets_per_subtype.csv", index=False)
    top_unfiltered_df.to_csv(OUT_DIR / "top_targets_per_subtype_unfiltered.csv", index=False)
    with open(OUT_DIR / "phase3_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nOutputs saved to {OUT_DIR}/")
    print("Done.")


if __name__ == "__main__":
    main()
