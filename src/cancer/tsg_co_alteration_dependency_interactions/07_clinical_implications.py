"""Phase 5: Clinical implication framework for non-additive interactions.

Maps double-mutant dependencies to combination therapy opportunities, estimates
co-alteration patient populations per cancer type from TCGA data, and flags
potential contraindications from TP53 modulation and antagonistic interactions.

Integrates results from all prior phases:
  - Phase 1a/b: co-alteration frequencies and patient populations
  - Phase 3a: interaction classifications (synergistic/antagonistic)
  - Phase 3b: emergent double-mutant dependencies
  - Phase 4: TP53 modulation effects

Output:
  - phase5_clinical_implications.csv: per-pair clinical summary
  - phase5_clinical_implications_summary.json: meta-summary

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.07_clinical_implications
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"

# Drug target annotations from published literature / atlas work
DRUG_TARGETS = {
    "MTAP": {
        "drug_class": "PRMT5 inhibitor",
        "agents": ["JNJ-64619178", "GSK3326595"],
        "stage": "Phase I/II",
    },
    "ARID1A": {
        "drug_class": "EZH2 inhibitor",
        "agents": ["tazemetostat"],
        "stage": "FDA-approved (epithelioid sarcoma), Phase II (others)",
    },
    "PIK3CA": {
        "drug_class": "PI3Kalpha inhibitor",
        "agents": ["inavolisib", "alpelisib"],
        "stage": "FDA-approved (breast)",
    },
    "PTEN": {
        "drug_class": "AKT inhibitor / PI3Kbeta inhibitor",
        "agents": ["capivasertib", "AZD8186"],
        "stage": "FDA-approved (breast, capivasertib)",
    },
    "RB1": {
        "drug_class": "CDK2 inhibitor / CHK1 inhibitor",
        "agents": ["PF-07104091", "prexasertib"],
        "stage": "Phase I/II",
    },
    "BRCA1": {
        "drug_class": "PARP inhibitor",
        "agents": ["olaparib", "talazoparib"],
        "stage": "FDA-approved",
    },
    "BRCA2": {
        "drug_class": "PARP inhibitor",
        "agents": ["olaparib", "talazoparib"],
        "stage": "FDA-approved",
    },
    "TP53": {
        "drug_class": "MDM2 inhibitor / WEE1 inhibitor",
        "agents": ["KT-253", "adavosertib"],
        "stage": "Phase I/II",
    },
    "NF1": {
        "drug_class": "MEK inhibitor",
        "agents": ["trametinib", "selumetinib"],
        "stage": "FDA-approved (NF1 plexiform neurofibroma)",
    },
    "KEAP1": {
        "drug_class": "NRF2 pathway inhibitor",
        "agents": ["investigational"],
        "stage": "Preclinical",
    },
    "CDKN2A": {
        "drug_class": "CDK4/6 inhibitor",
        "agents": ["palbociclib", "ribociclib", "abemaciclib"],
        "stage": "FDA-approved (breast, others)",
    },
    "SMARCA4": {
        "drug_class": "CDK4/6 inhibitor / EZH2 inhibitor",
        "agents": ["palbociclib", "tazemetostat"],
        "stage": "Phase II (investigational)",
    },
}

# TCGA abbreviation to full cancer type name
TCGA_CANCER_NAMES = {
    "BRCA": "Breast invasive carcinoma",
    "GBM": "Glioblastoma",
    "OV": "Ovarian serous cystadenocarcinoma",
    "LUAD": "Lung adenocarcinoma",
    "UCEC": "Uterine corpus endometrial carcinoma",
    "KIRC": "Kidney renal clear cell carcinoma",
    "HNSC": "Head and neck squamous cell carcinoma",
    "LGG": "Brain lower grade glioma",
    "LUSC": "Lung squamous cell carcinoma",
    "PRAD": "Prostate adenocarcinoma",
    "SKCM": "Skin cutaneous melanoma",
    "COAD": "Colon adenocarcinoma",
    "STAD": "Stomach adenocarcinoma",
    "BLCA": "Bladder urothelial carcinoma",
    "LIHC": "Liver hepatocellular carcinoma",
    "CESC": "Cervical squamous cell carcinoma",
    "PAAD": "Pancreatic adenocarcinoma",
    "SARC": "Sarcoma",
    "ESCA": "Esophageal carcinoma",
    "READ": "Rectum adenocarcinoma",
}


def load_coalteration_data() -> pd.DataFrame:
    """Load Phase 1b pairwise co-alteration matrix."""
    return pd.read_csv(OUTPUT_DIR / "phase1_coalteration_matrix.csv")


def load_alteration_matrix() -> pd.DataFrame:
    """Load Phase 1a patient-level alteration matrix for cancer type breakdown."""
    return pd.read_csv(OUTPUT_DIR / "phase1_alteration_matrix.csv")


def load_interaction_summary() -> dict:
    """Load Phase 3a interaction test summary."""
    with open(OUTPUT_DIR / "phase3_interaction_results" / "interaction_summary.json") as f:
        return json.load(f)


def load_emergent_summary() -> dict:
    """Load Phase 3b emergent dependency summary."""
    with open(OUTPUT_DIR / "phase3_emergent_dependencies" / "emergent_summary.json") as f:
        return json.load(f)


def load_tp53_modulation() -> pd.DataFrame | None:
    """Load Phase 4 TP53 modulation results if available."""
    path = OUTPUT_DIR / "phase4_tp53_modulation" / "phase4_tp53_modulation.csv"
    if path.exists():
        return pd.read_csv(path)
    return None


def get_top_cancer_types(
    alt_matrix: pd.DataFrame, gene_a: str, gene_b: str, top_n: int = 5
) -> list[dict]:
    """Find cancer types with highest co-alteration frequency for a pair."""
    # Handle BRCA1_2 combined
    ga = gene_a if gene_a != "BRCA1_2" else "BRCA1"
    gb = gene_b if gene_b != "BRCA1_2" else "BRCA1"

    # For BRCA1_2, combine BRCA1 and BRCA2
    df = alt_matrix.copy()
    if gene_a == "BRCA1_2" or gene_b == "BRCA1_2":
        df["BRCA1_2"] = ((df["BRCA1"] == 1) | (df["BRCA2"] == 1)).astype(int)
        if gene_a == "BRCA1_2":
            ga = "BRCA1_2"
        if gene_b == "BRCA1_2":
            gb = "BRCA1_2"

    ct_groups = df.groupby("cancer_type").apply(
        lambda g: pd.Series({
            "n_total": len(g),
            "n_co_alt": int(((g[ga] == 1) & (g[gb] == 1)).sum()),
        })
    ).reset_index()
    ct_groups["co_alt_pct"] = ct_groups["n_co_alt"] / ct_groups["n_total"] * 100
    ct_groups = ct_groups[ct_groups["n_co_alt"] > 0].sort_values("co_alt_pct", ascending=False)

    results = []
    for _, row in ct_groups.head(top_n).iterrows():
        ct = row["cancer_type"]
        results.append({
            "cancer_type": ct,
            "cancer_name": TCGA_CANCER_NAMES.get(ct, ct),
            "n_co_altered": int(row["n_co_alt"]),
            "n_total": int(row["n_total"]),
            "co_alt_pct": round(float(row["co_alt_pct"]), 1),
        })
    return results


def build_combination_rationale(
    gene_a: str, gene_b: str, interaction_type: str
) -> str:
    """Build clinical rationale for combination therapy."""
    drug_a = DRUG_TARGETS.get(gene_a, {})
    drug_b = DRUG_TARGETS.get(gene_b, {})

    if not drug_a or not drug_b:
        return "Insufficient drug target data for combination rationale"

    rationale_parts = []

    if interaction_type == "synergistic":
        rationale_parts.append(
            f"Synergistic interaction: double-mutant lines show enhanced dependency "
            f"beyond individual effects, supporting combination of "
            f"{drug_a.get('drug_class', '?')} + {drug_b.get('drug_class', '?')}"
        )
    elif interaction_type == "antagonistic":
        rationale_parts.append(
            f"Antagonistic interaction: caution warranted — double-mutant dependency "
            f"is WEAKER than expected from individual effects. "
            f"{drug_a.get('drug_class', '?')} + {drug_b.get('drug_class', '?')} "
            f"may show reduced efficacy in co-altered tumors"
        )
    else:
        rationale_parts.append(
            f"Additive/non-significant interaction: dependencies appear independent. "
            f"{drug_a.get('drug_class', '?')} and {drug_b.get('drug_class', '?')} "
            f"may be combined without synergy/antagonism concerns"
        )

    return "; ".join(rationale_parts)


def main() -> None:
    out_dir = OUTPUT_DIR / "phase5_clinical_implications"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=== Phase 5: Clinical Implication Framework ===\n")

    # Load all prior phase data
    coalt = load_coalteration_data()
    alt_matrix = load_alteration_matrix()
    interactions = load_interaction_summary()
    emergent = load_emergent_summary()
    tp53_mod = load_tp53_modulation()

    print(f"  Co-alteration pairs: {len(coalt)}")
    print(f"  Interaction-tested pairs: {len(interactions)}")
    print(f"  Emergent-screened pairs: {len(emergent)}")
    print(f"  TP53 modulation data: {'available' if tp53_mod is not None else 'not found'}\n")

    # Build clinical implication rows
    rows = []

    # Process all interaction-tested pairs (these are the most informative)
    all_pair_names = set(interactions.keys()) | set(emergent.keys())

    for pair_name in sorted(all_pair_names):
        # Parse gene names
        inter = interactions.get(pair_name, {})
        emerg = emergent.get(pair_name, {})

        gene_a = inter.get("gene_a") or emerg.get("gene_a", pair_name.split("+")[0])
        gene_b = inter.get("gene_b") or emerg.get("gene_b", pair_name.split("+")[1])

        # Get co-alteration stats from Phase 1b
        coalt_row = coalt[
            ((coalt["gene_1"] == gene_a) & (coalt["gene_2"] == gene_b))
            | ((coalt["gene_1"] == gene_b) & (coalt["gene_2"] == gene_a))
        ]

        n_co_alt = 0
        co_alt_pct = 0.0
        odds_ratio = 0.0
        tendency = "unknown"
        if len(coalt_row) > 0:
            r = coalt_row.iloc[0]
            n_co_alt = int(r["co_altered"])
            co_alt_pct = float(r["co_alt_pct"])
            odds_ratio = float(r["odds_ratio"])
            tendency = str(r["tendency"])

        # Interaction classification
        n_synergistic = inter.get("n_synergistic", 0)
        n_antagonistic = inter.get("n_antagonistic", 0)
        n_sig_fdr = inter.get("n_sig_fdr", 0)

        if n_synergistic > n_antagonistic and n_sig_fdr > 0:
            interaction_type = "synergistic"
        elif n_antagonistic > n_synergistic and n_sig_fdr > 0:
            interaction_type = "antagonistic"
        elif n_sig_fdr > 0:
            interaction_type = "mixed"
        else:
            interaction_type = "additive"

        # Emergent dependencies
        n_novel_emergent = emerg.get("n_novel_emergent", 0)
        n_genomewide_hits = emerg.get("n_genomewide_hits", 0)

        # Drug targets
        drug_a = DRUG_TARGETS.get(gene_a, {})
        drug_b = DRUG_TARGETS.get(gene_b, {})
        both_druggable = bool(drug_a) and bool(drug_b)

        drug_targets_str = ""
        if drug_a:
            drug_targets_str += f"{gene_a}: {drug_a.get('drug_class', '?')}"
        if drug_b:
            drug_targets_str += f"; {gene_b}: {drug_b.get('drug_class', '?')}"

        combination_agents = []
        if drug_a:
            combination_agents.extend(drug_a.get("agents", []))
        if drug_b:
            combination_agents.extend(drug_b.get("agents", []))

        # Get top cancer types
        top_cts = get_top_cancer_types(alt_matrix, gene_a, gene_b)
        top_ct_str = "; ".join(
            f"{ct['cancer_type']} ({ct['co_alt_pct']}%)" for ct in top_cts[:3]
        )

        # Patient population estimate (from TCGA total)
        n_total_patients = len(alt_matrix)
        est_patients_per_100k = round(n_co_alt / n_total_patients * 100000)

        # Contraindication flags
        contraindications = []
        if interaction_type == "antagonistic":
            contraindications.append("ANTAGONISTIC: combination may be less effective than monotherapy")
        if gene_a == "TP53" or gene_b == "TP53":
            if tp53_mod is not None:
                # Check if relevant atlases show TP53 weakening
                other_gene = gene_b if gene_a == "TP53" else gene_a
                mod_row = tp53_mod[tp53_mod["atlas"].str.contains(other_gene, case=False, na=False)]
                if len(mod_row) > 0 and mod_row.iloc[0].get("tp53_weakens_sl", False):
                    contraindications.append(
                        f"TP53_MODULATION: TP53 co-mutation may weaken {other_gene} SL effects"
                    )
        if tendency == "mutual_exclusivity":
            contraindications.append("MUTUAL_EXCLUSIVITY: co-alteration rarer than expected")

        # Clinical rationale
        rationale = build_combination_rationale(gene_a, gene_b, interaction_type)

        # Priority score: combines actionability, interaction strength, patient population
        priority = 0
        if both_druggable:
            priority += 3
        if interaction_type == "synergistic":
            priority += 3
        elif interaction_type == "mixed":
            priority += 1
        elif interaction_type == "antagonistic":
            priority -= 1
        if n_novel_emergent > 10:
            priority += 2
        elif n_novel_emergent > 0:
            priority += 1
        if co_alt_pct > 2:
            priority += 2
        elif co_alt_pct > 1:
            priority += 1
        if not contraindications:
            priority += 1

        rows.append({
            "pair": pair_name,
            "gene_a": gene_a,
            "gene_b": gene_b,
            "interaction_type": interaction_type,
            "n_synergistic": n_synergistic,
            "n_antagonistic": n_antagonistic,
            "n_co_altered_tcga": n_co_alt,
            "co_alt_pct": round(co_alt_pct, 2),
            "odds_ratio": round(odds_ratio, 2),
            "tendency": tendency,
            "n_novel_emergent": n_novel_emergent,
            "n_genomewide_hits": n_genomewide_hits,
            "both_druggable": both_druggable,
            "drug_targets": drug_targets_str,
            "combination_agents": "; ".join(combination_agents),
            "top_cancer_types": top_ct_str,
            "est_patients_per_100k": est_patients_per_100k,
            "contraindications": " | ".join(contraindications) if contraindications else "none",
            "clinical_rationale": rationale,
            "priority_score": priority,
        })

    # Sort by priority score
    df = pd.DataFrame(rows).sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.to_csv(out_dir / "phase5_clinical_implications.csv", index=False)
    # Also save to the main output dir per task spec
    df.to_csv(OUTPUT_DIR / "phase5_clinical_implications.csv", index=False)

    print(f"{'=' * 70}")
    print(f"Clinical Implication Framework — {len(df)} pairs analyzed\n")

    print(f"{'Pair':25s}  {'Type':12s}  {'Emerg':>5s}  {'CoAlt%':>6s}  {'Priority':>8s}  {'Contra':>6s}")
    print("-" * 75)
    for _, row in df.iterrows():
        contra = "YES" if row["contraindications"] != "none" else "-"
        print(f"{row['pair']:25s}  {row['interaction_type']:12s}  "
              f"{row['n_novel_emergent']:5d}  {row['co_alt_pct']:6.2f}  "
              f"{row['priority_score']:8d}  {contra:>6s}")

    # Summary
    summary = {
        "total_pairs": len(df),
        "synergistic": int((df["interaction_type"] == "synergistic").sum()),
        "antagonistic": int((df["interaction_type"] == "antagonistic").sum()),
        "mixed": int((df["interaction_type"] == "mixed").sum()),
        "additive": int((df["interaction_type"] == "additive").sum()),
        "both_druggable": int(df["both_druggable"].sum()),
        "with_emergent_deps": int((df["n_novel_emergent"] > 0).sum()),
        "with_contraindications": int((df["contraindications"] != "none").sum()),
        "top_priority_pairs": df.head(5)[["pair", "priority_score", "interaction_type"]].to_dict("records"),
    }

    with open(out_dir / "phase5_clinical_implications_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"\n{'=' * 70}")
    print("Summary:")
    print(f"  Synergistic pairs:     {summary['synergistic']}")
    print(f"  Antagonistic pairs:    {summary['antagonistic']}")
    print(f"  Mixed:                 {summary['mixed']}")
    print(f"  Additive:              {summary['additive']}")
    print(f"  Both druggable:        {summary['both_druggable']}")
    print(f"  With emergent deps:    {summary['with_emergent_deps']}")
    print(f"  Contraindications:     {summary['with_contraindications']}")

    print(f"\nTop 3 priority pairs:")
    for _, row in df.head(3).iterrows():
        print(f"  {row['pair']:25s}  score={row['priority_score']}  {row['interaction_type']}")
        print(f"    Drugs: {row['drug_targets']}")
        print(f"    Cancer types: {row['top_cancer_types']}")

    print(f"\nSaved: {out_dir / 'phase5_clinical_implications.csv'}")
    print(f"Saved: {OUTPUT_DIR / 'phase5_clinical_implications.csv'}")
    print(f"Saved: {out_dir / 'phase5_clinical_implications_summary.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
