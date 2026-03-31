"""Phase 2: Identify DepMap 25Q3 double-mutant cell lines for top 10 co-alteration pairs.

For each of the top 10 co-alteration pairs from Phase 1b, identifies DepMap cell
lines carrying BOTH alterations. Classifies lines into four groups: double-mutant,
gene-A-only, gene-B-only, and wild-type-both.

Uses same classification criteria as Phase 1a:
  - TSGs: truncating (LikelyLoF) + deep deletion (CN log2 <= 0.3)
  - TP53: truncating + missense
  - PIK3CA: hotspot missense
  - MTAP/CDKN2A: deep deletion + truncating

Output:
  - phase2_double_mutant_lines.json: cell line IDs per group per pair
  - phase2_double_mutant_summary.csv: summary statistics

Usage:
    uv run python -m cancer.tsg_co_alteration_dependency_interactions.03_depmap_double_mutant_lines
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import (
    load_depmap_matrix,
    load_depmap_model_metadata,
)

OUTPUT_DIR = REPO_ROOT / "output" / "cancer" / "tsg-co-alteration-dependency-interactions"
DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"

# CN threshold for deep deletion (log2 ratio scale: 1.0 = diploid)
HOMDEL_CN_THRESHOLD = 0.3

# PIK3CA hotspot residues
PIK3CA_HOTSPOT_PREFIXES = ["E542", "E545", "H1047", "C420", "N345", "R88", "Q546"]

# Same gene rules as Phase 1a
GENE_RULES = {
    "MTAP": "deletion",
    "ARID1A": "tsg",
    "SMARCA4": "tsg",
    "PIK3CA": "oncogene",
    "PTEN": "tsg",
    "RB1": "tsg",
    "BRCA1": "tsg",
    "BRCA2": "tsg",
    "TP53": "tp53",
    "KEAP1": "tsg",
    "NF1": "tsg",
    "CDKN2A": "deletion",
}

# Top 10 pairs to analyze (from Phase 1b composite rank, BRCA1/2 expanded)
TOP_PAIRS = [
    ("MTAP", "CDKN2A"),
    ("TP53", "CDKN2A"),
    ("MTAP", "TP53"),
    ("PIK3CA", "TP53"),
    ("RB1", "TP53"),
    ("PTEN", "TP53"),
    ("ARID1A", "PTEN"),
    ("ARID1A", "TP53"),
    ("TP53", "NF1"),
    ("ARID1A", "PIK3CA"),
]

# Priority validation pairs from the plan
PRIORITY_PAIRS = [
    ("PTEN", "RB1"),
    ("ARID1A", "PIK3CA"),
    ("TP53", "PTEN"),
    ("TP53", "BRCA1"),
    ("TP53", "BRCA2"),
    ("TP53", "RB1"),
]


def load_mutation_status() -> pd.DataFrame:
    """Load DepMap mutations and classify per gene."""
    cols = [
        "ModelID", "HugoSymbol", "VariantInfo", "ProteinChange",
        "VepImpact", "LikelyLoF",
    ]
    muts = pd.read_csv(
        DEPMAP_DIR / "OmicsSomaticMutations.csv",
        usecols=lambda c: c in cols,
    )
    return muts


def load_cn_data() -> pd.DataFrame:
    """Load copy number data for target genes."""
    cn = load_depmap_matrix(DEPMAP_DIR / "PortalOmicsCNGeneLog2.csv")
    target_genes = list(GENE_RULES.keys())
    available = [g for g in target_genes if g in cn.columns]
    return cn[available]


def classify_gene_status(
    gene: str,
    muts: pd.DataFrame,
    cn: pd.DataFrame,
    crispr_lines: set[str],
) -> dict[str, set[str]]:
    """Classify all cell lines as altered or WT for a single gene.

    Returns dict with 'altered' and 'wt' sets of ModelIDs.
    """
    rule = GENE_RULES[gene]
    altered: set[str] = set()

    # Check mutations
    gene_muts = muts[muts["HugoSymbol"] == gene]

    if rule == "tsg" or rule == "deletion":
        lof_lines = gene_muts[gene_muts["LikelyLoF"] == True]["ModelID"].unique()
        altered.update(lof_lines)

    elif rule == "tp53":
        # Truncating
        lof_lines = gene_muts[gene_muts["LikelyLoF"] == True]["ModelID"].unique()
        altered.update(lof_lines)
        # Missense (HIGH or MODERATE impact)
        missense = gene_muts[
            (gene_muts["VepImpact"].isin(["HIGH", "MODERATE"]))
        ]["ModelID"].unique()
        altered.update(missense)

    elif rule == "oncogene":
        # PIK3CA hotspot missense
        for _, row in gene_muts.iterrows():
            pc = str(row.get("ProteinChange", ""))
            if pc and any(pc.lstrip("p.").startswith(hp) for hp in PIK3CA_HOTSPOT_PREFIXES):
                altered.add(row["ModelID"])

    # Check copy number: deep deletion
    if gene in cn.columns:
        homdel_lines = cn.index[cn[gene] <= HOMDEL_CN_THRESHOLD].tolist()
        if rule in ("tsg", "deletion", "tp53"):
            altered.update(homdel_lines)

    # WT = lines with CRISPR data that are NOT altered
    wt = crispr_lines - altered

    return {"altered": altered, "wt": wt}


def identify_groups(
    gene_a: str,
    gene_b: str,
    status_a: dict[str, set[str]],
    status_b: dict[str, set[str]],
    crispr_lines: set[str],
) -> dict[str, list[str]]:
    """Identify four groups for a gene pair: double, A-only, B-only, WT-both."""
    alt_a = status_a["altered"] & crispr_lines
    alt_b = status_b["altered"] & crispr_lines

    double = sorted(alt_a & alt_b)
    a_only = sorted(alt_a - alt_b)
    b_only = sorted(alt_b - alt_a)
    wt_both = sorted(crispr_lines - alt_a - alt_b)

    return {
        "double_mutant": double,
        f"{gene_a}_only": a_only,
        f"{gene_b}_only": b_only,
        "wt_both": wt_both,
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 2: DepMap Double-Mutant Cell Line Identification ===\n")

    # Load data
    print("Loading DepMap data...")
    meta = load_depmap_model_metadata(DEPMAP_DIR / "Model.csv")
    muts = load_mutation_status()
    cn = load_cn_data()

    # Get CRISPR lines
    crispr = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", usecols=[0])
    crispr_lines = set(crispr.iloc[:, 0])
    print(f"  {len(crispr_lines)} cell lines with CRISPR data")
    print(f"  {len(muts)} total mutation records")
    print(f"  {len(cn)} cell lines with CN data\n")

    # Classify all genes
    print("Classifying gene status across all lines...")
    gene_status: dict[str, dict[str, set[str]]] = {}
    all_genes = set()
    for pair in TOP_PAIRS + PRIORITY_PAIRS:
        all_genes.update(pair)

    for gene in sorted(all_genes):
        status = classify_gene_status(gene, muts, cn, crispr_lines)
        gene_status[gene] = status
        n_alt = len(status["altered"] & crispr_lines)
        print(f"  {gene:10s}  {n_alt:4d} altered lines (with CRISPR)")

    # Combine pairs to analyze (top 10 + priority, deduplicated)
    all_pairs = list(dict.fromkeys(TOP_PAIRS + PRIORITY_PAIRS))

    # Identify groups for each pair
    print(f"\nIdentifying double-mutant groups for {len(all_pairs)} pairs...")
    results: dict[str, dict] = {}
    summary_rows = []

    for gene_a, gene_b in all_pairs:
        pair_name = f"{gene_a}+{gene_b}"
        groups = identify_groups(
            gene_a, gene_b, gene_status[gene_a], gene_status[gene_b], crispr_lines,
        )

        n_double = len(groups["double_mutant"])
        n_a_only = len(groups[f"{gene_a}_only"])
        n_b_only = len(groups[f"{gene_b}_only"])
        n_wt = len(groups["wt_both"])
        powered = n_double >= 5

        # Add cancer type annotations for double-mutant lines
        double_lines = groups["double_mutant"]
        cancer_types = []
        for mid in double_lines:
            if mid in meta.index:
                ct = meta.loc[mid].get("OncotreeLineage", "Unknown")
                cancer_types.append(ct)

        results[pair_name] = {
            "gene_a": gene_a,
            "gene_b": gene_b,
            "groups": groups,
            "n_double": n_double,
            "powered": powered,
            "double_mutant_cancer_types": cancer_types,
        }

        summary_rows.append({
            "pair": pair_name,
            "gene_a": gene_a,
            "gene_b": gene_b,
            "n_double_mutant": n_double,
            f"n_{gene_a}_only": n_a_only,
            f"n_{gene_b}_only": n_b_only,
            "n_wt_both": n_wt,
            "powered_N_gte_5": powered,
        })

        status = "POWERED" if powered else "UNDERPOWERED"
        print(f"  {pair_name:25s}  double={n_double:3d}  "
              f"{gene_a}-only={n_a_only:3d}  {gene_b}-only={n_b_only:3d}  "
              f"WT={n_wt:3d}  [{status}]")

    # Save JSON with full cell line IDs
    out_json = OUTPUT_DIR / "phase2_double_mutant_lines.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_json}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_rows)
    out_csv = OUTPUT_DIR / "phase2_double_mutant_summary.csv"
    summary_df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Validation
    n_powered = sum(1 for r in results.values() if r["powered"])
    n_total = len(results)
    print(f"\nValidation:")
    print(f"  Powered pairs (N>=5): {n_powered}/{n_total}")

    # Check TP53+RB1 for SCLC lines
    tp53_rb1 = results.get("TP53+RB1", {})
    if tp53_rb1:
        sclc_count = sum(1 for ct in tp53_rb1.get("double_mutant_cancer_types", [])
                         if "Lung" in str(ct) or "Small Cell" in str(ct))
        print(f"  TP53+RB1 double-mutant lung lines: {sclc_count} "
              f"({'PASS' if sclc_count > 0 else 'check needed'})")

    print("\nDone.")


if __name__ == "__main__":
    main()
