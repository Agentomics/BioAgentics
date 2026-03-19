"""Phase 1b: KEGG metabolic gene filter + differential dependency screen.

Filters DepMap CRISPR gene effect data to KEGG metabolic pathway genes (~1,800).
For each qualifying cancer type, compares dependency scores between:
  - ARID1A-mutant vs WT
  - SMARCA4-mutant vs WT
  - Combined SWI/SNF-mutant (any) vs WT

Computes Cohen's d, Mann-Whitney p-values, and Benjamini-Hochberg FDR.

Usage:
    uv run python -m swisnf_metabolic_convergence.02_metabolic_dependency_screen
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

from bioagentics.config import REPO_ROOT
from bioagentics.data.gene_ids import load_depmap_matrix

logger = logging.getLogger(__name__)

DEPMAP_DIR = REPO_ROOT / "data" / "depmap" / "25q3"
PHASE1A_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1a"
OUTPUT_DIR = REPO_ROOT / "data" / "results" / "swisnf-metabolic-convergence" / "phase1b"

# Minimum samples per group
MIN_SAMPLES = 3

# KEGG metabolic pathway gene cache
KEGG_CACHE = OUTPUT_DIR / "kegg_metabolic_genes.json"

# Key KEGG metabolic pathway IDs (human) covering the plan's requirements
KEGG_METABOLIC_PATHWAYS = {
    "hsa00010": "Glycolysis / Gluconeogenesis",
    "hsa00020": "Citrate cycle (TCA cycle)",
    "hsa00030": "Pentose phosphate pathway",
    "hsa00040": "Pentose and glucuronate interconversions",
    "hsa00051": "Fructose and mannose metabolism",
    "hsa00052": "Galactose metabolism",
    "hsa00053": "Ascorbate and aldarate metabolism",
    "hsa00061": "Fatty acid biosynthesis",
    "hsa00062": "Fatty acid elongation",
    "hsa00071": "Fatty acid degradation",
    "hsa00072": "Synthesis and degradation of ketone bodies",
    "hsa00100": "Steroid biosynthesis",
    "hsa00120": "Primary bile acid biosynthesis",
    "hsa00130": "Ubiquinone and other terpenoid-quinone biosynthesis",
    "hsa00140": "Steroid hormone biosynthesis",
    "hsa00190": "Oxidative phosphorylation",
    "hsa00220": "Arginine biosynthesis",
    "hsa00230": "Purine metabolism",
    "hsa00240": "Pyrimidine metabolism",
    "hsa00250": "Alanine, aspartate and glutamate metabolism",
    "hsa00260": "Glycine, serine and threonine metabolism",
    "hsa00270": "Cysteine and methionine metabolism",
    "hsa00280": "Valine, leucine and isoleucine degradation",
    "hsa00290": "Valine, leucine and isoleucine biosynthesis",
    "hsa00300": "Lysine biosynthesis",
    "hsa00310": "Lysine degradation",
    "hsa00330": "Arginine and proline metabolism",
    "hsa00340": "Histidine metabolism",
    "hsa00350": "Tyrosine metabolism",
    "hsa00360": "Phenylalanine metabolism",
    "hsa00380": "Tryptophan metabolism",
    "hsa00400": "Phenylalanine, tyrosine and tryptophan biosynthesis",
    "hsa00410": "beta-Alanine metabolism",
    "hsa00430": "Taurine and hypotaurine metabolism",
    "hsa00440": "Phosphonate and phosphinate metabolism",
    "hsa00450": "Selenocompound metabolism",
    "hsa00460": "Cyanoamino acid metabolism",
    "hsa00471": "D-Glutamine and D-glutamate metabolism",
    "hsa00472": "D-Arginine and D-ornithine metabolism",
    "hsa00480": "Glutathione metabolism",
    "hsa00500": "Starch and sucrose metabolism",
    "hsa00510": "N-Glycan biosynthesis",
    "hsa00512": "Mucin type O-glycan biosynthesis",
    "hsa00514": "Other types of O-glycan biosynthesis",
    "hsa00515": "Mannose type O-glycan biosynthesis",
    "hsa00520": "Amino sugar and nucleotide sugar metabolism",
    "hsa00524": "Neomycin, kanamycin and gentamicin biosynthesis",
    "hsa00531": "Glycosaminoglycan degradation",
    "hsa00532": "Glycosaminoglycan biosynthesis - chondroitin sulfate / dermatan sulfate",
    "hsa00534": "Glycosaminoglycan biosynthesis - heparan sulfate / heparin",
    "hsa00561": "Glycerolipid metabolism",
    "hsa00562": "Inositol phosphate metabolism",
    "hsa00563": "Glycosylphosphatidylinositol (GPI)-anchor biosynthesis",
    "hsa00564": "Glycerophospholipid metabolism",
    "hsa00565": "Ether lipid metabolism",
    "hsa00590": "Arachidonic acid metabolism",
    "hsa00591": "Linoleic acid metabolism",
    "hsa00592": "alpha-Linolenic acid metabolism",
    "hsa00600": "Sphingolipid metabolism",
    "hsa00601": "Glycosphingolipid biosynthesis - lacto and neolacto series",
    "hsa00603": "Glycosphingolipid biosynthesis - globo and isoglobo series",
    "hsa00604": "Glycosphingolipid biosynthesis - ganglio series",
    "hsa00620": "Pyruvate metabolism",
    "hsa00630": "Glyoxylate and dicarboxylate metabolism",
    "hsa00640": "Propanoate metabolism",
    "hsa00650": "Butanoate metabolism",
    "hsa00670": "One carbon pool by folate",
    "hsa00730": "Thiamine metabolism",
    "hsa00740": "Riboflavin metabolism",
    "hsa00750": "Vitamin B6 metabolism",
    "hsa00760": "Nicotinate and nicotinamide metabolism",
    "hsa00770": "Pantothenate and CoA biosynthesis",
    "hsa00780": "Biotin metabolism",
    "hsa00785": "Lipoic acid metabolism",
    "hsa00790": "Folate biosynthesis",
    "hsa00860": "Porphyrin metabolism",
    "hsa00900": "Terpenoid backbone biosynthesis",
    "hsa00910": "Nitrogen metabolism",
    "hsa00920": "Sulfur metabolism",
    "hsa01040": "Biosynthesis of unsaturated fatty acids",
    "hsa01200": "Carbon metabolism",
    "hsa01210": "2-Oxocarboxylic acid metabolism",
    "hsa01212": "Fatty acid metabolism",
    "hsa01230": "Biosynthesis of amino acids",
    "hsa01232": "Nucleotide metabolism",
    "hsa01250": "Biosynthesis of nucleotide sugars",
}


def fetch_kegg_pathway_genes(pathway_id: str) -> list[str]:
    """Fetch gene symbols for a KEGG pathway via REST API."""
    import re
    import requests

    url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()

    # Format: "hsa:ENTREZID\tpathway:hsa00010"
    entrez_ids = []
    for line in resp.text.strip().split("\n"):
        if not line:
            continue
        parts = line.split("\t")
        if len(parts) >= 1:
            gene_id = parts[1].replace("hsa:", "") if len(parts) > 1 else parts[0].replace("hsa:", "")
            entrez_ids.append(gene_id)

    if not entrez_ids:
        return []

    # Convert Entrez IDs to gene symbols in batches
    symbols = []
    for i in range(0, len(entrez_ids), 10):
        batch = entrez_ids[i:i + 10]
        ids_str = "+".join(f"hsa:{eid}" for eid in batch)
        url2 = f"https://rest.kegg.jp/get/{ids_str}"
        try:
            resp2 = requests.get(url2, timeout=30)
            resp2.raise_for_status()
            for line in resp2.text.split("\n"):
                m = re.match(r"SYMBOL\s+(.+)", line.strip())
                if m:
                    for sym in m.group(1).split(","):
                        sym = sym.strip()
                        if sym:
                            symbols.append(sym)
        except Exception:
            continue

    return symbols


def load_kegg_metabolic_genes(cache_path: Path | None = None) -> dict[str, list[str]]:
    """Load KEGG metabolic pathway genes, with caching.

    Returns dict mapping pathway_name -> list of gene symbols.
    Also returns flat set of all metabolic gene symbols.
    """
    if cache_path is None:
        cache_path = KEGG_CACHE

    if cache_path.exists():
        with open(cache_path) as f:
            return json.load(f)

    print("Downloading KEGG metabolic pathway genes (first run only)...")
    import requests

    pathway_genes: dict[str, list[str]] = {}

    for pathway_id, pathway_name in KEGG_METABOLIC_PATHWAYS.items():
        # Use KEGG link API to get gene list
        url = f"https://rest.kegg.jp/link/hsa/{pathway_id}"
        try:
            resp = requests.get(url, timeout=30)
            resp.raise_for_status()
        except Exception as e:
            print(f"  Warning: failed to fetch {pathway_id}: {e}")
            continue

        entrez_ids = []
        for line in resp.text.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                eid = parts[1].replace("hsa:", "")
                entrez_ids.append(eid)

        if not entrez_ids:
            continue

        # Convert Entrez IDs to symbols using KEGG conv API
        url2 = f"https://rest.kegg.jp/conv/ncbi-geneid/{pathway_id}"
        try:
            resp2 = requests.get(url2, timeout=30)
            resp2.raise_for_status()
        except Exception:
            continue

        # Build entrez-to-kegg mapping
        kegg_to_entrez: dict[str, str] = {}
        for line in resp2.text.strip().split("\n"):
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                kegg_id = parts[0].replace("hsa:", "")
                ncbi_id = parts[1].replace("ncbi-geneid:", "")
                kegg_to_entrez[kegg_id] = ncbi_id

        pathway_genes[pathway_name] = entrez_ids
        print(f"  {pathway_name}: {len(entrez_ids)} genes")

    # Cache results
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(pathway_genes, f, indent=2)

    return pathway_genes


def get_metabolic_gene_symbols(
    crispr_columns: pd.Index, pathway_genes: dict[str, list[str]],
) -> tuple[set[str], dict[str, str]]:
    """Map KEGG Entrez IDs to symbols available in CRISPR data.

    Uses the CRISPRGeneEffect column names (HUGO symbols from load_depmap_matrix)
    as the ground truth for available genes.

    Returns (set of matched symbols, dict of symbol -> pathway membership).
    """
    # The pathway_genes values are Entrez IDs from KEGG link API
    # We need to match them to HUGO symbols in the CRISPR data
    # Strategy: Use a KEGG-to-symbol mapping file or the GMT file
    # For now, load all metabolic genes from the GMT file as a complement

    # First approach: extract all unique Entrez IDs across pathways
    all_entrez: set[str] = set()
    for genes in pathway_genes.values():
        all_entrez.update(genes)

    # We need to map Entrez IDs to HUGO symbols
    # Load the original CRISPRGeneEffect with raw headers to build the map
    raw_df = pd.read_csv(DEPMAP_DIR / "CRISPRGeneEffect.csv", index_col=0, nrows=0)
    import re
    entrez_to_symbol: dict[str, str] = {}
    for col in raw_df.columns:
        m = re.match(r"^(.+?)\s+\((\d+)\)$", col)
        if m:
            symbol, entrez = m.group(1), m.group(2)
            entrez_to_symbol[entrez] = symbol

    # Map pathway Entrez IDs to symbols
    matched_symbols: set[str] = set()
    symbol_to_pathway: dict[str, str] = {}
    for pathway_name, entrez_ids in pathway_genes.items():
        for eid in entrez_ids:
            symbol = entrez_to_symbol.get(eid)
            if symbol and symbol in crispr_columns:
                matched_symbols.add(symbol)
                if symbol not in symbol_to_pathway:
                    symbol_to_pathway[symbol] = pathway_name
                else:
                    symbol_to_pathway[symbol] += f"; {pathway_name}"

    return matched_symbols, symbol_to_pathway


def cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size (group1 - group2)."""
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(ddof=1), group2.var(ddof=1)
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_std == 0:
        return 0.0
    return float((group1.mean() - group2.mean()) / pooled_std)


def fdr_correction(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])
    sorted_idx = np.argsort(pvalues)
    sorted_p = pvalues[sorted_idx]
    fdr = np.empty(n)
    for i in range(n):
        fdr[sorted_idx[i]] = sorted_p[i] * n / (i + 1)
    fdr_sorted = fdr[sorted_idx]
    for i in range(n - 2, -1, -1):
        fdr_sorted[i] = min(fdr_sorted[i], fdr_sorted[i + 1])
    fdr[sorted_idx] = fdr_sorted
    return np.minimum(fdr, 1.0)


def run_dependency_screen(
    crispr: pd.DataFrame,
    classified: pd.DataFrame,
    metabolic_genes: set[str],
    comparison_name: str,
    mutant_col: str,
) -> pd.DataFrame:
    """Run differential dependency for metabolic genes.

    For each qualifying cancer type, tests each metabolic gene
    for differential dependency between mutant and WT groups.
    """
    # Get cancer type summary from Phase 1a
    summary = pd.read_csv(PHASE1A_DIR / "cancer_type_summary.csv")

    # Determine qualifying column
    if comparison_name == "ARID1A":
        qual_col = "qualifies_arid1a"
    elif comparison_name == "SMARCA4":
        qual_col = "qualifies_smarca4"
    else:
        qual_col = "qualifies_combined"

    qualifying_types = summary[summary[qual_col]]["cancer_type"].tolist()

    # Filter to genes available in CRISPR data
    available_genes = sorted(metabolic_genes & set(crispr.columns))

    results = []
    for cancer_type in qualifying_types:
        # Get cell lines for this cancer type
        ct_lines = classified[classified["OncotreeLineage"] == cancer_type]

        if mutant_col == "swisnf_any_mutant":
            mut_ids = ct_lines[ct_lines[mutant_col] == True].index
            wt_ids = ct_lines[ct_lines[mutant_col] == False].index
        else:
            mut_ids = ct_lines[ct_lines[mutant_col] == True].index
            wt_ids = ct_lines[ct_lines["swisnf_any_mutant"] == False].index

        # Filter to lines with CRISPR data
        mut_ids = mut_ids.intersection(crispr.index)
        wt_ids = wt_ids.intersection(crispr.index)

        if len(mut_ids) < MIN_SAMPLES or len(wt_ids) < MIN_SAMPLES:
            continue

        # Test each metabolic gene
        for gene in available_genes:
            mut_vals = crispr.loc[mut_ids, gene].dropna().values
            wt_vals = crispr.loc[wt_ids, gene].dropna().values

            if len(mut_vals) < MIN_SAMPLES or len(wt_vals) < MIN_SAMPLES:
                continue

            d = cohens_d(mut_vals, wt_vals)
            _, pval = stats.mannwhitneyu(mut_vals, wt_vals, alternative="two-sided")

            results.append({
                "cancer_type": cancer_type,
                "gene": gene,
                "comparison": comparison_name,
                "cohens_d": round(d, 4),
                "p_value": pval,
                "n_mut": len(mut_vals),
                "n_wt": len(wt_vals),
                "median_dep_mut": round(float(np.median(mut_vals)), 4),
                "median_dep_wt": round(float(np.median(wt_vals)), 4),
            })

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)

    # FDR correction per cancer type
    fdr_values = []
    for _, group in df.groupby("cancer_type"):
        pvals = group["p_value"].values
        fdrs = fdr_correction(pvals)
        fdr_values.extend(zip(group.index, fdrs))

    fdr_series = pd.Series(dict(fdr_values))
    df["fdr"] = fdr_series

    df = df.sort_values(["cancer_type", "p_value"]).reset_index(drop=True)
    return df


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=== Phase 1b: Metabolic Gene Dependency Screen ===\n")

    # Step 1: Load KEGG metabolic pathway genes
    print("Loading KEGG metabolic pathway genes...")
    pathway_genes = load_kegg_metabolic_genes()
    all_entrez = set()
    for genes in pathway_genes.values():
        all_entrez.update(genes)
    print(f"  {len(pathway_genes)} pathways, {len(all_entrez)} unique Entrez IDs")

    # Step 2: Load CRISPR gene effect data
    print("Loading CRISPRGeneEffect.csv...")
    crispr = load_depmap_matrix(DEPMAP_DIR / "CRISPRGeneEffect.csv")
    print(f"  {crispr.shape[0]} cell lines, {crispr.shape[1]} genes")

    # Step 3: Map KEGG Entrez IDs to CRISPR gene symbols
    print("Mapping metabolic genes to CRISPR data...")
    metabolic_symbols, symbol_pathway = get_metabolic_gene_symbols(
        crispr.columns, pathway_genes,
    )
    print(f"  {len(metabolic_symbols)} metabolic genes available in CRISPR data")

    # Save metabolic gene list with pathway annotations
    gene_list = pd.DataFrame([
        {"gene": sym, "pathways": symbol_pathway.get(sym, "")}
        for sym in sorted(metabolic_symbols)
    ])
    gene_list.to_csv(OUTPUT_DIR / "metabolic_gene_list.csv", index=False)

    # Step 4: Load classified cell lines from Phase 1a
    print("Loading classified cell lines from Phase 1a...")
    classified = pd.read_csv(PHASE1A_DIR / "swisnf_classified_lines.csv", index_col=0)
    print(f"  {len(classified)} classified cell lines")

    # Step 5: Run dependency screens
    print("\n--- Running ARID1A-mutant vs WT screen ---")
    arid1a_results = run_dependency_screen(
        crispr, classified, metabolic_symbols,
        comparison_name="ARID1A",
        mutant_col="ARID1A_disrupted",
    )
    if len(arid1a_results) > 0:
        n_sig = ((arid1a_results["fdr"] < 0.05) & (arid1a_results["cohens_d"].abs() > 0.3)).sum()
        print(f"  {len(arid1a_results)} tests, {n_sig} significant (FDR<0.05, |d|>0.3)")
        arid1a_results.to_csv(OUTPUT_DIR / "screen_arid1a_vs_wt.csv", index=False)

    print("\n--- Running SMARCA4-mutant vs WT screen ---")
    smarca4_results = run_dependency_screen(
        crispr, classified, metabolic_symbols,
        comparison_name="SMARCA4",
        mutant_col="SMARCA4_disrupted",
    )
    if len(smarca4_results) > 0:
        n_sig = ((smarca4_results["fdr"] < 0.05) & (smarca4_results["cohens_d"].abs() > 0.3)).sum()
        print(f"  {len(smarca4_results)} tests, {n_sig} significant (FDR<0.05, |d|>0.3)")
        smarca4_results.to_csv(OUTPUT_DIR / "screen_smarca4_vs_wt.csv", index=False)

    print("\n--- Running combined SWI/SNF-mutant vs WT screen ---")
    combined_results = run_dependency_screen(
        crispr, classified, metabolic_symbols,
        comparison_name="SWI/SNF_combined",
        mutant_col="swisnf_any_mutant",
    )
    if len(combined_results) > 0:
        n_sig = ((combined_results["fdr"] < 0.05) & (combined_results["cohens_d"].abs() > 0.3)).sum()
        print(f"  {len(combined_results)} tests, {n_sig} significant (FDR<0.05, |d|>0.3)")
        combined_results.to_csv(OUTPUT_DIR / "screen_combined_vs_wt.csv", index=False)

    # Step 6: Summary of top hits across all comparisons
    print("\n=== Top metabolic dependency hits ===")
    for name, df in [("ARID1A", arid1a_results), ("SMARCA4", smarca4_results),
                      ("Combined", combined_results)]:
        if len(df) == 0:
            print(f"\n{name}: No results")
            continue
        sig = df[(df["fdr"] < 0.05) & (df["cohens_d"].abs() > 0.3)]
        sig_neg = sig[sig["cohens_d"] < 0].sort_values("cohens_d")
        print(f"\n{name} — {len(sig_neg)} SL hits (FDR<0.05, d<-0.3):")
        for _, row in sig_neg.head(15).iterrows():
            print(f"  {row['cancer_type']}: {row['gene']} d={row['cohens_d']:.2f} "
                  f"FDR={row['fdr']:.2e} (n_mut={row['n_mut']}, n_wt={row['n_wt']})")

    print(f"\nResults saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
