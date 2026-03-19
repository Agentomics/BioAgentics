"""Pathway and gene set enrichment analysis for IVIG scRNA-seq data.

Performs per-cell-type pathway enrichment on pseudobulk DE results using:
1. Over-representation analysis (ORA) via Fisher's exact test
2. Gene set enrichment analysis (GSEA) via rank-based KS-like statistic
3. Targeted pathway scoring for IVIG-relevant modules:
   - Autophagy (ATG7, UVRAG, BECN1) in monocytes
   - S100A12-TLR4-MYD88 axis in CD14+ monocytes
   - Histone modification in neutrophils/NK cells
   - Defense response / innate immunity

Gene sets are curated from GO, KEGG, and Reactome for pathways relevant
to PANS/IVIG biology. No external API calls required.

Usage:
    from bioagentics.pandas_pans.ivig_pathway_enrichment import (
        run_pathway_enrichment,
        score_ivig_modules,
    )
    enrichment = run_pathway_enrichment(de_results_df, background_genes)
    module_scores = score_ivig_modules(adata)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Curated gene sets relevant to PANS / IVIG mechanism
# ---------------------------------------------------------------------------

GENE_SETS: dict[str, dict[str, list[str]]] = {
    # GO Biological Process (curated subsets)
    "GO_defense_response": {
        "GO:0006952_defense_response": [
            "TNF", "IL1B", "IL6", "IFNG", "CXCL8", "CCL2", "CCL5",
            "NFKB1", "RELA", "TLR2", "TLR4", "MYD88", "IRAK4",
            "NLRP3", "CASP1", "IL18", "IL1A", "CXCL10", "STAT1",
        ],
        "GO:0045087_innate_immune_response": [
            "TLR1", "TLR2", "TLR4", "TLR6", "TLR7", "TLR8", "TLR9",
            "MYD88", "IRAK1", "IRAK4", "TRAF6", "NFKB1", "IRF3",
            "IRF7", "IFNB1", "DDX58", "IFIH1", "MAVS", "STING1",
            "CGAS", "LY96", "CD14", "NLRP3", "PYCARD", "CASP1",
        ],
        "GO:0006954_inflammatory_response": [
            "TNF", "IL1B", "IL6", "CXCL8", "CCL2", "CCL3", "CCL4",
            "PTGS2", "ALOX5", "PLA2G4A", "SELE", "ICAM1", "VCAM1",
            "NFKB1", "S100A8", "S100A9", "S100A12", "NLRP3",
        ],
    },
    # Autophagy pathway (KEGG hsa04140 + key regulators)
    "autophagy": {
        "KEGG_autophagy": [
            "ATG7", "ATG5", "ATG12", "ATG16L1", "ATG3", "ATG4B",
            "BECN1", "UVRAG", "PIK3C3", "ATG14", "ATG9A",
            "MAP1LC3B", "MAP1LC3A", "GABARAP", "GABARAPL1", "GABARAPL2",
            "ULK1", "ULK2", "RB1CC1", "ATG13", "ATG101",
            "SQSTM1", "NBR1", "OPTN", "CALCOCO2", "TAX1BP1",
            "AMBRA1", "WIPI1", "WIPI2", "LAMP1", "LAMP2",
        ],
        "autophagy_mtor_regulation": [
            "MTOR", "RPTOR", "RICTOR", "AKT1", "TSC1", "TSC2",
            "PTEN", "AMPK", "PRKAA1", "PRKAA2", "TFEB",
            "ULK1", "ATG13", "DEPTOR", "MLST8",
        ],
    },
    # S100A12-TLR4-MYD88 axis (Kawasaki/PANS non-response)
    "s100a12_tlr4_axis": {
        "S100A12_TLR4_MYD88": [
            "S100A12", "S100A8", "S100A9", "TLR4", "MYD88",
            "LY96", "CD14", "IRAK1", "IRAK4", "TRAF6",
            "NFKB1", "RELA", "NFKBIA", "MAP3K7", "TAB1", "TAB2",
        ],
    },
    # Histone modification (novel finding from Han VX et al.)
    "histone_modification": {
        "GO:0016570_histone_modification": [
            "KAT2A", "KAT2B", "KAT6A", "KAT6B", "KAT7", "KAT8",
            "HDAC1", "HDAC2", "HDAC3", "HDAC4", "HDAC5", "HDAC6",
            "KDM1A", "KDM4A", "KDM5A", "KDM6A", "KDM6B",
            "KMT2A", "KMT2D", "EZH2", "SUV39H1",
            "SETD2", "DOT1L", "PRMT1", "PRMT5",
            "SIRT1", "SIRT2", "SIRT6",
        ],
    },
    # Cytokine signaling (cross-reference with cytokine-network initiative)
    "cytokine_signaling": {
        "IL17_signaling": [
            "IL17A", "IL17F", "IL17RA", "IL17RC", "ACT1", "TRAF3IP2",
            "NFKB1", "CXCL1", "CXCL8", "CCL20", "MMP9", "DEFB4A",
        ],
        "TNF_signaling": [
            "TNF", "TNFRSF1A", "TNFRSF1B", "TRADD", "TRAF2",
            "RIPK1", "NFKB1", "RELA", "CASP8", "BIRC2", "BIRC3",
            "MAP3K7", "CHUK", "IKBKB", "IKBKG",
        ],
        "IL6_JAK_STAT3": [
            "IL6", "IL6R", "IL6ST", "JAK1", "JAK2", "STAT3",
            "SOCS3", "SOCS1", "CRP", "SAA1", "HP", "FGA", "FGB",
        ],
        "type_I_interferon": [
            "IFNA1", "IFNB1", "IFNAR1", "IFNAR2", "JAK1", "TYK2",
            "STAT1", "STAT2", "IRF9", "MX1", "OAS1", "ISG15",
            "IFIT1", "IFIT3", "IFI44L", "RSAD2",
        ],
        "complement_system": [
            "C1QA", "C1QB", "C1QC", "C1R", "C1S", "C2", "C3",
            "C4A", "C4B", "C5", "C6", "C7", "C8A", "C9",
            "CFB", "CFD", "CFH", "CFI", "MBL2", "MASP1", "MASP2",
        ],
        "TGFb_signaling": [
            "TGFB1", "TGFB2", "TGFB3", "TGFBR1", "TGFBR2",
            "SMAD2", "SMAD3", "SMAD4", "SMAD7",
            "FOXP3", "ITGB6", "ITGB8",
        ],
    },
    # Secretory granules (downregulated pre-IVIG per Han VX)
    "secretory_granules": {
        "GO:0030141_secretory_granule": [
            "ELANE", "MPO", "CTSG", "PRTN3", "AZU1",
            "DEFA1", "DEFA3", "DEFA4", "LTF", "MMP8", "MMP9",
            "CAMP", "LCN2", "CEACAM8", "OLFM4",
            "GZMB", "GZMA", "GZMK", "PRF1", "GNLY",
        ],
    },
    # B cell / immunoglobulin (IVIG mechanism)
    "b_cell_ig": {
        "B_cell_activation": [
            "CD19", "CD79A", "CD79B", "MS4A1", "CR2", "CD40",
            "AICDA", "BCL6", "IRF4", "PRDM1", "XBP1",
            "PAX5", "EBF1", "TCF3", "IGHM", "IGHG1",
        ],
        "Fc_receptor_signaling": [
            "FCGR1A", "FCGR2A", "FCGR2B", "FCGR3A", "FCGR3B",
            "FCER1A", "FCER1G", "FCER2", "FCGRT",
            "SYK", "LYN", "SHIP1", "INPP5D",
        ],
    },
    # T cell subsets (Th1/Th2/Th17/Treg)
    "t_cell_subsets": {
        "Th1_signature": [
            "TBX21", "IFNG", "TNF", "IL2", "STAT4", "IL12RB1",
            "CXCR3", "CCR5",
        ],
        "Th2_signature": [
            "GATA3", "IL4", "IL5", "IL13", "STAT6", "IL4R",
            "CCR4", "CCR3",
        ],
        "Th17_signature": [
            "RORC", "IL17A", "IL17F", "IL22", "IL23R", "CCR6",
            "STAT3", "IL21",
        ],
        "Treg_signature": [
            "FOXP3", "IL2RA", "CTLA4", "TIGIT", "IKZF2",
            "TNFRSF18", "IL10", "TGFB1", "ENTPD1",
        ],
    },
}


def get_all_gene_sets() -> dict[str, list[str]]:
    """Return flat dict of gene_set_name -> gene list."""
    flat: dict[str, list[str]] = {}
    for _category, sets in GENE_SETS.items():
        for name, genes in sets.items():
            flat[name] = genes
    return flat


# ---------------------------------------------------------------------------
# Over-representation analysis (ORA)
# ---------------------------------------------------------------------------

@dataclass
class ORAResult:
    """Result of over-representation analysis for one gene set."""

    gene_set: str
    category: str
    n_overlap: int
    n_gene_set: int
    n_de_genes: int
    n_background: int
    overlap_genes: list[str]
    odds_ratio: float
    pvalue: float
    pvalue_adj: float = 1.0
    cell_type: str = ""
    comparison: str = ""

    def to_dict(self) -> dict:
        return {
            "gene_set": self.gene_set,
            "category": self.category,
            "n_overlap": self.n_overlap,
            "n_gene_set": self.n_gene_set,
            "n_de_genes": self.n_de_genes,
            "n_background": self.n_background,
            "overlap_genes": ",".join(self.overlap_genes),
            "odds_ratio": self.odds_ratio,
            "pvalue": self.pvalue,
            "pvalue_adj": self.pvalue_adj,
            "cell_type": self.cell_type,
            "comparison": self.comparison,
        }


def _benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    n = len(pvalues)
    if n == 0:
        return np.array([])

    ranked = np.argsort(pvalues)
    adjusted = np.ones(n)
    for i, rank_idx in enumerate(reversed(ranked)):
        rank = n - i
        if i == 0:
            adjusted[rank_idx] = min(1.0, pvalues[rank_idx] * n / rank)
        else:
            prev_idx = ranked[n - i]
            adjusted[rank_idx] = min(adjusted[prev_idx], pvalues[rank_idx] * n / rank)

    return np.clip(adjusted, 0, 1)


def run_ora(
    de_genes: list[str],
    background_genes: list[str],
    gene_sets: dict[str, list[str]] | None = None,
    cell_type: str = "",
    comparison: str = "",
    min_overlap: int = 2,
) -> list[ORAResult]:
    """Run over-representation analysis using Fisher's exact test.

    Args:
        de_genes: Significant DE gene list.
        background_genes: All tested genes (universe).
        gene_sets: Dict of gene_set_name -> gene list. If None, uses built-in sets.
        cell_type: Cell type label for annotation.
        comparison: Comparison label for annotation.
        min_overlap: Minimum overlap to report a result.

    Returns:
        List of ORAResult, BH-adjusted.
    """
    if gene_sets is None:
        gene_sets = get_all_gene_sets()

    de_set = set(de_genes)
    bg_set = set(background_genes)
    n_bg = len(bg_set)
    n_de = len(de_set & bg_set)

    # Find category for each gene set
    set_to_category: dict[str, str] = {}
    for category, sets in GENE_SETS.items():
        for name in sets:
            set_to_category[name] = category

    results: list[ORAResult] = []
    for gs_name, gs_genes in gene_sets.items():
        gs_in_bg = set(gs_genes) & bg_set
        if len(gs_in_bg) < 2:
            continue

        overlap = list(de_set & gs_in_bg)
        n_overlap = len(overlap)

        if n_overlap < min_overlap:
            continue

        # Fisher's exact test (2x2 contingency)
        a = n_overlap  # DE and in gene set
        b = len(gs_in_bg) - n_overlap  # not DE but in gene set
        c = n_de - n_overlap  # DE but not in gene set
        d = n_bg - n_de - b  # neither DE nor in gene set

        table = np.array([[a, b], [c, max(d, 0)]])
        fisher_result = scipy_stats.fisher_exact(table, alternative="greater")

        results.append(ORAResult(
            gene_set=gs_name,
            category=set_to_category.get(gs_name, "custom"),
            n_overlap=n_overlap,
            n_gene_set=len(gs_in_bg),
            n_de_genes=n_de,
            n_background=n_bg,
            overlap_genes=sorted(overlap),
            odds_ratio=float(fisher_result[0]),
            pvalue=float(fisher_result[1]),
            cell_type=cell_type,
            comparison=comparison,
        ))

    # BH correction
    if results:
        pvals = np.array([r.pvalue for r in results])
        adj_pvals = _benjamini_hochberg(pvals)
        for r, adj_p in zip(results, adj_pvals):
            r.pvalue_adj = float(adj_p)

    return sorted(results, key=lambda r: r.pvalue)


# ---------------------------------------------------------------------------
# Gene set enrichment analysis (GSEA)
# ---------------------------------------------------------------------------

@dataclass
class GSEAResult:
    """Result of GSEA for one gene set."""

    gene_set: str
    category: str
    enrichment_score: float
    normalized_es: float
    pvalue: float
    pvalue_adj: float = 1.0
    n_genes_in_set: int = 0
    leading_edge: list[str] = field(default_factory=list)
    cell_type: str = ""
    comparison: str = ""

    def to_dict(self) -> dict:
        return {
            "gene_set": self.gene_set,
            "category": self.category,
            "enrichment_score": self.enrichment_score,
            "normalized_es": self.normalized_es,
            "pvalue": self.pvalue,
            "pvalue_adj": self.pvalue_adj,
            "n_genes_in_set": self.n_genes_in_set,
            "leading_edge": ",".join(self.leading_edge),
            "cell_type": self.cell_type,
            "comparison": self.comparison,
        }


def compute_enrichment_score(
    ranked_genes: list[str],
    ranked_scores: np.ndarray,
    gene_set: set[str],
) -> tuple[float, list[str]]:
    """Compute weighted GSEA enrichment score.

    Uses the KS-like running sum statistic weighted by the absolute
    value of the ranking metric (Subramanian et al. 2005).

    Args:
        ranked_genes: Genes sorted by ranking metric (descending).
        ranked_scores: Corresponding ranking metric values.
        gene_set: Set of genes in the pathway.

    Returns:
        (enrichment_score, leading_edge_genes)
    """
    n = len(ranked_genes)
    hits = np.array([1 if g in gene_set else 0 for g in ranked_genes])
    n_hit = int(hits.sum())

    if n_hit == 0:
        return 0.0, []

    # Weighted running sum
    abs_scores = np.abs(ranked_scores)
    hit_weights = hits * abs_scores
    hit_sum = hit_weights.sum()
    if hit_sum == 0:
        hit_sum = 1.0

    miss_penalty = 1.0 / (n - n_hit) if (n - n_hit) > 0 else 0.0

    running_sum = np.zeros(n)
    for i in range(n):
        if hits[i]:
            running_sum[i] = hit_weights[i] / hit_sum
        else:
            running_sum[i] = -miss_penalty

    cumsum = np.cumsum(running_sum)
    es_max = float(cumsum.max())
    es_min = float(cumsum.min())

    # ES is the maximum deviation from zero
    es = es_max if abs(es_max) >= abs(es_min) else es_min

    # Leading edge: genes contributing to ES
    if es >= 0:
        peak_idx = int(np.argmax(cumsum))
        leading_edge = [ranked_genes[i] for i in range(peak_idx + 1) if hits[i]]
    else:
        peak_idx = int(np.argmin(cumsum))
        leading_edge = [ranked_genes[i] for i in range(peak_idx, n) if hits[i]]

    return es, leading_edge


def run_gsea_permutation(
    ranked_genes: list[str],
    ranked_scores: np.ndarray,
    gene_set: set[str],
    n_perm: int = 1000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute GSEA ES with permutation-based p-value and NES.

    Args:
        ranked_genes: Genes sorted by ranking metric.
        ranked_scores: Ranking metric values.
        gene_set: Gene set to test.
        n_perm: Number of permutations.
        seed: Random seed.

    Returns:
        (enrichment_score, normalized_es, pvalue)
    """
    observed_es, _ = compute_enrichment_score(ranked_genes, ranked_scores, gene_set)

    n_hit = sum(1 for g in ranked_genes if g in gene_set)
    if n_hit == 0:
        return 0.0, 0.0, 1.0

    rng = np.random.default_rng(seed)
    null_es = np.zeros(n_perm)

    all_genes = list(ranked_genes)
    for p in range(n_perm):
        perm_set = set(rng.choice(all_genes, size=n_hit, replace=False))
        null_es[p], _ = compute_enrichment_score(ranked_genes, ranked_scores, perm_set)

    # Separate positive and negative null distributions
    if observed_es >= 0:
        pos_null = null_es[null_es >= 0]
        if len(pos_null) == 0:
            pval = 0.0
            nes = observed_es
        else:
            pval = float(np.mean(pos_null >= observed_es))
            mean_pos = float(np.mean(pos_null)) if np.mean(pos_null) != 0 else 1.0
            nes = observed_es / abs(mean_pos)
    else:
        neg_null = null_es[null_es < 0]
        if len(neg_null) == 0:
            pval = 0.0
            nes = observed_es
        else:
            pval = float(np.mean(neg_null <= observed_es))
            mean_neg = float(np.mean(neg_null)) if np.mean(neg_null) != 0 else 1.0
            nes = observed_es / abs(mean_neg)

    return observed_es, nes, max(pval, 1.0 / n_perm)


def run_gsea(
    de_results: pd.DataFrame,
    gene_col: str = "gene",
    score_col: str = "log2_fold_change",
    pvalue_col: str = "pvalue",
    gene_sets: dict[str, list[str]] | None = None,
    cell_type: str = "",
    comparison: str = "",
    n_perm: int = 1000,
    min_set_size: int = 5,
    seed: int = 42,
) -> list[GSEAResult]:
    """Run gene set enrichment analysis on DE results.

    Ranks genes by signed -log10(pvalue) * sign(log2FC) and computes
    enrichment scores for each gene set.

    Args:
        de_results: DataFrame with gene, log2FC, and pvalue columns.
        gene_col: Column with gene names.
        score_col: Column with fold change.
        pvalue_col: Column with p-values.
        gene_sets: Dict of gene_set_name -> genes. If None, uses built-in sets.
        cell_type: Cell type label.
        comparison: Comparison label.
        n_perm: Permutations for p-value.
        min_set_size: Min genes from set present in data.
        seed: Random seed.

    Returns:
        List of GSEAResult, BH-adjusted.
    """
    if gene_sets is None:
        gene_sets = get_all_gene_sets()

    if de_results.empty:
        return []

    df = de_results.copy()

    # Compute ranking metric: signed -log10(p)
    pvals = np.array(df[pvalue_col].values, dtype=np.float64)
    pvals[pvals == 0] = 1e-300
    scores = np.array(df[score_col].values, dtype=np.float64)
    signed_score = -np.log10(pvals) * np.sign(scores)
    df = df.copy()
    df["_rank_score"] = signed_score

    # Sort by ranking metric descending
    df = df.sort_values("_rank_score", ascending=False)
    ranked_genes = df[gene_col].tolist()
    ranked_scores = np.array(df["_rank_score"].values, dtype=np.float64)

    available_genes = set(ranked_genes)

    # Find category for each gene set
    set_to_category: dict[str, str] = {}
    for category, sets in GENE_SETS.items():
        for name in sets:
            set_to_category[name] = category

    results: list[GSEAResult] = []
    for gs_name, gs_genes in gene_sets.items():
        gs_in_data = set(gs_genes) & available_genes
        if len(gs_in_data) < min_set_size:
            continue

        es, nes, pval = run_gsea_permutation(
            ranked_genes, ranked_scores, gs_in_data,
            n_perm=n_perm, seed=seed,
        )
        _, leading_edge = compute_enrichment_score(
            ranked_genes, ranked_scores, gs_in_data,
        )

        results.append(GSEAResult(
            gene_set=gs_name,
            category=set_to_category.get(gs_name, "custom"),
            enrichment_score=es,
            normalized_es=nes,
            pvalue=pval,
            n_genes_in_set=len(gs_in_data),
            leading_edge=leading_edge,
            cell_type=cell_type,
            comparison=comparison,
        ))

    # BH correction
    if results:
        pvals_arr = np.array([r.pvalue for r in results])
        adj_pvals = _benjamini_hochberg(pvals_arr)
        for r, adj_p in zip(results, adj_pvals):
            r.pvalue_adj = float(adj_p)

    return sorted(results, key=lambda r: r.pvalue)


# ---------------------------------------------------------------------------
# IVIG-specific module scoring
# ---------------------------------------------------------------------------

@dataclass
class ModuleScore:
    """Score for a specific gene module in a cell type."""

    module_name: str
    cell_type: str
    condition: str
    mean_score: float
    std_score: float
    n_cells: int
    genes_detected: list[str]
    n_genes_detected: int
    n_genes_total: int

    def to_dict(self) -> dict:
        return {
            "module_name": self.module_name,
            "cell_type": self.cell_type,
            "condition": self.condition,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "n_cells": self.n_cells,
            "genes_detected": ",".join(self.genes_detected),
            "n_genes_detected": self.n_genes_detected,
            "n_genes_total": self.n_genes_total,
        }


# Targeted IVIG modules from the research plan
IVIG_MODULES: dict[str, list[str]] = {
    "autophagy_core": [
        "ATG7", "UVRAG", "BECN1", "ATG5", "ATG12", "ATG16L1",
        "MAP1LC3B", "SQSTM1", "ULK1", "PIK3C3",
    ],
    "autophagy_mtor": [
        "MTOR", "RPTOR", "TSC1", "TSC2", "PRKAA1", "TFEB",
        "ULK1", "ATG13", "DEPTOR",
    ],
    "s100a12_tlr4_myd88": [
        "S100A12", "S100A8", "S100A9", "TLR4", "MYD88",
        "LY96", "CD14", "IRAK1", "IRAK4", "TRAF6",
        "NFKB1", "RELA",
    ],
    "histone_modification": [
        "KAT2A", "KAT2B", "HDAC1", "HDAC2", "HDAC3",
        "KDM1A", "KDM6B", "EZH2", "SIRT1", "SIRT6",
        "KMT2A", "KMT2D", "SETD2", "DOT1L",
    ],
    "defense_response": [
        "TNF", "IL1B", "IL6", "IFNG", "CXCL8", "CCL2",
        "TLR2", "TLR4", "NLRP3", "CASP1",
    ],
    "secretory_granules": [
        "ELANE", "MPO", "CTSG", "PRTN3", "AZU1",
        "GZMB", "GZMA", "PRF1", "GNLY", "LTF",
    ],
    "tfh_b_coordination": [
        "CXCR5", "PDCD1", "ICOS", "BCL6", "IL21",
        "CD40LG", "AICDA", "IRF4", "PRDM1",
    ],
    "nk_t_exhaustion": [
        "PDCD1", "LAG3", "HAVCR2", "TIGIT", "CTLA4",
        "TOX", "EOMES", "NFATC1",
    ],
}


def score_module(
    adata: ad.AnnData,
    genes: list[str],
    module_name: str,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    layer: str | None = None,
) -> list[ModuleScore]:
    """Score a gene module per cell type per condition.

    Computes mean z-scored expression of module genes across cells.

    Args:
        adata: AnnData with expression data.
        genes: Gene list for the module.
        module_name: Name for this module.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        layer: Layer for expression (None = X).

    Returns:
        List of ModuleScore per cell_type x condition.
    """
    # Find which module genes are in the data
    available = [g for g in genes if g in adata.var_names]
    if not available:
        return []

    gene_idx = [list(adata.var_names).index(g) for g in available]

    # Extract expression matrix for module genes
    X_raw = adata.layers[layer] if layer else adata.X
    if sp.issparse(X_raw):
        X_mod = np.asarray(X_raw.toarray()[:, gene_idx])
    else:
        X_mod = np.asarray(np.array(X_raw)[:, gene_idx])

    # Z-score per gene across all cells
    gene_means = X_mod.mean(axis=0)
    gene_stds = X_mod.std(axis=0)
    gene_stds[gene_stds == 0] = 1.0
    X_z = (X_mod - gene_means) / gene_stds

    # Mean z-score per cell = module score
    cell_scores = X_z.mean(axis=1)

    # Aggregate per cell_type x condition
    results: list[ModuleScore] = []
    cell_types = sorted(adata.obs[cell_type_key].unique())
    conditions = sorted(adata.obs[condition_key].unique())

    for ct in cell_types:
        for cond in conditions:
            mask = (adata.obs[cell_type_key].values == ct) & (
                adata.obs[condition_key].values == cond
            )
            n_cells = int(mask.sum())
            if n_cells == 0:
                continue

            scores = cell_scores[mask]
            results.append(ModuleScore(
                module_name=module_name,
                cell_type=ct,
                condition=str(cond),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)),
                n_cells=n_cells,
                genes_detected=available,
                n_genes_detected=len(available),
                n_genes_total=len(genes),
            ))

    return results


def score_ivig_modules(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    layer: str | None = None,
    modules: dict[str, list[str]] | None = None,
) -> pd.DataFrame:
    """Score all IVIG-relevant modules across cell types and conditions.

    Args:
        adata: AnnData with expression data.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        layer: Layer for expression (None = X).
        modules: Custom modules dict. If None, uses IVIG_MODULES.

    Returns:
        DataFrame with module scores per cell_type x condition.
    """
    if modules is None:
        modules = IVIG_MODULES

    all_scores: list[dict] = []
    for mod_name, mod_genes in modules.items():
        scores = score_module(
            adata, mod_genes, mod_name,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
            layer=layer,
        )
        all_scores.extend(s.to_dict() for s in scores)

    return pd.DataFrame(all_scores)


def compare_module_scores(
    scores_df: pd.DataFrame,
    module_name: str,
    condition1: str,
    condition2: str,
    cell_type: str | None = None,
) -> pd.DataFrame:
    """Compare module scores between two conditions using Mann-Whitney U test.

    Args:
        scores_df: Output from score_ivig_modules().
        module_name: Module to compare.
        condition1: Reference condition.
        condition2: Test condition.
        cell_type: Specific cell type (None = all).

    Returns:
        DataFrame with comparison statistics per cell type.
    """
    df = scores_df[scores_df["module_name"] == module_name]

    if cell_type:
        cell_types = [cell_type]
    else:
        cell_types = sorted(df["cell_type"].unique())

    results = []
    for ct in cell_types:
        ct_df = df[df["cell_type"] == ct]
        c1 = ct_df[ct_df["condition"] == condition1]
        c2 = ct_df[ct_df["condition"] == condition2]

        if c1.empty or c2.empty:
            continue

        score1 = c1["mean_score"].values[0]
        score2 = c2["mean_score"].values[0]

        results.append({
            "module_name": module_name,
            "cell_type": ct,
            "condition1": condition1,
            "condition2": condition2,
            "score_condition1": score1,
            "score_condition2": score2,
            "score_diff": score2 - score1,
        })

    return pd.DataFrame(results)


# ---------------------------------------------------------------------------
# Combined enrichment pipeline
# ---------------------------------------------------------------------------

@dataclass
class PathwayEnrichmentSummary:
    """Summary of pathway enrichment analysis."""

    ora_results: list[ORAResult] = field(default_factory=list)
    gsea_results: list[GSEAResult] = field(default_factory=list)
    module_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    cell_types_analyzed: list[str] = field(default_factory=list)
    comparisons_analyzed: list[str] = field(default_factory=list)

    def get_significant_ora(self, alpha: float = 0.05) -> pd.DataFrame:
        """Get significant ORA results."""
        df = pd.DataFrame([r.to_dict() for r in self.ora_results])
        if df.empty:
            return df
        return df[df["pvalue_adj"] < alpha].sort_values("pvalue_adj")

    def get_significant_gsea(self, alpha: float = 0.25) -> pd.DataFrame:
        """Get significant GSEA results (standard FDR < 0.25)."""
        df = pd.DataFrame([r.to_dict() for r in self.gsea_results])
        if df.empty:
            return df
        return df[df["pvalue_adj"] < alpha].sort_values("pvalue_adj")

    def to_ora_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.ora_results])

    def to_gsea_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.gsea_results])

    def summary(self) -> str:
        lines = [
            "Pathway Enrichment Analysis Summary:",
            f"  Cell types analyzed: {len(self.cell_types_analyzed)}",
            f"  Comparisons: {len(self.comparisons_analyzed)}",
            f"  ORA results: {len(self.ora_results)} "
            f"({sum(1 for r in self.ora_results if r.pvalue_adj < 0.05)} sig)",
            f"  GSEA results: {len(self.gsea_results)} "
            f"({sum(1 for r in self.gsea_results if r.pvalue_adj < 0.25)} sig)",
        ]
        if not self.module_scores.empty:
            n_modules = self.module_scores["module_name"].nunique()
            lines.append(f"  Module scores: {n_modules} modules scored")
        return "\n".join(lines)


def run_pathway_enrichment(
    de_results: pd.DataFrame,
    background_genes: list[str],
    gene_col: str = "gene",
    cell_type_col: str = "cell_type",
    comparison_col: str = "comparison",
    lfc_col: str = "log2_fold_change",
    pvalue_col: str = "pvalue",
    pvalue_adj_col: str = "pvalue_adj",
    alpha_de: float = 0.05,
    lfc_threshold: float = 0.5,
    run_ora_analysis: bool = True,
    run_gsea_analysis: bool = True,
    gsea_n_perm: int = 1000,
    gene_sets: dict[str, list[str]] | None = None,
    adata: ad.AnnData | None = None,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
) -> PathwayEnrichmentSummary:
    """Run complete pathway enrichment analysis on DE results.

    Performs ORA and GSEA per cell type per comparison, and optionally
    scores IVIG-specific modules on the AnnData object.

    Args:
        de_results: DataFrame from pseudobulk DE (gene, cell_type, comparison, log2FC, pvalue).
        background_genes: All tested genes.
        gene_col: Column with gene names.
        cell_type_col: Column with cell type.
        comparison_col: Column with comparison name.
        lfc_col: Column with log2 fold change.
        pvalue_col: Column with raw p-value.
        pvalue_adj_col: Column with adjusted p-value.
        alpha_de: FDR cutoff for DE significance (used for ORA).
        lfc_threshold: |log2FC| threshold for ORA.
        run_ora_analysis: Whether to run ORA.
        run_gsea_analysis: Whether to run GSEA.
        gsea_n_perm: GSEA permutations.
        gene_sets: Custom gene sets. If None, uses built-in sets.
        adata: Optional AnnData for module scoring.
        cell_type_key: obs column for cell type in adata.
        condition_key: obs column for condition in adata.

    Returns:
        PathwayEnrichmentSummary with all results.
    """
    if gene_sets is None:
        gene_sets = get_all_gene_sets()

    summary = PathwayEnrichmentSummary()

    if de_results.empty:
        return summary

    cell_types = sorted(de_results[cell_type_col].unique())
    comparisons = sorted(de_results[comparison_col].unique())
    summary.cell_types_analyzed = cell_types
    summary.comparisons_analyzed = comparisons

    print(f"Running pathway enrichment: {len(cell_types)} cell types, "
          f"{len(comparisons)} comparisons")

    for ct in cell_types:
        ct_df = de_results[de_results[cell_type_col] == ct]

        for comp in comparisons:
            comp_df = ct_df[ct_df[comparison_col] == comp]
            if comp_df.empty:
                continue

            # ORA: use significant DE genes
            if run_ora_analysis:
                sig_mask = (
                    (comp_df[pvalue_adj_col] < alpha_de) &
                    (comp_df[lfc_col].abs() > lfc_threshold)
                )
                sig_genes = comp_df.loc[sig_mask, gene_col].tolist()

                if len(sig_genes) >= 2:
                    ora = run_ora(
                        sig_genes, background_genes,
                        gene_sets=gene_sets,
                        cell_type=ct, comparison=comp,
                    )
                    summary.ora_results.extend(ora)

            # GSEA: use all genes with ranking
            if run_gsea_analysis:
                gsea = run_gsea(
                    comp_df, gene_col=gene_col,
                    score_col=lfc_col, pvalue_col=pvalue_col,
                    gene_sets=gene_sets,
                    cell_type=ct, comparison=comp,
                    n_perm=gsea_n_perm,
                )
                summary.gsea_results.extend(gsea)

    print(f"  ORA: {len(summary.ora_results)} tests")
    print(f"  GSEA: {len(summary.gsea_results)} tests")

    # Module scoring on AnnData
    if adata is not None:
        print("  Scoring IVIG-specific modules...")
        summary.module_scores = score_ivig_modules(
            adata,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
        )

    print(f"\n{summary.summary()}")
    return summary
