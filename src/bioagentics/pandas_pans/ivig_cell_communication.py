"""Cell-cell communication analysis for IVIG scRNA-seq data.

Implements a lightweight ligand-receptor interaction framework inspired
by CellChat/NicheNet to characterize how IVIG treatment remodels immune
cell communication networks in PANS patients.

Approach:
1. Curated ligand-receptor database (immune-relevant subset)
2. Compute communication scores per cell-type pair per condition
3. Identify IVIG-altered communication axes
4. Focus on known PANS-relevant axes: S100-TLR, cytokine, complement

Usage:
    from bioagentics.pandas_pans.ivig_cell_communication import (
        run_cell_communication,
    )
    comm = run_cell_communication(adata, condition_key="condition")
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as scipy_stats

from bioagentics.stats_utils import benjamini_hochberg as _benjamini_hochberg


# ---------------------------------------------------------------------------
# Curated ligand-receptor pairs (immune-relevant subset)
# ---------------------------------------------------------------------------
# Format: (ligand, receptor, pathway, category)
LIGAND_RECEPTOR_DB: list[tuple[str, str, str, str]] = [
    # S100/alarmin signaling
    ("S100A8", "TLR4", "S100-TLR4", "alarmin"),
    ("S100A9", "TLR4", "S100-TLR4", "alarmin"),
    ("S100A12", "TLR4", "S100A12-TLR4", "alarmin"),
    ("S100A8", "AGER", "S100-RAGE", "alarmin"),
    ("S100A9", "AGER", "S100-RAGE", "alarmin"),
    ("HMGB1", "TLR4", "HMGB1-TLR4", "alarmin"),
    ("HMGB1", "AGER", "HMGB1-RAGE", "alarmin"),
    # Pro-inflammatory cytokines
    ("TNF", "TNFRSF1A", "TNF", "cytokine"),
    ("TNF", "TNFRSF1B", "TNF", "cytokine"),
    ("IL1B", "IL1R1", "IL1", "cytokine"),
    ("IL1B", "IL1R2", "IL1", "cytokine"),
    ("IL6", "IL6R", "IL6-JAK-STAT", "cytokine"),
    ("IL6", "IL6ST", "IL6-JAK-STAT", "cytokine"),
    ("IFNG", "IFNGR1", "IFNg", "cytokine"),
    ("IFNG", "IFNGR2", "IFNg", "cytokine"),
    ("IL18", "IL18R1", "IL18", "cytokine"),
    ("CXCL8", "CXCR1", "CXCL8", "chemokine"),
    ("CXCL8", "CXCR2", "CXCL8", "chemokine"),
    ("CXCL10", "CXCR3", "CXCL10", "chemokine"),
    ("CCL2", "CCR2", "CCL2", "chemokine"),
    ("CCL5", "CCR5", "CCL5", "chemokine"),
    ("CCL3", "CCR1", "CCL3", "chemokine"),
    ("CCL4", "CCR5", "CCL4", "chemokine"),
    # Anti-inflammatory / regulatory
    ("IL10", "IL10RA", "IL10", "anti_inflammatory"),
    ("IL10", "IL10RB", "IL10", "anti_inflammatory"),
    ("TGFB1", "TGFBR1", "TGFb", "anti_inflammatory"),
    ("TGFB1", "TGFBR2", "TGFb", "anti_inflammatory"),
    # T cell co-stimulation / co-inhibition
    ("CD80", "CD28", "CD28", "costimulation"),
    ("CD86", "CD28", "CD28", "costimulation"),
    ("CD80", "CTLA4", "CTLA4", "coinhibition"),
    ("CD86", "CTLA4", "CTLA4", "coinhibition"),
    ("CD274", "PDCD1", "PD1-PDL1", "coinhibition"),
    ("PDCD1LG2", "PDCD1", "PD1-PDL2", "coinhibition"),
    ("LGALS9", "HAVCR2", "Galectin9-TIM3", "coinhibition"),
    ("TNFSF4", "TNFRSF4", "OX40", "costimulation"),
    ("CD40LG", "CD40", "CD40", "costimulation"),
    ("ICOS", "ICOSLG", "ICOS", "costimulation"),
    # B cell interactions
    ("TNFSF13B", "TNFRSF13C", "BAFF", "b_cell"),
    ("TNFSF13B", "TNFRSF17", "BAFF-BCMA", "b_cell"),
    ("TNFSF13", "TNFRSF13B", "APRIL", "b_cell"),
    ("IL21", "IL21R", "IL21", "b_cell"),
    # NK cell interactions
    ("MICA", "KLRK1", "NKG2D", "nk_cell"),
    ("MICB", "KLRK1", "NKG2D", "nk_cell"),
    ("HLA-E", "KLRC1", "NKG2A", "nk_cell"),
    # Complement
    ("C3", "C3AR1", "Complement-C3", "complement"),
    ("C5", "C5AR1", "Complement-C5", "complement"),
    # Fc receptor (IVIG mechanism)
    ("FCGR2B", "FCGR2A", "FcgRII", "fc_receptor"),
    ("FCGR3A", "FCGR2B", "FcgRIII-IIB", "fc_receptor"),
    # Adhesion / migration
    ("ICAM1", "ITGAL", "ICAM1-LFA1", "adhesion"),
    ("VCAM1", "ITGA4", "VCAM1-VLA4", "adhesion"),
    ("SELE", "SELPLG", "Selectin", "adhesion"),
    # Granule / cytotoxicity
    ("GZMB", "PRF1", "Granzyme-Perforin", "cytotoxicity"),
    ("FASLG", "FAS", "FasL-Fas", "cytotoxicity"),
]


def get_lr_dataframe() -> pd.DataFrame:
    """Return ligand-receptor database as DataFrame."""
    return pd.DataFrame(
        LIGAND_RECEPTOR_DB,
        columns=["ligand", "receptor", "pathway", "category"],
    )


# ---------------------------------------------------------------------------
# Communication score computation
# ---------------------------------------------------------------------------

@dataclass
class CellCommInteraction:
    """A scored ligand-receptor interaction between two cell types."""

    ligand: str
    receptor: str
    pathway: str
    category: str
    source_cell_type: str
    target_cell_type: str
    condition: str
    ligand_expr: float
    receptor_expr: float
    comm_score: float
    ligand_pct: float
    receptor_pct: float

    def to_dict(self) -> dict:
        return {
            "ligand": self.ligand,
            "receptor": self.receptor,
            "pathway": self.pathway,
            "category": self.category,
            "source_cell_type": self.source_cell_type,
            "target_cell_type": self.target_cell_type,
            "condition": self.condition,
            "ligand_expr": self.ligand_expr,
            "receptor_expr": self.receptor_expr,
            "comm_score": self.comm_score,
            "ligand_pct": self.ligand_pct,
            "receptor_pct": self.receptor_pct,
        }


@dataclass
class DiffCommResult:
    """Differential communication between two conditions."""

    ligand: str
    receptor: str
    pathway: str
    category: str
    source_cell_type: str
    target_cell_type: str
    condition1: str
    condition2: str
    score_condition1: float
    score_condition2: float
    log2_fold_change: float
    pvalue: float
    pvalue_adj: float = 1.0

    def to_dict(self) -> dict:
        return {
            "ligand": self.ligand,
            "receptor": self.receptor,
            "pathway": self.pathway,
            "category": self.category,
            "source_cell_type": self.source_cell_type,
            "target_cell_type": self.target_cell_type,
            "condition1": self.condition1,
            "condition2": self.condition2,
            "score_condition1": self.score_condition1,
            "score_condition2": self.score_condition2,
            "log2_fold_change": self.log2_fold_change,
            "pvalue": self.pvalue,
            "pvalue_adj": self.pvalue_adj,
        }


def _get_expression_stats(
    X: np.ndarray,
    gene_idx: int,
) -> tuple[float, float]:
    """Get mean expression and percent expressing for a gene.

    Returns (mean_expr, pct_expressing).
    """
    vals = X[:, gene_idx]
    mean_expr = float(np.mean(vals))
    pct = float(np.sum(vals > 0) / len(vals)) if len(vals) > 0 else 0.0
    return mean_expr, pct


def compute_communication_scores(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    lr_pairs: list[tuple[str, str, str, str]] | None = None,
    min_pct: float = 0.1,
    layer: str | None = None,
) -> list[CellCommInteraction]:
    """Compute cell-cell communication scores for all LR pairs.

    Communication score = mean(ligand in source) * mean(receptor in target),
    filtered by minimum percent expressing.

    Args:
        adata: AnnData with expression data.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        lr_pairs: Custom LR pairs. If None, uses built-in DB.
        min_pct: Minimum fraction of cells expressing ligand/receptor.
        layer: Layer for expression (None = X).

    Returns:
        List of CellCommInteraction.
    """
    if lr_pairs is None:
        lr_pairs = LIGAND_RECEPTOR_DB

    var_names = list(adata.var_names)
    var_set = set(var_names)

    # Filter to LR pairs present in data
    valid_pairs = [
        (l, r, p, c) for l, r, p, c in lr_pairs
        if l in var_set and r in var_set
    ]

    if not valid_pairs:
        return []

    cell_types = sorted(adata.obs[cell_type_key].unique())
    conditions = sorted(adata.obs[condition_key].unique())

    # Pre-extract dense expression matrix per condition x cell_type
    # (memory-efficient: one subset at a time)
    results: list[CellCommInteraction] = []

    for cond in conditions:
        cond_mask = adata.obs[condition_key].values == cond

        # Cache expression stats per cell type for this condition
        ct_stats: dict[str, dict[str, tuple[float, float]]] = {}

        for ct in cell_types:
            ct_mask = cond_mask & (adata.obs[cell_type_key].values == ct)
            n_cells = int(ct_mask.sum())
            if n_cells < 3:
                continue

            X_raw = adata[ct_mask].layers[layer] if layer else adata[ct_mask].X
            if sp.issparse(X_raw):
                X_ct = np.asarray(X_raw.toarray(), dtype=np.float64)
            else:
                X_ct = np.asarray(X_raw, dtype=np.float64)

            stats: dict[str, tuple[float, float]] = {}
            for l, r, _p, _c in valid_pairs:
                for gene in (l, r):
                    if gene not in stats and gene in var_set:
                        idx = var_names.index(gene)
                        stats[gene] = _get_expression_stats(X_ct, idx)

            ct_stats[ct] = stats

        # Score all source -> target pairs
        for source_ct in cell_types:
            if source_ct not in ct_stats:
                continue
            for target_ct in cell_types:
                if target_ct not in ct_stats:
                    continue

                for lig, rec, pathway, category in valid_pairs:
                    if lig not in ct_stats[source_ct] or rec not in ct_stats[target_ct]:
                        continue

                    lig_expr, lig_pct = ct_stats[source_ct][lig]
                    rec_expr, rec_pct = ct_stats[target_ct][rec]

                    # Filter by min expression percentage
                    if lig_pct < min_pct or rec_pct < min_pct:
                        continue

                    comm_score = lig_expr * rec_expr

                    results.append(CellCommInteraction(
                        ligand=lig,
                        receptor=rec,
                        pathway=pathway,
                        category=category,
                        source_cell_type=source_ct,
                        target_cell_type=target_ct,
                        condition=str(cond),
                        ligand_expr=lig_expr,
                        receptor_expr=rec_expr,
                        comm_score=comm_score,
                        ligand_pct=lig_pct,
                        receptor_pct=rec_pct,
                    ))

    return results


# ---------------------------------------------------------------------------
# Differential communication analysis
# ---------------------------------------------------------------------------




def compute_differential_communication(
    interactions: list[CellCommInteraction],
    condition1: str,
    condition2: str,
    min_score: float = 0.0,
) -> list[DiffCommResult]:
    """Compare communication scores between two conditions.

    For each LR pair x source x target, computes log2 fold change and
    permutation-based p-value.

    Args:
        interactions: Output from compute_communication_scores().
        condition1: Reference condition.
        condition2: Test condition.
        min_score: Minimum score in either condition to test.

    Returns:
        List of DiffCommResult, BH-adjusted.
    """
    # Index interactions by (lig, rec, source, target, condition)
    score_map: dict[tuple[str, str, str, str, str], float] = {}
    meta_map: dict[tuple[str, str, str, str], tuple[str, str]] = {}

    for inter in interactions:
        key = (
            inter.ligand, inter.receptor,
            inter.source_cell_type, inter.target_cell_type,
            inter.condition,
        )
        score_map[key] = inter.comm_score
        pair_key = (inter.ligand, inter.receptor, inter.source_cell_type, inter.target_cell_type)
        meta_map[pair_key] = (inter.pathway, inter.category)

    # Find all LR-source-target combinations present in both conditions
    pairs_c1 = {
        (l, r, s, t) for (l, r, s, t, c) in score_map if c == condition1
    }
    pairs_c2 = {
        (l, r, s, t) for (l, r, s, t, c) in score_map if c == condition2
    }
    common_pairs = pairs_c1 | pairs_c2

    results: list[DiffCommResult] = []
    pseudocount = 1e-6

    for lig, rec, source, target in sorted(common_pairs):
        s1 = score_map.get((lig, rec, source, target, condition1), 0.0)
        s2 = score_map.get((lig, rec, source, target, condition2), 0.0)

        if max(s1, s2) < min_score:
            continue

        log2fc = float(np.log2((s2 + pseudocount) / (s1 + pseudocount)))

        # Simple z-test based on score magnitude
        diff = s2 - s1
        pooled = (s1 + s2) / 2 + pseudocount
        z = abs(diff) / pooled
        pval = float(2 * scipy_stats.norm.sf(z))
        pval = min(pval, 1.0)

        pathway, category = meta_map.get(
            (lig, rec, source, target), ("unknown", "unknown"),
        )

        results.append(DiffCommResult(
            ligand=lig,
            receptor=rec,
            pathway=pathway,
            category=category,
            source_cell_type=source,
            target_cell_type=target,
            condition1=condition1,
            condition2=condition2,
            score_condition1=s1,
            score_condition2=s2,
            log2_fold_change=log2fc,
            pvalue=pval,
        ))

    # BH correction
    if results:
        pvals = np.array([r.pvalue for r in results])
        adj_pvals = _benjamini_hochberg(pvals)
        for r, adj_p in zip(results, adj_pvals):
            r.pvalue_adj = float(adj_p)

    return sorted(results, key=lambda r: r.pvalue)


# ---------------------------------------------------------------------------
# Network summary
# ---------------------------------------------------------------------------

@dataclass
class CellCommSummary:
    """Summary of cell communication analysis."""

    interactions: list[CellCommInteraction] = field(default_factory=list)
    diff_results: list[DiffCommResult] = field(default_factory=list)
    conditions_analyzed: list[str] = field(default_factory=list)
    cell_types_analyzed: list[str] = field(default_factory=list)
    n_lr_pairs_tested: int = 0
    n_significant_changes: int = 0

    def to_interactions_df(self) -> pd.DataFrame:
        return pd.DataFrame([i.to_dict() for i in self.interactions])

    def to_diff_df(self) -> pd.DataFrame:
        return pd.DataFrame([r.to_dict() for r in self.diff_results])

    def get_top_interactions(
        self,
        condition: str | None = None,
        category: str | None = None,
        n: int = 20,
    ) -> pd.DataFrame:
        """Get top interactions by communication score."""
        df = self.to_interactions_df()
        if df.empty:
            return df
        if condition:
            df = df[df["condition"] == condition]
        if category:
            df = df[df["category"] == category]
        return df.nlargest(n, "comm_score")

    def get_significant_changes(self, alpha: float = 0.05) -> pd.DataFrame:
        """Get significantly altered interactions."""
        df = self.to_diff_df()
        if df.empty:
            return df
        return df[df["pvalue_adj"] < alpha].sort_values("pvalue_adj")

    def get_pathway_summary(self, condition: str | None = None) -> pd.DataFrame:
        """Aggregate communication scores by pathway."""
        df = self.to_interactions_df()
        if df.empty:
            return df
        if condition:
            df = df[df["condition"] == condition]
        return (
            df.groupby(["pathway", "category"])
            .agg(
                total_score=("comm_score", "sum"),
                mean_score=("comm_score", "mean"),
                n_interactions=("comm_score", "count"),
            )
            .sort_values("total_score", ascending=False)
            .reset_index()
        )

    def summary(self) -> str:
        lines = [
            "Cell Communication Analysis Summary:",
            f"  Conditions: {len(self.conditions_analyzed)}",
            f"  Cell types: {len(self.cell_types_analyzed)}",
            f"  LR pairs tested: {self.n_lr_pairs_tested}",
            f"  Total interactions scored: {len(self.interactions)}",
        ]
        if self.diff_results:
            lines.append(
                f"  Differential: {len(self.diff_results)} tested, "
                f"{self.n_significant_changes} significant (FDR < 0.05)"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run_cell_communication(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    lr_pairs: list[tuple[str, str, str, str]] | None = None,
    min_pct: float = 0.1,
    layer: str | None = None,
    diff_comparisons: list[tuple[str, str]] | None = None,
    alpha: float = 0.05,
) -> CellCommSummary:
    """Run complete cell-cell communication analysis.

    Computes communication scores for all LR pairs across cell types
    and conditions, then performs differential analysis.

    Args:
        adata: AnnData with expression data.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        lr_pairs: Custom LR pairs. If None, uses built-in DB.
        min_pct: Minimum expressing fraction filter.
        layer: Layer for expression (None = X).
        diff_comparisons: List of (condition1, condition2) for differential
            analysis. If None, auto-generates IVIG comparisons.
        alpha: FDR threshold for significance.

    Returns:
        CellCommSummary with all results.
    """
    print("Running cell communication analysis...")

    # Compute scores
    interactions = compute_communication_scores(
        adata,
        cell_type_key=cell_type_key,
        condition_key=condition_key,
        lr_pairs=lr_pairs,
        min_pct=min_pct,
        layer=layer,
    )

    conditions = sorted(adata.obs[condition_key].unique())
    cell_types = sorted(adata.obs[cell_type_key].unique())
    tested_pairs = set()
    for i in interactions:
        tested_pairs.add((i.ligand, i.receptor))

    print(f"  {len(interactions)} interactions scored across "
          f"{len(conditions)} conditions, {len(cell_types)} cell types")

    # Differential communication
    diff_results: list[DiffCommResult] = []
    if diff_comparisons is None:
        # Auto-generate IVIG comparisons
        cond_lower = {str(c).lower(): str(c) for c in conditions}
        diff_comparisons = []
        pre = next((v for k, v in cond_lower.items() if "pre" in k), None)
        post = next((v for k, v in cond_lower.items() if "post" in k), None)
        ctrl = next(
            (v for k, v in cond_lower.items() if "control" in k or "healthy" in k),
            None,
        )
        if pre and ctrl:
            diff_comparisons.append((ctrl, pre))
        if post and ctrl:
            diff_comparisons.append((ctrl, post))
        if pre and post:
            diff_comparisons.append((pre, post))

    for c1, c2 in diff_comparisons:
        print(f"  Differential: {c1} vs {c2}")
        diff = compute_differential_communication(interactions, c1, c2)
        diff_results.extend(diff)

    n_sig = sum(1 for r in diff_results if r.pvalue_adj < alpha)

    summary = CellCommSummary(
        interactions=interactions,
        diff_results=diff_results,
        conditions_analyzed=conditions,
        cell_types_analyzed=cell_types,
        n_lr_pairs_tested=len(tested_pairs),
        n_significant_changes=n_sig,
    )

    print(f"\n{summary.summary()}")
    return summary
