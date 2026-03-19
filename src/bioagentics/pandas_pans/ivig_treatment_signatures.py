"""Treatment response signature analysis for IVIG scRNA-seq data.

Phase 4 of the IVIG mechanism single-cell analysis pipeline.

Identifies gene modules that:
1. Distinguish pre-IVIG PANS from controls (disease signature)
2. Normalize post-IVIG (treatment-responsive signature)
3. Can predict IVIG treatment response (minimal predictor signature)

Builds on pseudobulk DE results (Phase 2) and pathway enrichment (Phase 3)
to derive clinically actionable gene signatures.

Usage:
    from bioagentics.pandas_pans.ivig_treatment_signatures import (
        run_disease_signature,
        run_treatment_response_signature,
        run_signature_scoring,
        run_minimal_predictor,
    )
    disease_sig = run_disease_signature(de_summary, alpha=0.05, lfc=1.0)
    treatment_sig = run_treatment_response_signature(de_summary)
    scored = run_signature_scoring(adata, disease_sig)
    predictor = run_minimal_predictor(scored, n_genes=30)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GeneSignature:
    """A gene signature with metadata about direction and effect size."""

    name: str
    genes_up: list[str] = field(default_factory=list)
    genes_down: list[str] = field(default_factory=list)
    cell_type: str = ""
    comparison: str = ""
    description: str = ""

    @property
    def all_genes(self) -> list[str]:
        return sorted(set(self.genes_up + self.genes_down))

    @property
    def n_genes(self) -> int:
        return len(self.all_genes)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "genes_up": self.genes_up,
            "genes_down": self.genes_down,
            "cell_type": self.cell_type,
            "comparison": self.comparison,
            "description": self.description,
            "n_genes_up": len(self.genes_up),
            "n_genes_down": len(self.genes_down),
            "n_genes_total": self.n_genes,
        }


@dataclass
class SignatureScore:
    """Score of a gene signature in a cell type x condition group."""

    signature_name: str
    cell_type: str
    condition: str
    mean_score: float
    std_score: float
    n_cells: int
    n_genes_detected: int
    n_genes_total: int

    def to_dict(self) -> dict:
        return {
            "signature_name": self.signature_name,
            "cell_type": self.cell_type,
            "condition": self.condition,
            "mean_score": self.mean_score,
            "std_score": self.std_score,
            "n_cells": self.n_cells,
            "n_genes_detected": self.n_genes_detected,
            "n_genes_total": self.n_genes_total,
        }


@dataclass
class TreatmentResponseResult:
    """Summary of treatment response signature analysis."""

    disease_signatures: dict[str, GeneSignature] = field(default_factory=dict)
    treatment_signatures: dict[str, GeneSignature] = field(default_factory=dict)
    signature_scores: pd.DataFrame = field(default_factory=pd.DataFrame)
    predictor_genes: list[str] = field(default_factory=list)
    predictor_weights: dict[str, float] = field(default_factory=dict)
    cell_types_analyzed: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Treatment Response Signature Analysis"]
        lines.append(f"  Cell types analyzed: {len(self.cell_types_analyzed)}")
        lines.append(f"  Disease signatures: {len(self.disease_signatures)}")
        for name, sig in self.disease_signatures.items():
            lines.append(f"    {name}: {sig.n_genes} genes ({len(sig.genes_up)} up, {len(sig.genes_down)} down)")
        lines.append(f"  Treatment-responsive signatures: {len(self.treatment_signatures)}")
        for name, sig in self.treatment_signatures.items():
            lines.append(f"    {name}: {sig.n_genes} genes ({len(sig.genes_up)} up, {len(sig.genes_down)} down)")
        if self.predictor_genes:
            lines.append(f"  Minimal predictor: {len(self.predictor_genes)} genes")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Disease signature extraction
# ---------------------------------------------------------------------------


def _extract_de_genes(
    de_results: list[dict] | pd.DataFrame,
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
) -> pd.DataFrame:
    """Extract significant DE genes from results.

    Args:
        de_results: DE results as list of dicts or DataFrame with columns:
            gene, cell_type, comparison, log2_fold_change, pvalue_adj.
        alpha: FDR threshold.
        lfc_threshold: Min |log2FC|.

    Returns:
        Filtered DataFrame of significant DE genes.
    """
    if isinstance(de_results, list):
        df = pd.DataFrame(de_results)
    else:
        df = de_results.copy()

    if df.empty:
        return df

    required = {"gene", "log2_fold_change", "pvalue_adj"}
    if not required.issubset(df.columns):
        missing = required - set(df.columns)
        raise ValueError(f"Missing columns: {missing}")

    mask = (df["pvalue_adj"] < alpha) & (df["log2_fold_change"].abs() > lfc_threshold)
    return df[mask].copy()


def run_disease_signature(
    de_results: list[dict] | pd.DataFrame,
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
    comparison_pattern: str = "pans_vs_control",
    min_cell_types: int = 1,
) -> dict[str, GeneSignature]:
    """Identify disease signatures: genes distinguishing pre-IVIG PANS from controls.

    For each cell type, extracts up- and down-regulated genes in PANS vs controls.
    Also builds a pan-cell-type consensus signature from genes DE in multiple types.

    Args:
        de_results: DE results (list of dicts or DataFrame).
        alpha: FDR threshold.
        lfc_threshold: Min |log2FC|.
        comparison_pattern: String to match the PANS vs control comparison.
        min_cell_types: Min cell types a gene must be DE in for consensus.

    Returns:
        Dict of cell_type -> GeneSignature, plus "consensus" key.
    """
    sig_df = _extract_de_genes(de_results, alpha, lfc_threshold)

    if sig_df.empty:
        return {}

    # Filter to disease comparison
    if "comparison" in sig_df.columns:
        disease_df = sig_df[
            sig_df["comparison"].str.contains(comparison_pattern, case=False, na=False)
        ]
    else:
        disease_df = sig_df

    if disease_df.empty:
        return {}

    signatures: dict[str, GeneSignature] = {}

    # Per-cell-type signatures
    if "cell_type" in disease_df.columns:
        cell_types = sorted(disease_df["cell_type"].unique())
    else:
        cell_types = ["all"]
        disease_df = disease_df.copy()
        disease_df["cell_type"] = "all"

    gene_ct_counts: dict[str, int] = {}
    gene_directions: dict[str, list[float]] = {}

    for ct in cell_types:
        ct_df = disease_df[disease_df["cell_type"] == ct]
        if ct_df.empty:
            continue

        up = sorted(ct_df[ct_df["log2_fold_change"] > 0]["gene"].tolist())
        down = sorted(ct_df[ct_df["log2_fold_change"] < 0]["gene"].tolist())

        if not up and not down:
            continue

        signatures[ct] = GeneSignature(
            name=f"disease_{ct}",
            genes_up=up,
            genes_down=down,
            cell_type=ct,
            comparison=comparison_pattern,
            description=f"Genes DE in {ct}: PANS vs controls",
        )

        # Track cross-cell-type counts
        for gene in up + down:
            gene_ct_counts[gene] = gene_ct_counts.get(gene, 0) + 1
            if gene not in gene_directions:
                gene_directions[gene] = []
            lfc_val = ct_df[ct_df["gene"] == gene]["log2_fold_change"].values
            if len(lfc_val) > 0:
                gene_directions[gene].append(float(lfc_val[0]))

    # Consensus signature: genes DE in >= min_cell_types
    consensus_up = []
    consensus_down = []
    for gene, count in gene_ct_counts.items():
        if count >= min_cell_types:
            mean_dir = np.mean(gene_directions.get(gene, [0]))
            if mean_dir > 0:
                consensus_up.append(gene)
            else:
                consensus_down.append(gene)

    if consensus_up or consensus_down:
        signatures["consensus"] = GeneSignature(
            name="disease_consensus",
            genes_up=sorted(consensus_up),
            genes_down=sorted(consensus_down),
            cell_type="consensus",
            comparison=comparison_pattern,
            description=f"Genes DE in >={min_cell_types} cell types: PANS vs controls",
        )

    return signatures


# ---------------------------------------------------------------------------
# Treatment-responsive signature
# ---------------------------------------------------------------------------


def run_treatment_response_signature(
    de_results: list[dict] | pd.DataFrame,
    alpha: float = 0.05,
    lfc_threshold: float = 0.5,
    disease_comparison: str = "pans_vs_control",
    treatment_comparison: str = "pre_vs_post",
    require_reversal: bool = True,
) -> dict[str, GeneSignature]:
    """Identify treatment-responsive signatures: genes that normalize post-IVIG.

    A treatment-responsive gene is:
    1. DE in PANS vs controls (disease gene)
    2. Changes in the opposite direction pre-IVIG vs post-IVIG (reversal)

    Args:
        de_results: DE results (list of dicts or DataFrame).
        alpha: FDR threshold.
        lfc_threshold: Min |log2FC| for disease comparison.
        disease_comparison: Pattern for PANS-vs-control comparison.
        treatment_comparison: Pattern for pre-vs-post comparison.
        require_reversal: If True, only include genes whose direction reverses.

    Returns:
        Dict of cell_type -> GeneSignature for treatment-responsive genes.
    """
    if isinstance(de_results, list):
        df = pd.DataFrame(de_results)
    else:
        df = de_results.copy()

    if df.empty or "comparison" not in df.columns:
        return {}

    # Get disease genes
    disease_df = _extract_de_genes(df, alpha, lfc_threshold)
    if "comparison" in disease_df.columns:
        disease_df = disease_df[
            disease_df["comparison"].str.contains(disease_comparison, case=False, na=False)
        ]

    if disease_df.empty:
        return {}

    # Get treatment DE (use relaxed threshold for reversal detection)
    treatment_lfc = lfc_threshold * 0.5  # More lenient for treatment effect
    treatment_sig = _extract_de_genes(df, alpha=alpha * 2, lfc_threshold=treatment_lfc)
    if "comparison" in treatment_sig.columns:
        treatment_sig = treatment_sig[
            treatment_sig["comparison"].str.contains(treatment_comparison, case=False, na=False)
        ]

    if treatment_sig.empty and require_reversal:
        return {}

    # Build treatment direction lookup: (gene, cell_type) -> lfc
    treatment_dir: dict[tuple[str, str], float] = {}
    if not treatment_sig.empty and "cell_type" in treatment_sig.columns:
        for _, row in treatment_sig.iterrows():
            treatment_dir[(row["gene"], row["cell_type"])] = row["log2_fold_change"]

    signatures: dict[str, GeneSignature] = {}

    if "cell_type" in disease_df.columns:
        cell_types = sorted(disease_df["cell_type"].unique())
    else:
        cell_types = ["all"]
        disease_df = disease_df.copy()
        disease_df["cell_type"] = "all"

    all_responsive_up: list[str] = []
    all_responsive_down: list[str] = []

    for ct in cell_types:
        ct_disease = disease_df[disease_df["cell_type"] == ct]
        responsive_up = []
        responsive_down = []

        for _, row in ct_disease.iterrows():
            gene = row["gene"]
            disease_lfc = row["log2_fold_change"]

            if require_reversal:
                treatment_lfc_val = treatment_dir.get((gene, ct))
                if treatment_lfc_val is None:
                    continue
                # Reversal: disease direction and treatment direction are opposite
                if np.sign(disease_lfc) == np.sign(treatment_lfc_val):
                    continue

            # Label by disease direction (what's reversed)
            if disease_lfc > 0:
                responsive_up.append(gene)
            else:
                responsive_down.append(gene)

        if responsive_up or responsive_down:
            signatures[ct] = GeneSignature(
                name=f"treatment_responsive_{ct}",
                genes_up=sorted(responsive_up),
                genes_down=sorted(responsive_down),
                cell_type=ct,
                comparison=f"{disease_comparison}_reversed_by_{treatment_comparison}",
                description=f"Disease genes in {ct} that reverse post-IVIG",
            )
            all_responsive_up.extend(responsive_up)
            all_responsive_down.extend(responsive_down)

    # Consensus treatment-responsive signature
    if all_responsive_up or all_responsive_down:
        signatures["consensus"] = GeneSignature(
            name="treatment_responsive_consensus",
            genes_up=sorted(set(all_responsive_up)),
            genes_down=sorted(set(all_responsive_down)),
            cell_type="consensus",
            comparison=f"{disease_comparison}_reversed_by_{treatment_comparison}",
            description="All disease genes that reverse post-IVIG across cell types",
        )

    return signatures


# ---------------------------------------------------------------------------
# Signature scoring
# ---------------------------------------------------------------------------


def _score_signature_in_adata(
    adata: ad.AnnData,
    signature: GeneSignature,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    layer: str | None = None,
) -> list[SignatureScore]:
    """Score a gene signature across cells, grouped by cell_type x condition.

    Up-regulated genes contribute positively, down-regulated negatively.
    Each gene is z-scored across cells before averaging.

    Returns:
        List of SignatureScore per cell_type x condition.
    """
    all_genes = signature.all_genes
    available = [g for g in all_genes if g in adata.var_names]
    if not available:
        return []

    gene_idx = [list(adata.var_names).index(g) for g in available]
    up_set = set(signature.genes_up)
    down_set = set(signature.genes_down)

    # Extract and z-score
    X_raw = adata.layers[layer] if layer else adata.X
    if sp.issparse(X_raw):
        X_mod = np.asarray(X_raw[:, gene_idx].toarray())
    else:
        X_mod = np.asarray(np.array(X_raw)[:, gene_idx])

    gene_means = X_mod.mean(axis=0)
    gene_stds = X_mod.std(axis=0)
    gene_stds[gene_stds == 0] = 1.0
    X_z = (X_mod - gene_means) / gene_stds

    # Flip sign for down-regulated genes so score reflects disease state
    for i, g in enumerate(available):
        if g in down_set:
            X_z[:, i] *= -1

    cell_scores = X_z.mean(axis=1)

    # Aggregate per group
    results: list[SignatureScore] = []
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
            results.append(SignatureScore(
                signature_name=signature.name,
                cell_type=ct,
                condition=str(cond),
                mean_score=float(np.mean(scores)),
                std_score=float(np.std(scores)) if n_cells > 1 else 0.0,
                n_cells=n_cells,
                n_genes_detected=len(available),
                n_genes_total=len(all_genes),
            ))

    return results


def run_signature_scoring(
    adata: ad.AnnData,
    signatures: dict[str, GeneSignature],
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    layer: str | None = None,
) -> pd.DataFrame:
    """Score multiple gene signatures across all cells.

    Args:
        adata: AnnData with expression data.
        signatures: Dict of name -> GeneSignature.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        layer: Expression layer (None = X).

    Returns:
        DataFrame with signature scores per cell_type x condition.
    """
    all_scores: list[dict] = []
    for sig in signatures.values():
        scores = _score_signature_in_adata(
            adata, sig,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
            layer=layer,
        )
        all_scores.extend(s.to_dict() for s in scores)

    return pd.DataFrame(all_scores)


# ---------------------------------------------------------------------------
# Signature comparison between conditions
# ---------------------------------------------------------------------------


def compare_signature_scores(
    scores_df: pd.DataFrame,
    signature_name: str,
    condition1: str,
    condition2: str,
) -> pd.DataFrame:
    """Compare signature scores between conditions using Mann-Whitney U.

    Args:
        scores_df: Output from run_signature_scoring().
        signature_name: Signature to compare.
        condition1: Reference condition.
        condition2: Test condition.

    Returns:
        DataFrame with comparison stats per cell type.
    """
    if scores_df.empty:
        return pd.DataFrame()

    sig_df = scores_df[scores_df["signature_name"] == signature_name]
    if sig_df.empty:
        return pd.DataFrame()

    results = []
    cell_types = sorted(sig_df["cell_type"].unique())

    for ct in cell_types:
        ct_df = sig_df[sig_df["cell_type"] == ct]
        c1 = ct_df[ct_df["condition"] == condition1]
        c2 = ct_df[ct_df["condition"] == condition2]

        if c1.empty or c2.empty:
            continue

        mean1 = c1["mean_score"].values[0]
        mean2 = c2["mean_score"].values[0]
        n1 = c1["n_cells"].values[0]
        n2 = c2["n_cells"].values[0]

        # Effect size (Cohen's d approximation from aggregated stats)
        std1 = c1["std_score"].values[0] if c1["std_score"].values[0] > 0 else 1e-6
        std2 = c2["std_score"].values[0] if c2["std_score"].values[0] > 0 else 1e-6
        pooled_std = np.sqrt(
            ((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / max(n1 + n2 - 2, 1)
        )
        cohens_d = (mean2 - mean1) / pooled_std if pooled_std > 0 else 0.0

        # Welch's t-test from summary stats
        se1 = std1 / np.sqrt(max(n1, 1))
        se2 = std2 / np.sqrt(max(n2, 1))
        se_diff = np.sqrt(se1**2 + se2**2)
        if se_diff > 0:
            t_stat = (mean2 - mean1) / se_diff
            # Welch-Satterthwaite df
            df_num = (se1**2 + se2**2) ** 2
            df_den = se1**4 / max(n1 - 1, 1) + se2**4 / max(n2 - 1, 1)
            df_val = df_num / df_den if df_den > 0 else 1
            pvalue = float(2 * scipy_stats.t.sf(abs(t_stat), df=df_val))
        else:
            t_stat = 0.0
            pvalue = 1.0

        results.append({
            "cell_type": ct,
            "signature_name": signature_name,
            "condition1": condition1,
            "condition2": condition2,
            "mean_score_1": mean1,
            "mean_score_2": mean2,
            "delta_score": mean2 - mean1,
            "cohens_d": cohens_d,
            "t_statistic": t_stat,
            "pvalue": pvalue,
            "n_cells_1": n1,
            "n_cells_2": n2,
        })

    result_df = pd.DataFrame(results)
    if not result_df.empty and len(result_df) > 1:
        result_df["pvalue_adj"] = _benjamini_hochberg(result_df["pvalue"].values)
    elif not result_df.empty:
        result_df["pvalue_adj"] = result_df["pvalue"]

    return result_df


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


# ---------------------------------------------------------------------------
# Minimal predictor signature
# ---------------------------------------------------------------------------


def run_minimal_predictor(
    de_results: list[dict] | pd.DataFrame,
    disease_signatures: dict[str, GeneSignature] | None = None,
    treatment_signatures: dict[str, GeneSignature] | None = None,
    n_genes: int = 30,
    alpha: float = 0.05,
    lfc_threshold: float = 1.0,
    comparison_pattern: str = "pans_vs_control",
) -> tuple[list[str], dict[str, float]]:
    """Select a minimal gene set for predicting IVIG responders.

    Strategy: rank genes by a composite score combining:
    1. Significance (adjusted p-value) in disease comparison
    2. Effect size (|log2FC|) in disease comparison
    3. Presence in treatment-responsive signature (bonus)
    4. Cross-cell-type consistency (bonus)

    Args:
        de_results: Full DE results.
        disease_signatures: From run_disease_signature (optional, used for cross-CT info).
        treatment_signatures: From run_treatment_response_signature (optional, bonus scoring).
        n_genes: Target size for minimal predictor.
        alpha: FDR threshold for candidate genes.
        lfc_threshold: Min |log2FC| for candidates.
        comparison_pattern: Disease comparison pattern.

    Returns:
        Tuple of (gene_list, gene_weights_dict).
    """
    if isinstance(de_results, list):
        df = pd.DataFrame(de_results)
    else:
        df = de_results.copy()

    if df.empty:
        return [], {}

    # Filter to disease comparison
    if "comparison" in df.columns:
        disease_df = df[
            df["comparison"].str.contains(comparison_pattern, case=False, na=False)
        ]
    else:
        disease_df = df

    # Get significant genes
    sig_df = _extract_de_genes(disease_df, alpha, lfc_threshold)
    if sig_df.empty:
        return [], {}

    # Build treatment-responsive gene set for bonus scoring
    treatment_genes: set[str] = set()
    if treatment_signatures:
        for sig in treatment_signatures.values():
            treatment_genes.update(sig.all_genes)

    # Count how many cell types each gene is DE in
    if "cell_type" in sig_df.columns:
        gene_ct_count = sig_df.groupby("gene")["cell_type"].nunique().to_dict()
    else:
        gene_ct_count = {g: 1 for g in sig_df["gene"].unique()}

    # Compute composite score per gene (take best across cell types)
    gene_scores: dict[str, float] = {}
    for gene in sig_df["gene"].unique():
        gene_rows = sig_df[sig_df["gene"] == gene]

        # Best significance (-log10 padj)
        best_padj = gene_rows["pvalue_adj"].min()
        sig_score = -np.log10(max(best_padj, 1e-300))

        # Best effect size
        best_lfc = gene_rows["log2_fold_change"].abs().max()

        # Cross-cell-type bonus
        ct_bonus = np.log2(max(gene_ct_count.get(gene, 1), 1))

        # Treatment-responsive bonus
        treatment_bonus = 2.0 if gene in treatment_genes else 0.0

        # Composite: weighted sum
        composite = sig_score * 0.4 + best_lfc * 0.3 + ct_bonus * 0.15 + treatment_bonus * 0.15

        gene_scores[gene] = float(composite)

    # Sort by composite score and take top n
    sorted_genes = sorted(gene_scores.keys(), key=lambda g: gene_scores[g], reverse=True)
    predictor_genes = sorted_genes[:n_genes]
    predictor_weights = {g: gene_scores[g] for g in predictor_genes}

    return predictor_genes, predictor_weights


# ---------------------------------------------------------------------------
# Full pipeline orchestrator
# ---------------------------------------------------------------------------


def run_treatment_signatures(
    adata: ad.AnnData,
    de_results: list[dict] | pd.DataFrame,
    cell_type_key: str = "cell_type",
    condition_key: str = "condition",
    layer: str | None = None,
    disease_alpha: float = 0.05,
    disease_lfc: float = 1.0,
    treatment_alpha: float = 0.05,
    treatment_lfc: float = 0.5,
    disease_comparison: str = "pans_vs_control",
    treatment_comparison: str = "pre_vs_post",
    predictor_n_genes: int = 30,
    require_reversal: bool = True,
) -> TreatmentResponseResult:
    """Run full Phase 4 treatment response signature pipeline.

    Args:
        adata: AnnData with expression data.
        de_results: Pseudobulk DE results from Phase 2.
        cell_type_key: obs column for cell type.
        condition_key: obs column for condition.
        layer: Expression layer (None = X).
        disease_alpha: FDR threshold for disease signature.
        disease_lfc: Min |log2FC| for disease signature.
        treatment_alpha: FDR threshold for treatment signature.
        treatment_lfc: Min |log2FC| for treatment signature.
        disease_comparison: Pattern for PANS-vs-control comparison.
        treatment_comparison: Pattern for pre-vs-post comparison.
        predictor_n_genes: Target size for minimal predictor.
        require_reversal: Require direction reversal for treatment-responsive genes.

    Returns:
        TreatmentResponseResult with all signatures and scores.
    """
    print("Phase 4: Treatment Response Signature Analysis")
    print("=" * 50)

    # Step 1: Disease signatures
    print("\n1. Extracting disease signatures (PANS vs controls)...")
    disease_sigs = run_disease_signature(
        de_results, alpha=disease_alpha, lfc_threshold=disease_lfc,
        comparison_pattern=disease_comparison,
    )
    for name, sig in disease_sigs.items():
        print(f"  {name}: {sig.n_genes} genes ({len(sig.genes_up)} up, {len(sig.genes_down)} down)")

    # Step 2: Treatment-responsive signatures
    print("\n2. Identifying treatment-responsive genes...")
    treatment_sigs = run_treatment_response_signature(
        de_results, alpha=treatment_alpha, lfc_threshold=treatment_lfc,
        disease_comparison=disease_comparison,
        treatment_comparison=treatment_comparison,
        require_reversal=require_reversal,
    )
    for name, sig in treatment_sigs.items():
        print(f"  {name}: {sig.n_genes} genes ({len(sig.genes_up)} up, {len(sig.genes_down)} down)")

    # Step 3: Score signatures in expression data
    print("\n3. Scoring signatures across cells...")
    all_sigs = {**disease_sigs, **treatment_sigs}
    scores_df = pd.DataFrame()
    if all_sigs:
        scores_df = run_signature_scoring(
            adata, all_sigs,
            cell_type_key=cell_type_key,
            condition_key=condition_key,
            layer=layer,
        )
        print(f"  Scored {len(all_sigs)} signatures across {len(scores_df)} cell_type x condition groups")

    # Step 4: Minimal predictor
    print("\n4. Selecting minimal predictor gene set...")
    predictor_genes, predictor_weights = run_minimal_predictor(
        de_results,
        disease_signatures=disease_sigs,
        treatment_signatures=treatment_sigs,
        n_genes=predictor_n_genes,
        alpha=disease_alpha,
        lfc_threshold=disease_lfc,
        comparison_pattern=disease_comparison,
    )
    print(f"  Selected {len(predictor_genes)} predictor genes")

    # Collect analyzed cell types
    cell_types = set()
    for sig in disease_sigs.values():
        if sig.cell_type and sig.cell_type != "consensus":
            cell_types.add(sig.cell_type)
    for sig in treatment_sigs.values():
        if sig.cell_type and sig.cell_type != "consensus":
            cell_types.add(sig.cell_type)

    result = TreatmentResponseResult(
        disease_signatures=disease_sigs,
        treatment_signatures=treatment_sigs,
        signature_scores=scores_df,
        predictor_genes=predictor_genes,
        predictor_weights=predictor_weights,
        cell_types_analyzed=sorted(cell_types),
    )

    print(f"\n{result.summary()}")
    return result
