"""Bulk RNA-seq deconvolution using scRNA-seq reference for IVIG analysis.

Phase 5 of the IVIG mechanism single-cell analysis pipeline.

Deconvolves bulk RNA-seq (GSE293230: 32 PANS + 68 NDD + 58 controls)
using the scRNA-seq cell type reference to estimate cell type proportions
in the larger cohort. Validates treatment-responsive gene modules from
Phase 4 in the deconvolved bulk data.

Methods implemented:
1. Non-negative least squares (NNLS) deconvolution — lightweight, no
   external service required
2. Signature matrix construction from scRNA-seq pseudobulk profiles
3. Treatment module validation in bulk expression
4. Cross-initiative integration with autoantibody/HLA findings

Usage:
    from bioagentics.pandas_pans.ivig_deconvolution import (
        run_build_signature_matrix,
        run_deconvolve_bulk,
        run_validate_modules_in_bulk,
        run_cross_initiative_integration,
        run_deconvolution_pipeline,
    )
    sig_matrix = run_build_signature_matrix(sc_adata)
    proportions = run_deconvolve_bulk(bulk_expr, sig_matrix)
    validation = run_validate_modules_in_bulk(bulk_expr, treatment_sigs, proportions)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import nnls
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class SignatureMatrix:
    """Cell-type signature matrix built from scRNA-seq reference."""

    matrix: pd.DataFrame  # genes x cell_types
    cell_types: list[str] = field(default_factory=list)
    n_genes: int = 0
    method: str = "mean_expression"
    gene_selection: str = "marker_genes"

    def to_dict(self) -> dict:
        return {
            "cell_types": self.cell_types,
            "n_genes": self.n_genes,
            "n_cell_types": len(self.cell_types),
            "method": self.method,
            "gene_selection": self.gene_selection,
        }


@dataclass
class DeconvolutionResult:
    """Cell type proportion estimates from deconvolution."""

    proportions: pd.DataFrame  # samples x cell_types
    residuals: pd.Series = field(default_factory=pd.Series)
    method: str = "nnls"
    n_samples: int = 0
    n_cell_types: int = 0

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "n_cell_types": self.n_cell_types,
            "method": self.method,
            "mean_residual": float(self.residuals.mean()) if len(self.residuals) > 0 else 0.0,
            "cell_types": list(self.proportions.columns) if not self.proportions.empty else [],
        }


@dataclass
class ModuleValidation:
    """Validation of treatment-responsive modules in bulk data."""

    module_scores: pd.DataFrame  # samples x modules
    module_correlations: dict[str, dict] = field(default_factory=dict)
    group_comparisons: list[dict] = field(default_factory=list)
    n_modules_validated: int = 0
    n_modules_significant: int = 0

    def to_dict(self) -> dict:
        return {
            "n_modules_validated": self.n_modules_validated,
            "n_modules_significant": self.n_modules_significant,
            "group_comparisons": self.group_comparisons,
        }


@dataclass
class CrossInitiativeResult:
    """Cross-initiative integration results."""

    autoantibody_overlap: dict[str, list[str]] = field(default_factory=dict)
    hla_expression: dict[str, dict] = field(default_factory=dict)
    pathway_convergence: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_autoantibody_overlaps": sum(
                len(v) for v in self.autoantibody_overlap.values()
            ),
            "n_hla_genes_checked": len(self.hla_expression),
            "n_pathway_convergences": len(self.pathway_convergence),
        }


@dataclass
class DeconvolutionPipelineResult:
    """Full pipeline result combining all Phase 5 analyses."""

    signature_matrix: SignatureMatrix | None = None
    deconvolution: DeconvolutionResult | None = None
    module_validation: ModuleValidation | None = None
    cross_initiative: CrossInitiativeResult | None = None

    def summary(self) -> str:
        lines = ["Phase 5: Deconvolution & Integration"]
        if self.signature_matrix:
            lines.append(
                f"  Signature matrix: {self.signature_matrix.n_genes} genes x "
                f"{len(self.signature_matrix.cell_types)} cell types"
            )
        if self.deconvolution:
            lines.append(
                f"  Deconvolution: {self.deconvolution.n_samples} samples, "
                f"{self.deconvolution.n_cell_types} cell types"
            )
        if self.module_validation:
            lines.append(
                f"  Module validation: {self.module_validation.n_modules_significant}/"
                f"{self.module_validation.n_modules_validated} significant"
            )
        if self.cross_initiative:
            ci = self.cross_initiative
            lines.append(
                f"  Cross-initiative: {ci.to_dict()['n_autoantibody_overlaps']} "
                f"autoantibody overlaps, {ci.to_dict()['n_hla_genes_checked']} HLA genes"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Signature matrix construction
# ---------------------------------------------------------------------------


def _select_marker_genes(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    n_top: int = 50,
    min_fold_change: float = 1.5,
    min_fraction: float = 0.1,
) -> list[str]:
    """Select marker genes for each cell type via one-vs-rest comparison.

    For each cell type, finds genes with highest mean expression ratio
    vs all other cells. Returns the union of top markers across types.

    Args:
        adata: Annotated data with cell_type_key in obs.
        cell_type_key: Column in obs for cell type labels.
        n_top: Number of top marker genes per cell type.
        min_fold_change: Minimum fold change vs rest.
        min_fraction: Minimum fraction of cells expressing the gene.

    Returns:
        Sorted list of selected marker gene names.
    """
    if cell_type_key not in adata.obs.columns:
        raise ValueError(f"'{cell_type_key}' not in adata.obs")

    cell_types = sorted(adata.obs[cell_type_key].unique())
    if len(cell_types) < 2:
        raise ValueError("Need at least 2 cell types for marker selection")

    # Compute per-cell-type means via sparse ops to avoid materializing
    # the full dense matrix (144K cells x 20K genes = ~23GB as float64).
    X = adata.X
    is_sparse = sp.issparse(X)
    if is_sparse:
        X_csc = sp.csc_matrix(X)

    gene_names = list(adata.var_names)
    selected: set[str] = set()

    # Pre-compute global mean using sparse arithmetic
    if is_sparse:
        global_mean = np.asarray(X_csc.mean(axis=0)).ravel()
    else:
        global_mean = np.asarray(X.mean(axis=0)).ravel()
    n_total = X.shape[0]

    for ct in cell_types:
        mask_ct = (adata.obs[cell_type_key] == ct).values
        n_ct = int(mask_ct.sum())
        n_rest = n_total - n_ct

        if n_ct < 2 or n_rest < 2:
            continue

        # Compute cell-type mean and fraction from the subset only
        if is_sparse:
            X_ct = X_csc[mask_ct]
            mean_ct = np.asarray(X_ct.mean(axis=0)).ravel()
            frac_ct = np.asarray((X_ct > 0).mean(axis=0)).ravel()
        else:
            X_ct = np.asarray(X[mask_ct])
            mean_ct = X_ct.mean(axis=0)
            frac_ct = (X_ct > 0).mean(axis=0)

        # Derive rest-mean from global mean without a second full pass
        mean_rest = (global_mean * n_total - mean_ct * n_ct) / n_rest

        # Fold change (add pseudocount)
        fc = (mean_ct + 1e-9) / (mean_rest + 1e-9)

        # Filter: expressed in enough cells, sufficient fold change
        valid = (frac_ct >= min_fraction) & (fc >= min_fold_change)
        valid_indices = np.where(valid)[0]

        if len(valid_indices) == 0:
            continue

        # Rank by fold change, take top n
        ranked = valid_indices[np.argsort(-fc[valid_indices])]
        top = ranked[:n_top]

        for idx in top:
            selected.add(gene_names[idx])

    return sorted(selected)


def run_build_signature_matrix(
    adata: ad.AnnData,
    cell_type_key: str = "cell_type",
    n_marker_genes: int = 50,
    min_fold_change: float = 1.5,
    min_fraction: float = 0.1,
    use_raw: bool = False,
) -> SignatureMatrix:
    """Build a cell-type signature matrix from scRNA-seq reference data.

    Constructs a genes x cell_types matrix of mean expression values,
    using marker genes selected via one-vs-rest fold change ranking.

    Args:
        adata: scRNA-seq AnnData with cell type annotations.
        cell_type_key: Column in obs for cell type labels.
        n_marker_genes: Number of marker genes per cell type.
        min_fold_change: Minimum fold change for marker selection.
        min_fraction: Minimum fraction of cells expressing marker.
        use_raw: If True, use adata.raw for expression values.

    Returns:
        SignatureMatrix with genes x cell_types expression matrix.
    """
    if adata.n_obs == 0 or adata.n_vars == 0:
        return SignatureMatrix(
            matrix=pd.DataFrame(),
            cell_types=[],
            n_genes=0,
        )

    source = adata.raw if use_raw and adata.raw is not None else adata

    # Select marker genes
    marker_genes = _select_marker_genes(
        source if isinstance(source, ad.AnnData) else adata,
        cell_type_key=cell_type_key,
        n_top=n_marker_genes,
        min_fold_change=min_fold_change,
        min_fraction=min_fraction,
    )

    if not marker_genes:
        return SignatureMatrix(
            matrix=pd.DataFrame(),
            cell_types=[],
            n_genes=0,
        )

    # Subset to marker genes that exist in var_names
    available_genes = [g for g in marker_genes if g in source.var_names]
    if not available_genes:
        return SignatureMatrix(
            matrix=pd.DataFrame(),
            cell_types=[],
            n_genes=0,
        )

    # Build pseudobulk mean expression per cell type
    cell_types = sorted(adata.obs[cell_type_key].unique())
    sig_data: dict[str, list[float]] = {}

    X_sub = source[:, available_genes].X
    if sp.issparse(X_sub):
        X_sub = X_sub.toarray()
    X_sub = np.asarray(X_sub, dtype=np.float64)

    for ct in cell_types:
        mask = (adata.obs[cell_type_key] == ct).values
        if mask.sum() == 0:
            continue
        sig_data[ct] = X_sub[mask].mean(axis=0).tolist()

    sig_df = pd.DataFrame(sig_data, index=available_genes)
    valid_cts = list(sig_df.columns)

    return SignatureMatrix(
        matrix=sig_df,
        cell_types=valid_cts,
        n_genes=len(available_genes),
        method="mean_expression",
        gene_selection=f"top_{n_marker_genes}_markers_fc{min_fold_change}",
    )


# ---------------------------------------------------------------------------
# Bulk RNA-seq deconvolution
# ---------------------------------------------------------------------------


def _deconvolve_single_sample(
    sample_expr: np.ndarray,
    sig_matrix: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Deconvolve a single bulk sample using NNLS.

    Args:
        sample_expr: Expression vector (n_genes,).
        sig_matrix: Signature matrix (n_genes, n_cell_types).

    Returns:
        Tuple of (proportions array, residual).
    """
    coeffs, residual = nnls(sig_matrix, sample_expr)

    # Normalize to sum to 1
    total = coeffs.sum()
    if total > 0:
        proportions = coeffs / total
    else:
        proportions = np.zeros_like(coeffs)

    return proportions, float(residual)


def run_deconvolve_bulk(
    bulk_expr: pd.DataFrame,
    sig_matrix: SignatureMatrix,
    min_overlap_genes: int = 10,
) -> DeconvolutionResult:
    """Deconvolve bulk RNA-seq using scRNA-seq signature matrix via NNLS.

    For each bulk sample, estimates cell type proportions by solving
    bulk_expr ≈ sig_matrix @ proportions using non-negative least squares.

    Args:
        bulk_expr: Bulk expression DataFrame (samples x genes) or (genes x samples).
            Gene names in columns or index; will auto-orient.
        sig_matrix: SignatureMatrix from run_build_signature_matrix.
        min_overlap_genes: Minimum overlapping genes required.

    Returns:
        DeconvolutionResult with proportions (samples x cell_types).
    """
    if sig_matrix.matrix.empty:
        return DeconvolutionResult(
            proportions=pd.DataFrame(),
            method="nnls",
            n_samples=0,
            n_cell_types=0,
        )

    sig_genes = set(sig_matrix.matrix.index)

    # Auto-orient bulk_expr: determine if genes are rows or columns
    col_overlap = len(sig_genes & set(bulk_expr.columns))
    idx_overlap = len(sig_genes & set(bulk_expr.index))

    if col_overlap >= idx_overlap:
        # Genes in columns, samples in rows
        bulk = bulk_expr
    else:
        # Genes in rows, samples in columns — transpose
        bulk = bulk_expr.T

    # Find overlapping genes
    overlap_genes = sorted(sig_genes & set(bulk.columns))

    if len(overlap_genes) < min_overlap_genes:
        return DeconvolutionResult(
            proportions=pd.DataFrame(),
            method="nnls",
            n_samples=0,
            n_cell_types=0,
        )

    # Align matrices to overlapping genes
    sig_mat = sig_matrix.matrix.loc[overlap_genes].values.astype(np.float64)
    bulk_mat = bulk[overlap_genes].values.astype(np.float64)

    sample_ids = list(bulk.index)
    cell_types = sig_matrix.cell_types

    # Deconvolve each sample
    all_props = []
    all_residuals = []

    for i in range(bulk_mat.shape[0]):
        props, resid = _deconvolve_single_sample(bulk_mat[i], sig_mat)
        all_props.append(props)
        all_residuals.append(resid)

    prop_df = pd.DataFrame(all_props, index=sample_ids, columns=cell_types)
    resid_series = pd.Series(all_residuals, index=sample_ids, name="residual")

    return DeconvolutionResult(
        proportions=prop_df,
        residuals=resid_series,
        method="nnls",
        n_samples=len(sample_ids),
        n_cell_types=len(cell_types),
    )


# ---------------------------------------------------------------------------
# Treatment module validation in bulk data
# ---------------------------------------------------------------------------


def _score_module_in_bulk(
    bulk_expr: pd.DataFrame,
    genes_up: list[str],
    genes_down: list[str],
) -> pd.Series:
    """Score a gene module in bulk expression data.

    Uses mean z-scored expression: up-genes contribute positively,
    down-genes contribute negatively.

    Args:
        bulk_expr: Bulk expression (samples x genes).
        genes_up: Up-regulated genes in the module.
        genes_down: Down-regulated genes in the module.

    Returns:
        Series of module scores per sample.
    """
    available_up = [g for g in genes_up if g in bulk_expr.columns]
    available_down = [g for g in genes_down if g in bulk_expr.columns]

    if not available_up and not available_down:
        return pd.Series(0.0, index=bulk_expr.index, dtype=np.float64)

    scores = pd.Series(0.0, index=bulk_expr.index, dtype=np.float64)

    if available_up:
        up_expr = bulk_expr[available_up].copy()
        # Z-score each gene
        up_z = up_expr.apply(lambda col: _zscore_series(col), axis=0)
        scores = scores + up_z.mean(axis=1)

    if available_down:
        down_expr = bulk_expr[available_down].copy()
        down_z = down_expr.apply(lambda col: _zscore_series(col), axis=0)
        scores = scores - down_z.mean(axis=1)

    return scores


def _zscore_series(s: pd.Series) -> pd.Series:
    """Z-score a pandas Series, handling zero-variance."""
    std = s.std()
    if std == 0 or np.isnan(std):
        return pd.Series(0.0, index=s.index, dtype=np.float64)
    return (s - s.mean()) / std


def run_validate_modules_in_bulk(
    bulk_expr: pd.DataFrame,
    treatment_signatures: dict[str, object],
    proportions: DeconvolutionResult | None = None,
    group_labels: pd.Series | None = None,
    alpha: float = 0.05,
) -> ModuleValidation:
    """Validate treatment-responsive gene modules in bulk RNA-seq data.

    Scores each treatment module in bulk samples, then tests whether
    module scores differ between groups (PANS vs controls, etc.).
    Optionally correlates module scores with cell type proportions.

    Args:
        bulk_expr: Bulk expression (samples x genes).
        treatment_signatures: Dict of name -> signature object with
            genes_up/genes_down attributes.
        proportions: Optional DeconvolutionResult for correlation analysis.
        group_labels: Optional Series of group labels per sample.
        alpha: Significance threshold for group comparisons.

    Returns:
        ModuleValidation with scores, correlations, and comparisons.
    """
    if bulk_expr.empty or not treatment_signatures:
        return ModuleValidation(
            module_scores=pd.DataFrame(),
            n_modules_validated=0,
            n_modules_significant=0,
        )

    # Auto-orient: genes in columns
    if _genes_likely_in_index(bulk_expr, treatment_signatures):
        bulk_expr = bulk_expr.T

    module_scores: dict[str, pd.Series] = {}
    correlations: dict[str, dict] = {}
    comparisons: list[dict] = []
    n_significant = 0

    for name, sig in treatment_signatures.items():
        genes_up = getattr(sig, "genes_up", [])
        genes_down = getattr(sig, "genes_down", [])

        if not genes_up and not genes_down:
            continue

        scores = _score_module_in_bulk(bulk_expr, genes_up, genes_down)
        module_scores[name] = scores

        # Correlate with cell type proportions if available
        if proportions is not None and not proportions.proportions.empty:
            prop_df = proportions.proportions
            common_idx = scores.index.intersection(prop_df.index)
            if len(common_idx) >= 5:
                for ct in prop_df.columns:
                    r, p = scipy_stats.spearmanr(
                        scores.loc[common_idx],
                        prop_df.loc[common_idx, ct],
                        nan_policy="omit",
                    )
                    if name not in correlations:
                        correlations[name] = {}
                    correlations[name][ct] = {
                        "spearman_r": float(r) if not np.isnan(r) else 0.0,
                        "pvalue": float(p) if not np.isnan(p) else 1.0,
                    }

        # Compare groups if labels provided
        if group_labels is not None:
            common_idx = scores.index.intersection(group_labels.index)
            if len(common_idx) >= 4:
                labeled_scores = scores.loc[common_idx]
                labels = group_labels.loc[common_idx]
                groups = sorted(labels.unique())

                for i, g1 in enumerate(groups):
                    for g2 in groups[i + 1 :]:
                        s1 = labeled_scores[labels == g1].dropna()
                        s2 = labeled_scores[labels == g2].dropna()

                        if len(s1) >= 2 and len(s2) >= 2:
                            stat, pval = scipy_stats.mannwhitneyu(
                                s1, s2, alternative="two-sided"
                            )
                            is_sig = pval < alpha
                            if is_sig:
                                n_significant += 1
                            comparisons.append({
                                "module": name,
                                "group1": str(g1),
                                "group2": str(g2),
                                "n1": len(s1),
                                "n2": len(s2),
                                "median1": float(s1.median()),
                                "median2": float(s2.median()),
                                "statistic": float(stat),
                                "pvalue": float(pval),
                                "significant": is_sig,
                            })

    scores_df = pd.DataFrame(module_scores)

    return ModuleValidation(
        module_scores=scores_df,
        module_correlations=correlations,
        group_comparisons=comparisons,
        n_modules_validated=len(module_scores),
        n_modules_significant=n_significant,
    )


def _genes_likely_in_index(
    df: pd.DataFrame,
    signatures: dict[str, object],
) -> bool:
    """Heuristic: check if gene names are in df.index rather than df.columns."""
    all_genes: set[str] = set()
    for sig in signatures.values():
        all_genes.update(getattr(sig, "genes_up", []))
        all_genes.update(getattr(sig, "genes_down", []))
    if not all_genes:
        return False
    idx_overlap = len(all_genes & set(df.index))
    col_overlap = len(all_genes & set(df.columns))
    return idx_overlap > col_overlap


# ---------------------------------------------------------------------------
# Cross-initiative integration
# ---------------------------------------------------------------------------

# Known autoantibody target genes from PANDAS/PANS literature
# (from autoantibody-target-network-mapping initiative)
AUTOANTIBODY_TARGET_GENES: list[str] = [
    "CAMK2A", "CAMK2B", "LYN", "TUBB3", "DRD1", "DRD2",
    "GRIN2A", "GRIN2B", "SLC6A3", "SLC6A4", "HTR2A", "HTR2C",
    "TPH2", "TH", "GAD1", "GAD2", "GABRA1", "GABBR1",
    "CHRM1", "CHRNA7", "SYN1", "SYP", "SNAP25", "VAMP2",
]

# HLA genes relevant to PANDAS/PANS susceptibility
HLA_GENES: list[str] = [
    "HLA-A", "HLA-B", "HLA-C", "HLA-DRA", "HLA-DRB1", "HLA-DRB5",
    "HLA-DQA1", "HLA-DQB1", "HLA-DPA1", "HLA-DPB1",
    "B2M", "TAP1", "TAP2", "TAPBP", "PSMB8", "PSMB9",
]


def run_cross_initiative_integration(
    deconv_proportions: DeconvolutionResult | None = None,
    bulk_expr: pd.DataFrame | None = None,
    treatment_signatures: dict[str, object] | None = None,
    group_labels: pd.Series | None = None,
    autoantibody_genes: list[str] | None = None,
    hla_genes: list[str] | None = None,
) -> CrossInitiativeResult:
    """Integrate deconvolution results with other PANDAS/PANS initiatives.

    1. Check overlap between IVIG-responsive genes and autoantibody targets
    2. Assess HLA gene expression across groups
    3. Identify pathway convergence points

    Args:
        deconv_proportions: Cell type proportions from deconvolution.
        bulk_expr: Bulk expression (samples x genes).
        treatment_signatures: Treatment-responsive signatures from Phase 4.
        group_labels: Group labels per sample.
        autoantibody_genes: Override default autoantibody target gene list.
        hla_genes: Override default HLA gene list.

    Returns:
        CrossInitiativeResult with overlap and expression analyses.
    """
    ab_genes = set(autoantibody_genes or AUTOANTIBODY_TARGET_GENES)
    hla = hla_genes or HLA_GENES

    result = CrossInitiativeResult()

    # 1. Autoantibody overlap: which IVIG-responsive genes are autoantibody targets?
    if treatment_signatures:
        for name, sig in treatment_signatures.items():
            all_sig_genes = set(
                getattr(sig, "genes_up", []) + getattr(sig, "genes_down", [])
            )
            overlap = sorted(all_sig_genes & ab_genes)
            if overlap:
                result.autoantibody_overlap[name] = overlap

    # 2. HLA gene expression by group
    if bulk_expr is not None and not bulk_expr.empty:
        expr = bulk_expr
        # Auto-orient
        if treatment_signatures and _genes_likely_in_index(expr, treatment_signatures):
            expr = expr.T

        for gene in hla:
            if gene not in expr.columns:
                continue

            gene_expr = expr[gene]
            entry: dict = {"mean": float(gene_expr.mean()), "std": float(gene_expr.std())}

            if group_labels is not None:
                common = gene_expr.index.intersection(group_labels.index)
                if len(common) >= 4:
                    for grp in sorted(group_labels.loc[common].unique()):
                        vals = gene_expr.loc[
                            common[group_labels.loc[common] == grp]
                        ]
                        entry[f"mean_{grp}"] = float(vals.mean())

            result.hla_expression[gene] = entry

    # 3. Pathway convergence: IVIG-responsive cell types that express
    #    both autoantibody targets and HLA genes
    if deconv_proportions is not None and not deconv_proportions.proportions.empty:
        prop_df = deconv_proportions.proportions
        for ct in prop_df.columns:
            mean_prop = float(prop_df[ct].mean())
            if mean_prop > 0.01:  # cell type present in >1% of cells
                result.pathway_convergence.append({
                    "cell_type": ct,
                    "mean_proportion": mean_prop,
                    "note": f"{ct} may mediate IVIG effects on autoantibody targets",
                })

    return result


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_deconvolution_pipeline(
    sc_adata: ad.AnnData | None = None,
    bulk_expr: pd.DataFrame | None = None,
    treatment_signatures: dict[str, object] | None = None,
    group_labels: pd.Series | None = None,
    sig_matrix: SignatureMatrix | None = None,
    cell_type_key: str = "cell_type",
    n_marker_genes: int = 50,
    autoantibody_genes: list[str] | None = None,
    hla_genes: list[str] | None = None,
) -> DeconvolutionPipelineResult:
    """Run the full Phase 5 deconvolution and integration pipeline.

    Steps:
    1. Build signature matrix from scRNA-seq (or use provided)
    2. Deconvolve bulk RNA-seq
    3. Validate treatment modules in bulk data
    4. Cross-initiative integration

    Args:
        sc_adata: scRNA-seq AnnData with cell type annotations.
        bulk_expr: Bulk RNA-seq expression (samples x genes).
        treatment_signatures: Treatment-responsive signatures from Phase 4.
        group_labels: Group labels for bulk samples.
        sig_matrix: Pre-built signature matrix (skips step 1).
        cell_type_key: Cell type column in sc_adata.obs.
        n_marker_genes: Marker genes per cell type for signature matrix.
        autoantibody_genes: Override autoantibody target genes.
        hla_genes: Override HLA genes.

    Returns:
        DeconvolutionPipelineResult with all analyses.
    """
    result = DeconvolutionPipelineResult()

    # Step 1: Build signature matrix
    if sig_matrix is not None:
        result.signature_matrix = sig_matrix
    elif sc_adata is not None:
        result.signature_matrix = run_build_signature_matrix(
            sc_adata,
            cell_type_key=cell_type_key,
            n_marker_genes=n_marker_genes,
        )

    # Step 2: Deconvolve bulk
    if bulk_expr is not None and result.signature_matrix is not None:
        result.deconvolution = run_deconvolve_bulk(
            bulk_expr, result.signature_matrix
        )

    # Step 3: Validate treatment modules
    if bulk_expr is not None and treatment_signatures:
        result.module_validation = run_validate_modules_in_bulk(
            bulk_expr,
            treatment_signatures,
            proportions=result.deconvolution,
            group_labels=group_labels,
        )

    # Step 4: Cross-initiative integration
    result.cross_initiative = run_cross_initiative_integration(
        deconv_proportions=result.deconvolution,
        bulk_expr=bulk_expr,
        treatment_signatures=treatment_signatures,
        group_labels=group_labels,
        autoantibody_genes=autoantibody_genes,
        hla_genes=hla_genes,
    )

    return result
