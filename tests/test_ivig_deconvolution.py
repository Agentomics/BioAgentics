"""Tests for IVIG bulk RNA-seq deconvolution pipeline (Phase 5)."""

import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse as sp

from bioagentics.pandas_pans.ivig_deconvolution import (
    SignatureMatrix,
    DeconvolutionResult,
    ModuleValidation,
    CrossInitiativeResult,
    DeconvolutionPipelineResult,
    _select_marker_genes,
    _deconvolve_single_sample,
    _score_module_in_bulk,
    _zscore_series,
    _genes_likely_in_index,
    run_build_signature_matrix,
    run_deconvolve_bulk,
    run_validate_modules_in_bulk,
    run_cross_initiative_integration,
    run_deconvolution_pipeline,
    AUTOANTIBODY_TARGET_GENES,
    HLA_GENES,
)

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sc_adata(n_cells=200, n_genes=100, n_types=4, seed=42):
    """Create synthetic scRNA-seq AnnData with distinct cell types."""
    rng = np.random.RandomState(seed)
    types = [f"CellType_{i}" for i in range(n_types)]
    cells_per_type = n_cells // n_types

    gene_names = [f"Gene_{i}" for i in range(n_genes)]
    X = rng.poisson(2, size=(n_cells, n_genes)).astype(np.float32)

    # Make each cell type have distinct markers (higher expression)
    obs_types = []
    for i, ct in enumerate(types):
        start = i * cells_per_type
        end = start + cells_per_type
        # Marker genes for this type: genes i*10 to i*10+10
        marker_start = i * (n_genes // n_types)
        marker_end = marker_start + min(10, n_genes // n_types)
        X[start:end, marker_start:marker_end] += rng.poisson(15, size=(cells_per_type, marker_end - marker_start))
        obs_types.extend([ct] * cells_per_type)

    obs = pd.DataFrame({"cell_type": obs_types}, index=[f"cell_{i}" for i in range(n_cells)])
    var = pd.DataFrame(index=gene_names)
    return ad.AnnData(X=X, obs=obs, var=var)


def _make_bulk_expr(n_samples=20, gene_names=None, n_genes=100, seed=42):
    """Create synthetic bulk RNA-seq expression (samples x genes)."""
    rng = np.random.RandomState(seed)
    if gene_names is None:
        gene_names = [f"Gene_{i}" for i in range(n_genes)]
    expr = rng.lognormal(3, 1.5, size=(n_samples, len(gene_names)))
    sample_ids = [f"Sample_{i}" for i in range(n_samples)]
    return pd.DataFrame(expr, index=sample_ids, columns=gene_names)


def _make_treatment_sigs():
    """Create mock treatment signatures with genes_up/genes_down."""
    class MockSig:
        def __init__(self, up, down):
            self.genes_up = up
            self.genes_down = down

    return {
        "Mono_treatment": MockSig(
            up=["Gene_0", "Gene_1", "Gene_2"],
            down=["Gene_10", "Gene_11"],
        ),
        "NK_treatment": MockSig(
            up=["Gene_30", "Gene_31"],
            down=["Gene_40", "Gene_41", "Gene_42"],
        ),
    }


# ---------------------------------------------------------------------------
# SignatureMatrix dataclass
# ---------------------------------------------------------------------------


class TestSignatureMatrix:
    def test_to_dict(self):
        sm = SignatureMatrix(
            matrix=pd.DataFrame({"A": [1, 2], "B": [3, 4]}, index=["g1", "g2"]),
            cell_types=["A", "B"],
            n_genes=2,
        )
        d = sm.to_dict()
        assert d["n_cell_types"] == 2
        assert d["n_genes"] == 2
        assert d["cell_types"] == ["A", "B"]

    def test_empty_signature(self):
        sm = SignatureMatrix(matrix=pd.DataFrame(), cell_types=[], n_genes=0)
        assert sm.to_dict()["n_cell_types"] == 0


# ---------------------------------------------------------------------------
# DeconvolutionResult dataclass
# ---------------------------------------------------------------------------


class TestDeconvolutionResult:
    def test_to_dict(self):
        dr = DeconvolutionResult(
            proportions=pd.DataFrame({"A": [0.5], "B": [0.5]}),
            residuals=pd.Series([0.1]),
            n_samples=1,
            n_cell_types=2,
        )
        d = dr.to_dict()
        assert d["n_samples"] == 1
        assert d["n_cell_types"] == 2
        assert d["method"] == "nnls"
        assert d["mean_residual"] == pytest.approx(0.1)

    def test_empty_result(self):
        dr = DeconvolutionResult(proportions=pd.DataFrame())
        assert dr.to_dict()["n_samples"] == 0


# ---------------------------------------------------------------------------
# ModuleValidation dataclass
# ---------------------------------------------------------------------------


class TestModuleValidation:
    def test_to_dict(self):
        mv = ModuleValidation(
            module_scores=pd.DataFrame(),
            n_modules_validated=3,
            n_modules_significant=1,
        )
        d = mv.to_dict()
        assert d["n_modules_validated"] == 3
        assert d["n_modules_significant"] == 1


# ---------------------------------------------------------------------------
# CrossInitiativeResult
# ---------------------------------------------------------------------------


class TestCrossInitiativeResult:
    def test_to_dict(self):
        cr = CrossInitiativeResult(
            autoantibody_overlap={"sig1": ["CAMK2A", "DRD1"]},
            hla_expression={"HLA-A": {"mean": 5.0}},
        )
        d = cr.to_dict()
        assert d["n_autoantibody_overlaps"] == 2
        assert d["n_hla_genes_checked"] == 1


# ---------------------------------------------------------------------------
# Pipeline result
# ---------------------------------------------------------------------------


class TestDeconvolutionPipelineResult:
    def test_summary_empty(self):
        r = DeconvolutionPipelineResult()
        s = r.summary()
        assert "Phase 5" in s

    def test_summary_with_data(self):
        r = DeconvolutionPipelineResult(
            signature_matrix=SignatureMatrix(
                matrix=pd.DataFrame(), cell_types=["A", "B"], n_genes=50
            ),
            deconvolution=DeconvolutionResult(
                proportions=pd.DataFrame(), n_samples=20, n_cell_types=2
            ),
        )
        s = r.summary()
        assert "50 genes" in s
        assert "20 samples" in s


# ---------------------------------------------------------------------------
# _zscore_series
# ---------------------------------------------------------------------------


class TestZscoreSeries:
    def test_normal(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        z = _zscore_series(s)
        assert z.mean() == pytest.approx(0.0, abs=1e-10)
        assert z.std() == pytest.approx(1.0, abs=0.1)

    def test_zero_variance(self):
        s = pd.Series([5.0, 5.0, 5.0])
        z = _zscore_series(s)
        assert (z == 0).all()

    def test_single_value(self):
        s = pd.Series([3.0])
        z = _zscore_series(s)
        assert z.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# _select_marker_genes
# ---------------------------------------------------------------------------


class TestSelectMarkerGenes:
    def test_basic(self):
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        markers = _select_marker_genes(adata, n_top=5, min_fold_change=1.2)
        assert len(markers) > 0
        assert all(isinstance(g, str) for g in markers)

    def test_sorted_output(self):
        adata = _make_sc_adata()
        markers = _select_marker_genes(adata, n_top=5)
        assert markers == sorted(markers)

    def test_missing_key_raises(self):
        adata = _make_sc_adata()
        with pytest.raises(ValueError, match="not in adata.obs"):
            _select_marker_genes(adata, cell_type_key="nonexistent")

    def test_single_cell_type_raises(self):
        adata = _make_sc_adata()
        adata.obs["cell_type"] = "OnlyType"
        with pytest.raises(ValueError, match="at least 2"):
            _select_marker_genes(adata)

    def test_sparse_input(self):
        adata = _make_sc_adata()
        adata.X = sp.csr_matrix(adata.X)
        markers = _select_marker_genes(adata, n_top=5, min_fold_change=1.2)
        assert len(markers) > 0


# ---------------------------------------------------------------------------
# run_build_signature_matrix
# ---------------------------------------------------------------------------


class TestBuildSignatureMatrix:
    def test_basic(self):
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        assert sm.n_genes > 0
        assert len(sm.cell_types) == 4
        assert sm.matrix.shape[1] == 4

    def test_empty_adata(self):
        adata = ad.AnnData(
            X=np.zeros((0, 10)),
            obs=pd.DataFrame(columns=["cell_type"]),
            var=pd.DataFrame(index=[f"g{i}" for i in range(10)]),
        )
        sm = run_build_signature_matrix(adata)
        assert sm.n_genes == 0
        assert sm.matrix.empty

    def test_sparse_adata(self):
        adata = _make_sc_adata()
        adata.X = sp.csr_matrix(adata.X)
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        assert sm.n_genes > 0

    def test_signature_values_positive(self):
        adata = _make_sc_adata()
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        # Mean expression should be non-negative
        assert (sm.matrix.values >= 0).all()

    def test_to_dict(self):
        adata = _make_sc_adata()
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        d = sm.to_dict()
        assert "cell_types" in d
        assert d["n_genes"] == sm.n_genes


# ---------------------------------------------------------------------------
# _deconvolve_single_sample
# ---------------------------------------------------------------------------


class TestDeconvolveSingleSample:
    def test_pure_cell_type(self):
        # Signature: 2 cell types, 3 genes
        sig = np.array([[10.0, 1.0], [1.0, 10.0], [5.0, 5.0]])
        # Sample is 100% type 0
        sample = np.array([10.0, 1.0, 5.0])
        props, resid = _deconvolve_single_sample(sample, sig)
        assert props.sum() == pytest.approx(1.0, abs=1e-6)
        assert props[0] > props[1]

    def test_mixed_sample(self):
        sig = np.array([[10.0, 1.0], [1.0, 10.0]])
        # 50/50 mix
        sample = np.array([5.5, 5.5])
        props, resid = _deconvolve_single_sample(sample, sig)
        assert props.sum() == pytest.approx(1.0, abs=1e-6)
        assert abs(props[0] - props[1]) < 0.2

    def test_zero_sample(self):
        sig = np.array([[10.0, 1.0], [1.0, 10.0]])
        sample = np.array([0.0, 0.0])
        props, resid = _deconvolve_single_sample(sample, sig)
        assert props.sum() == pytest.approx(0.0, abs=1e-6)

    def test_proportions_sum_to_one(self):
        rng = np.random.RandomState(42)
        sig = rng.rand(20, 5) + 0.1
        sample = rng.rand(20) + 0.1
        props, resid = _deconvolve_single_sample(sample, sig)
        assert props.sum() == pytest.approx(1.0, abs=1e-6)
        assert (props >= 0).all()


# ---------------------------------------------------------------------------
# run_deconvolve_bulk
# ---------------------------------------------------------------------------


class TestDeconvolveBulk:
    def test_basic(self):
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        bulk = _make_bulk_expr(n_samples=10, gene_names=list(adata.var_names))
        result = run_deconvolve_bulk(bulk, sm)
        assert result.n_samples == 10
        assert result.n_cell_types == 4
        assert result.proportions.shape == (10, 4)
        # Proportions should sum to ~1 per sample
        row_sums = result.proportions.sum(axis=1)
        assert all(abs(s - 1.0) < 1e-6 for s in row_sums)

    def test_transposed_bulk(self):
        """Bulk with genes in rows should auto-orient."""
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        bulk = _make_bulk_expr(n_samples=10, gene_names=list(adata.var_names))
        result = run_deconvolve_bulk(bulk.T, sm)  # genes x samples
        assert result.n_samples == 10

    def test_empty_signature(self):
        sm = SignatureMatrix(matrix=pd.DataFrame(), cell_types=[], n_genes=0)
        bulk = _make_bulk_expr()
        result = run_deconvolve_bulk(bulk, sm)
        assert result.n_samples == 0
        assert result.proportions.empty

    def test_insufficient_overlap(self):
        adata = _make_sc_adata()
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        # Bulk with completely different gene names
        bulk = _make_bulk_expr(gene_names=[f"Other_{i}" for i in range(100)])
        result = run_deconvolve_bulk(bulk, sm)
        assert result.n_samples == 0

    def test_residuals_reported(self):
        adata = _make_sc_adata()
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        bulk = _make_bulk_expr(n_samples=5, gene_names=list(adata.var_names))
        result = run_deconvolve_bulk(bulk, sm)
        assert len(result.residuals) == 5
        assert all(r >= 0 for r in result.residuals)


# ---------------------------------------------------------------------------
# _score_module_in_bulk
# ---------------------------------------------------------------------------


class TestScoreModuleInBulk:
    def test_up_genes_only(self):
        bulk = pd.DataFrame(
            {"G1": [10, 20, 30], "G2": [5, 10, 15], "G3": [1, 1, 1]},
            index=["s1", "s2", "s3"],
        )
        scores = _score_module_in_bulk(bulk, genes_up=["G1", "G2"], genes_down=[])
        assert len(scores) == 3
        # Higher expression -> higher score
        assert scores.iloc[2] > scores.iloc[0]

    def test_down_genes_only(self):
        bulk = pd.DataFrame(
            {"G1": [10, 20, 30], "G2": [5, 10, 15]},
            index=["s1", "s2", "s3"],
        )
        scores = _score_module_in_bulk(bulk, genes_up=[], genes_down=["G1", "G2"])
        # Higher expression of down genes -> lower score
        assert scores.iloc[0] > scores.iloc[2]

    def test_no_genes_available(self):
        bulk = pd.DataFrame({"G1": [1, 2]}, index=["s1", "s2"])
        scores = _score_module_in_bulk(bulk, genes_up=["MISSING"], genes_down=[])
        assert (scores == 0.0).all()

    def test_mixed_up_down(self):
        bulk = pd.DataFrame(
            {"UP1": [1, 5, 10], "DOWN1": [10, 5, 1]},
            index=["s1", "s2", "s3"],
        )
        scores = _score_module_in_bulk(bulk, genes_up=["UP1"], genes_down=["DOWN1"])
        # s3 has high UP1 and low DOWN1, should score highest
        assert scores.iloc[2] > scores.iloc[0]


# ---------------------------------------------------------------------------
# run_validate_modules_in_bulk
# ---------------------------------------------------------------------------


class TestValidateModulesInBulk:
    def test_basic_validation(self):
        bulk = _make_bulk_expr(n_samples=20)
        sigs = _make_treatment_sigs()
        result = run_validate_modules_in_bulk(bulk, sigs)
        assert result.n_modules_validated == 2
        assert not result.module_scores.empty

    def test_with_group_labels(self):
        bulk = _make_bulk_expr(n_samples=20)
        labels = pd.Series(
            ["PANS"] * 10 + ["Control"] * 10,
            index=bulk.index,
        )
        sigs = _make_treatment_sigs()
        result = run_validate_modules_in_bulk(bulk, sigs, group_labels=labels)
        assert len(result.group_comparisons) > 0
        assert all("pvalue" in c for c in result.group_comparisons)

    def test_with_proportions_correlation(self):
        bulk = _make_bulk_expr(n_samples=20)
        sigs = _make_treatment_sigs()
        props = DeconvolutionResult(
            proportions=pd.DataFrame(
                np.random.rand(20, 3),
                index=bulk.index,
                columns=["A", "B", "C"],
            ),
            n_samples=20,
            n_cell_types=3,
        )
        result = run_validate_modules_in_bulk(bulk, sigs, proportions=props)
        assert len(result.module_correlations) > 0

    def test_empty_signatures(self):
        bulk = _make_bulk_expr()
        result = run_validate_modules_in_bulk(bulk, {})
        assert result.n_modules_validated == 0

    def test_empty_bulk(self):
        result = run_validate_modules_in_bulk(pd.DataFrame(), _make_treatment_sigs())
        assert result.n_modules_validated == 0


# ---------------------------------------------------------------------------
# _genes_likely_in_index
# ---------------------------------------------------------------------------


class TestGenesLikelyInIndex:
    def test_genes_in_columns(self):
        df = pd.DataFrame(
            {"Gene_0": [1], "Gene_1": [2]},
            index=["sample1"],
        )
        sigs = _make_treatment_sigs()
        assert not _genes_likely_in_index(df, sigs)

    def test_genes_in_index(self):
        df = pd.DataFrame(
            {"sample1": [1, 2]},
            index=["Gene_0", "Gene_1"],
        )
        sigs = _make_treatment_sigs()
        assert _genes_likely_in_index(df, sigs)

    def test_empty_sigs(self):
        df = pd.DataFrame({"a": [1]})

        class Empty:
            genes_up = []
            genes_down = []

        assert not _genes_likely_in_index(df, {"e": Empty()})


# ---------------------------------------------------------------------------
# Cross-initiative integration
# ---------------------------------------------------------------------------


class TestCrossInitiativeIntegration:
    def test_autoantibody_overlap(self):
        class Sig:
            def __init__(self, up, down):
                self.genes_up = up
                self.genes_down = down

        sigs = {"mono": Sig(up=["CAMK2A", "TNF"], down=["DRD1", "IL6"])}
        result = run_cross_initiative_integration(treatment_signatures=sigs)
        assert "mono" in result.autoantibody_overlap
        assert "CAMK2A" in result.autoantibody_overlap["mono"]
        assert "DRD1" in result.autoantibody_overlap["mono"]

    def test_hla_expression(self):
        gene_names = HLA_GENES[:5] + [f"G{i}" for i in range(10)]
        bulk = _make_bulk_expr(n_samples=10, gene_names=gene_names)
        result = run_cross_initiative_integration(bulk_expr=bulk)
        assert len(result.hla_expression) > 0

    def test_hla_with_groups(self):
        gene_names = HLA_GENES[:3] + [f"G{i}" for i in range(10)]
        bulk = _make_bulk_expr(n_samples=10, gene_names=gene_names)
        labels = pd.Series(
            ["PANS"] * 5 + ["Control"] * 5, index=bulk.index
        )
        result = run_cross_initiative_integration(
            bulk_expr=bulk, group_labels=labels
        )
        for gene_info in result.hla_expression.values():
            assert "mean_PANS" in gene_info or "mean_Control" in gene_info

    def test_pathway_convergence(self):
        props = DeconvolutionResult(
            proportions=pd.DataFrame(
                {"Mono": [0.3, 0.2], "NK": [0.05, 0.04], "Rare": [0.005, 0.003]},
                index=["s1", "s2"],
            ),
            n_samples=2,
            n_cell_types=3,
        )
        result = run_cross_initiative_integration(deconv_proportions=props)
        # Mono and NK should appear (>1%), Rare should not
        ct_names = [p["cell_type"] for p in result.pathway_convergence]
        assert "Mono" in ct_names
        assert "NK" in ct_names
        assert "Rare" not in ct_names

    def test_empty_inputs(self):
        result = run_cross_initiative_integration()
        assert result.to_dict()["n_autoantibody_overlaps"] == 0

    def test_known_gene_lists(self):
        assert len(AUTOANTIBODY_TARGET_GENES) > 10
        assert len(HLA_GENES) > 10
        assert "CAMK2A" in AUTOANTIBODY_TARGET_GENES
        assert "HLA-A" in HLA_GENES


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


class TestDeconvolutionPipeline:
    def test_full_pipeline(self):
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        bulk = _make_bulk_expr(n_samples=20, gene_names=list(adata.var_names))
        sigs = _make_treatment_sigs()
        labels = pd.Series(
            ["PANS"] * 10 + ["Control"] * 10, index=bulk.index
        )

        result = run_deconvolution_pipeline(
            sc_adata=adata,
            bulk_expr=bulk,
            treatment_signatures=sigs,
            group_labels=labels,
            n_marker_genes=5,
        )

        assert result.signature_matrix is not None
        assert result.signature_matrix.n_genes > 0
        assert result.deconvolution is not None
        assert result.deconvolution.n_samples == 20
        assert result.module_validation is not None
        assert result.module_validation.n_modules_validated == 2
        assert result.cross_initiative is not None

    def test_pipeline_with_prebuilt_sig_matrix(self):
        adata = _make_sc_adata(n_cells=200, n_genes=100, n_types=4)
        sm = run_build_signature_matrix(adata, n_marker_genes=5, min_fold_change=1.2)
        bulk = _make_bulk_expr(n_samples=10, gene_names=list(adata.var_names))

        result = run_deconvolution_pipeline(
            bulk_expr=bulk, sig_matrix=sm
        )
        assert result.signature_matrix is sm
        assert result.deconvolution is not None

    def test_pipeline_sc_only(self):
        adata = _make_sc_adata()
        result = run_deconvolution_pipeline(sc_adata=adata, n_marker_genes=5)
        assert result.signature_matrix is not None
        assert result.deconvolution is None  # no bulk data

    def test_pipeline_empty(self):
        result = run_deconvolution_pipeline()
        assert result.signature_matrix is None
        assert result.deconvolution is None
        s = result.summary()
        assert "Phase 5" in s

    def test_pipeline_summary(self):
        adata = _make_sc_adata()
        bulk = _make_bulk_expr(n_samples=5, gene_names=list(adata.var_names))
        result = run_deconvolution_pipeline(
            sc_adata=adata, bulk_expr=bulk, n_marker_genes=5
        )
        s = result.summary()
        assert "Deconvolution" in s
