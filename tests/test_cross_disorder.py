"""Tests for cross-disorder pathway comparison module."""

from __future__ import annotations

import pandas as pd
import pytest

from bioagentics.tourettes.ts_gwas_functional_annotation.cross_disorder import (
    DISORDERS,
    GENOMIC_FACTORS,
    CrossDisorderPathway,
    CrossDisorderSummary,
    LocusOverlap,
    PleioLocus,
    _classify_pathway,
    _disorders_to_factors,
    _normalize_enrichment_df,
    build_enrichment_matrix,
    build_factor_matrix,
    compare_pathway_enrichment,
    compute_locus_overlaps,
    load_pleiotropic_loci,
    summarize_cross_disorder,
    write_cross_disorder_results,
    write_cross_disorder_summary,
    write_enrichment_heatmap,
    write_locus_overlaps,
)


# --- Fixtures ---


@pytest.fixture
def ts_enrichment_df():
    """TS pathway enrichment results (convergence format)."""
    return pd.DataFrame({
        "PATHWAY": [
            "dopamine_signaling", "synaptic_adhesion", "axon_guidance",
            "neurodevelopment", "gaba_signaling", "striatal_msn",
        ],
        "SOURCE": ["builtin"] * 6,
        "CONVERGENCE_P": [0.001, 0.005, 0.01, 0.02, 0.03, 0.04],
        "CONVERGENCE_FDR": [0.006, 0.015, 0.03, 0.06, 0.09, 0.12],
    })


@pytest.fixture
def disorder_enrichments():
    """Per-disorder pathway enrichment results."""
    ocd_df = pd.DataFrame({
        "PATHWAY": [
            "dopamine_signaling", "synaptic_adhesion", "axon_guidance",
            "neurodevelopment", "gaba_signaling",
        ],
        "P": [0.01, 0.02, 0.8, 0.5, 0.03],
        "FDR": [0.05, 0.08, 0.9, 0.7, 0.10],
    })
    adhd_df = pd.DataFrame({
        "PATHWAY": [
            "dopamine_signaling", "synaptic_adhesion", "neurodevelopment",
            "gaba_signaling",
        ],
        "P": [0.005, 0.6, 0.01, 0.7],
        "FDR": [0.02, 0.8, 0.04, 0.85],
    })
    asd_df = pd.DataFrame({
        "PATHWAY": ["neurodevelopment", "synaptic_adhesion"],
        "P": [0.003, 0.02],
        "FDR": [0.01, 0.07],
    })
    return {"OCD": ocd_df, "ADHD": adhd_df, "ASD": asd_df}


@pytest.fixture
def ts_loci():
    """Sample TS loci for overlap testing."""
    return [
        {"locus_id": 1, "chr": 5, "start": 174800000, "stop": 174900000,
         "genes": ["DRD1"]},
        {"locus_id": 2, "chr": 11, "start": 113200000, "stop": 113400000,
         "genes": ["DRD2", "TH"]},
        {"locus_id": 3, "chr": 2, "start": 50000000, "stop": 51000000,
         "genes": ["NRXN1"]},
    ]


@pytest.fixture
def pleio_loci():
    """Sample pleiotropic loci."""
    return [
        PleioLocus(1, 5, 174750000, 174950000, "rs123", ["TS", "OCD"], ["compulsive"], 2),
        PleioLocus(2, 11, 113000000, 113500000, "rs456", ["TS", "ADHD", "SCZ"],
                   ["compulsive", "neurodevelopmental", "psychotic"], 3),
        PleioLocus(3, 1, 10000000, 10500000, "rs789", ["OCD", "MDD"], ["compulsive", "internalizing"], 2),
    ]


# --- Factor model tests ---


class TestFactorModel:
    def test_disorders_list(self):
        assert "TS" in DISORDERS
        assert "OCD" in DISORDERS
        assert len(DISORDERS) >= 10

    def test_all_disorders_in_factors(self):
        factor_disorders = set()
        for factor_d in GENOMIC_FACTORS.values():
            factor_disorders.update(factor_d)
        for d in DISORDERS:
            assert d in factor_disorders, f"{d} not assigned to any factor"

    def test_ts_in_compulsive_factor(self):
        assert "TS" in GENOMIC_FACTORS["compulsive"]
        assert "OCD" in GENOMIC_FACTORS["compulsive"]

    def test_disorders_to_factors(self):
        assert _disorders_to_factors(["TS", "OCD"]) == ["compulsive"]
        assert "neurodevelopmental" in _disorders_to_factors(["ADHD"])
        assert sorted(_disorders_to_factors(["TS", "ADHD"])) == ["compulsive", "neurodevelopmental"]

    def test_empty_disorders(self):
        assert _disorders_to_factors([]) == []

    def test_unknown_disorder(self):
        assert _disorders_to_factors(["UNKNOWN"]) == []


# --- Locus overlap tests ---


class TestLocusOverlap:
    def test_basic_overlap(self, ts_loci, pleio_loci):
        overlaps = compute_locus_overlaps(ts_loci, pleio_loci)
        assert len(overlaps) >= 2  # loci 1 & 2 overlap with pleio 1 & 2

    def test_overlap_chromosomes_match(self, ts_loci, pleio_loci):
        overlaps = compute_locus_overlaps(ts_loci, pleio_loci)
        for o in overlaps:
            assert o.chr in {ts["chr"] for ts in ts_loci}

    def test_no_overlap_different_chr(self):
        ts = [{"locus_id": 1, "chr": 1, "start": 100, "stop": 200, "genes": ["A"]}]
        pleio = [PleioLocus(1, 2, 100, 200, "rs1", ["TS"], ["compulsive"], 1)]
        assert compute_locus_overlaps(ts, pleio) == []

    def test_no_overlap_distant_regions(self):
        ts = [{"locus_id": 1, "chr": 1, "start": 100, "stop": 200, "genes": ["A"]}]
        pleio = [PleioLocus(1, 1, 10000, 20000, "rs1", ["TS"], ["compulsive"], 1)]
        assert compute_locus_overlaps(ts, pleio) == []

    def test_overlap_with_window(self):
        ts = [{"locus_id": 1, "chr": 1, "start": 1000, "stop": 2000, "genes": ["A"]}]
        pleio = [PleioLocus(1, 1, 3000, 4000, "rs1", ["TS"], ["compulsive"], 1)]
        # No overlap without window
        assert compute_locus_overlaps(ts, pleio, window_kb=0) == []
        # With 2kb window, TS extends to 4000 -> overlap at 3000
        assert len(compute_locus_overlaps(ts, pleio, window_kb=2)) == 1

    def test_empty_inputs(self):
        assert compute_locus_overlaps([], []) == []
        ts = [{"locus_id": 1, "chr": 1, "start": 100, "stop": 200, "genes": ["A"]}]
        assert compute_locus_overlaps(ts, []) == []
        assert compute_locus_overlaps([], [PleioLocus(1, 1, 100, 200, "", [], [], 0)]) == []

    def test_overlap_carries_disorders(self, ts_loci, pleio_loci):
        overlaps = compute_locus_overlaps(ts_loci, pleio_loci)
        for o in overlaps:
            assert len(o.shared_disorders) > 0

    def test_overlap_carries_factors(self, ts_loci, pleio_loci):
        overlaps = compute_locus_overlaps(ts_loci, pleio_loci)
        for o in overlaps:
            assert len(o.shared_factors) > 0


# --- Load pleiotropic loci tests ---


class TestLoadPleioLoci:
    def test_load_from_tsv(self, tmp_path):
        df = pd.DataFrame({
            "LOCUS_ID": [1, 2],
            "CHR": [1, 5],
            "START": [100000, 200000],
            "STOP": [150000, 250000],
            "LEAD_SNP": ["rs1", "rs2"],
            "DISORDERS": ["TS;OCD", "ADHD;ASD"],
            "FACTORS": ["compulsive", "neurodevelopmental"],
        })
        path = tmp_path / "pleio.tsv"
        df.to_csv(path, sep="\t", index=False)

        loci = load_pleiotropic_loci(path)
        assert len(loci) == 2
        assert loci[0].disorders == ["TS", "OCD"]
        assert loci[0].chr == 1

    def test_missing_columns(self, tmp_path):
        df = pd.DataFrame({"LOCUS_ID": [1], "CHR": [1]})
        path = tmp_path / "bad.tsv"
        df.to_csv(path, sep="\t", index=False)
        loci = load_pleiotropic_loci(path)
        assert loci == []


# --- Pathway classification tests ---


class TestClassifyPathway:
    def test_ts_specific(self):
        disorder_p = {"TS": 0.001, "OCD": 0.5, "ADHD": 0.8, "ASD": 0.9}
        assert _classify_pathway(disorder_p, 0.05) == "ts_specific"

    def test_compulsive_shared(self):
        disorder_p = {"TS": 0.001, "OCD": 0.01, "ADHD": 0.8, "ASD": 0.9}
        assert _classify_pathway(disorder_p, 0.05) == "compulsive_shared"

    def test_broadly_shared(self):
        disorder_p = {"TS": 0.001, "OCD": 0.01, "ADHD": 0.005, "ASD": 0.9}
        assert _classify_pathway(disorder_p, 0.05) == "broadly_shared"

    def test_no_enrichment(self):
        disorder_p = {"TS": 0.5, "OCD": 0.8}
        assert _classify_pathway(disorder_p, 0.05) == "ts_specific"


# --- Cross-disorder comparison tests ---


class TestComparePathwayEnrichment:
    def test_basic_comparison(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        assert len(results) > 0
        for r in results:
            assert r.ts_p > 0
            assert r.classification in ("ts_specific", "compulsive_shared", "broadly_shared")
            assert r.specificity_score >= 0

    def test_dopamine_broadly_shared(self, ts_enrichment_df, disorder_enrichments):
        """Dopamine enriched in TS, OCD (compulsive), and ADHD (neurodevelopmental)."""
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        dopamine = [r for r in results if r.pathway == "dopamine_signaling"]
        assert len(dopamine) == 1
        assert dopamine[0].classification == "broadly_shared"

    def test_axon_guidance_ts_specific(self, ts_enrichment_df, disorder_enrichments):
        """Axon guidance enriched in TS only (OCD P=0.8, others missing)."""
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        axon = [r for r in results if r.pathway == "axon_guidance"]
        assert len(axon) == 1
        assert axon[0].classification == "ts_specific"

    def test_disorder_p_values_present(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        for r in results:
            assert "TS" in r.disorder_p
            # Disorders we provided enrichments for
            for d in disorder_enrichments:
                assert d in r.disorder_p

    def test_factor_min_p(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        for r in results:
            for factor in GENOMIC_FACTORS:
                assert factor in r.factor_min_p
                assert 0 <= r.factor_min_p[factor] <= 1

    def test_empty_ts_enrichment(self, disorder_enrichments):
        empty_df = pd.DataFrame()
        results = compare_pathway_enrichment(empty_df, disorder_enrichments)
        assert results == []

    def test_no_disorder_enrichments(self, ts_enrichment_df):
        results = compare_pathway_enrichment(ts_enrichment_df, {})
        assert len(results) > 0
        for r in results:
            assert r.classification == "ts_specific"

    def test_specificity_score_ordering(self, ts_enrichment_df, disorder_enrichments):
        """Results should be sorted by specificity score descending."""
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        scores = [r.specificity_score for r in results]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1] - 1e-10


# --- Normalize enrichment tests ---


class TestNormalizeEnrichment:
    def test_convergence_format(self):
        df = pd.DataFrame({
            "PATHWAY": ["a"], "CONVERGENCE_P": [0.01], "CONVERGENCE_FDR": [0.05],
        })
        out = _normalize_enrichment_df(df)
        assert "P" in out.columns
        assert "FDR" in out.columns
        assert float(out.iloc[0]["P"]) == pytest.approx(0.01)

    def test_gene_set_format(self):
        df = pd.DataFrame({
            "GENE_SET": ["a"], "P": [0.01], "FDR_Q": [0.05],
        })
        out = _normalize_enrichment_df(df)
        assert "PATHWAY" in out.columns
        assert "FDR" in out.columns

    def test_standard_format(self):
        df = pd.DataFrame({"PATHWAY": ["a"], "P": [0.01], "FDR": [0.05]})
        out = _normalize_enrichment_df(df)
        assert len(out) == 1

    def test_missing_columns(self):
        df = pd.DataFrame({"X": [1]})
        out = _normalize_enrichment_df(df)
        assert out.empty


# --- Enrichment matrix tests ---


class TestEnrichmentMatrix:
    def test_build_matrix(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        matrix = build_enrichment_matrix(results)
        assert "PATHWAY" in matrix.columns
        assert "TS" in matrix.columns
        assert len(matrix) == len(results)

    def test_neglog10_values(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        matrix = build_enrichment_matrix(results, use_neglog10=True)
        # -log10(P) should be positive for P < 1
        for col in matrix.columns:
            if col not in ("PATHWAY", "SOURCE", "CLASSIFICATION"):
                assert (matrix[col] >= 0).all()

    def test_raw_p_values(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        matrix = build_enrichment_matrix(results, use_neglog10=False)
        for col in matrix.columns:
            if col not in ("PATHWAY", "SOURCE", "CLASSIFICATION"):
                assert (matrix[col] >= 0).all()
                assert (matrix[col] <= 1).all()

    def test_empty_results(self):
        assert build_enrichment_matrix([]).empty

    def test_factor_matrix(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        matrix = build_factor_matrix(results)
        assert "PATHWAY" in matrix.columns
        for factor in GENOMIC_FACTORS:
            assert factor in matrix.columns
        assert len(matrix) == len(results)


# --- Summary tests ---


class TestSummary:
    def test_basic_summary(self, ts_enrichment_df, disorder_enrichments):
        results = compare_pathway_enrichment(
            ts_enrichment_df, disorder_enrichments,
        )
        summary = summarize_cross_disorder(results)
        assert summary.n_ts_pathways == len(results)
        assert summary.n_ts_specific + summary.n_compulsive_shared + summary.n_broadly_shared == len(results)

    def test_summary_with_overlaps(self):
        overlaps = [
            LocusOverlap(1, 1, 1, 100, 200, ["A"], ["TS"], ["compulsive"]),
        ]
        summary = summarize_cross_disorder([], overlaps)
        assert summary.n_pleio_loci_overlap == 1

    def test_empty_summary(self):
        summary = summarize_cross_disorder([])
        assert summary.n_ts_pathways == 0


# --- Output writer tests ---


class TestOutputWriters:
    def test_write_cross_disorder_results(self, tmp_path):
        results = [
            CrossDisorderPathway(
                pathway="test_pathway",
                source="builtin",
                ts_p=0.001,
                ts_fdr=0.005,
                disorder_p={"TS": 0.001, "OCD": 0.01, "ADHD": 0.5},
                disorder_fdr={"TS": 0.005, "OCD": 0.04, "ADHD": 0.8},
                factor_min_p={"compulsive": 0.001, "neurodevelopmental": 0.5,
                              "psychotic": 1.0, "internalizing": 1.0, "substance_use": 1.0},
                classification="compulsive_shared",
                specificity_score=2.5,
                n_disorders_enriched=1,
            ),
        ]
        path = write_cross_disorder_results(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["PATHWAY"] == "test_pathway"
        assert df.iloc[0]["CLASSIFICATION"] == "compulsive_shared"

    def test_write_enrichment_heatmap(self, tmp_path):
        results = [
            CrossDisorderPathway(
                pathway="test", source="builtin", ts_p=0.01, ts_fdr=0.05,
                disorder_p={"TS": 0.01, "OCD": 0.02},
                disorder_fdr={"TS": 0.05, "OCD": 0.08},
                factor_min_p={"compulsive": 0.01},
                classification="compulsive_shared",
                specificity_score=1.0, n_disorders_enriched=1,
            ),
        ]
        path = write_enrichment_heatmap(results, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert "TS" in df.columns
        assert "OCD" in df.columns

    def test_write_locus_overlaps(self, tmp_path):
        overlaps = [
            LocusOverlap(1, 1, 5, 100, 200, ["DRD1"], ["TS", "OCD"], ["compulsive"]),
        ]
        path = write_locus_overlaps(overlaps, tmp_path)
        assert path.exists()
        df = pd.read_csv(path, sep="\t")
        assert len(df) == 1
        assert df.iloc[0]["TS_LOCUS_ID"] == 1

    def test_write_summary(self, tmp_path):
        results = [
            CrossDisorderPathway(
                pathway="test", source="builtin", ts_p=0.01, ts_fdr=0.05,
                disorder_p={"TS": 0.01}, disorder_fdr={"TS": 0.05},
                factor_min_p={"compulsive": 0.01},
                classification="ts_specific",
                specificity_score=2.0, n_disorders_enriched=0,
            ),
        ]
        overlaps = [
            LocusOverlap(1, 1, 5, 100, 200, ["A"], ["TS"], ["compulsive"]),
        ]
        summary = summarize_cross_disorder(results, overlaps)
        path = write_cross_disorder_summary(results, overlaps, summary, tmp_path)
        assert path.exists()
        content = path.read_text()
        assert "Cross-Disorder" in content
        assert "TS-Specific" in content
