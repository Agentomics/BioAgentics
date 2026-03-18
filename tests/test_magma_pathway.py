"""Tests for the MAGMA gene-set analysis pipeline."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bioagentics.pipelines.magma_pathway.pipeline import (
    ALL_BUILTIN_GENE_SETS,
    GeneResult,
    GeneSetResult,
    PathwayAnalysisResult,
    gene_analysis_snp_wise,
    gene_set_analysis,
    load_gene_annotations,
    load_gene_sets,
    map_snps_to_genes,
    run_pathway_analysis,
)

RNG = np.random.default_rng(42)

N_SNPS = 1000
N_GENES = 50


def _make_snp_ids(n: int) -> list[str]:
    return [f"rs{i}" for i in range(1, n + 1)]


def _make_gwas_with_positions(
    n_snps: int = N_SNPS,
    n_chromosomes: int = 5,
) -> pd.DataFrame:
    """Generate synthetic GWAS with positional information."""
    from scipy import stats as sp_stats

    snp_ids = _make_snp_ids(n_snps)
    z = RNG.normal(0, 1.5, size=n_snps)
    p = 2 * sp_stats.norm.sf(np.abs(z))
    chrs = RNG.integers(1, n_chromosomes + 1, size=n_snps)
    bps = RNG.integers(1, 250_000_000, size=n_snps)

    return pd.DataFrame({
        "SNP": snp_ids,
        "A1": RNG.choice(["A", "C", "G", "T"], size=n_snps),
        "A2": RNG.choice(["A", "C", "G", "T"], size=n_snps),
        "BETA": z / np.sqrt(50000),
        "SE": np.full(n_snps, 1 / np.sqrt(50000)),
        "Z": z,
        "P": p,
        "N": np.full(n_snps, 50000, dtype=int),
        "CHR": chrs,
        "BP": bps,
    })


def _make_gene_annotations(
    n_genes: int = N_GENES,
    n_chromosomes: int = 5,
    gene_names: list[str] | None = None,
) -> pd.DataFrame:
    """Generate synthetic gene annotations."""
    if gene_names is None:
        gene_names = [f"GENE{i}" for i in range(1, n_genes + 1)]
    n = len(gene_names)
    chrs = RNG.integers(1, n_chromosomes + 1, size=n)
    starts = RNG.integers(1, 200_000_000, size=n)
    stops = starts + RNG.integers(5000, 200000, size=n)

    return pd.DataFrame({
        "GENE": gene_names,
        "CHR": chrs,
        "START": starts,
        "STOP": stops,
    })


def _make_gwas_for_gene_analysis(
    gene_annot: pd.DataFrame,
    n_snps_per_gene: int = 20,
    signal_genes: list[str] | None = None,
    signal_strength: float = 3.0,
) -> pd.DataFrame:
    """Generate GWAS data with SNPs positioned within gene boundaries.

    Optionally enriches signal in specific genes.
    """
    from scipy import stats as sp_stats

    if signal_genes is None:
        signal_genes = []
    signal_set = set(signal_genes)

    rows = []
    snp_counter = 1
    for _, gene in gene_annot.iterrows():
        gene_chr = gene["CHR"]
        gene_start = gene["START"]
        gene_stop = gene["STOP"]
        gene_name = gene["GENE"]

        positions = RNG.integers(gene_start, max(gene_stop, gene_start + 1), size=n_snps_per_gene)

        if gene_name in signal_set:
            z = RNG.normal(signal_strength, 1.0, size=n_snps_per_gene)
        else:
            z = RNG.normal(0, 1.0, size=n_snps_per_gene)

        p = 2 * sp_stats.norm.sf(np.abs(z))

        for j in range(n_snps_per_gene):
            rows.append({
                "SNP": f"rs{snp_counter}",
                "CHR": gene_chr,
                "BP": int(positions[j]),
                "BETA": z[j] / np.sqrt(50000),
                "SE": 1 / np.sqrt(50000),
                "Z": z[j],
                "P": p[j],
                "N": 50000,
                "A1": "A",
                "A2": "G",
            })
            snp_counter += 1

    return pd.DataFrame(rows)


@pytest.fixture()
def gwas_df():
    return _make_gwas_with_positions()


@pytest.fixture()
def gene_annot():
    return _make_gene_annotations()


@pytest.fixture()
def gene_sets():
    return {
        "pathway_a": [f"GENE{i}" for i in range(1, 11)],
        "pathway_b": [f"GENE{i}" for i in range(5, 20)],
        "pathway_c": [f"GENE{i}" for i in range(20, 35)],
    }


# ---------------------------------------------------------------------------
# Tests: load_gene_annotations
# ---------------------------------------------------------------------------


class TestLoadGeneAnnotations:
    def test_load_from_file(self, tmp_path):
        df = _make_gene_annotations(10)
        path = tmp_path / "genes.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_gene_annotations(path)
        assert len(loaded) == 10
        assert "GENE" in loaded.columns

    def test_load_with_aliases(self, tmp_path):
        df = _make_gene_annotations(5)
        df = df.rename(columns={"GENE": "SYMBOL", "STOP": "END"})
        path = tmp_path / "genes.tsv"
        df.to_csv(path, sep="\t", index=False)
        loaded = load_gene_annotations(path)
        assert "GENE" in loaded.columns
        assert "STOP" in loaded.columns

    def test_no_file_returns_empty(self):
        from pathlib import Path
        loaded = load_gene_annotations(Path("/nonexistent"))
        assert len(loaded) == 0

    def test_none_returns_empty(self):
        loaded = load_gene_annotations(None)
        assert len(loaded) == 0


# ---------------------------------------------------------------------------
# Tests: load_gene_sets
# ---------------------------------------------------------------------------


class TestLoadGeneSets:
    def test_builtin_sets(self):
        gs = load_gene_sets(None)
        assert len(gs) > 0
        assert "glutamate_signaling" in gs
        assert "dopamine_signaling" in gs
        assert "cstc_striatal" in gs

    def test_load_gmt(self, tmp_path):
        path = tmp_path / "sets.gmt"
        with open(path, "w") as f:
            f.write("pathway1\tdescription1\tGENE1\tGENE2\tGENE3\n")
            f.write("pathway2\tdescription2\tGENE4\tGENE5\n")
        gs = load_gene_sets(path)
        assert "pathway1" in gs
        assert len(gs["pathway1"]) == 3
        assert len(gs["pathway2"]) == 2

    def test_load_tsv(self, tmp_path):
        df = pd.DataFrame({
            "GENE_SET": ["setA", "setA", "setB", "setB", "setB"],
            "GENE": ["G1", "G2", "G3", "G4", "G5"],
        })
        path = tmp_path / "sets.tsv"
        df.to_csv(path, sep="\t", index=False)
        gs = load_gene_sets(path)
        assert "setA" in gs
        assert len(gs["setA"]) == 2
        assert len(gs["setB"]) == 3


# ---------------------------------------------------------------------------
# Tests: map_snps_to_genes
# ---------------------------------------------------------------------------


class TestMapSNPsToGenes:
    def test_basic_mapping(self):
        gene_annot = _make_gene_annotations(5)
        gwas = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=10)
        mapping = map_snps_to_genes(gwas, gene_annot, window_kb=10)
        assert len(mapping) > 0
        for gene, snps in mapping.items():
            assert len(snps) > 0

    def test_empty_annotations(self, gwas_df):
        mapping = map_snps_to_genes(gwas_df, pd.DataFrame())
        assert len(mapping) == 0

    def test_no_position_columns(self):
        df = pd.DataFrame({
            "SNP": ["rs1", "rs2"],
            "P": [0.01, 0.05],
        })
        gene_annot = _make_gene_annotations(2)
        mapping = map_snps_to_genes(df, gene_annot)
        assert len(mapping) == 0


# ---------------------------------------------------------------------------
# Tests: gene_analysis_snp_wise
# ---------------------------------------------------------------------------


class TestGeneAnalysis:
    def test_with_gene_annotations(self):
        gene_annot = _make_gene_annotations(20)
        gwas = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=15)
        results = gene_analysis_snp_wise(gwas, gene_annot=gene_annot)
        assert len(results) > 0
        for r in results:
            assert isinstance(r, GeneResult)
            assert r.n_snps > 0
            assert 0 < r.p_value <= 1

    def test_synthetic_blocks_fallback(self, gwas_df):
        # No gene annotations → synthetic blocks
        results = gene_analysis_snp_wise(gwas_df)
        assert len(results) > 0

    def test_sorted_by_pvalue(self):
        gene_annot = _make_gene_annotations(10)
        gwas = _make_gwas_for_gene_analysis(gene_annot)
        results = gene_analysis_snp_wise(gwas, gene_annot=gene_annot)
        p_values = [r.p_value for r in results]
        assert p_values == sorted(p_values)

    def test_signal_genes_rank_higher(self):
        gene_annot = _make_gene_annotations(20)
        signal_genes = ["GENE1", "GENE2"]
        gwas = _make_gwas_for_gene_analysis(
            gene_annot, signal_genes=signal_genes, signal_strength=5.0
        )
        results = gene_analysis_snp_wise(gwas, gene_annot=gene_annot)
        # Top genes should include signal genes
        top_5 = {r.gene for r in results[:5]}
        assert len(top_5 & set(signal_genes)) > 0

    def test_pre_computed_mapping(self):
        gene_annot = _make_gene_annotations(5)
        gwas = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=10)
        mapping = map_snps_to_genes(gwas, gene_annot)
        results = gene_analysis_snp_wise(gwas, gene_snps=mapping)
        assert len(results) == len(mapping)


# ---------------------------------------------------------------------------
# Tests: gene_set_analysis
# ---------------------------------------------------------------------------


class TestGeneSetAnalysis:
    def test_basic_enrichment(self, gene_sets):
        # Create gene results where pathway_a genes have stronger signal
        gene_results = []
        for i in range(1, 51):
            gene_name = f"GENE{i}"
            if i <= 10:  # pathway_a genes
                z = 3.0
            else:
                z = 0.0
            from scipy import stats as sp_stats
            p = float(sp_stats.norm.sf(z))
            gene_results.append(GeneResult(
                gene=gene_name, n_snps=10, stat=z**2, p_value=p, z_score=z,
            ))

        results = gene_set_analysis(gene_results, gene_sets)
        assert len(results) > 0
        # pathway_a should be the most enriched
        assert results[0].gene_set == "pathway_a"
        assert results[0].p_value < 0.05

    def test_no_enrichment(self, gene_sets):
        # All genes have same Z-score → no enrichment
        gene_results = [
            GeneResult(gene=f"GENE{i}", n_snps=10, stat=1.0, p_value=0.5, z_score=0.0)
            for i in range(1, 51)
        ]
        results = gene_set_analysis(gene_results, gene_sets)
        # No pathway should be significant
        for r in results:
            assert r.p_value > 0.01

    def test_fdr_correction(self, gene_sets):
        gene_results = [
            GeneResult(
                gene=f"GENE{i}", n_snps=10, stat=1.0,
                p_value=float(RNG.uniform(0.01, 0.99)),
                z_score=float(RNG.normal()),
            )
            for i in range(1, 51)
        ]
        results = gene_set_analysis(gene_results, gene_sets)
        # FDR Q-values should be >= P-values
        for r in results:
            assert r.fdr_q >= r.p_value - 1e-10

    def test_empty_gene_results(self, gene_sets):
        results = gene_set_analysis([], gene_sets)
        assert results == []

    def test_gene_set_too_small(self):
        gene_results = [
            GeneResult(gene="GENE1", n_snps=10, stat=1.0, p_value=0.5, z_score=0.0),
        ]
        gene_sets = {"tiny_set": ["GENE1"]}  # only 1 gene
        results = gene_set_analysis(gene_results, gene_sets)
        # Set with <2 genes should be skipped
        assert len(results) == 0

    def test_category_assignment(self):
        gene_results = [
            GeneResult(gene=g, n_snps=5, stat=1.0, p_value=0.5, z_score=0.0)
            for g in ALL_BUILTIN_GENE_SETS.get("cstc_cortical", [])
        ]
        # Add more genes so regression works
        for i in range(20):
            gene_results.append(
                GeneResult(gene=f"OTHER{i}", n_snps=5, stat=1.0, p_value=0.5, z_score=0.0)
            )

        results = gene_set_analysis(gene_results, ALL_BUILTIN_GENE_SETS)
        categories = {r.category for r in results}
        assert "cstc" in categories or "neurotransmitter" in categories


# ---------------------------------------------------------------------------
# Tests: run_pathway_analysis (integration)
# ---------------------------------------------------------------------------


class TestRunPathwayAnalysis:
    def test_with_builtin_sets(self, tmp_path):
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        # Use gene names from built-in sets so enrichment can be computed
        builtin_genes = []
        for genes in ALL_BUILTIN_GENE_SETS.values():
            builtin_genes.extend(genes)
        builtin_genes = sorted(set(builtin_genes))[:40]

        gene_annot = _make_gene_annotations(gene_names=builtin_genes, n_chromosomes=5)
        annot_path = tmp_path / "genes.tsv"
        gene_annot.to_csv(annot_path, sep="\t", index=False)

        for name in ["compulsive", "neurodevelopmental", "ts_specific"]:
            df = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=15)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)

        results = run_pathway_analysis(
            gwas_dir=gwas_dir,
            gene_annot_path=annot_path,
            output_dir=output_dir,
        )

        assert len(results) == 3
        for r in results:
            assert isinstance(r, PathwayAnalysisResult)
            assert len(r.gene_results) > 0
            assert len(r.gene_set_results) > 0

        # Check output files
        for stratum in ["compulsive", "neurodevelopmental", "ts_specific"]:
            assert (output_dir / f"gene_results_{stratum}.tsv").exists()
            assert (output_dir / f"gene_set_results_{stratum}.tsv").exists()
        assert (output_dir / "pathway_comparison.tsv").exists()

    def test_with_gene_annotations(self, tmp_path):
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        gene_annot = _make_gene_annotations(30)
        annot_path = tmp_path / "genes.tsv"
        gene_annot.to_csv(annot_path, sep="\t", index=False)

        gwas = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=10)
        gwas.to_csv(gwas_dir / "compulsive.tsv", sep="\t", index=False)

        results = run_pathway_analysis(
            gwas_dir=gwas_dir,
            gene_annot_path=annot_path,
            output_dir=output_dir,
        )

        assert len(results) == 1
        assert results[0].stratum == "compulsive"

    def test_missing_stratum(self, tmp_path):
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        df = _make_gwas_with_positions()
        df.to_csv(gwas_dir / "compulsive.tsv", sep="\t", index=False)

        results = run_pathway_analysis(
            gwas_dir=gwas_dir,
            output_dir=output_dir,
        )

        assert len(results) == 1

    def test_pathway_comparison_output(self, tmp_path):
        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        # Use builtin gene names for proper enrichment
        builtin_genes = []
        for genes in ALL_BUILTIN_GENE_SETS.values():
            builtin_genes.extend(genes)
        builtin_genes = sorted(set(builtin_genes))[:40]

        gene_annot = _make_gene_annotations(gene_names=builtin_genes)
        annot_path = tmp_path / "genes.tsv"
        gene_annot.to_csv(annot_path, sep="\t", index=False)

        for name in ["compulsive", "ts_specific"]:
            df = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=15)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)

        run_pathway_analysis(
            gwas_dir=gwas_dir,
            gene_annot_path=annot_path,
            output_dir=output_dir,
        )

        comparison = pd.read_csv(output_dir / "pathway_comparison.tsv", sep="\t")
        assert "gene_set" in comparison.columns
        assert "p_compulsive" in comparison.columns
        assert "p_ts_specific" in comparison.columns


# ---------------------------------------------------------------------------
# Tests: CLI
# ---------------------------------------------------------------------------


class TestCLI:
    def test_main_runs(self, tmp_path):
        from bioagentics.pipelines.magma_pathway.pipeline import main

        gwas_dir = tmp_path / "gwas"
        gwas_dir.mkdir()
        output_dir = tmp_path / "output"

        # Use builtin gene names so enrichment works
        builtin_genes = []
        for genes in ALL_BUILTIN_GENE_SETS.values():
            builtin_genes.extend(genes)
        builtin_genes = sorted(set(builtin_genes))[:40]

        gene_annot = _make_gene_annotations(gene_names=builtin_genes)
        annot_path = tmp_path / "genes.tsv"
        gene_annot.to_csv(annot_path, sep="\t", index=False)

        for name in ["compulsive", "neurodevelopmental", "ts_specific"]:
            df = _make_gwas_for_gene_analysis(gene_annot, n_snps_per_gene=15)
            df.to_csv(gwas_dir / f"{name}.tsv", sep="\t", index=False)

        main([
            "--gwas-dir", str(gwas_dir),
            "--gene-annot", str(annot_path),
            "--output-dir", str(output_dir),
            "--log-level", "WARNING",
        ])

        assert (output_dir / "pathway_comparison.tsv").exists()
