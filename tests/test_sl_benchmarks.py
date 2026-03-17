"""Tests for SL benchmark integration module."""

from __future__ import annotations

import pandas as pd
import pytest

from bioagentics.models.sl_benchmarks import (
    _load_desjardins,
    _load_genome_biology,
    _load_vermeulen,
    load_sl_benchmarks,
)


@pytest.fixture
def benchmark_dir(tmp_path):
    """Create minimal benchmark data for testing."""
    # Genome Biology
    gb_dir = tmp_path / "sl_pairs"
    gb_dir.mkdir()
    pd.DataFrame({
        "gene_pair": ["A|B", "C|D", "E|F"],
        "gene_1": ["A", "C", "E"],
        "gene_2": ["B", "D", "F"],
        "hit_count": [3, 0, 1],
        "cell_lines_tested": [10, 10, 10],
    }).to_csv(gb_dir / "sl_pairs_summary.tsv", sep="\t", index=False)

    # Vermeulen
    vm_dir = tmp_path / "vermeulen_sl"
    vm_dir.mkdir()
    pd.DataFrame({
        "source": ["KRAS", "TP53", "BRAF"],
        "target": ["G", "H", "I"],
        "cancer_type": ["PAN", "PAN", "PAN"],
        "included": ["included", "included", "excluded"],
    }).to_csv(vm_dir / "vermeulen_sl_interactions.tsv", sep="\t", index=False)

    # Desjardins — write a small xlsx with 2 sheets
    dj_dir = tmp_path / "desjardins_isogenic_sl"
    dj_dir.mkdir()
    with pd.ExcelWriter(dj_dir / "Table1_isogenic_depmap_SL_screen_data.xlsx") as writer:
        pd.DataFrame({
            "Gene": ["X", "Y", "Z"],
            "Number of sgRNAs": [4, 4, 4],
            "CCA": [1.5, 0.8, 1.1],
            "BAGEL2 BFS (Test-Control)": [-10, 2, -5],
            "Median Chronos Score (Test-Control)": [-0.5, 0.1, -0.3],
            "Lesion": ["STK11", "STK11", "STK11"],
            "Parental Cell Line": ["RPE1", "RPE1", "RPE1"],
        }).to_excel(writer, sheet_name="STK11", index=False)

        pd.DataFrame({
            "Gene": ["A", "W"],
            "Number of sgRNAs": [4, 4],
            "CCA": [1.2, 0.5],
            "BAGEL2 BFS (Test-Control)": [-8, 3],
            "Median Chronos Score (Test-Control)": [-0.4, 0.2],
            "Lesion": ["KEAP1", "KEAP1"],
            "Parental Cell Line": ["RPE1", "RPE1"],
        }).to_excel(writer, sheet_name="KEAP1", index=False)

    return tmp_path


def test_load_genome_biology(benchmark_dir):
    df = _load_genome_biology(benchmark_dir, min_hits=1)
    assert len(df) == 2  # hit_count 3 and 1
    assert set(df.columns) >= {"gene_a", "gene_b", "source"}
    assert (df["source"] == "genome_biology").all()


def test_load_vermeulen(benchmark_dir):
    df = _load_vermeulen(benchmark_dir)
    assert len(df) == 2  # 2 included
    assert set(df["gene_a"]) == {"KRAS", "TP53"}


def test_load_desjardins(benchmark_dir):
    df = _load_desjardins(benchmark_dir, min_cca=1.0)
    assert len(df) == 3  # STK11: X,Z; KEAP1: A
    assert set(df["gene_a"]) == {"STK11", "KEAP1"}


def test_load_sl_benchmarks(benchmark_dir):
    df = load_sl_benchmarks(benchmark_dir)
    # 2 GB + 2 Vermeulen + 3 Desjardins = 7 total, minus any cross-source overlaps
    assert len(df) > 0
    assert "n_sources" in df.columns
    assert "sources" in df.columns
    assert set(df["confidence_tier"].unique()) <= {"experimental", "isogenic", "computational"}


def test_deduplication(benchmark_dir):
    df = load_sl_benchmarks(benchmark_dir)
    # Each unique gene pair should appear only once
    pairs = df.apply(lambda r: tuple(sorted([r["gene_a"], r["gene_b"]])), axis=1)
    assert pairs.duplicated().sum() == 0
