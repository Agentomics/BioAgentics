"""Tests for Phase 1a MTAP classifier."""

from pathlib import Path

import pandas as pd
import pytest

from bioagentics.config import REPO_ROOT

OUTPUT_PATH = REPO_ROOT / "output" / "mtap-prmt5-nsclc-sl" / "nsclc_cell_lines_classified.csv"


@pytest.fixture(scope="module")
def classified() -> pd.DataFrame:
    """Load the classified cell lines output."""
    if not OUTPUT_PATH.exists():
        pytest.skip("Run 01_mtap_classifier.py first to generate output")
    return pd.read_csv(OUTPUT_PATH, index_col=0)


def test_minimum_nsclc_lines(classified: pd.DataFrame) -> None:
    """Assert >= 80 NSCLC lines found."""
    assert len(classified) >= 80, f"Only {len(classified)} NSCLC lines found"


def test_mtap_deletion_rate(classified: pd.DataFrame) -> None:
    """Assert MTAP deletion rate is roughly 15-18% (allow 8-30% range)."""
    rate = classified["MTAP_deleted"].mean()
    assert 0.08 <= rate <= 0.30, f"MTAP deletion rate {rate:.1%} outside expected range"


def test_deleted_lines_have_lower_expression(classified: pd.DataFrame) -> None:
    """Assert MTAP-deleted lines have significantly lower expression."""
    from scipy import stats

    has_data = classified.dropna(subset=["MTAP_deleted", "MTAP_expression"])
    deleted = has_data[has_data["MTAP_deleted"]]["MTAP_expression"]
    intact = has_data[~has_data["MTAP_deleted"]]["MTAP_expression"]

    assert len(deleted) >= 5, "Too few deleted lines for statistical test"
    stat, pval = stats.mannwhitneyu(deleted, intact, alternative="less")
    assert pval < 0.01, f"Deleted lines not significantly lower expression (p={pval:.4f})"


def test_required_columns(classified: pd.DataFrame) -> None:
    """Assert all required columns are present."""
    required = [
        "CellLineName", "MTAP_CN_log2", "MTAP_deleted", "MTAP_expression",
        "KRAS_status", "KRAS_allele", "STK11_mut", "KEAP1_mut",
        "TP53_mut", "NFE2L2_mut",
    ]
    missing = [c for c in required if c not in classified.columns]
    assert not missing, f"Missing columns: {missing}"


def test_nfe2l2_column_present(classified: pd.DataFrame) -> None:
    """Assert NFE2L2 mutation column is present and boolean."""
    assert "NFE2L2_mut" in classified.columns
    assert classified["NFE2L2_mut"].dtype == bool or set(classified["NFE2L2_mut"].unique()).issubset(
        {True, False}
    )
