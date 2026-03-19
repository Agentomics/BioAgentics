"""Tests for NOD2 variant functional impact pipeline scripts.

Tests the project-specific pipeline scripts at
src/crohns/nod2_variant_functional_impact/.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import pandas as pd
import pytest


def _import_pipeline(name: str):
    """Import a pipeline module by numeric name (e.g. '01_data_collection')."""
    return importlib.import_module(f"crohns.nod2_variant_functional_impact.{name}")


# ---- 01_data_collection ----

def test_add_domain_location():
    """Test domain location annotation."""
    mod = _import_pipeline("01_data_collection")

    df = pd.DataFrame({
        "hgvs_p": ["R702W", "R334Q", "L1007fs", "", "G908R"],
    })
    result = mod.add_domain_location(df)
    assert "domain_location" in result.columns
    assert len(result) == 5
    assert all(isinstance(v, str) for v in result["domain_location"])


# ---- 03_conservation_features ----

def test_extract_conservation_features(tmp_path):
    """Test conservation feature extraction from predictor scores."""
    mod = _import_pipeline("03_conservation_features")

    scores_df = pd.DataFrame({
        "chrom": ["16", "16", "16"],
        "pos": [50710000, 50720000, 50730000],
        "ref": ["A", "G", "C"],
        "alt": ["G", "A", "T"],
        "phylop_100way": [5.2, -1.0, 8.5],
        "gerp_rs": [4.1, -0.5, 5.8],
        "phastcons_100way": [0.95, 0.2, 0.99],
    })
    scores_path = tmp_path / "scores.tsv"
    scores_df.to_csv(scores_path, sep="\t", index=False)

    output_path = tmp_path / "conservation.tsv"
    result = mod.extract_conservation_features(scores_path, output_path)

    assert len(result) == 3
    assert "phylop_100way" in result.columns
    assert "gerp_rs" in result.columns
    assert "conservation_score" in result.columns
    assert output_path.exists()


# ---- 06_training_set ----

def test_extract_fs_pos():
    """Test frameshift position extraction."""
    mod = _import_pipeline("06_training_set")

    assert mod._extract_fs_pos("L1007fs") == 1007
    assert mod._extract_fs_pos("R702W") is None
    assert mod._extract_fs_pos("") is None


# ---- Pipeline modules are importable ----

@pytest.mark.parametrize("module_name", [
    "crohns.nod2_variant_functional_impact",
    "crohns.nod2_variant_functional_impact.01_data_collection",
    "crohns.nod2_variant_functional_impact.02_structure_scores",
    "crohns.nod2_variant_functional_impact.03_conservation_features",
    "crohns.nod2_variant_functional_impact.04_structural_features",
    "crohns.nod2_variant_functional_impact.05_girdin_features",
    "crohns.nod2_variant_functional_impact.06_training_set",
    "crohns.nod2_variant_functional_impact.07_model",
    "crohns.nod2_variant_functional_impact.08_vus_classification",
])
def test_pipeline_modules_importable(module_name):
    """Verify all pipeline modules can be imported."""
    mod = importlib.import_module(module_name)
    assert mod is not None
