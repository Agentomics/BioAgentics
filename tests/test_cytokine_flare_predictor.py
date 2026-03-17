"""Tests for cytokine_flare_predictor module."""

from __future__ import annotations

from bioagentics.cytokine_extraction import CytokineDataset, CytokineRecord
from bioagentics.cytokine_flare_predictor import (
    FlarePredictor,
    build_feature_matrix,
    results_summary,
)


def _make_dataset() -> CytokineDataset:
    """Create a test dataset with 4 studies, 3 analytes, flare + remission."""
    records = []
    studies = [
        ("S1", {"IL-6": (15, 3), "TNF-α": (10, 2), "IL-10": (5, 1)},
         {"IL-6": (8, 2), "TNF-α": (6, 1.5), "IL-10": (8, 2)}),
        ("S2", {"IL-6": (18, 5), "TNF-α": (12, 3), "IL-10": (4, 1.5)},
         {"IL-6": (9, 3), "TNF-α": (7, 2), "IL-10": (9, 2.5)}),
        ("S3", {"IL-6": (14, 4), "TNF-α": (9, 2.5), "IL-10": (6, 1)},
         {"IL-6": (7, 2.5), "TNF-α": (5, 1), "IL-10": (7, 1.5)}),
        ("S4", {"IL-6": (16, 3.5), "TNF-α": (11, 2), "IL-10": (3, 1)},
         {"IL-6": (10, 2), "TNF-α": (8, 2), "IL-10": (10, 3)}),
    ]
    for sid, flare_vals, rem_vals in studies:
        for analyte, (m, s) in flare_vals.items():
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="flare", sample_size_n=20,
                mean_or_median=m, sd_or_iqr=s,
            ))
        for analyte, (m, s) in rem_vals.items():
            records.append(CytokineRecord(
                study_id=sid, analyte_name=analyte, measurement_method="ELISA",
                sample_type="serum", condition="remission", sample_size_n=20,
                mean_or_median=m, sd_or_iqr=s,
            ))
    return CytokineDataset(records)


def test_build_feature_matrix():
    ds = _make_dataset()
    X, y, study_ids = build_feature_matrix(ds)
    assert X.shape[0] == 8  # 4 studies × 2 conditions
    assert X.shape[1] == 3  # 3 analytes
    assert y.sum() == 4  # 4 flare samples
    assert len(study_ids.unique()) == 4


def test_flare_predictor_run():
    ds = _make_dataset()
    predictor = FlarePredictor(ds)
    results = predictor.run()
    assert len(results) == 2  # LR + RF
    for r in results:
        assert 0 <= r.auc_score <= 1
        assert r.cv_folds == 4


def test_roc_plot(tmp_path):
    ds = _make_dataset()
    predictor = FlarePredictor(ds)
    results = predictor.run()
    out = predictor.plot_roc(results, output_path=tmp_path / "roc.png")
    assert out is not None
    assert out.exists()


def test_feature_importance_plot(tmp_path):
    ds = _make_dataset()
    predictor = FlarePredictor(ds)
    results = predictor.run()
    # Random forest result
    rf_result = [r for r in results if "RandomForest" in r.model_name][0]
    out = predictor.plot_feature_importance(rf_result, output_path=tmp_path / "fi.png")
    assert out is not None
    assert out.exists()


def test_results_summary():
    ds = _make_dataset()
    predictor = FlarePredictor(ds)
    results = predictor.run()
    df = results_summary(results)
    assert len(df) == 2
    assert "auc" in df.columns
    assert "model" in df.columns
