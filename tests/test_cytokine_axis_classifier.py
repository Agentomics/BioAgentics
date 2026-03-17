"""Tests for cytokine_axis_classifier module."""

from __future__ import annotations

from bioagentics.cytokine_axis_classifier import (
    IMMUNE_AXES,
    classify_axes,
    classification_to_dict,
    radar_plot,
)
from bioagentics.cytokine_meta_analysis import MetaAnalysisResult


def _make_results() -> list[MetaAnalysisResult]:
    """Create mock meta-analysis results."""
    return [
        MetaAnalysisResult(analyte="IL-6", k=4, pooled_effect=1.5, p_value=0.001),
        MetaAnalysisResult(analyte="IL-1β", k=3, pooled_effect=0.9, p_value=0.01),
        MetaAnalysisResult(analyte="TNF-α", k=3, pooled_effect=0.7, p_value=0.02),
        MetaAnalysisResult(analyte="IFN-γ", k=3, pooled_effect=0.5, p_value=0.04),
        MetaAnalysisResult(analyte="IL-17A", k=3, pooled_effect=0.8, p_value=0.005),
        MetaAnalysisResult(analyte="IL-10", k=3, pooled_effect=-0.3, p_value=0.1),
        MetaAnalysisResult(analyte="IL-4", k=3, pooled_effect=0.1, p_value=0.5),
    ]


def test_classify_axes():
    results = _make_results()
    classification = classify_axes(results)
    assert classification.dominant_axis != ""
    assert len(classification.scores) == len(IMMUNE_AXES)


def test_innate_axis_highest():
    """With IL-6=1.5, IL-1β=0.9, TNF-α=0.7, innate should dominate."""
    results = _make_results()
    classification = classify_axes(results)
    innate = classification.scores["Innate"]
    assert innate.composite_score > 0
    assert innate.n_measured >= 2
    assert classification.dominant_axis == "Innate"


def test_significant_count():
    results = _make_results()
    classification = classify_axes(results)
    innate = classification.scores["Innate"]
    # IL-6, IL-1β, TNF-α all significant
    assert innate.n_significant >= 2


def test_unmeasured_axis():
    """An axis with no measured members should have score 0."""
    results = [MetaAnalysisResult(analyte="IL-6", k=3, pooled_effect=1.0, p_value=0.01)]
    classification = classify_axes(results)
    th2 = classification.scores["Th2"]
    assert th2.composite_score == 0.0
    assert th2.n_measured == 0


def test_radar_plot(tmp_path):
    results = _make_results()
    classification = classify_axes(results)
    out = radar_plot(classification, output_path=tmp_path / "radar.png")
    assert out.exists()


def test_classification_to_dict():
    results = _make_results()
    classification = classify_axes(results)
    d = classification_to_dict(classification)
    assert "dominant_axis" in d
    assert "axes" in d
    assert "Innate" in d["axes"]
    assert "composite_score" in d["axes"]["Innate"]
