"""Tests for LSTM/GRU temporal models."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from bioagentics.diagnostics.sepsis.models.temporal import (
    SepsisRNN,
    build_sequences,
    evaluate_temporal_nested_cv,
    train_and_evaluate,
    TEMPORAL_PARAM_GRID,
)


@pytest.fixture
def synthetic_dataset():
    """Synthetic binary classification dataset."""
    rng = np.random.default_rng(42)
    n = 150
    n_features = 8
    X = rng.standard_normal((n, n_features)).astype(np.float32)
    y = ((X[:, 0] + X[:, 1]) > 0).astype(np.int64)
    return X, y


def test_sepsis_rnn_lstm_forward():
    """LSTM model produces correct output shape."""
    model = SepsisRNN(input_size=8, hidden_size=16, rnn_type="lstm")
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.shape == (4,)


def test_sepsis_rnn_gru_forward():
    """GRU model produces correct output shape."""
    model = SepsisRNN(input_size=8, hidden_size=16, rnn_type="gru")
    x = torch.randn(4, 24, 8)
    out = model(x)
    assert out.shape == (4,)


def test_build_sequences(synthetic_dataset):
    """Sequences have correct shape (n, lookback, features)."""
    X, y = synthetic_dataset
    X_seq, y_seq = build_sequences(X, y, lookback=12)
    assert X_seq.shape == (len(X), 12, X.shape[1])
    assert len(y_seq) == len(y)
    assert X_seq.dtype == np.float32


def test_build_sequences_last_timestep(synthetic_dataset):
    """Last timestep of sequence should approximate the original features."""
    X, y = synthetic_dataset
    X_seq, _ = build_sequences(X, y, lookback=24)
    # Last timestep (alpha=1.0) should be close to original
    np.testing.assert_allclose(X_seq[:, -1, :], X, atol=0.05)


def test_train_and_evaluate(synthetic_dataset):
    """Training and evaluation produces valid metrics."""
    X, y = synthetic_dataset
    X_seq, _ = build_sequences(X, y, lookback=12)

    n_train = 100
    _, metrics = train_and_evaluate(
        X_seq[:n_train], y[:n_train],
        X_seq[n_train:], y[n_train:],
        TEMPORAL_PARAM_GRID[0], rnn_type="lstm",
    )
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert 0.0 <= metrics["auprc"] <= 1.0


def test_evaluate_temporal_nested_cv_lstm(synthetic_dataset):
    """LSTM nested CV returns valid metrics."""
    X, y = synthetic_dataset
    result = evaluate_temporal_nested_cv(
        X, y, rnn_type="lstm", n_outer_folds=2, n_inner_folds=2,
    )
    assert result["model"] == "lstm"
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert 0.0 <= result["auprc_mean"] <= 1.0
    assert len(result["fold_results"]) == 2


def test_evaluate_temporal_nested_cv_gru(synthetic_dataset):
    """GRU nested CV returns valid metrics."""
    X, y = synthetic_dataset
    result = evaluate_temporal_nested_cv(
        X, y, rnn_type="gru", n_outer_folds=2, n_inner_folds=2,
    )
    assert result["model"] == "gru"
    assert 0.0 <= result["auroc_mean"] <= 1.0
    assert len(result["fold_results"]) == 2
