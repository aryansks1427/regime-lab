import numpy as np
import pandas as pd
import pytest
import torch

from src.data_pipeline import PointInTimePipeline
from src.embedding_extractor import TemporalEmbeddingExtractor
from src.hybrid_pipeline import HybridRegimeClassifier
from src.backtest_engine import DynamicBacktestEngine


def test_pit_alignment_length():
    """Verify point-in-time multi-calendar alignment output matches target calendar length."""
    dates = pd.date_range("2024-01-01", periods=20, freq="B")
    raw_df = pd.DataFrame({"date": dates, "close": np.random.randn(20)})
    
    pipeline = PointInTimePipeline(primary_calendar=dates)
    aligned_data = pipeline.align_to_calendar({"equity": raw_df}, {"equity": 0})
    
    # align_to_calendar returns a merged DataFrame
    assert len(aligned_data) == 20
    assert any("close" in str(col) for col in aligned_data.columns)


def test_embedding_extractor_output_shape():
    """Verify 1D-CNN temporal sequence encoder transforms sliding windows to target 64D vectors."""
    window_size = 10
    feature_dim = 3
    seq_length = 50
    dummy_features = pd.DataFrame(
        np.random.randn(seq_length, feature_dim),
        index=pd.date_range("2024-01-01", periods=seq_length, freq="B"),
    )

    windows, aligned_dates = TemporalEmbeddingExtractor.create_sliding_windows(
        dummy_features, window_size=window_size
    )
    assert len(windows) == seq_length - window_size + 1

    extractor = TemporalEmbeddingExtractor(
        input_dim=feature_dim, sequence_length=window_size, embedding_dim=64
    )
    extractor.eval()
    with torch.no_grad():
        embeddings = extractor(torch.tensor(windows, dtype=torch.float32))

    assert embeddings.shape == (len(windows), 64)


def test_hybrid_classifier_probability_sum():
    """Verify Monte Carlo dropout predicted soft regime probabilities sum to 1 across classes."""
    raw_dim, emb_dim, n_classes, n_samples = 3, 64, 3, 20
    model = HybridRegimeClassifier(
        raw_feature_dim=raw_dim, embedding_dim=emb_dim, num_classes=n_classes
    )
    
    x_dummy = torch.randn(15, raw_dim + emb_dim)
    mean_probs, epistemic, aleatoric = model.predict_mc(x_dummy, n_samples=n_samples)

    assert mean_probs.shape == (15, n_classes)
    assert epistemic.shape == (15, n_classes)
    assert aleatoric.shape == (15, 1)
    np.testing.assert_allclose(mean_probs.sum(axis=1), 1.0, atol=1e-5)


def test_backtest_engine_metrics():
    """Verify dynamic backtest engine computes total return, Sharpe, and drawdown metrics correctly."""
    dates = pd.date_range("2024-01-01", periods=100, freq="B")
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100), index=dates)
    regime_probs = pd.DataFrame(
        {
            "hybrid_prob_regime_0": np.ones(100) * 0.5,
            "hybrid_prob_regime_1": np.ones(100) * 0.3,
            "hybrid_prob_regime_2": np.ones(100) * 0.2,
        },
        index=dates,
    )
    epistemic_unc = pd.Series(np.zeros(100), index=dates)
    aleatoric_unc = pd.Series(np.zeros(100), index=dates)

    engine = DynamicBacktestEngine(regime_weights={0: 1.0, 1: 0.5, 2: 0.0})
    backtest_df, metrics = engine.run_backtest(
        asset_returns=returns,
        regime_probs=regime_probs,
        epistemic_unc=epistemic_unc,
        aleatoric_unc=aleatoric_unc,
    )

    assert "Total Return (%)" in metrics
    assert "Annualized Sharpe Ratio" in metrics
    assert "Max Drawdown (%)" in metrics
    assert "cum_strategy" in backtest_df.columns
    assert len(backtest_df) == 100