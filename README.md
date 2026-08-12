# Regime-Lab: Quantitative Market Regime Detection & Backtesting Engine

An end-to-end quantitative trading framework that classifies financial market regimes using temporal CNN embeddings and Bayesian Neural Networks (BNN) with Monte Carlo Dropout, executing dynamic risk-managed asset allocation strategies.

## Key Features
- **Point-In-Time Data Engine**: Multi-calendar alignment eliminating lookahead bias across global equity and volatility markets.
- **1D-CNN Temporal Sequence Extractor**: Learns 64-dimensional latent temporal embeddings across sliding lookback windows.
- **Hybrid Bayesian Classifier**: Combines raw point-in-time features with temporal embeddings to output regime probabilities and separate epistemic vs. aleatoric uncertainty.
- **Dynamic Backtest Engine**: Backtests market regimes against benchmark indices with configurable transaction costs (bps), computing Sharpe ratio, maximum drawdown, and volatility.
- **Interactive Streamlit Dashboard**: Real-time visualization of latent regime probabilities, uncertainty metrics, and dynamic allocation performance.

## Project Structure
```text
regime-lab/
├── src/
│   ├── app.py                   # Streamlit quantitative dashboard
│   ├── backtest_engine.py       # Dynamic allocation & backtesting engine
│   ├── bnn_classifier.py        # Bayesian Neural Network implementation
│   ├── data_ingestion.py        # Real-time data loader (yfinance)
│   ├── data_pipeline.py         # Point-in-time synchronization pipeline
│   ├── embedding_extractor.py   # 1D-CNN sliding window encoder
│   ├── hybrid_pipeline.py       # Combined embedding + raw classifier
│   └── regime_models.py         # Baseline GMM / HMM models
├── tests/
│   └── test_pipeline.py         # PyTest automated unit test suite
├── main.py                      # CLI entry point
├── requirements.txt             # Environment dependencies
└── README.md                    # Project documentation
