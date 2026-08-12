# Regime Lab: Quantitative Market Regime Detection Engine

A quantitative framework using Hidden Markov Models (HMM) to classify financial market regimes (Bull, Neutral, Bear) and execute dynamic asset allocation strategies.

## Key Features
- **Point-In-Time (PIT) Alignment:** Prevents look-ahead bias in macroeconomic feature processing.
- **Unsupervised Regime Identification:** Uses `hmmlearn` to fit Gaussian Hidden Markov Models.
- **Automated State Ordering & Smoothing:** Maps states strictly by mean return expectation and applies rolling probability smoothing to minimize whipsaw trading.
- **Backtesting & Analytics Engine:** Computes risk-adjusted performance including Sharpe Ratio, Max Drawdown, and CAGR.
- **Interactive Monitoring Dashboard:** Built with Streamlit and Plotly for real-time scenario modeling.

## Installation & Execution

1. Clone the repository:
   ```bash
   git clone [https://github.com/aryansks1427/regime-lab.git](https://github.com/aryansks1427/regime-lab.git)
   cd regime-lab