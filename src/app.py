import os
import sys

# Add project root directory to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.pit_data_engine import fetch_market_data
from src.regime_models import GaussianHMMRegimeDetector
from src.backtest_engine import RegimeAwareBacktestEngine as RegimeBacktester

st.set_page_config(page_title="Regime Lab | Dynamic Market Allocation", layout="wide")

st.title("📈 Regime Lab: Market Regime Detection & Dynamic Backtesting")

# --- Sidebar Controls ---
st.sidebar.header("Data & Model Settings")
ticker_option = st.sidebar.selectbox("Select Asset / Index", ["^NSEI (Nifty 50)", "^GSPC (S&P 500)", "AAPL", "NVDA"], index=0)
ticker_symbol = ticker_option.split(" ")[0]

n_states = st.sidebar.slider("Number of Regimes (HMM States)", min_value=2, max_value=4, value=3)
smooth_win = st.sidebar.slider("Regime Probability Smoothing Window", min_value=1, max_value=21, value=5)
initial_cap = st.sidebar.number_input("Initial Capital ($)", value=100000)

# --- Fetch Data & Run Model ---
with st.spinner("Downloading market data and running regime estimation..."):
    df = fetch_market_data(ticker=ticker_symbol, start_date="2018-01-01", end_date="2024-01-01")
    
    detector = GaussianHMMRegimeDetector(n_components=n_states)
    regime_df = detector.fit_predict(df, smooth_window=smooth_win)
    
    # Weight map: State 0 (Bull) -> 100%, State 1 -> 50%, State 2 -> 0%
    weight_map = {0: 1.0, 1: 0.5, 2: 0.0, 3: 0.0}
    backtester = RegimeBacktester(initial_capital=initial_cap)
    final_df = backtester.run_backtest(regime_df, weight_map)
    metrics = backtester.calculate_metrics(final_df)

# --- Top Key Performance Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Strategy Value", f"${final_df['equity_curve'].iloc[-1]:,.2f}")
c2.metric("Benchmark Value", f"${final_df['benchmark_curve'].iloc[-1]:,.2f}")
c3.metric("Strategy Sharpe Ratio", metrics["Strategy Sharpe"])
c4.metric("Strategy Max Drawdown", metrics["Strategy Max Drawdown"])

# --- Interactive Charts ---
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08, subplot_titles=("Portfolio Performance vs Benchmark", "Detected Market Regimes"))

fig.add_trace(go.Scatter(x=final_df.index, y=final_df['equity_curve'], name="Regime Strategy", line=dict(color='#00CC96', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=final_df.index, y=final_df['benchmark_curve'], name="Buy & Hold Benchmark", line=dict(color='#636EFA', width=1.5, dash='dash')), row=1, col=1)

# Heatmap regime line
fig.add_trace(go.Scatter(x=final_df.index, y=final_df['regime_state'], name="Regime State (0=Bull, 2=Bear)", line=dict(color='#EF553B', width=1.5)), row=2, col=1)

fig.update_layout(height=650, template="plotly_dark", margin=dict(l=40, r=40, t=60, b=40))
st.plotly_chart(fig, use_container_width=True)

# --- Detailed Risk Metrics Table ---
st.subheader("📊 Performance & Risk Metrics Comparison")
metrics_df = pd.DataFrame({
    "Metric": ["CAGR", "Annualized Volatility", "Sharpe Ratio", "Max Drawdown"],
    "Regime Strategy": [metrics["Strategy CAGR"], metrics["Strategy Volatility"], metrics["Strategy Sharpe"], metrics["Strategy Max Drawdown"]],
    "Benchmark (Buy & Hold)": [metrics["Benchmark CAGR"], metrics["Benchmark Volatility"], metrics["Benchmark Sharpe"], metrics["Benchmark Max Drawdown"]]
})
st.table(metrics_df)
