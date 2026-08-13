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

    # Robustly find or calculate returns dataframe
    if 'Return' in regime_df.columns:
        returns_df = regime_df[['Return']]
    elif 'Close' in regime_df.columns:
        returns_df = regime_df[['Close']].pct_change().dropna()
    elif 'Close' in df.columns:
        returns_df = df[['Close']].pct_change().dropna()
    else:
        # Fallback to the first numeric column
        first_num_col = df.select_dtypes(include=[np.number]).columns[0]
        returns_df = df[[first_num_col]].pct_change().dropna()

   # Align regime probabilities to matching index
aligned_regime_df = regime_df.loc[returns_df.index]
prob_cols = [col for col in aligned_regime_df.columns if 'prob' in str(col).lower() or isinstance(col, int)]
regime_probs_df = aligned_regime_df[prob_cols] if prob_cols else aligned_regime_df

prediction_sets = [list(range(n_states))] * len(returns_df)

# Initialize backtester (unindented at module level)
backtester = RegimeBacktester(initial_capital=initial_cap)

# Execute backtest with matching dimensions
results = backtester.run_backtest(
    returns_df=returns_df,
    regime_probs_df=regime_probs_df,
    prediction_sets=prediction_sets
)

# Extract performance dataframe
perf_df = results["performance"]

# Compute metrics using backtest_engine's exact static method
metrics = RegimeBacktester.calculate_performance_metrics(perf_df)

# --- Top Key Performance Metrics ---
c1, c2, c3, c4 = st.columns(4)
c1.metric("Final Portfolio Value", f"${perf_df['portfolio_value'].iloc[-1]:,.2f}")
c2.metric("Annualized Return", f"{metrics['Annualized Return']*100:.2f}%")
c3.metric("Sharpe Ratio", f"{metrics['Sharpe Ratio']:.2f}")
c4.metric("Max Drawdown", f"{metrics['Max Drawdown']*100:.2f}%")

# --- Interactive Charts ---
fig = make_subplots(
    rows=2, cols=1, 
    shared_xaxes=True, 
    vertical_spacing=0.08, 
    subplot_titles=("Portfolio Performance", "Detected Market Regimes")
)

# Portfolio Equity Curve
fig.add_trace(
    go.Scatter(
        x=perf_df.index, 
        y=perf_df['portfolio_value'], 
        name="Regime Strategy", 
        line=dict(color='#00CC96', width=2)
    ), 
    row=1, col=1
)

# Regime State Line
regime_states = regime_probs_df.values.argmax(axis=1) if hasattr(regime_probs_df, 'values') else 0
fig.add_trace(
    go.Scatter(
        x=perf_df.index, 
        y=regime_states, 
        name="Regime State", 
        line=dict(color='#EF553B', width=1.5)
    ), 
    row=2, col=1
)

fig.update_layout(height=650, template="plotly_dark", margin=dict(l=40, r=40, t=60, b=40))
st.plotly_chart(fig, use_container_width=True)

# --- Detailed Risk Metrics Table ---
st.subheader("📊 Performance & Risk Metrics")
metrics_df = pd.DataFrame({
    "Metric": ["Annualized Return", "Annualized Volatility", "Sharpe Ratio", "Sortino Ratio", "Max Drawdown"],
    "Value": [
        f"{metrics['Annualized Return']*100:.2f}%",
        f"{metrics['Annualized Volatility']*100:.2f}%",
        f"{metrics['Sharpe Ratio']:.2f}",
        f"{metrics['Sortino Ratio']:.2f}",
        f"{metrics['Max Drawdown']*100:.2f}%"
    ]
})
st.table(metrics_df)
