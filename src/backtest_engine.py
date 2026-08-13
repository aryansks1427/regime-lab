import numpy as np
import pandas as pd
from typing import Dict, Tuple

# =====================================================================
# 1. REGIME-AWARE PORTFOLIO OPTIMIZER & BACKTESTER
# =====================================================================

class RegimeAwareBacktestEngine:
    r"""
    Executes backtests by dynamically shifting portfolio risk allocation
    based on calibrated regime probabilities and conformal uncertainty bounds.
    """
    def __init__(
        self,
        initial_capital: float = 1000000.0,
        transaction_cost_bps: float = 10.0,  # 10 bps per trade
        max_turnover: float = 0.20           # 20% max turnover per rebalance
    ):
        self.initial_capital = initial_capital
        self.tc_factor = transaction_cost_bps / 10000.0
        self.max_turnover = max_turnover

    def compute_regime_weights(
        self,
        cov_matrix: np.ndarray,
        regime_probs: np.ndarray,
        prediction_set_size: int
    ) -> np.ndarray:
        r"""
        Calculates asset weights using Inverse Volatility / Minimum Variance,
        scaled down during high uncertainty (large conformal prediction sets).
        """
        n_assets = cov_matrix.shape[0]
        
        # Base inverse-variance allocation
        asset_vols = np.sqrt(np.diag(cov_matrix))
        asset_vols[asset_vols == 0] = 1e-5
        inv_vol_weights = (1.0 / asset_vols) / np.sum(1.0 / asset_vols)

        # Shift defensive bias based on regime probabilities
        # regime 0 = Bear, 1 = Neutral, 2 = Bull
        p_bear, p_neutral, p_bull = regime_probs[0], regime_probs[1], regime_probs[2]
        
        # Exposure multiplier: full exposure on Bull, scaled down on Bear
        exposure_multiplier = p_bull * 1.0 + p_neutral * 0.6 + p_bear * 0.1

        # Uncertainty penalty: drop exposure if conformal set includes multiple regimes
        if prediction_set_size > 1:
            exposure_multiplier *= 0.7  # 30% risk haircut on high uncertainty

        target_weights = inv_vol_weights * exposure_multiplier
        return target_weights

    def run_backtest(
        self,
        returns_df: pd.DataFrame,
        regime_probs_df: pd.DataFrame,
        prediction_sets: list
    ) -> Dict[str, pd.DataFrame]:
        r"""
        Executes dynamic rebalancing over time tracking equity curve and transaction costs.
        """
        n_obs, n_assets = returns_df.shape
        asset_names = returns_df.columns

        portfolio_values = [self.initial_capital]
        weights_history = []
        turnover_history = []
        tc_history = []

        current_weights = np.zeros(n_assets)

        for t in range(n_obs):
            date = returns_df.index[t]
            daily_returns = returns_df.iloc[t].values
            
            # Compute rolling covariance for asset allocation
            if t >= 30:
                roll_cov = returns_df.iloc[t - 30 : t].cov().values
            else:
                roll_cov = np.eye(n_assets) * 0.0001

            probs_t = regime_probs_df.iloc[t].values
            pred_set_len = len(prediction_sets[t]) if t < len(prediction_sets) else 1

            # Target weights calculation
            target_w = self.compute_regime_weights(roll_cov, probs_t, pred_set_len)

            # Apply Turnover Constraint
            weight_change = target_w - current_weights
            turnover = np.sum(np.abs(weight_change))
            if turnover > self.max_turnover and turnover > 0:
                target_w = current_weights + weight_change * (self.max_turnover / turnover)
                turnover = self.max_turnover

            # Transaction Costs
            cost = portfolio_values[-1] * turnover * self.tc_factor
            tc_history.append(cost)
            turnover_history.append(turnover)

            # Update Portfolio Value
            port_return = np.sum(target_w * daily_returns)
            new_val = (portfolio_values[-1] - cost) * (1.0 + port_return)
            portfolio_values.append(new_val)

            current_weights = target_w
            weights_history.append(target_w)

        # Performance DataFrame
        perf_df = pd.DataFrame({
            "portfolio_value": portfolio_values[1:],
            "turnover": turnover_history,
            "transaction_costs": tc_history
        }, index=returns_df.index)

        weights_df = pd.DataFrame(weights_history, index=returns_df.index, columns=asset_names)

        return {"performance": perf_df, "weights": weights_df}

    @staticmethod
    def calculate_performance_metrics(perf_df: pd.DataFrame) -> Dict[str, float]:
        r"""
        Computes Sharpe Ratio, Sortino Ratio, Max Drawdown, and Annualized Return.
        """
        p_vals = perf_df["portfolio_value"].values
        daily_returns = np.diff(p_vals) / p_vals[:-1]

        ann_return = np.mean(daily_returns) * 252
        ann_vol = np.std(daily_returns) * np.sqrt(252)
        sharpe = ann_return / ann_vol if ann_vol > 0 else 0.0

        downside_returns = daily_returns[daily_returns < 0]
        downside_std = np.std(downside_returns) * np.sqrt(252) if len(downside_returns) > 0 else 1e-5
        sortino = ann_return / downside_std

        cum_max = np.maximum.accumulate(p_vals)
        drawdowns = (p_vals - cum_max) / cum_max
        max_drawdown = float(np.min(drawdowns))

        return {
            "Annualized Return": float(ann_return),
            "Annualized Volatility": float(ann_vol),
            "Sharpe Ratio": float(sharpe),
            "Sortino Ratio": float(sortino),
            "Max Drawdown": float(max_drawdown)
        }


# =====================================================================
# 2. EXECUTABLE TEST BLOCK
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("Executing Backtest Engine Verification...")

    dates = pd.date_range(start="2025-01-01", periods=150, freq="B")
    n_assets = 4
    
    # Synthetic asset returns
    returns = np.random.normal(0.0003, 0.012, size=(150, n_assets))
    returns_df = pd.DataFrame(returns, index=dates, columns=[f"ASSET_{i}" for i in range(n_assets)])

    # Synthetic calibrated probabilities (Bear, Neutral, Bull)
    raw_p = np.random.uniform(size=(150, 3))
    probs = raw_p / np.sum(raw_p, axis=1, keepdims=True)
    probs_df = pd.DataFrame(probs, index=dates, columns=["Bear", "Neutral", "Bull"])

    # Synthetic conformal sets
    pred_sets = [[0, 1] if i % 5 == 0 else [2] for i in range(150)]

    engine = RegimeAwareBacktestEngine(initial_capital=1000000.0, transaction_cost_bps=10.0)
    results = engine.run_backtest(returns_df, probs_df, pred_sets)
    metrics = engine.calculate_performance_metrics(results["performance"])

    print("\n--- Backtest Performance Results ---")
    for k, v in metrics.items():
        if "Ratio" in k:
            print(f"{k:<22}: {v:.4f}")
        else:
            print(f"{k:<22}: {v * 100:.2f}%")

    print(f"\nFinal Portfolio Equity: ${results['performance']['portfolio_value'].iloc[-1]:,.2f}")