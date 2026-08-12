import numpy as np
import pandas as pd

class RegimeBacktester:
    """
    Backtests regime-conditioned strategy and computes key risk/return metrics.
    """
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(self, df: pd.DataFrame, target_weights: dict) -> pd.DataFrame:
        bt_df = df.copy()
        
        # Map target weight to regime state
        bt_df['target_weight'] = bt_df['regime_state'].map(target_weights).fillna(0.0)
        
        # Shift position by 1 day to execute without look-ahead bias
        bt_df['position_weight'] = bt_df['target_weight'].shift(1).fillna(0.0)
        
        # Transaction costs drag
        bt_df['turnover'] = (bt_df['position_weight'] - bt_df['position_weight'].shift(1)).abs().fillna(0.0)
        bt_df['cost_drag'] = bt_df['turnover'] * self.transaction_cost
        
        # Daily strategy return
        bt_df['strategy_ret'] = (bt_df['position_weight'] * bt_df['log_ret']) - bt_df['cost_drag']
        
        # Equity curves
        bt_df['equity_curve'] = self.initial_capital * np.exp(bt_df['strategy_ret'].cumsum())
        bt_df['benchmark_curve'] = self.initial_capital * np.exp(bt_df['log_ret'].cumsum())
        
        return bt_df

    def calculate_metrics(self, df: pd.DataFrame) -> dict:
        """
        Calculates Sharpe Ratio, Max Drawdown, CAGR, and Volatility for strategy vs benchmark.
        """
        N = len(df) / 252  # Years
        
        # CAGR
        strat_cagr = (df['equity_curve'].iloc[-1] / self.initial_capital) ** (1 / N) - 1
        bench_cagr = (df['benchmark_curve'].iloc[-1] / self.initial_capital) ** (1 / N) - 1
        
        # Annualized Volatility
        strat_vol = df['strategy_ret'].std() * np.sqrt(252)
        bench_vol = df['log_ret'].std() * np.sqrt(252)
        
        # Sharpe Ratio (assumes 0% risk-free rate)
        strat_sharpe = strat_cagr / (strat_vol + 1e-6)
        bench_sharpe = bench_cagr / (bench_vol + 1e-6)
        
        # Max Drawdown
        strat_peak = df['equity_curve'].cummax()
        strat_dd = ((df['equity_curve'] - strat_peak) / strat_peak).min()
        
        bench_peak = df['benchmark_curve'].cummax()
        bench_dd = ((df['benchmark_curve'] - bench_peak) / bench_peak).min()
        
        return {
            "Strategy CAGR": f"{strat_cagr:.2%}",
            "Benchmark CAGR": f"{bench_cagr:.2%}",
            "Strategy Sharpe": round(strat_sharpe, 2),
            "Benchmark Sharpe": round(bench_sharpe, 2),
            "Strategy Max Drawdown": f"{strat_dd:.2%}",
            "Benchmark Max Drawdown": f"{bench_dd:.2%}",
            "Strategy Volatility": f"{strat_vol:.2%}",
            "Benchmark Volatility": f"{bench_vol:.2%}"
        }