import numpy as np
import pandas as pd

class RegimeBacktester:
    """
    Backtests dynamic asset allocation based on predicted regime probabilities.
    """
    def __init__(self, initial_capital: float = 100000.0, transaction_cost: float = 0.001):
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost

    def run_backtest(self, df: pd.DataFrame, target_weights: dict) -> pd.DataFrame:
        """
        Executes regime-conditioned portfolio rebalancing.
        
        target_weights: map of regime state (0, 1, 2) to equity allocation (e.g., {0: 1.0, 1: 0.5, 2: 0.0})
        """
        bt_df = df.copy()
        
        # Determine target equity weight based on predicted regime
        bt_df['target_weight'] = bt_df['regime_state'].map(target_weights).fillna(0.0)
        
        # Lag weight by 1 day to execute trades at next period without look-ahead
        bt_df['position_weight'] = bt_df['target_weight'].shift(1).fillna(0.0)
        
        # Calculate turnover and transaction cost drag
        bt_df['turnover'] = (bt_df['position_weight'] - bt_df['position_weight'].shift(1)).abs().fillna(0.0)
        bt_df['cost_drag'] = bt_df['turnover'] * self.transaction_cost
        
        # Portfolio daily returns
        bt_df['strategy_ret'] = (bt_df['position_weight'] * bt_df['log_ret']) - bt_df['cost_drag']
        
        # Cumulative performance curves
        bt_df['equity_curve'] = self.initial_capital * np.exp(bt_df['strategy_ret'].cumsum())
        bt_df['benchmark_curve'] = self.initial_capital * np.exp(bt_df['log_ret'].cumsum())
        
        return bt_df

if __name__ == "__main__":
    print("Regime Backtester initialized successfully.")