import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Now your existing imports will work correctly:
from src.pit_data_engine import preprocess_nifty_features
from src.regime_models import GaussianHMMRegimeDetector
from src.backtest_engine import RegimeBacktester

def main():
    print("--- Running Regime Lab Pipeline ---")
    
    # 1. Generate dummy market data for verification
    dates = pd.date_range(start="2020-01-01", periods=1000, freq="B")
    np.random.seed(42)
    
    raw_df = pd.DataFrame({
        'nifty_close': 10000 * np.exp(np.cumsum(np.random.normal(0.0005, 0.01, 1000))),
        'india_vix': np.random.uniform(12, 28, 1000),
        'fii_net_flow': np.random.normal(100, 500, 1000),
        'advances': np.random.randint(800, 1400, 1000),
        'declines': np.random.randint(600, 1200, 1000)
    }, index=dates)
    
    # 2. Extract PIT features
    features_df = preprocess_nifty_features(raw_df)
    
    # 3. Fit HMM model
    detector = GaussianHMMRegimeDetector(n_components=3)
    regime_results = detector.fit_predict(features_df)
    
    # 4. Run Backtest (0: Bull = 100% Equity, 1: Neutral = 50% Equity, 2: Bear = 0% Equity)
    backtester = RegimeBacktester()
    target_weights = {0: 1.0, 1: 0.5, 2: 0.0}
    final_results = backtester.run_backtest(regime_results, target_weights)
    
    print("Pipeline Execution Complete. Final Strategy Capital:", round(final_results['equity_curve'].iloc[-1], 2))

if __name__ == "__main__":
    main()
