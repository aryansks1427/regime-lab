import pandas as pd
import numpy as np

def enforce_pit(df: pd.DataFrame, publication_lags: dict) -> pd.DataFrame:
    """
    Enforces Point-In-Time alignment by shifting macro and flow variables 
    by their actual publication lags to avoid look-ahead bias[cite: 1].
    """
    df_pit = df.copy()
    for col, lag_days in publication_lags.items():
        if col in df_pit.columns:
            df_pit[col] = df_pit[col].shift(lag_days)
    return df_pit

def preprocess_nifty_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineers standardized features for Indian market regime detection[cite: 1].
    """
    df = raw_df.copy()
    
    # Returns & Volatility
    df['log_ret'] = np.log(df['nifty_close'] / df['nifty_close'].shift(1))
    df['dma_50_200_ratio'] = (df['nifty_close'].rolling(50).mean() / 
                              df['nifty_close'].rolling(200).mean()) - 1.0
    df['vix_zscore'] = (df['india_vix'] - df['india_vix'].rolling(252).mean()) / df['india_vix'].rolling(252).std()
    
    # Flows & Breadth
    df['fii_flow_z'] = (df['fii_net_flow'] - df['fii_net_flow'].rolling(63).mean()) / df['fii_net_flow'].rolling(63).std()
    df['breadth_ratio'] = df['advances'] / (df['advances'] + df['declines'] + 1e-6)
    
    features = ['log_ret', 'dma_50_200_ratio', 'vix_zscore', 'fii_flow_z', 'breadth_ratio']
    return df.dropna(subset=features)[features]

if __name__ == "__main__":
    print("Point-In-Time Engine initialized successfully.")