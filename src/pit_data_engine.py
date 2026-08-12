import pandas as pd
import numpy as np
import yfinance as yf

def fetch_market_data(ticker: str = "^NSEI", start_date: str = "2018-01-01", end_date: str = "2024-01-01") -> pd.DataFrame:
    """
    Downloads historical market data and generates macroeconomic regime features.
    """
    df = yf.download(ticker, start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df = df.xs(ticker, level=1, axis=1)
        
    df = df[['Close']].rename(columns={'Close': 'close'})
    
    # Feature Engineering
    df['log_ret'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility_21d'] = df['log_ret'].rolling(21).std() * np.sqrt(252)
    df['dma_50_200'] = (df['close'].rolling(50).mean() / df['close'].rolling(200).mean()) - 1.0
    df['momentum_14d'] = (df['close'] / df['close'].shift(14)) - 1.0
    
    features = ['log_ret', 'volatility_21d', 'dma_50_200', 'momentum_14d']
    return df.dropna(subset=features)