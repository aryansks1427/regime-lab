import numpy as np
import pandas as pd
from hmmlearn import hmm

class GaussianHMMRegimeDetector:
    """
    Hidden Markov Model with automated state sorting and rolling probability smoothing.
    """
    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=1000,
            random_state=random_state
        )
        self.n_components = n_components

    def fit_predict(self, features: pd.DataFrame, smooth_window: int = 5) -> pd.DataFrame:
        X = features.values
        self.model.fit(X)
        
        # Extract raw predicted probabilities
        probs = self.model.predict_proba(X)
        
        # Sort states by the mean return feature (column index 0)
        # Highest mean return = State 0 (Bull), Lowest = State N (Bear)
        feature_means = self.model.means_[:, 0]
        sorted_order = np.argsort(feature_means)[::-1]
        sorted_probs = probs[:, sorted_order]
        
        result_df = features.copy()
        
        # Apply rolling window smoothing to eliminate daily regime flickering
        for i in range(self.n_components):
            col_name = f'prob_regime_{i}'
            raw_series = pd.Series(sorted_probs[:, i], index=features.index)
            result_df[col_name] = raw_series.rolling(smooth_window, min_periods=1).mean()
        
        # Determine dominant regime state based on smoothed probabilities
        prob_cols = [f'prob_regime_{i}' for i in range(self.n_components)]
        result_df['regime_state'] = result_df[prob_cols].values.argmax(axis=1)
            
        return result_df