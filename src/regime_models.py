import numpy as np
import pandas as pd
from hmmlearn import hmm

class GaussianHMMRegimeDetector:
    """
    Hidden Markov Model for identifying latent market regimes (e.g., Bull, Bear, Volatile)[cite: 1].
    """
    def __init__(self, n_components: int = 3, covariance_type: str = "full", random_state: int = 42):
        self.model = hmm.GaussianHMM(
            n_components=n_components,
            covariance_type=covariance_type,
            n_iter=1000,
            random_state=random_state
        )
        self.n_components = n_components

    def fit_predict(self, features: pd.DataFrame) -> pd.DataFrame:
        """
        Fits the HMM on feature data and returns regime probabilities along with hidden states.
        """
        X = features.values
        self.model.fit(X)
        hidden_states = self.model.predict(X)
        probs = self.model.predict_proba(X)

        result_df = features.copy()
        result_df['regime_state'] = hidden_states
        
        for i in range(self.n_components):
            result_df[f'prob_regime_{i}'] = probs[:, i]
            
        return result_df

if __name__ == "__main__":
    print("Regime Detection Model Module initialized successfully.")