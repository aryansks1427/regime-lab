import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple


class MCDropoutClassifier(nn.Module):
    """
    Deliverable 3: Monte Carlo (MC) Dropout Classifier (Bayesian Neural Network Proxy).
    Maintains active dropout during inference to compute uncertainty bounds over regimes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 32,
        num_classes: int = 3,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

    def predict_mc(
        self, x: torch.Tensor, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Runs Monte Carlo forward passes with dropout enabled during inference.

        Returns:
            mean_probs: Calibrated average probability per regime.
            epistemic_unc: Model uncertainty (variance of probabilities across MC runs).
            aleatoric_unc: Data noise uncertainty (predictive entropy).
        """
        self.train()  # Keep dropout active during inference
        probs_list = []

        with torch.no_grad():
            for _ in range(n_samples):
                logits = self.forward(x)
                probs = torch.softmax(logits, dim=-1)
                probs_list.append(probs.numpy())

        # Shape: (n_samples, batch_size, num_classes)
        probs_arr = np.array(probs_list)

        # 1. Mean Predictive Probability
        mean_probs = np.mean(probs_arr, axis=0)

        # 2. Epistemic Uncertainty (Model Variance across MC passes)
        epistemic_unc = np.var(probs_arr, axis=0)

        # 3. Aleatoric Uncertainty (Average Predictive Entropy)
        aleatoric_unc = -np.sum(
            mean_probs * np.log(mean_probs + 1e-12), axis=-1, keepdims=True
        )

        return mean_probs, epistemic_unc, aleatoric_unc


# --- Verification & Test Harness ---
if __name__ == "__main__":
    from src.data_pipeline import PointInTimePipeline
    from src.regime_models import BaselineRegimeModels

    # 1. Load pipeline & generate feature matrix
    trading_days = pd.date_range(start="2024-01-01", end="2026-08-01", freq="B")
    equity_data = pd.DataFrame({
        "date": trading_days,
        "close": np.cumprod(1 + np.random.normal(0.0003, 0.012, len(trading_days))) * 22000,
        "breadth_advances": np.random.randint(300, 1700, len(trading_days)),
        "breadth_declines": np.random.randint(300, 1700, len(trading_days)),
    })

    pipeline = PointInTimePipeline(primary_calendar=trading_days)
    aligned_data = pipeline.align_to_calendar({"equity": equity_data}, {"equity": 0})
    features = pipeline.compute_regime_features(aligned_data)

    # 2. Pseudo-label data using GMM from Deliverable 2
    gmm_engine = BaselineRegimeModels(n_regimes=3)
    gmm_preds = gmm_engine.fit_predict_gmm(features)
    y_labels = torch.tensor(gmm_preds["gmm_pred_state"].values, dtype=torch.long)
    X_tensor = torch.tensor(features.values, dtype=torch.float32)

    # 3. Train BNN Classifier
    model = MCDropoutClassifier(input_dim=X_tensor.shape[1], hidden_dim=32, num_classes=3)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(X_tensor)
        loss = criterion(out, y_labels)
        loss.backward()
        optimizer.step()

    # 4. Perform Monte Carlo Bayesian Inference
    mean_probs, epistemic_unc, aleatoric_unc = model.predict_mc(X_tensor, n_samples=50)

    results_df = pd.DataFrame(
        mean_probs,
        index=features.index,
        columns=["bnn_prob_regime_0", "bnn_prob_regime_1", "bnn_prob_regime_2"],
    )
    results_df["epistemic_unc_mean"] = epistemic_unc.mean(axis=1)
    results_df["aleatoric_unc"] = aleatoric_unc

    print("\n--- Deliverable 3: Bayesian Neural Network Output ---")
    print(results_df.head())