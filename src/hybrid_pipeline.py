import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple

from src.bnn_classifier import MCDropoutClassifier


class HybridRegimeClassifier(nn.Module):
    """
    Deliverable 5: Hybrid Architecture Pipeline.
    Fuses high-dimensional latent temporal embeddings with engineered point-in-time features 
    into a downstream Bayesian (MC-Dropout) classification head.
    """

    def __init__(
        self,
        raw_feature_dim: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        num_classes: int = 3,
    ):
        super().__init__()
        fused_dim = raw_feature_dim + embedding_dim
        self.bnn_head = MCDropoutClassifier(
            input_dim=fused_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout_rate=0.2,
        )

    def forward(self, fused_features: torch.Tensor) -> torch.Tensor:
        return self.bnn_head(fused_features)

    def predict_mc(
        self, fused_features: torch.Tensor, n_samples: int = 100
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Performs Monte Carlo forward passes over the fused features to generate 
        calibrated regime probabilities alongside epistemic and aleatoric uncertainties.
        """
        return self.bnn_head.predict_mc(fused_features, n_samples=n_samples)


# --- Verification & Test Harness ---
if __name__ == "__main__":
    from src.data_pipeline import PointInTimePipeline
    from src.regime_models import BaselineRegimeModels
    from src.embedding_extractor import TemporalEmbeddingExtractor

    # 1. Generate features via Deliverable 1
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

    # 2. Extract temporal embeddings via Deliverable 4
    window_size = 10
    windows, aligned_dates = TemporalEmbeddingExtractor.create_sliding_windows(
        features, window_size=window_size
    )
    extractor = TemporalEmbeddingExtractor(
        input_dim=features.shape[1],
        sequence_length=window_size,
        embedding_dim=64,
    )
    extractor.eval()
    with torch.no_grad():
        embeddings = extractor(torch.tensor(windows, dtype=torch.float32)).numpy()

    # 3. Align raw features to match embedding sequence timeline
    aligned_raw_features = features.loc[aligned_dates].values

    # 4. Concatenate raw features and temporal embeddings into a unified matrix
    fused_matrix = np.hstack([aligned_raw_features, embeddings])
    fused_tensor = torch.tensor(fused_matrix, dtype=torch.float32)

    # 5. Pseudo-label data using GMM target
    gmm_engine = BaselineRegimeModels(n_regimes=3)
    gmm_preds = gmm_engine.fit_predict_gmm(features.loc[aligned_dates])
    labels = torch.tensor(gmm_preds["gmm_pred_state"].values, dtype=torch.long)

    # 6. Train Hybrid Architecture Classifier
    model = HybridRegimeClassifier(
        raw_feature_dim=aligned_raw_features.shape[1],
        embedding_dim=embeddings.shape[1],
        hidden_dim=64,
        num_classes=3,
    )
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(50):
        optimizer.zero_grad()
        out = model(fused_tensor)
        loss = criterion(out, labels)
        loss.backward()
        optimizer.step()

    # 7. Perform Bayesian Inference over Fused Representations
    mean_probs, epistemic_unc, aleatoric_unc = model.predict_mc(fused_tensor, n_samples=50)

    results_df = pd.DataFrame(
        mean_probs,
        index=aligned_dates,
        columns=["hybrid_prob_regime_0", "hybrid_prob_regime_1", "hybrid_prob_regime_2"],
    )
    results_df["epistemic_unc_mean"] = epistemic_unc.mean(axis=1)
    results_df["aleatoric_unc"] = aleatoric_unc

    print("\n--- Deliverable 5: Hybrid Architecture Pipeline Output ---")
    print(f"Fused Input Tensor Shape: {fused_matrix.shape}")
    print(results_df.head())