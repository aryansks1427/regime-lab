import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from typing import Tuple


class TemporalEmbeddingExtractor(nn.Module):
    """
    Deliverable 4: Latent Temporal Embedding Extraction Engine.
    Transforms rolling sequence windows into dense temporal embeddings for downstream classifiers.
    """

    def __init__(self, input_dim: int, sequence_length: int = 10, embedding_dim: int = 64):
        super().__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim

        # 1D Temporal Encoder Layer
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, embedding_dim),
            nn.LayerNorm(embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input shape:  (batch_size, sequence_length, input_dim)
        Output shape: (batch_size, embedding_dim)
        """
        # Permute to (batch_size, input_dim, sequence_length) for Conv1d
        x_permuted = x.permute(0, 2, 1)
        return self.encoder(x_permuted)

    @staticmethod
    def create_sliding_windows(
        df: pd.DataFrame, window_size: int = 10
    ) -> Tuple[np.ndarray, pd.DatetimeIndex]:
        """
        Converts a 2D feature DataFrame into 3D sequential sliding windows.
        """
        values = df.values
        dates = df.index[window_size - 1 :]

        windows = []
        for i in range(len(values) - window_size + 1):
            windows.append(values[i : i + window_size])

        return np.array(windows), dates


# --- Verification & Test Harness ---
if __name__ == "__main__":
    from src.data_pipeline import PointInTimePipeline

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

    # 2. Form 3D sequence windows (10-day lookback)
    window_size = 10
    windows, aligned_dates = TemporalEmbeddingExtractor.create_sliding_windows(
        features, window_size=window_size
    )
    X_tensor = torch.tensor(windows, dtype=torch.float32)

    # 3. Extract 64-dimensional embeddings
    extractor = TemporalEmbeddingExtractor(
        input_dim=features.shape[1],
        sequence_length=window_size,
        embedding_dim=64,
    )

    extractor.eval()
    with torch.no_grad():
        embeddings = extractor(X_tensor).numpy()

    embedding_cols = [f"emb_dim_{i}" for i in range(embeddings.shape[1])]
    embeddings_df = pd.DataFrame(embeddings, index=aligned_dates, columns=embedding_cols)

    print("\n--- Deliverable 4: Latent Temporal Embeddings Output ---")
    print(f"Extracted Matrix Shape: {embeddings_df.shape}")
    print(embeddings_df.head())