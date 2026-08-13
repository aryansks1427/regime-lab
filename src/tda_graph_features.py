import numpy as np
import pandas as pd
from scipy.stats import entropy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List

# =====================================================================
# 1. TOPOLOGICAL DATA ANALYSIS (TDA) EXTRACTOR
# =====================================================================

class TopologicalFeatureExtractor:
    r"""
    Extracts topological features (Betti numbers, persistent entropy)
    from multi-asset return distance manifolds.
    """
    def __init__(self, window_size: int = 60):
        self.window_size = window_size

    def correlation_to_distance(self, corr_matrix: np.ndarray) -> np.ndarray:
        r"""
        Transforms a correlation matrix into a valid metric distance space:
        d_{i,j} = \sqrt{2(1 - \rho_{i,j})}
        """
        corr_clipped = np.clip(corr_matrix, -1.0, 1.0)
        dist_matrix = np.sqrt(2.0 * (1.0 - corr_clipped))
        np.fill_diagonal(dist_matrix, 0.0)
        return dist_matrix

    def compute_betti_approximations(self, dist_matrix: np.ndarray, epsilons: np.ndarray) -> Dict[str, List[float]]:
        r"""
        Computes persistent homology filtration across distance thresholds (epsilons).
        Tracks connected components (Betti-0) and cycle proxies (Betti-1).
        """
        n_assets = dist_matrix.shape[0]
        betti_0 = []
        betti_1 = []

        for eps in epsilons:
            # Adjacency matrix at scale eps
            adj = (dist_matrix <= eps).astype(int)
            
            # Betti 0 proxy via graph Laplacian zero eigenvalues
            deg = np.diag(adj.sum(axis=1))
            laplacian = deg - adj
            evals = np.linalg.eigvalsh(laplacian)
            b0 = np.sum(np.isclose(evals, 0.0, atol=1e-5))
            betti_0.append(b0)

            # Betti 1 proxy via Euler characteristic approximation: chi = V - E + F
            num_vertices = n_assets
            num_edges = np.sum(np.triu(adj, k=1))
            triangles = np.trace(np.matmul(adj, np.matmul(adj, adj))) / 6.0
            
            chi = num_vertices - num_edges + triangles
            b1 = max(0, b0 - chi)
            betti_1.append(b1)

        return {"betti_0": betti_0, "betti_1": betti_1}

    def compute_persistent_entropy(self, betti_series: List[float]) -> float:
        r"""
        Calculates topological entropy over filtration scales.
        Higher entropy signals structural disorganization/imminent regime shift.
        """
        betti_arr = np.array(betti_series, dtype=float)
        total = np.sum(betti_arr)
        if total == 0:
            return 0.0
        probs = betti_arr / total
        probs = probs[probs > 0]
        return float(entropy(probs))

    def transform(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        r"""
        Processes rolling window asset returns into a TDA feature DataFrame.
        """
        n_obs = len(returns_df)
        features = []
        epsilons = np.linspace(0.1, 1.8, 15)

        for i in range(self.window_size, n_obs):
            window_data = returns_df.iloc[i - self.window_size : i]
            corr_mat = window_data.corr().values
            dist_mat = self.correlation_to_distance(corr_mat)

            betti_dict = self.compute_betti_approximations(dist_mat, epsilons)
            
            b0_mean = np.mean(betti_dict["betti_0"])
            b1_max = np.max(betti_dict["betti_1"])
            topo_entropy = self.compute_persistent_entropy(betti_dict["betti_1"])

            features.append({
                "timestamp": returns_df.index[i],
                "tda_betti0_avg": b0_mean,
                "tda_betti1_max": b1_max,
                "tda_topological_entropy": topo_entropy,
                "tda_mean_distance": np.mean(dist_mat[np.triu_indices_from(dist_mat, k=1)])
            })

        return pd.DataFrame(features).set_index("timestamp")


# =====================================================================
# 2. DYNAMIC GRAPH NEURAL NETWORK (GNN) MODULE
# =====================================================================

class GraphConvolutionLayer(nn.Module):
    r"""
    Spectral Graph Convolution Layer: 
    H^{(l+1)} = \sigma(\tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2} H^{(l)} W^{(l)})
    """
    def __init__(self, in_features: int, out_features: int):
        super(GraphConvolutionLayer, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        adj_hat = adj + torch.eye(adj.size(0), device=adj.device)
        deg = torch.sum(adj_hat, dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0
        d_mat = torch.diag(deg_inv_sqrt)

        norm_adj = torch.mm(torch.mm(d_mat, adj_hat), d_mat)
        support = torch.mm(x, self.weight)
        return torch.mm(norm_adj, support)


class AssetGraphEncoder(nn.Module):
    r"""
    Temporal Graph Encoder mapping dynamic asset networks to latent regime representations.
    """
    def __init__(self, num_nodes: int, node_in_features: int, latent_dim: int = 4):
        super(AssetGraphEncoder, self).__init__()
        self.gcn1 = GraphConvolutionLayer(node_in_features, 16)
        self.gcn2 = GraphConvolutionLayer(16, latent_dim)
        self.readout = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        h = F.relu(self.gcn1(x, adj))
        h = F.relu(self.gcn2(h, adj))
        graph_repr = torch.mean(h, dim=0)
        return self.readout(graph_repr)


# =====================================================================
# 3. PIPELINE ORCHESTRATOR
# =====================================================================

class TDA_GNN_FeaturePipeline:
    r"""
    Combines TDA topological metrics and GNN spatial graph embeddings.
    """
    def __init__(self, window_size: int = 60, gnn_latent_dim: int = 4, corr_threshold: float = 0.3):
        self.window_size = window_size
        self.gnn_latent_dim = gnn_latent_dim
        self.corr_threshold = corr_threshold
        self.tda_extractor = TopologicalFeatureExtractor(window_size=window_size)

    def _build_adjacency_matrix(self, corr_matrix: np.ndarray) -> torch.Tensor:
        abs_corr = np.abs(corr_matrix)
        adj = (abs_corr > self.corr_threshold).astype(np.float32)
        np.fill_diagonal(adj, 0.0)
        return torch.tensor(adj, dtype=torch.float32)

    def run_pipeline(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        # Step 1: Compute TDA Features
        tda_df = self.tda_extractor.transform(returns_df)

        # Step 2: Initialize GNN Model
        n_assets = returns_df.shape[1]
        gnn_model = AssetGraphEncoder(num_nodes=n_assets, node_in_features=3, latent_dim=self.gnn_latent_dim)
        gnn_model.eval()

        gnn_embeddings = []
        timestamps = []

        # Step 3: Extract GNN Embeddings over rolling windows
        with torch.no_grad():
            for i in range(self.window_size, len(returns_df)):
                window = returns_df.iloc[i - self.window_size : i]
                
                node_means = window.mean().values
                node_vols = window.std().values
                node_skews = window.skew().values
                x_nodes = torch.tensor(np.column_stack([node_means, node_vols, node_skews]), dtype=torch.float32)
                
                corr_mat = window.corr().values
                adj_mat = self._build_adjacency_matrix(corr_mat)

                emb = gnn_model(x_nodes, adj_mat).numpy()
                gnn_embeddings.append(emb)
                timestamps.append(returns_df.index[i])

        gnn_cols = [f"gnn_embed_{j}" for j in range(self.gnn_latent_dim)]
        gnn_df = pd.DataFrame(gnn_embeddings, index=timestamps, columns=gnn_cols)

        # Merge TDA and GNN features
        return pd.concat([tda_df, gnn_df], axis=1).dropna()


# =====================================================================
# 4. EXECUTABLE TEST BLOCK
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)
    
    print("Generating synthetic multi-asset market returns...")
    dates = pd.date_range(start="2025-01-01", periods=150, freq="B")
    returns_data = np.random.normal(0.0001, 0.015, size=(150, 8))
    returns_df = pd.DataFrame(returns_data, index=dates, columns=[f"ASSET_{i}" for i in range(8)])

    print("Running TDA & GNN Feature Pipeline...")
    pipeline = TDA_GNN_FeaturePipeline(window_size=40, gnn_latent_dim=4)
    feature_matrix = pipeline.run_pipeline(returns_df)

    print("\n--- Pipeline Execution Successful! ---")
    print("Extracted Features Shape:", feature_matrix.shape)
    print("\nFeature Columns:")
    print(feature_matrix.columns.tolist())
    print("\nFirst 3 Rows Preview:")
    print(feature_matrix.head(3))