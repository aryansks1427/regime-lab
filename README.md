# 📈 Regime-Lab: Topological & Conformal Risk Engine

**Regime-Lab** is a quantitative finance framework for multi-asset regime detection, adaptive conformal risk calibration, and regime-aware portfolio optimization. 

It combines **Topological Data Analysis (TDA)** and **Graph Neural Networks (GNN)** with **Adaptive Conformal Inference (ACI)** to provide finite-sample coverage guarantees on regime predictions before feeding them into risk-parity execution models.

---

## 🏛 Architecture Overview
+----------------------------------+
           |   Multi-Asset Market Data        |
           +----------------------------------+
                            |
                            v
           +----------------------------------+
           |  TDA & Dynamic GNN Pipeline      |
           |  - Persistent Homology           |
           |  - Wasserstein Distances         |
           +----------------------------------+
                            |
                            v
           +----------------------------------+
           |  Probabilistic Regime Model      |
           |  (Bear, Neutral, Bull Inference)  |
           +----------------------------------+
                            |
                            v
           +----------------------------------+
           | Adaptive Conformal Inference     |
           |  - Finite-Sample Coverage (ACI)   |
           |  - Calibration Scoring           |
           +----------------------------------+
                            |
                            v
           +----------------------------------+
           |  Regime-Aware Risk Parity        |
           |  - Turnover Control              |
           |  - Transaction Cost Penalty      |
           +----------------------------------+
                            |
               +------------+------------+
               |                         |
               v                         v
    +---------------------+   +----------------------+
    | CLI (`main.py`)     |   | UI (`app.py`)        |
    +---------------------+   +----------------------+

---

## 🛠 Project Structure

```text
regime-lab/
├── main.py                     # CLI End-to-End Orchestrator
├── app.py                      # Interactive Streamlit Dashboard
├── requirements.txt            # Python Dependencies
├── README.md                   # Documentation
└── src/
    ├── features/
    │   └── tda_graph_features.py  # TDA Homology & Graph Feature Engineering
    ├── calibration.py          # Adaptive Conformal Inference Engine
    ├── backtest_engine.py      # Risk Parity & Portfolio Execution Model
    ├── regime_models.py        # Regime Inference Logic
    └── data_pipeline.py        # Data Ingestion & Preprocessing