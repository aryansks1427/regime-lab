import numpy as np
import pandas as pd
from typing import Dict, List

# =====================================================================
# 1. ADAPTIVE CONFORMAL INFERENCE (ACI) & PROPER SCORING
# =====================================================================

class ConformalRegimeCalibrator:
    r"""
    Adaptive Conformal Inference (ACI) engine providing finite-sample 
    valid coverage guarantees for regime probabilities over non-stationary time series.
    """
    def __init__(self, target_coverage: float = 0.90, gamma: float = 0.05):
        """
        :param target_coverage: Desired empirical coverage (e.g., 0.90 for 90%)
        :param gamma: Learning rate for updating non-conformity threshold alpha
        """
        self.target_alpha = 1.0 - target_coverage
        self.gamma = gamma
        self.alpha_t = self.target_alpha
        self.history_alpha = []

    def compute_nonconformity_scores(self, prob_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        r"""
        Calculates non-conformity score: s_i = 1 - P(y_true_i | X_i)
        """
        n_samples = len(y_true)
        scores = np.zeros(n_samples)
        for i in range(n_samples):
            true_class = int(y_true[i])
            scores[i] = 1.0 - prob_matrix[i, true_class]
        return scores

    def calibrate_adaptive(self, prob_matrix: np.ndarray, y_true: np.ndarray) -> Dict[str, np.ndarray]:
        r"""
        Online Adaptive Conformal Inference loop adjusting alpha_t dynamically.
        """
        n_samples = len(y_true)
        prediction_sets = []
        covered_list = []

        for t in range(n_samples):
            probs_t = prob_matrix[t]
            y_t = int(y_true[t])

            # Form prediction set at threshold 1 - alpha_t
            q_threshold = 1.0 - self.alpha_t
            pred_set = np.where(probs_t >= (1.0 - q_threshold))[0]
            if len(pred_set) == 0:
                pred_set = np.array([np.argmax(probs_t)])

            is_covered = int(y_t in pred_set)
            covered_list.append(is_covered)
            prediction_sets.append(pred_set)

            # Update adaptive alpha_t via ACI update rule
            err_t = 1 - is_covered
            self.alpha_t = np.clip(self.alpha_t + self.gamma * (self.target_alpha - err_t), 0.01, 0.5)
            self.history_alpha.append(self.alpha_t)

        empirical_coverage = np.mean(covered_list)
        return {
            "empirical_coverage": empirical_coverage,
            "prediction_sets": prediction_sets,
            "alpha_history": np.array(self.history_alpha)
        }


class ProperScoringEvaluator:
    r"""
    Evaluates probabilistic calibration using Proper Scoring Rules.
    """
    @staticmethod
    def brier_score(prob_matrix: np.ndarray, y_true: np.ndarray) -> float:
        r"""
        Multiclass Brier Score: \frac{1}{N} \sum_{i=1}^N \sum_{k=1}^K (p_{i,k} - y_{i,k})^2
        """
        n_samples, n_classes = prob_matrix.shape
        y_onehot = np.zeros((n_samples, n_classes))
        for i, label in enumerate(y_true):
            y_onehot[i, int(label)] = 1.0
        
        return float(np.mean(np.sum((prob_matrix - y_onehot) ** 2, axis=1)))

    @staticmethod
    def log_loss(prob_matrix: np.ndarray, y_true: np.ndarray, eps: float = 1e-15) -> float:
        r"""
        Cross-Entropy Log Loss with probability clipping for numerical stability.
        """
        probs_clipped = np.clip(prob_matrix, eps, 1 - eps)
        n_samples = len(y_true)
        correct_probs = probs_clipped[np.arange(n_samples), y_true.astype(int)]
        return float(-np.mean(np.log(correct_probs)))

    @staticmethod
    def ranked_probability_score(prob_matrix: np.ndarray, y_true: np.ndarray) -> float:
        r"""
        Ranked Probability Score (RPS) measuring multi-state ordinal calibration.
        """
        n_samples, n_classes = prob_matrix.shape
        rps_list = []
        for i in range(n_samples):
            cum_p = np.cumsum(prob_matrix[i])
            cum_y = np.zeros(n_classes)
            cum_y[int(y_true[i]):] = 1.0
            rps_list.append(np.mean((cum_p - cum_y) ** 2))
        return float(np.mean(rps_list))


# =====================================================================
# 2. EXECUTABLE TEST BLOCK
# =====================================================================

if __name__ == "__main__":
    np.random.seed(42)
    print("Initializing Calibration Verification Test...")

    n_samples = 200
    n_classes = 3  # Regimes: 0 = Bear, 1 = Neutral, 2 = Bull

    # Generate synthetic uncalibrated regime probabilities
    raw_logits = np.random.normal(size=(n_samples, n_classes))
    probs = np.exp(raw_logits) / np.sum(np.exp(raw_logits), axis=1, keepdims=True)
    y_true = np.random.choice([0, 1, 2], size=n_samples, p=[0.3, 0.4, 0.3])

    # Run Proper Scoring Evaluation
    evaluator = ProperScoringEvaluator()
    brier = evaluator.brier_score(probs, y_true)
    logloss = evaluator.log_loss(probs, y_true)
    rps = evaluator.ranked_probability_score(probs, y_true)

    print(f"\n--- Proper Scoring Metrics ---")
    print(f"Brier Score : {brier:.4f}")
    print(f"Log Loss    : {logloss:.4f}")
    print(f"RPS Score   : {rps:.4f}")

    # Run Adaptive Conformal Inference
    calibrator = ConformalRegimeCalibrator(target_coverage=0.90, gamma=0.05)
    results = calibrator.calibrate_adaptive(probs, y_true)

    print(f"\n--- Conformal Coverage Results ---")
    print(f"Target Coverage    : 90.00%")
    print(f"Empirical Coverage : {results['empirical_coverage'] * 100:.2f}%")
    print(f"Final Adaptive Alpha: {results['alpha_history'][-1]:.4f}")