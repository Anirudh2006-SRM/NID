"""Phase 11: Hybrid Decision Engine for Network Intrusion Detection.

Combines:
1. Supervised multiclass prediction & confidence
2. Unsupervised anomaly score from Isolation Forest
3. Configurable confidence and anomaly thresholds

Conceptual outputs:
- 'NORMAL'
- 'KNOWN ATTACK' (with specific attack category)
- 'UNSEEN / ANOMALOUS'
"""

from typing import Dict, List, Optional, Union
import numpy as np

class HybridDecisionEngine:
    """Configurable, reproducible hybrid decision engine combining supervised and anomaly outputs."""

    def __init__(
        self,
        supervised_model,
        anomaly_model,
        categories: List[str],
        anomaly_threshold: float = 0.47821,
        confidence_threshold: float = 0.65,
    ):
        self.supervised_model = supervised_model
        self.anomaly_model = anomaly_model
        self.categories = categories
        self.anomaly_threshold = float(anomaly_threshold)
        self.confidence_threshold = float(confidence_threshold)

    def evaluate_flow(self, x: np.ndarray) -> Dict:
        """Evaluate a single flow vector (1D array of shape [n_features])."""
        if x.ndim == 1:
            x_2d = x.reshape(1, -1)
        else:
            x_2d = x

        # 1. Supervised prediction and confidence
        probs = self.supervised_model.predict_proba(x_2d)[0]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_category = self.categories[pred_idx] if pred_idx < len(self.categories) else "Unknown"

        # 2. Anomaly score (-score_samples; higher = more anomalous)
        anomaly_score = float(-self.anomaly_model.score_samples(x_2d)[0])
        is_anomalous = anomaly_score >= self.anomaly_threshold

        # 3. Decision Logic
        if is_anomalous:
            if confidence < self.confidence_threshold or pred_category == "Normal":
                decision = "UNSEEN / ANOMALOUS"
                risk_level = "HIGH"
                explanation = (
                    f"Traffic exhibits high structural abnormality (anomaly score: {anomaly_score:.4f} >= "
                    f"threshold: {self.anomaly_threshold:.4f}) while supervised classifier displays uncertainty "
                    f"(confidence: {confidence:.2f} < {self.confidence_threshold:.2f}) or conflicting benign prediction. "
                    "Flagged as potentially novel/unseen threat."
                )
            else:
                decision = "KNOWN ATTACK"
                risk_level = "CRITICAL"
                explanation = (
                    f"Traffic matches confirmed known attack pattern '{pred_category}' with high confidence "
                    f"({confidence*100:.1f}%), accompanied by elevated anomaly score ({anomaly_score:.4f})."
                )
        else:
            if pred_category == "Normal":
                decision = "NORMAL"
                risk_level = "LOW"
                explanation = (
                    f"Traffic is consistent with benign baseline (anomaly score: {anomaly_score:.4f} < "
                    f"{self.anomaly_threshold:.4f}) and supervised model predicts Normal ({confidence*100:.1f}%)."
                )
            elif confidence >= self.confidence_threshold:
                decision = "KNOWN ATTACK"
                risk_level = "HIGH"
                explanation = (
                    f"Supervised model strongly matches known attack signature '{pred_category}' "
                    f"(confidence: {confidence*100:.1f}%) within standard flow envelope."
                )
            else:
                decision = "NORMAL"
                risk_level = "LOW"
                explanation = (
                    f"Traffic features lie within benign envelope (anomaly score: {anomaly_score:.4f}) "
                    f"and supervised attack confidence is inconclusive ({confidence:.2f}). Evaluated as benign."
                )

        return {
            "decision": decision,
            "predicted_category": pred_category,
            "confidence": round(confidence, 4),
            "anomaly_score": round(anomaly_score, 4),
            "anomaly_threshold": round(self.anomaly_threshold, 4),
            "is_anomalous": is_anomalous,
            "risk_level": risk_level,
            "explanation": explanation,
        }

    def evaluate_batch(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Vectorized evaluation across a 2D feature matrix [n_samples, n_features]."""
        # Supervised predictions
        probs = self.supervised_model.predict_proba(X)
        pred_indices = np.argmax(probs, axis=1)
        confidences = np.max(probs, axis=1)
        pred_categories = np.array([
            self.categories[idx] if idx < len(self.categories) else "Unknown"
            for idx in pred_indices
        ])

        # Anomaly scores
        anomaly_scores = -self.anomaly_model.score_samples(X)
        is_anomalous = anomaly_scores >= self.anomaly_threshold

        # Decisions
        decisions = np.empty(len(X), dtype=object)

        # Case 1: High Anomaly + Low Conf or Predicted Normal -> UNSEEN / ANOMALOUS
        unseen_mask = is_anomalous & (
            (confidences < self.confidence_threshold) | (pred_categories == "Normal")
        )
        decisions[unseen_mask] = "UNSEEN / ANOMALOUS"

        # Case 2: High Anomaly + High Conf Known Attack -> KNOWN ATTACK
        known_anom_mask = is_anomalous & ~unseen_mask
        decisions[known_anom_mask] = "KNOWN ATTACK"

        # Case 3: Low Anomaly + Normal -> NORMAL
        low_anom_mask = ~is_anomalous
        norm_mask = low_anom_mask & (pred_categories == "Normal")
        decisions[norm_mask] = "NORMAL"

        # Case 4: Low Anomaly + High Conf Attack -> KNOWN ATTACK
        known_low_mask = low_anom_mask & (pred_categories != "Normal") & (confidences >= self.confidence_threshold)
        decisions[known_low_mask] = "KNOWN ATTACK"

        # Case 5: Low Anomaly + Low Conf Attack -> NORMAL
        low_conf_mask = low_anom_mask & (pred_categories != "Normal") & (confidences < self.confidence_threshold)
        decisions[low_conf_mask] = "NORMAL"

        return {
            "decisions": decisions,
            "predicted_categories": pred_categories,
            "confidences": confidences,
            "anomaly_scores": anomaly_scores,
            "is_anomalous": is_anomalous,
        }
