# Phase 5: Supervised Baseline Classifier Report

**Model:** Random Forest (100 estimators, max_depth=20, balanced class weighting)  
**Task:** Binary Classification (Normal vs Attack)  
**Training Set:** 175,341 flows (68.06% Attack, 31.94% Normal)  
**Testing Set:** 82,332 flows (55.06% Attack, 44.94% Normal)  

## 1. Quantitative Performance Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | 90.66% | Overall proportion of correct predictions |
| **Macro F1-Score** | 0.9049 | Unweighted mean F1 across both classes |
| **Weighted F1-Score** | 0.9062 | Class-frequency weighted F1 |
| **Macro Precision** | 0.9093 | Mean precision across Normal and Attack |
| **Macro Recall** | 0.9024 | Mean recall across Normal and Attack |
| **ROC-AUC** | 0.9800 | Area under Receiver Operating Characteristic |
| **PR-AUC** | 0.9853 | Area under Precision-Recall curve |
| **False Positive Rate (FPR)** | 13.91% | Proportion of normal flows falsely flagged as attacks |
| **False Negative Rate (FNR)** | 5.61% | Proportion of attack flows missed |

## 2. Confusion Matrix
| True \ Predicted | Predicted Normal (0) | Predicted Attack (1) | Total |
| :--- | :--- | :--- | :--- |
| **True Normal (0)** | 31,853 (TN) | 5,147 (FP) | 37,000 |
| **True Attack (1)** | 2,544 (FN) | 42,788 (TP) | 45,332 |

## 3. Computational Latency
- **Training Time:** 12.283 seconds on 175,341 samples.
- **Inference Latency:** 0.0081 ms per network flow (0.667s for 82,332 samples).

## 4. Top 10 Most Influential Features
| Rank | Feature | Gini Importance |
| :--- | :--- | :--- |
| 1 | `num_rate` | 0.0907 |
| 2 | `num_dload` | 0.0843 |
| 3 | `num_sload` | 0.0791 |
| 4 | `num_tcprtt` | 0.0525 |
| 5 | `num_dinpkt` | 0.0510 |
| 6 | `num_sbytes` | 0.0492 |
| 7 | `num_dpkts` | 0.0472 |
| 8 | `num_dbytes` | 0.0471 |
| 9 | `num_synack` | 0.0470 |
| 10 | `num_dmean` | 0.0465 |
