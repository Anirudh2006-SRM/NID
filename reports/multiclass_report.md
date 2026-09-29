# Phase 6: Multiclass Supervised Classifier Report

**Model:** Random Forest (100 estimators, max_depth=22, balanced class weighting)  
**Task:** 10-Class Attack Categorization  
**Training Set:** 175,341 flows across 10 categories  
**Testing Set:** 82,332 flows across 10 categories  

## 1. Global Performance Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | 68.55% | Proportion of exact class matches |
| **Macro F1-Score** | 0.4925 | Unweighted average F1 across all 10 categories |
| **Weighted F1-Score** | 0.7381 | Support-weighted average F1 |
| **Macro Precision** | 0.5027 | Unweighted average precision |
| **Macro Recall** | 0.5812 | Unweighted average recall |
| **Training Time** | 12.559s | Latency to fit 175,341 flows across 65 features |
| **Inference Latency** | 0.0081 ms/flow | Latency per network flow (0.668s for 82,332 samples) |

## 2. Per-Class Performance Breakdown
| Category | Support (Test) | Precision | Recall | F1-Score | Class Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 37,000.0 | 0.9859 | 0.6075 | 0.7518 | Majority |
| **Generic** | 18,871.0 | 0.9996 | 0.9641 | 0.9815 | Majority |
| **Exploits** | 11,132.0 | 0.8073 | 0.6015 | 0.6894 | Majority |
| **Fuzzers** | 6,062.0 | 0.2661 | 0.7161 | 0.3880 | Moderate |
| **DoS** | 4,089.0 | 0.3071 | 0.2976 | 0.3023 | Moderate |
| **Reconnaissance** | 3,496.0 | 0.8628 | 0.8415 | 0.8520 | Moderate |
| **Analysis** | 677.0 | 0.0008 | 0.0030 | 0.0013 | Extreme Minority |
| **Backdoor** | 583.0 | 0.0460 | 0.4271 | 0.0831 | Extreme Minority |
| **Shellcode** | 378.0 | 0.1960 | 0.7857 | 0.3138 | Extreme Minority |
| **Worms** | 44.0 | 0.5556 | 0.5682 | 0.5618 | Extreme Minority |

## 3. Key Observations for Unseen Attack Experiments
1. High-frequency categories (`Normal`, `Generic`, `Exploits`) achieve strong classification metrics.
2. Low-frequency categories (`Worms` support=44, `Shellcode` support=378) demonstrate the difficulty supervised models encounter when samples are scarce.
3. This sets the exact baseline needed for Phase 8: testing whether holding out an attack category leaves the supervised model blind to it while the anomaly detector catches it.
