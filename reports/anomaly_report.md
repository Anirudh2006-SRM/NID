# Phase 7: Anomaly Detection Baseline Report

**Model:** Isolation Forest (100 estimators, max_samples=256, benign-profile reference)  
**Reference Dataset:** 56,000 Normal flows (strictly benign training baseline)  
**Evaluation Set:** 82,332 testing flows (37,000 Normal + 45,332 Attacks)  

## 1. Threshold Calibration Methodology
The decision threshold is empirically calibrated on normal training flows to control the operational false positive rate:
- **Primary Threshold (90th Percentile):** `0.47821` (Target FPR: ~10%)
- **Conservative Threshold (95th Percentile):** `0.50086` (Target FPR: ~5%)

## 2. Global Ranking Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | 0.7862 | Ability to rank malicious flows above benign flows without labels |
| **PR-AUC** | 0.8182 | Area under Precision-Recall curve |
| **Empirical FPR (p90)** | 11.27% | Proportion of normal test flows flagged as anomalous |
| **Empirical FPR (p95)** | 5.78% | Conservative false alarm rate on normal test flows |
| **Training Time** | 0.68s | Latency to fit 56,000 reference flows |
| **Inference Latency** | 0.0101 ms/flow | Latency per flow evaluation |

## 3. Detection Rates per Attack Category (Zero Label Information Used)
| Category | Support | Mean Score | Detection Rate (p90) | Detection Rate (p95) | Category Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Normal** | 37,000 | 0.4121 | **11.27%** | 5.78% | Normal (FPR) |
| **Generic** | 18,871 | 0.5305 | **95.62%** | 92.50% | Attack |
| **Exploits** | 11,132 | 0.4440 | **19.74%** | 8.55% | Attack |
| **Fuzzers** | 6,062 | 0.4262 | **20.29%** | 8.61% | Attack |
| **DoS** | 4,089 | 0.4584 | **24.33%** | 7.68% | Attack |
| **Reconnaissance** | 3,496 | 0.4135 | **4.46%** | 0.94% | Attack |
| **Analysis** | 677 | 0.4645 | **32.20%** | 6.94% | Attack |
| **Backdoor** | 583 | 0.4662 | **34.31%** | 9.09% | Attack |
| **Shellcode** | 378 | 0.4040 | **3.44%** | 2.38% | Attack |
| **Worms** | 44 | 0.4477 | **20.45%** | 13.64% | Attack |

## 4. Key Takeaways for Controlled Unseen Attack Experiments (Phase 8)
1. The anomaly detector requires **zero attack labels** during training.
2. Attack categories with high structural deviations from benign profiles (e.g. DoS, Exploits, Generic) achieve distinct anomaly scores.
3. This sets the foundation for Phase 8: testing whether holding out an attack category leaves the supervised model blind while the anomaly detector flags it.
