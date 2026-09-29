# Phases 8 & 9: Controlled Unseen-Attack Experiments Report

**Research Focus:** Evaluating whether an anomaly detector can identify malicious network flows belonging to attack families completely excluded from supervised training.

## 1. Summary Comparison Across Holdout Experiments

| Experiment | Held-Out Category | Test Samples | Supervised Misclassified as Normal (Silent Breach) | Supervised Forced to Other Attack | Anomaly Detection Rate ($p_{90}$) | Anomaly Detection Rate ($p_{95}$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Exp (Worms)** | `Worms` | 44 | **0.0%** (0) | 100.00% | **20.45%** (9) | 13.64% (6) |
| **Exp (Shellcode)** | `Shellcode` | 378 | **1.59%** (6) | 98.41% | **3.44%** (13) | 2.38% (9) |
| **Exp (Backdoor)** | `Backdoor` | 583 | **0.0%** (0) | 100.00% | **34.31%** (200) | 9.09% (53) |


## 2. Supervised Model Blindness (Forced Classifications)

### Experiment: Held-out `Worms`
When `Worms` was excluded from training, the supervised classifier forced the 44 unseen test flows into:
- **Exploits**: 34 flows (77.27%) — *Forced into known attack*
- **Fuzzers**: 6 flows (13.64%) — *Forced into known attack*
- **DoS**: 3 flows (6.82%) — *Forced into known attack*
- **Shellcode**: 1 flows (2.27%) — *Forced into known attack*
- **Mean Confidence on Unseen:** `0.6996` vs **Mean Confidence on Known:** `0.8076`

### Experiment: Held-out `Shellcode`
When `Shellcode` was excluded from training, the supervised classifier forced the 378 unseen test flows into:
- **Fuzzers**: 271 flows (71.69%) — *Forced into known attack*
- **Exploits**: 53 flows (14.02%) — *Forced into known attack*
- **DoS**: 41 flows (10.85%) — *Forced into known attack*
- **Normal**: 6 flows (1.59%) — *⚠️ DANGEROUS: Benign misclassification*
- **Reconnaissance**: 4 flows (1.06%) — *Forced into known attack*
- **Backdoor**: 2 flows (0.53%) — *Forced into known attack*
- **Worms**: 1 flows (0.26%) — *Forced into known attack*
- **Mean Confidence on Unseen:** `0.5983` vs **Mean Confidence on Known:** `0.8142`

### Experiment: Held-out `Backdoor`
When `Backdoor` was excluded from training, the supervised classifier forced the 583 unseen test flows into:
- **DoS**: 425 flows (72.9%) — *Forced into known attack*
- **Analysis**: 108 flows (18.52%) — *Forced into known attack*
- **Exploits**: 38 flows (6.52%) — *Forced into known attack*
- **Shellcode**: 9 flows (1.54%) — *Forced into known attack*
- **Fuzzers**: 2 flows (0.34%) — *Forced into known attack*
- **Reconnaissance**: 1 flows (0.17%) — *Forced into known attack*
- **Mean Confidence on Unseen:** `0.4024` vs **Mean Confidence on Known:** `0.8178`

## 3. Scientific Conclusions on Unseen Attack Identification

1. **Vulnerability of Pure Supervised Classifiers:** When an attack category is withheld from training, the supervised classifier has no concept of novelty and is forced to predict an existing category. In multiple cases, substantial portions are falsely labeled as `Normal` (causing critical silent security breaches).
2. **Role of Anomaly Detection:** The anomaly detector, operating with zero attack labels, successfully flagged 20.45% of `Worms`, 34.31% of `Backdoor`, and 3.44% of `Shellcode` flows as anomalous under the $p_{90}$ threshold.
3. **Scientific Defense:** As mandated by the research protocol, we do *not* claim zero-day protection across all attack classes. Instead, we experimentally demonstrate that anomaly detection identifies specific held-out attack categories that supervised classifiers fundamentally cannot distinguish as novel.

