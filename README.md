# Intelligent Hybrid Network Intrusion Detection System for Unseen Attack Identification

## 1. Project Overview
This repository contains a research-grade prototype designed to address the challenge of network intrusion detection, with specific focus on detecting malicious traffic belonging to attack categories deliberately held out (unseen) during supervised training.

## 2. Research Problem & Question
Traditional supervised intrusion detection systems achieve high accuracy on known attack classes but often fail when confronted with novel or zero-day attacks. Pure anomaly detection systems can flag novel patterns but suffer from high false-positive rates on complex network traffic.

**Central Research Question:**
> *Can a hybrid supervised and anomaly-detection framework identify malicious network traffic belonging to attack categories that were deliberately excluded from supervised model training?*

*Important Research Note:* We do not claim this system detects arbitrary real-world zero-day attacks. All evaluations are conducted under a scientifically defensible, controlled unseen-attack experimental framework.

## 3. Architecture Overview
```
Incoming Network Flow Data
         │
         ▼
┌─────────────────────────────────────────┐
│       Reproducible Preprocessing        │
│  (Imputation, Categorical Encoding,     │
│   Scaling fit strictly on Train data)   │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   Supervised    │ │ Anomaly Detector│
│   Classifier    │ │(Isolation Forest│
│(Random Forest / │ │  or benchmark)  │
│    XGBoost)     │ │                 │
└────────┬────────┘ └────────┬────────┘
         │                   │
  [Pred, Conf]        [Anomaly Score]
         │                   │
         └─────────┬─────────┘
                   ▼
┌─────────────────────────────────────────┐
│         Hybrid Decision Engine          │
│  Outputs: NORMAL / KNOWN ATTACK /       │
│           UNSEEN (ANOMALOUS)            │
└──────────────────┬──────────────────────┘
                   │
         ┌─────────┴─────────┐
         ▼                   ▼
┌─────────────────┐ ┌─────────────────┐
│   SHAP-based    │ │ FastAPI Service │
│ Explainability  │ │& React Dashboard│
└─────────────────┘ └─────────────────┘
```

## 4. Dataset: UNSW-NB15
- **Source:** The University of New South Wales (UNSW-NB15 dataset)
- **Target Categories:** Normal + 9 Attack Categories:
  1. Fuzzers
  2. Analysis
  3. Backdoors
  4. DoS
  5. Exploits
  6. Generic
  7. Reconnaissance
  8. Shellcode
  9. Worms
- **File Placement:**
  - `data/raw/UNSW_NB15_training-set.csv`
  - `data/raw/UNSW_NB15_testing-set.csv`

## 5. Development Phases
1. **Phase 1: Project Inspection & Setup** (Current)
2. **Phase 2: Dataset Understanding & Auditing**
3. **Phase 3: Reproducible Preprocessing Pipeline**
4. **Phase 4: Exploratory Data Analysis (EDA)**
5. **Phase 5: Baseline Supervised Classifier (Binary & Multiclass)**
6. **Phase 6: Multiclass Supervised Classification**
7. **Phase 7: Anomaly Detection Baseline**
8. **Phase 8 & 9: Controlled Unseen-Attack Experiments**
9. **Phase 10: Baseline Comparison (Supervised vs Anomaly vs Hybrid)**
10. **Phase 11: Hybrid Decision Engine**
11. **Phase 12: Explainability (SHAP)**
12. **Phase 13: FastAPI Backend**
13. **Phase 14: React Dashboard**
14. **Phase 15: Evaluation & Documentation**

## 6. Research Limitations
- **Benchmark Specificity:** UNSW-NB15 reflects simulated network testbeds; feature distributions may differ in real enterprise environments.
- **Controlled Unseen vs Real Zero-Day:** Simulating unseen attacks via category holdouts tests generalization across taxonomic boundaries, not arbitrary protocol anomalies.
- **Class Imbalance:** Significant disparity exists across attack categories (e.g. Worms/Shellcode vs Generic/Exploits).
- **False-Positive Tradeoffs:** Anomaly detectors inevitably incur higher false-positive rates on complex normal behaviors.
