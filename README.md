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

## 5. Development Phases & Current Status
1. [x] **Phase 1: Project Inspection & Setup**
2. [x] **Phase 2: Dataset Understanding & Auditing** (`reports/dataset_audit.md`)
3. [x] **Phase 3: Reproducible Preprocessing Pipeline** (`src/preprocessing/pipeline.py`)
4. [x] **Phase 4: Exploratory Data Analysis (EDA)** (`src/data/eda.py`, `reports/eda_report.md`)
5. [x] **Phase 5: Baseline Supervised Classifier** (`src/models/baseline_supervised.py`)
6. [x] **Phase 6: Multiclass Supervised Classification** (`src/models/multiclass_supervised.py`)
7. [x] **Phase 7: Anomaly Detection Baseline** (`src/anomaly/isolation_forest.py`)
8. [x] **Phase 8 & 9: Controlled Unseen-Attack Experiments** (`src/evaluation/unseen_experiments.py`)
9. [x] **Phase 10: Baseline Comparison (Supervised vs Anomaly vs Hybrid)** (`src/evaluation/baseline_comparison.py`)
10. [x] **Phase 11: Hybrid Decision Engine** (`src/hybrid/decision_engine.py`)
11. [x] **Phase 12: Explainability (SHAP)** (`src/explainability/explainer.py`)
12. [x] **Phase 13: FastAPI Backend** (`src/api/main.py`)
13. [x] **Phase 14: Interactive Light-Themed SOC Dashboard** (`src/api/static/`)
14. [x] **Phase 15: Evaluation, Unit Tests & Documentation** (`tests/test_nids_system.py`)

## 6. Running the System

### Start the FastAPI Service & SOC Dashboard
```bash
python -m uvicorn src.api.main:app --host 127.0.0.1 --port 8000
```
- **Interactive SOC Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000) (Executive Light Theme)
- **FastAPI Interactive Swagger Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Run the Automated Test Suite
```bash
python -m unittest tests/test_nids_system.py -v
```

## 6. Research Limitations
- **Benchmark Specificity:** UNSW-NB15 reflects simulated network testbeds; feature distributions may differ in real enterprise environments.
- **Controlled Unseen vs Real Zero-Day:** Simulating unseen attacks via category holdouts tests generalization across taxonomic boundaries, not arbitrary protocol anomalies.
- **Class Imbalance:** Significant disparity exists across attack categories (e.g. Worms/Shellcode vs Generic/Exploits).
- **False-Positive Tradeoffs:** Anomaly detectors inevitably incur higher false-positive rates on complex normal behaviors.
