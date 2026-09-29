# Dataset Directory

This directory stores datasets for the Network Intrusion Detection System:

- `raw/`: Stores the raw, untouched UNSW-NB15 dataset files:
  - `UNSW_NB15_training-set.csv` (175,341 records)
  - `UNSW_NB15_testing-set.csv` (82,332 records)
- `processed/`: Stores cached preprocessed splits, encoders, scalers, and reproducible feature sets.

## Safety & Integrity Rules
1. Never modify files in `raw/`.
2. Do not commit raw or processed CSV files to Git.
3. Preprocessing transformations must be fit strictly on training splits.
