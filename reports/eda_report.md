# Phase 4: Exploratory Data Analysis (EDA) Report

**Dataset:** UNSW-NB15 (Train: 175,341 rows | Test: 82,332 rows)

## 1. Class Distributions & Imbalance

- **Binary Split:** In the training split, Attack traffic constitutes 68.06% (119,341 flows) while Normal constitutes 31.94% (56,000 flows). In testing, Attack is 55.06% and Normal is 44.94%.
- **Multiclass Extremes:** While classes like `Generic` (40,000) and `Exploits` (33,393) are heavily represented, rare classes such as `Worms` (130 samples in train, 44 in test) and `Shellcode` (1,133 train, 378 test) present acute imbalance.
- **Implication for Unseen Attack Experiments:** Holding out `Worms` or `Shellcode` as an unseen attack is ecologically valid, as novel exploits in reality appear with low initial frequency.

## 2. Feature Multicollinearity & Redundancy

Pairs with Pearson correlation coefficient $|r| \ge 0.80$:

| Feature 1 | Feature 2 | Pearson Correlation ($r$) | Recommendation |
| :--- | :--- | :--- | :--- |
| `dbytes` | `dloss` | 0.997 | High collinearity; monitor feature importance in tree models. |
| `sbytes` | `sloss` | 0.996 | High collinearity; monitor feature importance in tree models. |
| `swin` | `dwin` | 0.990 | High collinearity; monitor feature importance in tree models. |
| `dpkts` | `dloss` | 0.979 | High collinearity; monitor feature importance in tree models. |
| `dpkts` | `dbytes` | 0.972 | High collinearity; monitor feature importance in tree models. |
| `spkts` | `sloss` | 0.971 | High collinearity; monitor feature importance in tree models. |
| `spkts` | `sbytes` | 0.964 | High collinearity; monitor feature importance in tree models. |
| `tcprtt` | `synack` | 0.949 | High collinearity; monitor feature importance in tree models. |
| `tcprtt` | `ackdat` | 0.942 | High collinearity; monitor feature importance in tree models. |


## 3. Distribution Characteristics & Scaling Implications

- **Heavy Tails & Extreme Outliers:** Features such as `sload`, `dload`, `sbytes`, `dbytes`, and `dur` span multiple orders of magnitude with severe positive skew.
- **Scaling Requirement:** Standard Z-score (`StandardScaler`) is vulnerable to severe distortion by extreme network flow outliers. `RobustScaler` (median & IQR centering) or logarithmic transformations are strongly favored.

## 4. Generated Artifacts

- `reports/figures/binary_distribution.png`: Binary label proportions.
- `reports/figures/class_imbalance.png`: Linear vs log multiclass distribution.
- `reports/figures/correlation_matrix.png`: Triangular correlation heatmap.
- `reports/figures/normal_vs_attack_features.png`: Boxplots comparing Normal vs Attack profiles.
