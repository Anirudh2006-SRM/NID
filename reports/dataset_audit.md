# UNSW-NB15 Dataset Understanding & Audit Report

**Phase:** Phase 2 (Dataset Understanding)

**Dataset:** UNSW-NB15

## 1. Overview & Shape

| Metric | Training Set | Testing Set | Combined |
| :--- | :--- | :--- | :--- |
| **Total Records (Rows)** | 175,341 | 82,332 | 257,673 |
| **Total Features (Columns)** | 36 | 36 | 36 |
| **Numeric Features** | 32 | 32 | - |
| **Categorical Features** | 4 | 4 | - |
| **Missing / Null Values** | None | None | - |
| **Infinite Values** | None | None | - |
| **Full Exact Duplicates** | 78519 | 32361 | - |
| **Duplicates (excluding `id`)** | 78519 | 32361 | - |

## 2. Target Variable Distributions

### 2.1 Binary Classification (`label`: 0 = Normal, 1 = Attack)

| Class | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- |
| **Normal (0)** | 56,000 | 31.94% | 37,000 | 44.94% |
| **Attack (1)** | 119,341 | 68.06% | 45,332 | 55.06% |


### 2.2 Multiclass Attack Categories (`attack_cat`)

| Category | Train Count | Train % | Test Count | Test % |
| :--- | :--- | :--- | :--- | :--- |
| **Analysis** | 2,000 | 1.14% | 677 | 0.82% |
| **Backdoor** | 1,746 | 1.00% | 583 | 0.71% |
| **DoS** | 12,264 | 6.99% | 4,089 | 4.97% |
| **Exploits** | 33,393 | 19.04% | 11,132 | 13.52% |
| **Fuzzers** | 18,184 | 10.37% | 6,062 | 7.36% |
| **Generic** | 40,000 | 22.81% | 18,871 | 22.92% |
| **Normal** | 56,000 | 31.94% | 37,000 | 44.94% |
| **Reconnaissance** | 10,491 | 5.98% | 3,496 | 4.25% |
| **Shellcode** | 1,133 | 0.65% | 378 | 0.46% |
| **Worms** | 130 | 0.07% | 44 | 0.05% |


## 3. Data Integrity & Safety Checks

- **Column Parity:** MATCH (Both sets share identical columns)
- **Identifier Columns (Potential Leakage):** `[]`
  > **Rule:** Identifier columns like `id` must be dropped prior to modeling to prevent data leakage.
- **Missing Value Handling:** Training missing: False, Testing missing: False
- **Infinite Values:** Training infinite: False, Testing infinite: False
- **Categorical Features:** `['proto', 'service', 'state', 'attack_cat']`

