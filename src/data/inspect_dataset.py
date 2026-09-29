"""Phase 2: Comprehensive Dataset Understanding & Auditing for UNSW-NB15."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

def audit_split(df: pd.DataFrame, split_name: str) -> dict:
    """Analyze a single split of the dataset."""
    total_rows, total_cols = df.shape

    # Data types and columns
    col_types = {col: str(dtype) for col, dtype in df.dtypes.items()}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    # Missing values
    missing_counts = df.isnull().sum().to_dict()
    cols_with_missing = {k: int(v) for k, v in missing_counts.items() if v > 0}

    # Infinite values in numeric columns
    inf_counts = {}
    for col in numeric_cols:
        inf_num = int(np.isneginf(df[col]).sum() + np.isposinf(df[col]).sum())
        if inf_num > 0:
            inf_counts[col] = inf_num

    # Duplicate rows (full dataset and without identifier if 'id' is present)
    full_duplicates = int(df.duplicated().sum())
    features_only_cols = [c for c in df.columns if c.lower() != "id"]
    feature_duplicates = int(df.duplicated(subset=features_only_cols).sum())

    # Binary label distribution
    binary_dist = {}
    if "label" in df.columns:
        binary_counts = df["label"].value_counts().to_dict()
        binary_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v / total_rows * 100), 2),
            }
            for k, v in binary_counts.items()
        }

    # Multiclass attack category distribution
    attack_cat_dist = {}
    if "attack_cat" in df.columns:
        # Standardize string whitespace for reporting
        cleaned_cat = df["attack_cat"].astype(str).str.strip()
        cat_counts = cleaned_cat.value_counts(dropna=False).to_dict()
        attack_cat_dist = {
            str(k): {
                "count": int(v),
                "percentage": round(float(v / total_rows * 100), 4),
            }
            for k, v in cat_counts.items()
        }

    return {
        "split_name": split_name,
        "total_rows": total_rows,
        "total_columns": total_cols,
        "columns": list(df.columns),
        "numeric_column_count": len(numeric_cols),
        "numeric_columns": numeric_cols,
        "categorical_column_count": len(categorical_cols),
        "categorical_columns": categorical_cols,
        "missing_values_by_column": cols_with_missing,
        "has_missing": len(cols_with_missing) > 0,
        "infinite_values_by_column": inf_counts,
        "has_infinite": len(inf_counts) > 0,
        "exact_duplicates": full_duplicates,
        "duplicates_excluding_id": feature_duplicates,
        "binary_label_distribution": binary_dist,
        "attack_category_distribution": attack_cat_dist,
    }

def run_dataset_audit():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    reports_dir = project_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    train_file = raw_dir / "UNSW_NB15_training-set.parquet"
    test_file = raw_dir / "UNSW_NB15_testing-set.parquet"

    if not train_file.exists() or not test_file.exists():
        raise FileNotFoundError(
            f"Dataset files not found in {raw_dir}. Expected parquet files."
        )

    print(f"[*] Loading training split from {train_file.name}...")
    df_train = pd.read_parquet(train_file)
    print(f"[*] Loading testing split from {test_file.name}...")
    df_test = pd.read_parquet(test_file)

    print("[*] Auditing training split...")
    train_audit = audit_split(df_train, "training_set")
    print("[*] Auditing testing split...")
    test_audit = audit_split(df_test, "testing_set")

    # Cross-split comparison
    train_cols_set = set(df_train.columns)
    test_cols_set = set(df_test.columns)
    columns_match = train_cols_set == test_cols_set
    missing_in_test = list(train_cols_set - test_cols_set)
    missing_in_train = list(test_cols_set - train_cols_set)

    train_cats = set(train_audit["attack_category_distribution"].keys())
    test_cats = set(test_audit["attack_category_distribution"].keys())

    audit_summary = {
        "dataset_name": "UNSW-NB15",
        "training_set": train_audit,
        "testing_set": test_audit,
        "cross_split_validation": {
            "columns_identical": columns_match,
            "columns_in_train_not_test": missing_in_train,
            "columns_in_test_not_train": missing_in_test,
            "attack_categories_in_train": sorted(list(train_cats)),
            "attack_categories_in_test": sorted(list(test_cats)),
            "attack_categories_overlap": sorted(list(train_cats.intersection(test_cats))),
            "potential_leakage_identifiers": [
                col for col in df_train.columns if col.lower() in ["id", "index", "uid"]
            ],
        },
    }

    # Save JSON report
    json_path = reports_dir / "dataset_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2)
    print(f"[+] Saved structured JSON audit to {json_path}")

    # Generate and save Markdown report
    md_report = generate_markdown_report(audit_summary)
    md_path = reports_dir / "dataset_audit.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_report)
    print(f"[+] Saved Markdown audit report to {md_path}")
    print("[✓] Phase 2 Dataset Audit Complete.")

def generate_markdown_report(audit: dict) -> str:
    train = audit["training_set"]
    test = audit["testing_set"]
    cross = audit["cross_split_validation"]

    md = []
    md.append("# UNSW-NB15 Dataset Understanding & Audit Report\n")
    md.append("**Phase:** Phase 2 (Dataset Understanding)\n")
    md.append(f"**Dataset:** {audit['dataset_name']}\n")

    md.append("## 1. Overview & Shape\n")
    md.append("| Metric | Training Set | Testing Set | Combined |")
    md.append("| :--- | :--- | :--- | :--- |")
    md.append(f"| **Total Records (Rows)** | {train['total_rows']:,} | {test['total_rows']:,} | {train['total_rows'] + test['total_rows']:,} |")
    md.append(f"| **Total Features (Columns)** | {train['total_columns']} | {test['total_columns']} | {train['total_columns']} |")
    md.append(f"| **Numeric Features** | {train['numeric_column_count']} | {test['numeric_column_count']} | - |")
    md.append(f"| **Categorical Features** | {train['categorical_column_count']} | {test['categorical_column_count']} | - |")
    md.append(f"| **Missing / Null Values** | {'None' if not train['has_missing'] else train['missing_values_by_column']} | {'None' if not test['has_missing'] else test['missing_values_by_column']} | - |")
    md.append(f"| **Infinite Values** | {'None' if not train['has_infinite'] else train['infinite_values_by_column']} | {'None' if not test['has_infinite'] else test['infinite_values_by_column']} | - |")
    md.append(f"| **Full Exact Duplicates** | {train['exact_duplicates']} | {test['exact_duplicates']} | - |")
    md.append(f"| **Duplicates (excluding `id`)** | {train['duplicates_excluding_id']} | {test['duplicates_excluding_id']} | - |\n")

    md.append("## 2. Target Variable Distributions\n")
    md.append("### 2.1 Binary Classification (`label`: 0 = Normal, 1 = Attack)\n")
    md.append("| Class | Train Count | Train % | Test Count | Test % |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    for cls_key in sorted(train["binary_label_distribution"].keys()):
        cls_name = "Normal (0)" if cls_key == "0" else "Attack (1)"
        tr_info = train["binary_label_distribution"].get(cls_key, {"count": 0, "percentage": 0.0})
        te_info = test["binary_label_distribution"].get(cls_key, {"count": 0, "percentage": 0.0})
        md.append(f"| **{cls_name}** | {tr_info['count']:,} | {tr_info['percentage']}% | {te_info['count']:,} | {te_info['percentage']}% |")
    md.append("\n")

    md.append("### 2.2 Multiclass Attack Categories (`attack_cat`)\n")
    md.append("| Category | Train Count | Train % | Test Count | Test % |")
    md.append("| :--- | :--- | :--- | :--- | :--- |")
    all_cats = sorted(list(set(train["attack_category_distribution"].keys()) | set(test["attack_category_distribution"].keys())))
    for cat in all_cats:
        tr = train["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        te = test["attack_category_distribution"].get(cat, {"count": 0, "percentage": 0.0})
        md.append(f"| **{cat}** | {tr['count']:,} | {tr['percentage']:.2f}% | {te['count']:,} | {te['percentage']:.2f}% |")
    md.append("\n")

    md.append("## 3. Data Integrity & Safety Checks\n")
    md.append(f"- **Column Parity:** {'MATCH (Both sets share identical columns)' if cross['columns_identical'] else 'MISMATCH'}")
    md.append(f"- **Identifier Columns (Potential Leakage):** `{cross['potential_leakage_identifiers']}`")
    md.append("  > **Rule:** Identifier columns like `id` must be dropped prior to modeling to prevent data leakage.")
    md.append(f"- **Missing Value Handling:** Training missing: {train['has_missing']}, Testing missing: {test['has_missing']}")
    md.append(f"- **Infinite Values:** Training infinite: {train['has_infinite']}, Testing infinite: {test['has_infinite']}")
    md.append(f"- **Categorical Features:** `{train['categorical_columns']}`")
    md.append("\n")

    return "\n".join(md)

if __name__ == "__main__":
    run_dataset_audit()
