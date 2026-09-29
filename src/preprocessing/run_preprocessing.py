import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from src.preprocessing.pipeline import (
    CANONICAL_ATTACK_CATEGORIES,
    IDX_TO_LABEL,
    LABEL_TO_IDX,
    NIDSPreprocessor,
)

def run_preprocessing():
    project_root = Path(__file__).resolve().parent.parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    models_dir = project_root / "models"
    processed_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    print("[*] Loading raw UNSW-NB15 splits from data/raw/...")
    train_df = pd.read_parquet(raw_dir / "UNSW_NB15_training-set.parquet")
    test_df = pd.read_parquet(raw_dir / "UNSW_NB15_testing-set.parquet")

    print(f"  -> Raw train shape: {train_df.shape}")
    print(f"  -> Raw test shape: {test_df.shape}")

    # Initialize and fit strictly on training data
    print("[*] Fitting NIDSPreprocessor strictly on training data (Zero Data Leakage)...")
    preprocessor = NIDSPreprocessor()
    X_train, y_train_bin, y_train_multi, raw_cats_train = preprocessor.fit_transform(train_df)

    print(f"[+] Fitted preprocessor: {len(preprocessor.numeric_features)} numeric + "
          f"{len(preprocessor.feature_names_out) - len(preprocessor.numeric_features)} encoded categorical = "
          f"{len(preprocessor.feature_names_out)} total features.")

    # Transform testing data using the pre-fitted transformer
    print("[*] Transforming testing data using fitted preprocessor...")
    X_test, y_test_bin, y_test_multi, raw_cats_test = preprocessor.transform(test_df)

    # Save fitted preprocessor
    preprocessor_path = models_dir / "preprocessor.joblib"
    preprocessor.save(preprocessor_path)
    print(f"[+] Saved serialized preprocessor to {preprocessor_path}")

    # Create processed DataFrames for storage
    df_train_proc = pd.DataFrame(X_train, columns=preprocessor.feature_names_out)
    df_train_proc["label"] = y_train_bin
    df_train_proc["attack_cat_idx"] = y_train_multi
    df_train_proc["attack_cat"] = raw_cats_train

    df_test_proc = pd.DataFrame(X_test, columns=preprocessor.feature_names_out)
    df_test_proc["label"] = y_test_bin
    df_test_proc["attack_cat_idx"] = y_test_multi
    df_test_proc["attack_cat"] = raw_cats_test

    # Save processed Parquet files
    train_proc_path = processed_dir / "train_processed.parquet"
    test_proc_path = processed_dir / "test_processed.parquet"
    print(f"[*] Saving processed datasets to {processed_dir}...")
    df_train_proc.to_parquet(train_proc_path, index=False)
    df_test_proc.to_parquet(test_proc_path, index=False)
    print(f"  -> Saved {train_proc_path.name} ({train_proc_path.stat().st_size / (1024*1024):.2f} MB)")
    print(f"  -> Saved {test_proc_path.name} ({test_proc_path.stat().st_size / (1024*1024):.2f} MB)")

    # Metadata & verification
    metadata = {
        "pipeline": "NIDSPreprocessor",
        "scaling": "RobustScaler",
        "categorical_encoding": "Top-K OneHotEncoder (handle_unknown='ignore')",
        "top_k_config": preprocessor.top_k_config,
        "total_transformed_features": len(preprocessor.feature_names_out),
        "numeric_features": preprocessor.numeric_features,
        "feature_names": preprocessor.feature_names_out,
        "canonical_attack_categories": CANONICAL_ATTACK_CATEGORIES,
        "label_to_idx": LABEL_TO_IDX,
        "train_samples": int(len(df_train_proc)),
        "test_samples": int(len(df_test_proc)),
        "nan_counts_train": int(df_train_proc.isna().sum().sum()),
        "nan_counts_test": int(df_test_proc.isna().sum().sum()),
    }

    metadata_path = processed_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[+] Saved preprocessing metadata to {metadata_path}")
    print("[✓] Phase 3 Preprocessing Pipeline executed successfully.")

if __name__ == "__main__":
    run_preprocessing()
