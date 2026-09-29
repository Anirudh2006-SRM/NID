"""Phase 3: Reproducible Preprocessing Pipeline for UNSW-NB15.

Handles:
- Frequency-gated One-Hot Encoding for categorical features ('proto', 'service', 'state')
- RobustScaler for numerical features (resistant to network flow outliers)
- Target representations for Binary (0: Normal, 1: Attack) and Multiclass (10 classes)
- Strict train-only fitting to guarantee zero data leakage
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, RobustScaler

CANONICAL_ATTACK_CATEGORIES: List[str] = [
    "Normal",
    "Generic",
    "Exploits",
    "Fuzzers",
    "DoS",
    "Reconnaissance",
    "Analysis",
    "Backdoor",
    "Shellcode",
    "Worms",
]

LABEL_TO_IDX: Dict[str, int] = {cat: idx for idx, cat in enumerate(CANONICAL_ATTACK_CATEGORIES)}
IDX_TO_LABEL: Dict[int, str] = {idx: cat for cat, idx in LABEL_TO_IDX.items()}

# Top-K thresholds for categorical features
TOP_K_CONFIG: Dict[str, int] = {
    "proto": 15,    # Top 15 protocols cover >99% of flows
    "service": 10,  # Top 10 services cover >99.5%
    "state": 6,     # Top 6 states cover >99.8%
}

class NIDSPreprocessor:
    """Production and research grade preprocessor for UNSW-NB15 network flows."""

    def __init__(self, top_k_config: Optional[Dict[str, int]] = None):
        self.top_k_config = top_k_config or TOP_K_CONFIG
        self.scaler = RobustScaler()
        self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        self.top_k_categories: Dict[str, List[str]] = {}
        self.numeric_features: List[str] = []
        self.categorical_features: List[str] = ["proto", "service", "state"]
        self.feature_names_out: List[str] = []
        self.is_fitted: bool = False

    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize strings and clean column data."""
        df_clean = df.copy()
        for cat_col in self.categorical_features:
            if cat_col in df_clean.columns:
                df_clean[cat_col] = df_clean[cat_col].astype(str).str.strip().str.lower()
        if "attack_cat" in df_clean.columns:
            df_clean["attack_cat"] = df_clean["attack_cat"].astype(str).str.strip()
            # Normalize minor spelling differences if any
            df_clean["attack_cat"] = df_clean["attack_cat"].replace({
                "Backdoors": "Backdoor",
                "Fuzzers ": "Fuzzers",
                "Reconnaissance ": "Reconnaissance",
                "Shellcode ": "Shellcode",
            })
        return df_clean

    def _apply_top_k(self, df: pd.DataFrame, is_train: bool = False) -> pd.DataFrame:
        """Group rare categories into 'other' token based on training frequencies."""
        df_top = df.copy()
        for col, k in self.top_k_config.items():
            if col not in df_top.columns:
                continue
            if is_train:
                top_vals = df_top[col].value_counts().nlargest(k).index.tolist()
                self.top_k_categories[col] = top_vals
            else:
                top_vals = self.top_k_categories[col]

            # Replace rare/unseen with 'other'
            df_top[col] = df_top[col].apply(lambda x: x if x in top_vals else "other")
        return df_top

    def fit(self, df_train: pd.DataFrame) -> "NIDSPreprocessor":
        """Fit scaler, top-K thresholds, and one-hot encoder strictly on training data."""
        df_clean = self._clean_dataframe(df_train)

        # Separate numeric features (exclude targets & identifiers)
        exclude_cols = set(self.categorical_features + ["label", "attack_cat", "id", "index"])
        self.numeric_features = [
            c for c in df_clean.columns
            if c not in exclude_cols and np.issubdtype(df_clean[c].dtype, np.number)
        ]

        # 1. Fit scaler on numeric features
        self.scaler.fit(df_clean[self.numeric_features])

        # 2. Learn top-K and fit encoder on categorical features
        df_cat = self._apply_top_k(df_clean[self.categorical_features], is_train=True)
        self.encoder.fit(df_cat)

        # Build clean output feature names
        num_names = [f"num_{col}" for col in self.numeric_features]
        cat_names = [f"cat_{name}" for name in self.encoder.get_feature_names_out(self.categorical_features)]
        self.feature_names_out = num_names + cat_names

        self.is_fitted = True
        return self

    def transform(
        self, df: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Transform network flows using fitted statistics.
        
        Returns:
            X (np.ndarray): Scaled numerical + one-hot encoded categorical features.
            y_binary (np.ndarray or None): Binary target (0 = Normal, 1 = Attack).
            y_multiclass (np.ndarray or None): Multiclass integer labels (0..9).
            attack_cat_raw (np.ndarray or None): Raw attack category strings (for holdout splitting).
        """
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fit before calling transform.")

        df_clean = self._clean_dataframe(df)

        # Scale numerical features
        X_num = self.scaler.transform(df_clean[self.numeric_features])

        # Encode categorical features
        df_cat = self._apply_top_k(df_clean[self.categorical_features], is_train=False)
        X_cat = self.encoder.transform(df_cat)

        # Combine features
        X = np.hstack([X_num, X_cat])

        # Target processing
        y_binary = None
        if "label" in df_clean.columns:
            y_binary = df_clean["label"].to_numpy(dtype=np.int64)

        y_multiclass = None
        attack_cat_raw = None
        if "attack_cat" in df_clean.columns:
            attack_cat_raw = df_clean["attack_cat"].to_numpy()
            y_multiclass = np.array(
                [LABEL_TO_IDX.get(cat, -1) for cat in attack_cat_raw],
                dtype=np.int64
            )

        return X, y_binary, y_multiclass, attack_cat_raw

    def fit_transform(
        self, df_train: pd.DataFrame
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
        """Fit on train and transform."""
        return self.fit(df_train).transform(df_train)

    def save(self, filepath: Union[str, Path]) -> None:
        """Serialize fitted preprocessor to disk."""
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: Union[str, Path]) -> "NIDSPreprocessor":
        """Load serialized preprocessor from disk."""
        return joblib.load(filepath)
