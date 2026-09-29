"""Phase 5: Supervised Baseline Model (Random Forest Binary Classifier)."""

import json
import sys
import time
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

def train_and_evaluate_baseline():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    for d in [models_dir, results_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Loading processed training and testing datasets...")
    train_df = pd.read_parquet(processed_dir / "train_processed.parquet")
    test_df = pd.read_parquet(processed_dir / "test_processed.parquet")

    with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["label"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["label"].to_numpy()

    print(f"  -> X_train shape: {X_train.shape}, y_train distribution: {np.bincount(y_train)}")
    print(f"  -> X_test shape: {X_test.shape}, y_test distribution: {np.bincount(y_test)}")

    # Initialize Random Forest with balanced class weights
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("[*] Training Random Forest Baseline Classifier...")
    t0_train = time.perf_counter()
    rf.fit(X_train, y_train)
    train_time_sec = round(time.perf_counter() - t0_train, 3)
    print(f"[+] Model trained in {train_time_sec} seconds.")

    # Evaluate inference
    print("[*] Evaluating on testing set (82,332 flows)...")
    t0_infer = time.perf_counter()
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    infer_time_sec = round(time.perf_counter() - t0_infer, 3)
    ms_per_flow = round((infer_time_sec / len(X_test)) * 1000, 4)
    print(f"[+] Inference complete: {infer_time_sec}s total ({ms_per_flow} ms/flow).")

    # Metrics computation
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro"))
    rec_macro = float(recall_score(y_test, y_pred, average="macro"))
    f1_macro = float(f1_score(y_test, y_pred, average="macro"))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted"))
    roc_auc = float(roc_auc_score(y_test, y_prob))
    pr_auc = float(average_precision_score(y_test, y_prob))

    # Confusion Matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    fpr = float(fp / (fp + tn))
    fnr = float(fn / (fn + tp))

    # Feature importances
    importances = rf.feature_importances_
    top_indices = np.argsort(importances)[::-1][:15]
    top_features = [
        {"feature": feature_cols[idx], "importance": round(float(importances[idx]), 4)}
        for idx in top_indices
    ]

    print("\n--- BASELINE EVALUATION METRICS ---")
    print(f"Accuracy:        {acc:.4f}")
    print(f"Macro F1-Score:  {f1_macro:.4f}")
    print(f"Weighted F1:     {f1_weighted:.4f}")
    print(f"Macro Precision: {prec_macro:.4f}")
    print(f"Macro Recall:    {rec_macro:.4f}")
    print(f"ROC-AUC:         {roc_auc:.4f}")
    print(f"PR-AUC:          {pr_auc:.4f}")
    print(f"False Positive Rate (FPR): {fpr:.4f} ({fp:,} false alarms / {fp+tn:,} normal)")
    print(f"False Negative Rate (FNR): {fnr:.4f} ({fn:,} missed attacks / {fn+tp:,} attacks)")
    print(f"Training Time:   {train_time_sec}s")
    print(f"Inference Latency: {ms_per_flow} ms/flow")

    # Plot Confusion Matrix
    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=",d",
        cmap="Blues",
        xticklabels=["Normal (0)", "Attack (1)"],
        yticklabels=["Normal (0)", "Attack (1)"],
        cbar=False,
        annot_kws={"size": 14, "weight": "bold"}
    )
    plt.xlabel("Predicted Label", fontsize=12, weight="bold")
    plt.ylabel("True Label", fontsize=12, weight="bold")
    plt.title(f"Random Forest Baseline Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro F1: {f1_macro:.4f}", fontsize=12)
    plt.tight_layout()
    cm_path = figures_dir / "baseline_confusion_matrix.png"
    plt.savefig(cm_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved confusion matrix heatmap to {cm_path}")

    # Save Model
    model_save_path = models_dir / "baseline_rf_binary.joblib"
    joblib.dump(rf, model_save_path)
    print(f"[+] Saved baseline model to {model_save_path}")

    # Save structured JSON results
    results_payload = {
        "model": "RandomForestClassifier",
        "task": "Binary Classification (Normal vs Attack)",
        "hyperparameters": {
            "n_estimators": 100,
            "max_depth": 20,
            "class_weight": "balanced",
            "random_state": 42
        },
        "metrics": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr": fpr,
            "fnr": fnr,
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp)
            },
            "training_time_sec": train_time_sec,
            "inference_time_sec": infer_time_sec,
            "ms_per_flow": ms_per_flow
        },
        "top_15_features": top_features
    }

    results_json_path = results_dir / "baseline_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[+] Saved metrics to {results_json_path}")

    # Generate Markdown Report
    report_md = f"""# Phase 5: Supervised Baseline Classifier Report

**Model:** Random Forest (100 estimators, max_depth=20, balanced class weighting)  
**Task:** Binary Classification (Normal vs Attack)  
**Training Set:** 175,341 flows (68.06% Attack, 31.94% Normal)  
**Testing Set:** 82,332 flows (55.06% Attack, 44.94% Normal)  

## 1. Quantitative Performance Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Accuracy** | {acc*100:.2f}% | Overall proportion of correct predictions |
| **Macro F1-Score** | {f1_macro:.4f} | Unweighted mean F1 across both classes |
| **Weighted F1-Score** | {f1_weighted:.4f} | Class-frequency weighted F1 |
| **Macro Precision** | {prec_macro:.4f} | Mean precision across Normal and Attack |
| **Macro Recall** | {rec_macro:.4f} | Mean recall across Normal and Attack |
| **ROC-AUC** | {roc_auc:.4f} | Area under Receiver Operating Characteristic |
| **PR-AUC** | {pr_auc:.4f} | Area under Precision-Recall curve |
| **False Positive Rate (FPR)** | {fpr*100:.2f}% | Proportion of normal flows falsely flagged as attacks |
| **False Negative Rate (FNR)** | {fnr*100:.2f}% | Proportion of attack flows missed |

## 2. Confusion Matrix
| True \\ Predicted | Predicted Normal (0) | Predicted Attack (1) | Total |
| :--- | :--- | :--- | :--- |
| **True Normal (0)** | {tn:,} (TN) | {fp:,} (FP) | {tn+fp:,} |
| **True Attack (1)** | {fn:,} (FN) | {tp:,} (TP) | {fn+tp:,} |

## 3. Computational Latency
- **Training Time:** {train_time_sec} seconds on 175,341 samples.
- **Inference Latency:** {ms_per_flow} ms per network flow ({infer_time_sec}s for 82,332 samples).

## 4. Top 10 Most Influential Features
| Rank | Feature | Gini Importance |
| :--- | :--- | :--- |
"""
    for i, item in enumerate(top_features[:10], 1):
        report_md += f"| {i} | `{item['feature']}` | {item['importance']:.4f} |\n"

    report_md_path = reports_dir / "baseline_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved markdown report to {report_md_path}")
    print("[✓] Phase 5 Baseline Supervised Classifier completed.")

if __name__ == "__main__":
    train_and_evaluate_baseline()
