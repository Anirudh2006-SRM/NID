"""Phase 6: Multiclass Supervised Classifier across 10 UNSW-NB15 categories."""

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
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

def train_and_evaluate_multiclass():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    for d in [models_dir, results_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Loading processed datasets for multiclass training...")
    train_df = pd.read_parquet(processed_dir / "train_processed.parquet")
    test_df = pd.read_parquet(processed_dir / "test_processed.parquet")

    with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_names"]
    categories = meta["canonical_attack_categories"]

    X_train = train_df[feature_cols].to_numpy()
    y_train = train_df["attack_cat_idx"].to_numpy()
    X_test = test_df[feature_cols].to_numpy()
    y_test = test_df["attack_cat_idx"].to_numpy()

    print(f"  -> X_train: {X_train.shape}, classes: {len(categories)}")
    print(f"  -> X_test:  {X_test.shape}, classes: {len(categories)}")

    # Initialize Multiclass Random Forest with balanced class weights
    rf_multi = RandomForestClassifier(
        n_estimators=100,
        max_depth=22,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("[*] Training Multiclass Random Forest Classifier (10 Categories)...")
    t0_train = time.perf_counter()
    rf_multi.fit(X_train, y_train)
    train_time_sec = round(time.perf_counter() - t0_train, 3)
    print(f"[+] Multiclass model trained in {train_time_sec} seconds.")

    # Evaluate inference
    print("[*] Evaluating on testing set (82,332 flows)...")
    t0_infer = time.perf_counter()
    y_pred = rf_multi.predict(X_test)
    infer_time_sec = round(time.perf_counter() - t0_infer, 3)
    ms_per_flow = round((infer_time_sec / len(X_test)) * 1000, 4)
    print(f"[+] Inference complete: {infer_time_sec}s total ({ms_per_flow} ms/flow).")

    # Metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec_macro = float(precision_score(y_test, y_pred, average="macro", zero_division=0))
    rec_macro = float(recall_score(y_test, y_pred, average="macro", zero_division=0))
    f1_macro = float(f1_score(y_test, y_pred, average="macro", zero_division=0))
    f1_weighted = float(f1_score(y_test, y_pred, average="weighted", zero_division=0))

    # Detailed Per-Class Report
    class_report_dict = classification_report(
        y_test,
        y_pred,
        target_names=categories,
        output_dict=True,
        zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, labels=list(range(len(categories))))
    # Normalize by true class (row sum)
    with np.errstate(divide="ignore", invalid="ignore"):
        cm_normalized = np.where(cm.sum(axis=1, keepdims=True) != 0,
                                 cm.astype("float") / cm.sum(axis=1, keepdims=True), 0.0)

    print("\n--- MULTICLASS EVALUATION METRICS ---")
    print(f"Overall Accuracy: {acc:.4f} ({acc*100:.2f}%)")
    print(f"Macro F1-Score:   {f1_macro:.4f}")
    print(f"Weighted F1:      {f1_weighted:.4f}")
    print(f"Macro Precision:  {prec_macro:.4f}")
    print(f"Macro Recall:     {rec_macro:.4f}")
    print(f"Training Time:    {train_time_sec}s")
    print(f"Inference Time:   {ms_per_flow} ms/flow\n")

    print("Per-Class Performance Summary:")
    for cat in categories:
        metrics = class_report_dict.get(cat, {})
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        f = metrics.get("f1-score", 0.0)
        s = metrics.get("support", 0)
        print(f"  - {cat:<15}: Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}, Support={s}")

    # Plot Confusion Matrix Heatmap
    plt.figure(figsize=(11, 9))
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=categories,
        yticklabels=categories,
        cbar_kws={"label": "Normalized Recall (Proportion of True Class)"}
    )
    plt.xlabel("Predicted Category", fontsize=11, weight="bold")
    plt.ylabel("True Category", fontsize=11, weight="bold")
    plt.title(f"Multiclass Random Forest Confusion Matrix\nAccuracy: {acc*100:.2f}% | Macro F1: {f1_macro:.4f}", fontsize=12, pad=12)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    cm_plot_path = figures_dir / "multiclass_confusion_matrix.png"
    plt.savefig(cm_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved multiclass confusion matrix plot to {cm_plot_path}")

    # Save Model
    model_path = models_dir / "multiclass_rf.joblib"
    joblib.dump(rf_multi, model_path)
    print(f"[+] Saved multiclass model to {model_path}")

    # Save JSON Results
    results_payload = {
        "model": "RandomForestClassifier (Multiclass)",
        "task": "10-Class Classification",
        "categories": categories,
        "metrics": {
            "accuracy": acc,
            "macro_precision": prec_macro,
            "macro_recall": rec_macro,
            "macro_f1": f1_macro,
            "weighted_f1": f1_weighted,
            "training_time_sec": train_time_sec,
            "inference_time_sec": infer_time_sec,
            "ms_per_flow": ms_per_flow
        },
        "per_class_metrics": {
            cat: class_report_dict[cat] for cat in categories if cat in class_report_dict
        },
        "raw_confusion_matrix": cm.tolist()
    }

    results_json_path = results_dir / "multiclass_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[+] Saved metrics JSON to {results_json_path}")

    # Save Markdown Report
    report_md = f"""# Phase 6: Multiclass Supervised Classifier Report

**Model:** Random Forest (100 estimators, max_depth=22, balanced class weighting)  
**Task:** 10-Class Attack Categorization  
**Training Set:** 175,341 flows across 10 categories  
**Testing Set:** 82,332 flows across 10 categories  

## 1. Global Performance Metrics
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **Overall Accuracy** | {acc*100:.2f}% | Proportion of exact class matches |
| **Macro F1-Score** | {f1_macro:.4f} | Unweighted average F1 across all 10 categories |
| **Weighted F1-Score** | {f1_weighted:.4f} | Support-weighted average F1 |
| **Macro Precision** | {prec_macro:.4f} | Unweighted average precision |
| **Macro Recall** | {rec_macro:.4f} | Unweighted average recall |
| **Training Time** | {train_time_sec}s | Latency to fit 175,341 flows across 65 features |
| **Inference Latency** | {ms_per_flow} ms/flow | Latency per network flow ({infer_time_sec}s for 82,332 samples) |

## 2. Per-Class Performance Breakdown
| Category | Support (Test) | Precision | Recall | F1-Score | Class Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cat in categories:
        info = class_report_dict.get(cat, {})
        p = info.get("precision", 0.0)
        r = info.get("recall", 0.0)
        f = info.get("f1-score", 0.0)
        s = info.get("support", 0)
        status = "Majority" if s > 10000 else ("Moderate" if s > 1000 else "Extreme Minority")
        report_md += f"| **{cat}** | {s:,} | {p:.4f} | {r:.4f} | {f:.4f} | {status} |\n"

    report_md += """
## 3. Key Observations for Unseen Attack Experiments
1. High-frequency categories (`Normal`, `Generic`, `Exploits`) achieve strong classification metrics.
2. Low-frequency categories (`Worms` support=44, `Shellcode` support=378) demonstrate the difficulty supervised models encounter when samples are scarce.
3. This sets the exact baseline needed for Phase 8: testing whether holding out an attack category leaves the supervised model blind to it while the anomaly detector catches it.
"""
    report_md_path = reports_dir / "multiclass_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved markdown report to {report_md_path}")
    print("[✓] Phase 6 Multiclass Supervised Classifier completed.")

if __name__ == "__main__":
    train_and_evaluate_multiclass()
