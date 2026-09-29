"""Phase 7: Anomaly Detection Baseline using Isolation Forest.

Answers: 'Does this traffic behave unusually compared with benign network reference behavior?'
Trained strictly on Normal training flows (56,000 samples).
Calibrates anomaly thresholds on normal training scores (90th and 95th percentiles).
"""

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
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

def train_and_evaluate_anomaly():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    for d in [models_dir, results_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Loading processed datasets for anomaly detection...")
    train_df = pd.read_parquet(processed_dir / "train_processed.parquet")
    test_df = pd.read_parquet(processed_dir / "test_processed.parquet")

    with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)

    feature_cols = meta["feature_names"]
    categories = meta["canonical_attack_categories"]

    # Filter normal traffic for reference training (56,000 flows)
    train_normal = train_df[train_df["label"] == 0]
    X_train_normal = train_normal[feature_cols].to_numpy()

    X_test = test_df[feature_cols].to_numpy()
    y_test_bin = test_df["label"].to_numpy()
    y_test_cats = test_df["attack_cat"].to_numpy()

    print(f"  -> Normal training reference shape: {X_train_normal.shape}")
    print(f"  -> Testing set shape: {X_test.shape} ({np.bincount(y_test_bin)[0]} Normal, {np.bincount(y_test_bin)[1]} Attack)")

    # Initialize Isolation Forest
    # score_samples returns opposite of anomaly score; we negate it so higher = more anomalous
    iso_forest = IsolationForest(
        n_estimators=100,
        max_samples=256,
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    print("[*] Training Isolation Forest on Normal Reference Traffic...")
    t0_train = time.perf_counter()
    iso_forest.fit(X_train_normal)
    train_time_sec = round(time.perf_counter() - t0_train, 3)
    print(f"[+] Anomaly model trained in {train_time_sec} seconds.")

    # Threshold calibration on Normal training scores
    print("[*] Calibrating reproducible threshold on Normal training flows...")
    train_scores = -iso_forest.score_samples(X_train_normal)
    thresh_p90 = float(np.percentile(train_scores, 90.0))
    thresh_p95 = float(np.percentile(train_scores, 95.0))
    print(f"  -> 90th percentile threshold (target 10% FPR): {thresh_p90:.5f}")
    print(f"  -> 95th percentile threshold (target  5% FPR): {thresh_p95:.5f}")

    # Evaluate on testing set
    print("[*] Scoring 82,332 testing flows...")
    t0_infer = time.perf_counter()
    test_scores = -iso_forest.score_samples(X_test)
    infer_time_sec = round(time.perf_counter() - t0_infer, 3)
    ms_per_flow = round((infer_time_sec / len(X_test)) * 1000, 4)
    print(f"[+] Scoring complete: {infer_time_sec}s ({ms_per_flow} ms/flow).")

    # Continuous ranking metrics
    roc_auc = float(roc_auc_score(y_test_bin, test_scores))
    pr_auc = float(average_precision_score(y_test_bin, test_scores))

    # Binary decisions at threshold (using primary 90th percentile threshold)
    pred_anomalous_p90 = (test_scores >= thresh_p90).astype(int)
    pred_anomalous_p95 = (test_scores >= thresh_p95).astype(int)

    # Calculate Detection Rate (Recall) per attack category
    per_category_stats = {}
    for cat in categories:
        mask = (y_test_cats == cat)
        count = int(mask.sum())
        if count == 0:
            continue
        det_count_p90 = int((pred_anomalous_p90[mask] == 1).sum())
        det_rate_p90 = round(float(det_count_p90 / count), 4)

        det_count_p95 = int((pred_anomalous_p95[mask] == 1).sum())
        det_rate_p95 = round(float(det_count_p95 / count), 4)

        mean_score = round(float(test_scores[mask].mean()), 5)
        median_score = round(float(np.median(test_scores[mask])), 5)

        per_category_stats[cat] = {
            "total_samples": count,
            "mean_anomaly_score": mean_score,
            "median_anomaly_score": median_score,
            "detected_p90": det_count_p90,
            "detection_rate_p90": det_rate_p90,
            "detected_p95": det_count_p95,
            "detection_rate_p95": det_rate_p95,
        }

    fpr_p90 = per_category_stats["Normal"]["detection_rate_p90"]
    fpr_p95 = per_category_stats["Normal"]["detection_rate_p95"]

    print("\n--- ANOMALY DETECTION EVALUATION ---")
    print(f"ROC-AUC (Anomaly Score): {roc_auc:.4f}")
    print(f"PR-AUC (Anomaly Score):  {pr_auc:.4f}")
    print(f"Empirical FPR on Normal Testing: {fpr_p90*100:.2f}% (p90) | {fpr_p95*100:.2f}% (p95)")
    print("\nPer-Category Anomaly Detection Rates (p90 Threshold):")
    for cat, stats in per_category_stats.items():
        role = "Benign (FPR)" if cat == "Normal" else "Attack Detection"
        print(f"  - {cat:<15}: Rate={stats['detection_rate_p90']*100:.2f}%, Mean Score={stats['mean_anomaly_score']:.4f} ({role})")

    # --- FIGURE 1: Anomaly Score Distribution Plot ---
    plt.figure(figsize=(10, 6))
    normal_scores = test_scores[y_test_bin == 0]
    attack_scores = test_scores[y_test_bin == 1]
    sns.kdeplot(normal_scores, label="Normal Traffic", color="#2b5c8f", fill=True, alpha=0.4, linewidth=2)
    sns.kdeplot(attack_scores, label="Attack Traffic (All Categories)", color="#d95f02", fill=True, alpha=0.4, linewidth=2)
    plt.axvline(thresh_p90, color="red", linestyle="--", linewidth=1.5, label=f"p90 Threshold ({thresh_p90:.3f})")
    plt.axvline(thresh_p95, color="darkred", linestyle=":", linewidth=1.5, label=f"p95 Threshold ({thresh_p95:.3f})")
    plt.title(f"Anomaly Score Distribution (Normal vs Attack Traffic)\nROC-AUC: {roc_auc:.4f} | PR-AUC: {pr_auc:.4f}", fontsize=12, pad=10)
    plt.xlabel("Anomaly Score (- score_samples; Higher = More Anomalous)", fontsize=11, weight="bold")
    plt.ylabel("Density", fontsize=11, weight="bold")
    plt.legend(loc="upper right", frameon=True)
    plt.tight_layout()
    dist_fig_path = figures_dir / "anomaly_score_distribution.png"
    plt.savefig(dist_fig_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved anomaly score distribution to {dist_fig_path}")

    # --- FIGURE 2: ROC and PR Curves ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fpr_arr, tpr_arr, _ = roc_curve(y_test_bin, test_scores)
    prec_arr, rec_arr, _ = precision_recall_curve(y_test_bin, test_scores)

    axes[0].plot(fpr_arr, tpr_arr, color="#1f77b4", lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="grey", linestyle="--")
    axes[0].set_title("ROC Curve (Anomaly Detector)", fontsize=12, weight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].legend(loc="lower right")

    axes[1].plot(rec_arr, prec_arr, color="#d62728", lw=2, label=f"PR (AUC = {pr_auc:.4f})")
    axes[1].set_title("Precision-Recall Curve (Anomaly Detector)", fontsize=12, weight="bold")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    curves_fig_path = figures_dir / "anomaly_roc_pr_curves.png"
    plt.savefig(curves_fig_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved ROC & PR curves to {curves_fig_path}")

    # Save Model & Thresholds
    anomaly_bundle = {
        "model": iso_forest,
        "reference_type": "Benign-Profile (Normal flows only)",
        "threshold_p90": thresh_p90,
        "threshold_p95": thresh_p95,
        "feature_names": feature_cols
    }
    model_path = models_dir / "isolation_forest.joblib"
    joblib.dump(anomaly_bundle, model_path)
    print(f"[+] Saved anomaly model bundle to {model_path}")

    # Save JSON Results
    results_payload = {
        "model": "IsolationForest",
        "reference_training": "Normal Flows (56,000 samples)",
        "thresholds": {
            "p90": thresh_p90,
            "p95": thresh_p95
        },
        "metrics": {
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "fpr_p90": fpr_p90,
            "fpr_p95": fpr_p95,
            "training_time_sec": train_time_sec,
            "inference_time_sec": infer_time_sec,
            "ms_per_flow": ms_per_flow
        },
        "per_category_performance": per_category_stats
    }

    results_json_path = results_dir / "anomaly_results.json"
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(results_payload, f, indent=2)
    print(f"[+] Saved anomaly results JSON to {results_json_path}")

    # Save Markdown Report
    report_md = f"""# Phase 7: Anomaly Detection Baseline Report

**Model:** Isolation Forest (100 estimators, max_samples=256, benign-profile reference)  
**Reference Dataset:** 56,000 Normal flows (strictly benign training baseline)  
**Evaluation Set:** 82,332 testing flows (37,000 Normal + 45,332 Attacks)  

## 1. Threshold Calibration Methodology
The decision threshold is empirically calibrated on normal training flows to control the operational false positive rate:
- **Primary Threshold (90th Percentile):** `{thresh_p90:.5f}` (Target FPR: ~10%)
- **Conservative Threshold (95th Percentile):** `{thresh_p95:.5f}` (Target FPR: ~5%)

## 2. Global Ranking Performance
| Metric | Value | Interpretation |
| :--- | :--- | :--- |
| **ROC-AUC** | {roc_auc:.4f} | Ability to rank malicious flows above benign flows without labels |
| **PR-AUC** | {pr_auc:.4f} | Area under Precision-Recall curve |
| **Empirical FPR (p90)** | {fpr_p90*100:.2f}% | Proportion of normal test flows flagged as anomalous |
| **Empirical FPR (p95)** | {fpr_p95*100:.2f}% | Conservative false alarm rate on normal test flows |
| **Training Time** | {train_time_sec}s | Latency to fit 56,000 reference flows |
| **Inference Latency** | {ms_per_flow} ms/flow | Latency per flow evaluation |

## 3. Detection Rates per Attack Category (Zero Label Information Used)
| Category | Support | Mean Score | Detection Rate (p90) | Detection Rate (p95) | Category Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for cat, s in per_category_stats.items():
        role = "Normal (FPR)" if cat == "Normal" else "Attack"
        report_md += f"| **{cat}** | {s['total_samples']:,} | {s['mean_anomaly_score']:.4f} | **{s['detection_rate_p90']*100:.2f}%** | {s['detection_rate_p95']*100:.2f}% | {role} |\n"

    report_md += """
## 4. Key Takeaways for Controlled Unseen Attack Experiments (Phase 8)
1. The anomaly detector requires **zero attack labels** during training.
2. Attack categories with high structural deviations from benign profiles (e.g. DoS, Exploits, Generic) achieve distinct anomaly scores.
3. This sets the foundation for Phase 8: testing whether holding out an attack category leaves the supervised model blind while the anomaly detector flags it.
"""
    report_md_path = reports_dir / "anomaly_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved anomaly report to {report_md_path}")
    print("[✓] Phase 7 Anomaly Detection Baseline completed.")

if __name__ == "__main__":
    train_and_evaluate_anomaly()
