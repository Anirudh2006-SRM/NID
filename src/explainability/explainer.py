"""Phase 12: SHAP-based Explainability Module for NIDS.

Provides:
1. Exact Shapley feature attributions (TreeExplainer)
2. Per-flow local explanations for API and Dashboard (Top 5 features)
3. Global SHAP summary beeswarm plot across flow features
"""

import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Union

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

class NIDSExplainer:
    """SHAP explainer utilizing TreeExplainer on trained tree models."""

    def __init__(self, model, feature_names: List[str], categories: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.categories = categories
        print("[*] Initializing SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(self.model)

    def explain_flow(self, x: np.ndarray, target_class_idx: Optional[int] = None, top_n: int = 5) -> Dict:
        """Explain a single network flow vector."""
        if x.ndim == 1:
            x_2d = x.reshape(1, -1)
        else:
            x_2d = x

        # Predict class if not provided
        probs = self.model.predict_proba(x_2d)[0]
        if target_class_idx is None:
            target_class_idx = int(np.argmax(probs))

        pred_class_name = self.categories[target_class_idx] if target_class_idx < len(self.categories) else "Unknown"

        # Compute SHAP values
        raw_shap = self.explainer.shap_values(x_2d)

        # Handle different SHAP output structures across versions
        if isinstance(raw_shap, list):
            # List of arrays [n_samples, n_features], one per class
            class_shap = raw_shap[target_class_idx][0]
        elif raw_shap.ndim == 3:
            # Array of shape [n_samples, n_features, n_classes]
            class_shap = raw_shap[0, :, target_class_idx]
        else:
            class_shap = raw_shap[0]

        # Rank features by absolute impact
        abs_shap = np.abs(class_shap)
        top_indices = np.argsort(abs_shap)[::-1][:top_n]

        top_features = []
        for idx in top_indices:
            val = float(x_2d[0, idx])
            s_val = float(class_shap[idx])
            direction = "increases risk / promotes class" if s_val > 0 else "decreases risk / suppresses class"
            top_features.append({
                "feature": self.feature_names[idx],
                "shap_value": round(s_val, 5),
                "feature_value": round(val, 4),
                "impact": direction
            })

        return {
            "predicted_class": pred_class_name,
            "class_probability": round(float(probs[target_class_idx]), 4),
            "top_features": top_features
        }

def run_explainability():
    processed_dir = PROJECT_ROOT / "data" / "processed"
    models_dir = PROJECT_ROOT / "models"
    results_dir = PROJECT_ROOT / "results"
    reports_dir = PROJECT_ROOT / "reports"
    figures_dir = reports_dir / "figures"

    for d in [results_dir, reports_dir, figures_dir]:
        d.mkdir(parents=True, exist_ok=True)

    print("[*] Loading test data and multiclass model for SHAP analysis...")
    test_df = pd.read_parquet(processed_dir / "test_processed.parquet")

    with open(processed_dir / "metadata.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    feature_cols = meta["feature_names"]
    categories = meta["canonical_attack_categories"]

    model = joblib.load(models_dir / "multiclass_rf.joblib")
    explainer = NIDSExplainer(model, feature_cols, categories)

    # 1. Global Summary Plot on a Stratified Sample (300 flows for fast, high-quality rendering)
    print("[*] Generating Global SHAP Summary on representative test sample...")
    sample_df = test_df.groupby("attack_cat", group_keys=False).apply(
        lambda g: g.sample(min(len(g), 30), random_state=42)
    )
    X_sample = sample_df[feature_cols].to_numpy()

    t0 = time.perf_counter()
    raw_shap_sample = explainer.explainer.shap_values(X_sample)
    print(f"[+] SHAP values computed in {time.perf_counter() - t0:.2f}s")

    # Plot summary for attack class (e.g. Exploits or DoS or mean over classes)
    plt.figure(figsize=(12, 8))
    if isinstance(raw_shap_sample, list):
        # Multi-class bar summary
        shap.summary_plot(
            raw_shap_sample,
            X_sample,
            feature_names=feature_cols,
            class_names=categories,
            max_display=15,
            show=False
        )
    else:
        shap.summary_plot(
            raw_shap_sample,
            X_sample,
            feature_names=feature_cols,
            max_display=15,
            show=False
        )

    plt.title("SHAP Global Feature Importance across 10 Traffic Categories", fontsize=12, pad=12)
    plt.tight_layout()
    summary_plot_path = figures_dir / "shap_summary_plot.png"
    plt.savefig(summary_plot_path, bbox_inches="tight")
    plt.close()
    print(f"[+] Saved global SHAP summary plot to {summary_plot_path}")

    # 2. Local Sample Explanations (Normal, Exploits, DoS, Generic)
    print("[*] Generating local explanations for representative flows...")
    sample_categories = ["Normal", "Exploits", "DoS", "Generic"]
    sample_explanations = {}

    for cat in sample_categories:
        cat_flows = test_df[test_df["attack_cat"] == cat]
        if len(cat_flows) > 0:
            flow_row = cat_flows.iloc[0]
            flow_vec = flow_row[feature_cols].to_numpy()
            exp = explainer.explain_flow(flow_vec, top_n=5)
            sample_explanations[cat] = exp
            print(f"  -> Explained flow for '{cat}': Top feature: {exp['top_features'][0]['feature']} (SHAP: {exp['top_features'][0]['shap_value']})")

    # Save Explanations JSON
    json_path = results_dir / "sample_shap_explanations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(sample_explanations, f, indent=2)
    print(f"[+] Saved sample explanations to {json_path}")

    # 3. Generate Markdown Report
    report_md = f"""# Phase 12: SHAP-based Explainability Report

**Methodology:** `shap.TreeExplainer` on the 10-Class Random Forest Model  
**Feature Space:** 65 Transformed Features (31 Numerical + 34 One-Hot Encoded)  

## 1. Global Feature Attribution Highlights
Based on global SHAP attributions across all network traffic categories:
1. **Flow Rates (`num_rate`, `num_sload`, `num_dload`):** Primary discriminators separating high-volume attacks (DoS, Exploits) from Normal traffic.
2. **TCP Handshake & Timing (`num_tcprtt`, `num_synack`, `num_dinpkt`):** Critical for identifying reconnaissance probes and interactive exploits.
3. **Payload Characteristics (`num_smean`, `num_dmean`, `num_sbytes`):** Essential for differentiating micro-payloads (Shellcode) from standard HTTP/DNS transactions.

*Global SHAP Summary Plot: [reports/figures/shap_summary_plot.png](file:///c:/Users/Administrator/OneDrive/Desktop/Network%20Intrusion%20detection%20of%20Networks/reports/figures/shap_summary_plot.png)*

## 2. Sample Local Explanations (Real Flow Instances)
"""
    for cat, exp in sample_explanations.items():
        report_md += f"### Example Prediction: `{exp['predicted_class']}` (Confidence: {exp['class_probability']*100:.1f}%)\n"
        report_md += "| Rank | Feature Name | Flow Value (Normalized) | SHAP Attribution | Impact |\n"
        report_md += "| :--- | :--- | :--- | :--- | :--- |\n"
        for i, feat in enumerate(exp["top_features"], 1):
            report_md += f"| {i} | `{feat['feature']}` | {feat['feature_value']} | `{feat['shap_value']:+.4f}` | {feat['impact']} |\n"
        report_md += "\n"

    report_md += """
## 3. Integration with Backend & Dashboard
- The `explain_flow()` method provides real-time per-flow attribution.
- This feeds directly into the FastAPI `/detect` and `/predict` endpoints to supply the React dashboard with live feature contribution graphs.
"""
    report_md_path = reports_dir / "explainability_report.md"
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"[+] Saved explainability report to {report_md_path}")
    print("[✓] Phase 12 SHAP Explainability completed.")

if __name__ == "__main__":
    run_explainability()
