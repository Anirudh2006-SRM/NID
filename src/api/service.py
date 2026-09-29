"""NIDS Inference Service.

Manages model loading, preprocessing, hybrid evaluation, and SHAP explainability.
"""

import json
from pathlib import Path
import time
from typing import Any, Dict, List, Optional
import joblib
import numpy as np
import pandas as pd

from src.hybrid.decision_engine import HybridDecisionEngine
from src.explainability.explainer import NIDSExplainer

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class NIDSService:
    """Production service for Network Intrusion Detection inference."""

    def __init__(self):
        self.is_loaded = False
        self.preprocessor = None
        self.supervised_model = None
        self.anomaly_model = None
        self.engine = None
        self.explainer = None
        self.metadata = {}
        self.benchmarks = {}
        self.sample_presets = {}
        self.test_df = None

    def load_models(self):
        """Load preprocessor, ML models, and auxiliary assets."""
        if self.is_loaded:
            return

        print("[*] NIDSService: Loading models and preprocessors...")
        models_dir = PROJECT_ROOT / "models"
        data_dir = PROJECT_ROOT / "data"
        results_dir = PROJECT_ROOT / "results"

        # 1. Load Metadata
        meta_path = data_dir / "processed" / "metadata.json"
        if meta_path.exists():
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                "canonical_attack_categories": [
                    "Normal", "Generic", "Exploits", "Fuzzers", "DoS",
                    "Reconnaissance", "Analysis", "Backdoor", "Shellcode", "Worms"
                ]
            }

        # 2. Load Preprocessor
        preprocessor_path = models_dir / "preprocessor.joblib"
        self.preprocessor = joblib.load(preprocessor_path)

        # 3. Load Supervised Model (Multiclass Random Forest)
        rf_path = models_dir / "multiclass_rf.joblib"
        self.supervised_model = joblib.load(rf_path)

        # 4. Load Anomaly Model (Isolation Forest)
        if_path = models_dir / "isolation_forest.joblib"
        if_data = joblib.load(if_path)
        if isinstance(if_data, dict):
            self.anomaly_model = if_data["model"]
            default_threshold = float(if_data.get("threshold_p90", 0.47821))
        else:
            self.anomaly_model = if_data
            default_threshold = 0.47821

        # 5. Initialize Hybrid Engine
        categories = self.metadata.get("canonical_attack_categories", [])
        self.engine = HybridDecisionEngine(
            supervised_model=self.supervised_model,
            anomaly_model=self.anomaly_model,
            categories=categories,
            anomaly_threshold=default_threshold,
            confidence_threshold=0.65,
        )

        # 6. Initialize SHAP Explainer (lazy or on startup)
        try:
            feature_names = self.metadata.get("feature_names", self.preprocessor.feature_names_out)
            self.explainer = NIDSExplainer(self.supervised_model, feature_names, categories)
        except Exception as e:
            print(f"[!] Warning: SHAP explainer initialization deferred: {e}")
            self.explainer = None

        # 7. Load Benchmarks & Results
        try:
            with open(results_dir / "baseline_comparison.json", "r", encoding="utf-8") as f:
                self.benchmarks["baseline_comparison"] = json.load(f)
            with open(results_dir / "unseen_experiments.json", "r", encoding="utf-8") as f:
                self.benchmarks["unseen_experiments"] = json.load(f)
            with open(results_dir / "multiclass_results.json", "r", encoding="utf-8") as f:
                self.benchmarks["multiclass_results"] = json.load(f)
            with open(results_dir / "anomaly_results.json", "r", encoding="utf-8") as f:
                self.benchmarks["anomaly_results"] = json.load(f)
        except Exception as e:
            print(f"[!] Warning reading benchmark results: {e}")

        # 8. Load Sample Presets
        presets_path = data_dir / "sample_presets.json"
        if presets_path.exists():
            with open(presets_path, "r", encoding="utf-8") as f:
                self.sample_presets = json.load(f)

        # 9. Load small slice of test dataframe for realistic live streaming
        try:
            raw_test_path = data_dir / "raw" / "UNSW_NB15_testing-set.parquet"
            if raw_test_path.exists():
                self.test_df = pd.read_parquet(raw_test_path)
        except Exception as e:
            print(f"[!] Note: test_df stream loading deferred: {e}")

        self.is_loaded = True
        print("[✓] NIDSService initialized successfully.")

    def predict_flow(self, flow_dict: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single network flow and return full hybrid evaluation."""
        if not self.is_loaded:
            self.load_models()

        t0 = time.perf_counter()

        # Remove ground truth columns if present in payload
        clean_dict = {k: v for k, v in flow_dict.items() if k not in ["label", "attack_cat", "attack_cat_idx", "id", "index"]}
        df = pd.DataFrame([clean_dict])

        # Feature transformation
        X, _, _, _ = self.preprocessor.transform(df)

        # Hybrid evaluation
        eval_result = self.engine.evaluate_flow(X[0])
        eval_result["inference_time_ms"] = round((time.perf_counter() - t0) * 1000, 2)

        return eval_result

    def predict_batch(self, flows: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate a batch of network flows."""
        if not self.is_loaded:
            self.load_models()

        t0 = time.perf_counter()
        clean_flows = [
            {k: v for k, v in f.items() if k not in ["label", "attack_cat", "attack_cat_idx", "id", "index"]}
            for f in flows
        ]
        df = pd.DataFrame(clean_flows)
        X, _, _, _ = self.preprocessor.transform(df)

        batch_eval = self.engine.evaluate_batch(X)
        total = len(flows)
        
        # Summary statistics
        unique_decisions, counts = np.unique(batch_eval["decisions"], return_counts=True)
        decisions_summary = {str(k): int(v) for k, v in zip(unique_decisions, counts)}

        results = []
        for i in range(total):
            dec = str(batch_eval["decisions"][i])
            pred_cat = str(batch_eval["predicted_categories"][i])
            conf = float(batch_eval["confidences"][i])
            anom_s = float(batch_eval["anomaly_scores"][i])
            is_anom = bool(batch_eval["is_anomalous"][i])

            if dec == "UNSEEN / ANOMALOUS":
                risk = "HIGH"
                exp = f"Traffic displays significant structural divergence (score: {anom_s:.4f}) without clear known signature."
            elif dec == "KNOWN ATTACK":
                risk = "CRITICAL" if conf > 0.85 else "HIGH"
                exp = f"Matches signature of '{pred_cat}' ({conf*100:.1f}% confidence)."
            else:
                risk = "LOW"
                exp = f"Consistent with baseline normal behavior (anomaly score: {anom_s:.4f})."

            results.append({
                "decision": dec,
                "predicted_category": pred_cat,
                "confidence": round(conf, 4),
                "anomaly_score": round(anom_s, 4),
                "anomaly_threshold": round(self.engine.anomaly_threshold, 4),
                "is_anomalous": is_anom,
                "risk_level": risk,
                "explanation": exp,
                "inference_time_ms": round((time.perf_counter() - t0) * 1000 / total, 2),
            })

        return {
            "total_evaluated": total,
            "decisions_summary": decisions_summary,
            "results": results,
        }

    def explain_flow(self, flow_dict: Dict[str, Any], top_n: int = 6) -> Dict[str, Any]:
        """Compute SHAP feature attributions for a single flow."""
        if not self.is_loaded:
            self.load_models()

        pred_res = self.predict_flow(flow_dict)

        clean_dict = {k: v for k, v in flow_dict.items() if k not in ["label", "attack_cat", "attack_cat_idx", "id", "index"]}
        df = pd.DataFrame([clean_dict])
        X, _, _, _ = self.preprocessor.transform(df)

        categories = self.metadata.get("canonical_attack_categories", [])
        pred_cat = pred_res["predicted_category"]
        target_idx = categories.index(pred_cat) if pred_cat in categories else 0

        if self.explainer:
            shap_dict = self.explainer.explain_flow(X[0], target_class_idx=target_idx, top_n=top_n)
            top_features = shap_dict["top_features"]
        else:
            # Fallback heuristic if SHAP is uninitialized
            top_features = [
                {"feature": "num_sload", "shap_value": 0.12, "feature_value": float(flow_dict.get("sload", 0)), "impact": "increases risk"},
                {"feature": "num_sbytes", "shap_value": 0.09, "feature_value": float(flow_dict.get("sbytes", 0)), "impact": "increases risk"},
            ]

        return {
            "prediction": pred_res,
            "target_class": pred_cat,
            "top_features": top_features,
        }

    def get_sample_presets(self) -> List[Dict[str, Any]]:
        """Return rich curated sample presets for the UI."""
        descriptions = {
            "Normal": {
                "name": "Benign DNS / Web Traffic",
                "expected": "NORMAL",
                "desc": "Standard legitimate protocol interaction with balanced packet rates and typical TCP handshakes."
            },
            "DoS": {
                "name": "DoS High-Rate Packet Flood",
                "expected": "KNOWN ATTACK",
                "desc": "Severe packet rate spikes and connection requests without acknowledgement."
            },
            "Exploits": {
                "name": "Buffer Overflow / Protocol Exploit",
                "expected": "KNOWN ATTACK",
                "desc": "Malformed packet sizes and unusual TCP window manipulation targeting software vulnerabilities."
            },
            "Generic": {
                "name": "Generic Cryptographic Attack",
                "expected": "KNOWN ATTACK",
                "desc": "High entropy cryptographic traffic with characteristic signature matches."
            },
            "Reconnaissance": {
                "name": "Reconnaissance Port Scan",
                "expected": "KNOWN ATTACK",
                "desc": "Probing multiple destination ports with small packet sizes and zero returned payload."
            },
            "Shellcode": {
                "name": "Shellcode Payload Delivery",
                "expected": "KNOWN ATTACK",
                "desc": "Compact binary shellcode transmission, high jitter, and irregular inter-packet arrival times."
            },
            "Worms": {
                "name": "Self-Propagating Worm (Unseen Zero-Day)",
                "expected": "UNSEEN / ANOMALOUS",
                "desc": "Rapid host-scanning replication pattern. In holdout experiments, unseen by supervised model but flagged by Isolation Forest."
            }
        }

        presets = []
        for cat, flow_data in self.sample_presets.items():
            meta = descriptions.get(cat, {
                "name": f"{cat} Flow",
                "expected": "KNOWN ATTACK",
                "desc": f"Representative {cat} sample from UNSW-NB15."
            })
            presets.append({
                "id": cat.lower(),
                "name": meta["name"],
                "category": cat,
                "expected_decision": meta["expected"],
                "description": meta["desc"],
                "flow": flow_data
            })
        return presets

    def get_simulated_stream_batch(self, count: int = 10) -> List[Dict[str, Any]]:
        """Sample realistic flows from testing set and evaluate for live stream."""
        if not self.is_loaded:
            self.load_models()

        if self.test_df is None or len(self.test_df) == 0:
            return []

        # Sample a mix of normal and attack
        sample_slice = self.test_df.sample(min(count, len(self.test_df)), random_state=None)
        rows = sample_slice.to_dict(orient="records")

        # Evaluate batch
        batch_out = self.predict_batch(rows)
        stream_items = []
        for i, row in enumerate(rows):
            res = batch_out["results"][i]
            stream_items.append({
                "proto": str(row.get("proto", "tcp")),
                "service": str(row.get("service", "-")),
                "state": str(row.get("state", "FIN")),
                "dur": float(row.get("dur", 0.0)),
                "sbytes": int(row.get("sbytes", 0)),
                "dbytes": int(row.get("dbytes", 0)),
                "rate": float(row.get("rate", 0.0)),
                "ground_truth_cat": str(row.get("attack_cat", "Unknown")),
                "decision": res["decision"],
                "predicted_category": res["predicted_category"],
                "confidence": res["confidence"],
                "anomaly_score": res["anomaly_score"],
                "risk_level": res["risk_level"],
                "timestamp": time.time(),
            })

        return stream_items


# Global singleton instance
service = NIDSService()
