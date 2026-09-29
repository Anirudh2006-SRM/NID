"""Unit and Integration Tests for Sentinel-Hybrid NIDS using standard library unittest."""

import json
from pathlib import Path
import unittest
from fastapi.testclient import TestClient

from src.api.main import app
from src.api.service import service

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestNIDSSystem(unittest.TestCase):
    """Test suite covering NIDS FastAPI endpoints, models, and hybrid engine."""

    @classmethod
    def setUpClass(cls):
        """Initialize TestClient and load models once."""
        cls.client = TestClient(app)
        service.load_models()

    def test_01_health_endpoint(self):
        """Test /health system status."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["models_loaded"])
        self.assertTrue(data["explainer_ready"])
        self.assertIn("anomaly_threshold", data)

    def test_02_categories_endpoint(self):
        """Test /api/categories returns canonical 10 classes."""
        response = self.client.get("/api/categories")
        self.assertEqual(response.status_code, 200)
        cats = response.json()["categories"]
        self.assertEqual(len(cats), 10)
        self.assertIn("Normal", cats)
        self.assertIn("Exploits", cats)
        self.assertIn("Worms", cats)

    def test_03_sample_flows_preset(self):
        """Test /api/sample-flows returns valid presets."""
        response = self.client.get("/api/sample-flows")
        self.assertEqual(response.status_code, 200)
        presets = response.json()
        self.assertGreaterEqual(len(presets), 5)
        first = presets[0]
        self.assertIn("id", first)
        self.assertIn("flow", first)
        self.assertIn("expected_decision", first)

    def test_04_single_flow_prediction(self):
        """Test POST /api/predict on a representative benign flow."""
        presets = self.client.get("/api/sample-flows").json()
        normal_preset = next((p for p in presets if p["category"] == "Normal"), presets[0])
        
        response = self.client.post("/api/predict", json=normal_preset["flow"])
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertIn("decision", result)
        self.assertIn(result["decision"], ["NORMAL", "KNOWN ATTACK", "UNSEEN / ANOMALOUS"])
        self.assertTrue(0.0 <= result["confidence"] <= 1.0)
        self.assertIn("anomaly_score", result)
        self.assertIn("risk_level", result)
        self.assertGreater(result["inference_time_ms"], 0)

    def test_05_batch_prediction(self):
        """Test POST /api/predict/batch with multiple flows."""
        presets = self.client.get("/api/sample-flows").json()
        flows = [p["flow"] for p in presets[:3]]
        
        response = self.client.post("/api/predict/batch", json={"flows": flows})
        self.assertEqual(response.status_code, 200)
        result = response.json()
        self.assertEqual(result["total_evaluated"], len(flows))
        self.assertEqual(len(result["results"]), len(flows))
        self.assertIn("decisions_summary", result)

    def test_06_explainability_shap(self):
        """Test POST /api/explain returns top SHAP feature attributions."""
        presets = self.client.get("/api/sample-flows").json()
        sample_flow = presets[0]["flow"]
        
        response = self.client.post("/api/explain?top_n=5", json=sample_flow)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("prediction", data)
        self.assertIn("top_features", data)
        self.assertLessEqual(len(data["top_features"]), 5)
        for feat in data["top_features"]:
            self.assertIn("feature", feat)
            self.assertIn("shap_value", feat)
            self.assertIn("impact", feat)

    def test_07_stream_simulation(self):
        """Test GET /api/stream returns live packet feed."""
        response = self.client.get("/api/stream?count=5")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("packets", data)
        self.assertLessEqual(len(data["packets"]), 5)
        if len(data["packets"]) > 0:
            pkt = data["packets"][0]
            self.assertIn("decision", pkt)
            self.assertIn("anomaly_score", pkt)
            self.assertIn("risk_level", pkt)

    def test_08_threshold_update(self):
        """Test POST /api/thresholds dynamically updates boundaries."""
        orig_thresh = service.engine.anomaly_threshold
        try:
            response = self.client.post("/api/thresholds", json={"anomaly_threshold": 0.52, "confidence_threshold": 0.70})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["anomaly_threshold"], 0.52)
            self.assertEqual(response.json()["confidence_threshold"], 0.70)
            self.assertEqual(service.engine.anomaly_threshold, 0.52)
        finally:
            service.engine.anomaly_threshold = orig_thresh
            service.engine.confidence_threshold = 0.65


if __name__ == "__main__":
    unittest.main()
