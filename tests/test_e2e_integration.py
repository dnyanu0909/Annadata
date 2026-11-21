"""
Annadata AI Diagnostics - End-to-End Integration Test Suite
Verifies:
  1. FastAPI backend /api/predict against original predict.py inference.
  2. Exact class label and confidence score parity.
  3. Error handling: invalid formats, empty files, corrupted payloads, missing fields.
  4. Missing model graceful degradation.
  5. API contract compliance with Repository A's TypeScript interface.
"""

import sys
import json
import unittest
from unittest import mock
from pathlib import Path
import numpy as np

from fastapi.testclient import TestClient
from api.main import app, AnnadataInferenceEngine, get_engine
from predict import AnnadataPredictor

REPO_B_DIR = Path(__file__).resolve().parent.parent
TEST_IMAGE = REPO_B_DIR / "test_leaf.jpg"
ONNX_MODEL = REPO_B_DIR / "models" / "annadata_production.onnx"
CLASS_MAP = REPO_B_DIR / "artifacts" / "disease_classes.json"

class TestAnnadataEndToEndIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.predictor = AnnadataPredictor(onnx_path=ONNX_MODEL, class_path=CLASS_MAP)

    def test_01_health_and_metadata(self):
        """Verifies GET / returns healthy status with ONNX model loaded."""
        res = self.client.get("/")
        self.assertEqual(res.status_code, 200)
        data = res.json()
        self.assertEqual(data["status"], "ok")
        self.assertTrue(data["model_loaded"])
        self.assertEqual(data["num_classes"], 38)
        self.assertEqual(data["input_shape"], [1, 224, 224, 3])

    def test_02_classes_endpoint(self):
        """Verifies GET /api/classes returns all 38 classes with raw_class and name."""
        res = self.client.get("/api/classes")
        self.assertEqual(res.status_code, 200)
        classes = res.json()
        self.assertEqual(len(classes), 38)
        self.assertIn("0", classes)
        self.assertIn("37", classes)
        self.assertIn("raw_class", classes["0"])
        self.assertIn("name", classes["0"])

    def test_03_prediction_parity_with_original_predictor(self):
        """
        Verifies that POST /api/predict generates identical predictions
        to the original standalone predict.py implementation.
        """
        self.assertTrue(TEST_IMAGE.exists(), f"Test image missing at {TEST_IMAGE}")

        # 1. Standalone predict.py output
        cli_preds = self.predictor.predict(str(TEST_IMAGE), top_k=3)

        # 2. FastAPI API output
        with open(TEST_IMAGE, "rb") as f:
            res = self.client.post("/api/predict?top_k=3", files={"image": ("test_leaf.jpg", f, "image/jpeg")})

        self.assertEqual(res.status_code, 200)
        api_data = res.json()

        # Contract checks
        self.assertTrue(api_data["success"])
        self.assertIn("predicted_class", api_data)
        self.assertIn("predicted_label", api_data)
        self.assertIn("confidence", api_data)
        self.assertIn("processing_time_ms", api_data)
        self.assertEqual(len(api_data["top_predictions"]), 3)

        # Compare top-1 prediction
        cli_top = cli_preds[0]
        self.assertEqual(api_data["predicted_label"], cli_top["name"])
        self.assertAlmostEqual(api_data["confidence"], cli_top["confidence_percent"] / 100.0, places=3)

        # Compare top-3 rankings
        for i in range(3):
            self.assertEqual(api_data["top_predictions"][i]["label"], cli_preds[i]["name"])
            self.assertAlmostEqual(
                api_data["top_predictions"][i]["confidence"],
                cli_preds[i]["confidence_percent"] / 100.0,
                places=3
            )

        print(f"\n[E2E Parity Check] CLI vs API Diagnosis: {api_data['predicted_label']} ({api_data['confidence']*100:.2f}%)")

    def test_04_fallback_field_name_file(self):
        """Verifies alternative 'file' form field name works for curl/CLI compatibility."""
        with open(TEST_IMAGE, "rb") as f:
            res = self.client.post("/api/predict", files={"file": ("test_leaf.jpg", f, "image/jpeg")})
        self.assertEqual(res.status_code, 200)
        self.assertTrue(res.json()["success"])

    def test_05_missing_image_payload(self):
        """Verifies 400 Bad Request when no file is sent."""
        res = self.client.post("/api/predict")
        self.assertEqual(res.status_code, 400)
        self.assertIn("No image uploaded", res.json()["detail"])

    def test_06_empty_image_payload(self):
        """Verifies 400 Bad Request when an empty file is sent."""
        res = self.client.post("/api/predict", files={"image": ("empty.jpg", b"", "image/jpeg")})
        self.assertEqual(res.status_code, 400)
        self.assertIn("empty", res.json()["detail"].lower())

    def test_07_unsupported_file_format(self):
        """Verifies 415 Unsupported Media Type for non-image files."""
        res = self.client.post("/api/predict", files={"image": ("notes.txt", b"plain text leaf description", "text/plain")})
        self.assertEqual(res.status_code, 415)
        self.assertIn("Unsupported image format", res.json()["detail"])

    def test_08_corrupted_image_payload(self):
        """Verifies 422 Unprocessable Content when image cannot be decoded."""
        corrupted_bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00CORRUPTED_GARBAGE_PAYLOAD"
        res = self.client.post("/api/predict", files={"image": ("broken.jpg", corrupted_bytes, "image/jpeg")})
        self.assertEqual(res.status_code, 422)
        self.assertIn("could not be decoded", res.json()["detail"])

    def test_09_missing_model_graceful_handling(self):
        """Verifies proper 503 error if model fails to load or is uninitialized."""
        with mock.patch("api.main.get_engine", return_value=None):
            with open(TEST_IMAGE, "rb") as f:
                res = self.client.post("/api/predict", files={"image": ("leaf.jpg", f, "image/jpeg")})
            self.assertEqual(res.status_code, 503)
            self.assertIn("unavailable", res.json()["detail"].lower())

if __name__ == "__main__":
    unittest.main()
