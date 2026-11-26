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

    def test_10_cors_allowed_origin(self):
        """Verifies that requests from allowed dev/configured origins receive CORS headers."""
        headers = {
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST"
        }
        res = self.client.options("/api/predict", headers=headers)
        self.assertEqual(res.status_code, 200)
        self.assertEqual(res.headers.get("access-control-allow-origin"), "http://localhost:3000")
        self.assertEqual(res.headers.get("access-control-allow-credentials"), "true")

    def test_11_cors_disallowed_origin(self):
        """Verifies that requests from unapproved origins do NOT receive CORS access."""
        headers = {
            "Origin": "https://unauthorized-malicious-site.com",
            "Access-Control-Request-Method": "POST"
        }
        res = self.client.options("/api/predict", headers=headers)
        # Starlette CORS middleware does not return Access-Control-Allow-Origin for unauthorized origins
        self.assertIsNone(res.headers.get("access-control-allow-origin"))

    def test_12_ensure_model_available_existing_local_model(self):
        """Verifies ensure_model_available succeeds silently when model is already staged."""
        from api.main import ensure_model_available
        # Should return immediately with zero network requests
        ensure_model_available(ONNX_MODEL)
        self.assertTrue(ONNX_MODEL.exists())

    def test_13_ensure_model_available_missing_no_url(self):
        """Verifies clear FileNotFoundError when model is missing and no download URL is set."""
        from api.main import ensure_model_available
        with mock.patch.dict("os.environ", {"MODEL_DOWNLOAD_URL": ""}, clear=False):
            missing_path = REPO_B_DIR / "models" / "missing_nonexistent_model.onnx"
            with self.assertRaises(FileNotFoundError) as ctx:
                ensure_model_available(missing_path)
            self.assertIn("MODEL_DOWNLOAD_URL", str(ctx.exception))

    def test_14_valid_zip_download_and_extraction(self):
        """
        Verifies downloading a ZIP release asset, safely extracting annadata_production.onnx,
        and validating extracted file size and SHA-256 checksum.
        """
        import io
        import zipfile
        import hashlib
        from api.main import ensure_model_available

        fake_model_bytes = b"MODEL_DATA_FOR_VALID_EXTRACTION_TEST_12345"
        fake_sha256 = hashlib.sha256(fake_model_bytes).hexdigest()
        fake_size = len(fake_model_bytes)

        # Create in-memory ZIP containing annadata_production.onnx
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", fake_model_bytes)
        zip_bytes = zip_buf.getvalue()

        target_path = REPO_B_DIR / "models" / "zip_test_model.onnx"
        if target_path.exists():
            target_path.unlink()

        mock_response = io.BytesIO(zip_bytes)
        mock_response.status = 200

        with mock.patch.dict("os.environ", {
            "MODEL_DOWNLOAD_URL": "https://github.com/dnyanu0909/Annadata/releases/download/v1.0.0/annadata_production.zip",
            "MODEL_EXPECTED_SIZE": str(fake_size),
            "MODEL_SHA256": fake_sha256
        }):
            with mock.patch("urllib.request.urlopen", return_value=mock_response):
                ensure_model_available(target_path)
                self.assertTrue(target_path.exists())
                self.assertEqual(target_path.stat().st_size, fake_size)
                # Verify extracted file checksum matches
                with open(target_path, "rb") as f:
                    self.assertEqual(hashlib.sha256(f.read()).hexdigest(), fake_sha256)
                # Cleanup
                target_path.unlink()

    def test_15_corrupted_zip_archive_rejection(self):
        """Verifies that non-ZIP / corrupted archive downloads are rejected and cleaned up."""
        import io
        from api.main import ensure_model_available

        corrupted_zip_bytes = b"CORRUPTED_NON_ZIP_STREAM_DATA"
        target_path = REPO_B_DIR / "models" / "corrupted_zip_test.onnx"
        temp_zip_path = target_path.with_suffix(".zip.tmp")
        temp_onnx_path = target_path.with_suffix(".onnx.tmp")

        mock_response = io.BytesIO(corrupted_zip_bytes)
        mock_response.status = 200

        with mock.patch.dict("os.environ", {
            "MODEL_DOWNLOAD_URL": "https://example.com/annadata_production.zip",
            "MODEL_EXPECTED_SIZE": "100",
            "MODEL_SHA256": "0000000000000000000000000000000000000000000000000000000000000000"
        }):
            with mock.patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_model_available(target_path)
                self.assertIn("not a valid zip archive", str(ctx.exception).lower())
                self.assertFalse(target_path.exists())
                self.assertFalse(temp_zip_path.exists())
                self.assertFalse(temp_onnx_path.exists())

    def test_16_missing_onnx_file_inside_zip(self):
        """Verifies rejection when ZIP is valid but lacks annadata_production.onnx."""
        import io
        import zipfile
        from api.main import ensure_model_available

        # Create valid ZIP missing annadata_production.onnx
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("some_unrelated_file.txt", b"Hello World")
        zip_bytes = zip_buf.getvalue()

        target_path = REPO_B_DIR / "models" / "missing_file_zip_test.onnx"
        mock_response = io.BytesIO(zip_bytes)
        mock_response.status = 200

        with mock.patch.dict("os.environ", {
            "MODEL_DOWNLOAD_URL": "https://example.com/annadata_production.zip",
        }):
            with mock.patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_model_available(target_path)
                self.assertIn("not found inside the zip archive", str(ctx.exception).lower())
                self.assertFalse(target_path.exists())

    def test_17_incorrect_extracted_model_checksum(self):
        """Verifies that an incorrect checksum on the extracted ONNX model triggers failure."""
        import io
        import zipfile
        from api.main import ensure_model_available

        model_bytes = b"TAMPERED_MODEL_BYTES"
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", model_bytes)
        zip_bytes = zip_buf.getvalue()

        target_path = REPO_B_DIR / "models" / "checksum_fail_test.onnx"
        mock_response = io.BytesIO(zip_bytes)
        mock_response.status = 200

        with mock.patch.dict("os.environ", {
            "MODEL_DOWNLOAD_URL": "https://example.com/annadata_production.zip",
            "MODEL_EXPECTED_SIZE": str(len(model_bytes)),
            "MODEL_SHA256": "ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"  # Mismatch
        }):
            with mock.patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_model_available(target_path)
                self.assertIn("checksum verification failed", str(ctx.exception).lower())
                self.assertFalse(target_path.exists())

    def test_18_path_traversal_zip_attempt_rejection(self):
        """Verifies that path traversal attempts (e.g. ../ or absolute paths) in ZIP are rejected."""
        import io
        import zipfile
        from api.main import ensure_model_available

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            # Traversal member
            zf.writestr("../annadata_production.onnx", b"MALICIOUS_TRAVERSAL_CONTENT")
        zip_bytes = zip_buf.getvalue()

        target_path = REPO_B_DIR / "models" / "traversal_test.onnx"
        mock_response = io.BytesIO(zip_bytes)
        mock_response.status = 200

        with mock.patch.dict("os.environ", {
            "MODEL_DOWNLOAD_URL": "https://example.com/annadata_production.zip",
        }):
            with mock.patch("urllib.request.urlopen", return_value=mock_response):
                with self.assertRaises(RuntimeError) as ctx:
                    ensure_model_available(target_path)
                self.assertIn("path traversal", str(ctx.exception).lower())
                self.assertFalse(target_path.exists())

    def test_19_corrupted_local_model_recovers_via_download(self):
        """Verifies that an existing corrupted local model is detected and safely recovered via download."""
        import io
        import zipfile
        import hashlib
        from api.main import ensure_model_available

        recovered_bytes = b"RECOVERED_VALID_MODEL_BYTES_AFTER_CORRUPTION"
        recovered_sha256 = hashlib.sha256(recovered_bytes).hexdigest()
        recovered_size = len(recovered_bytes)

        target_path = REPO_B_DIR / "models" / "corrupted_recovery_test.onnx"
        target_path.write_bytes(b"CORRUPTED_INITIAL_LOCAL_BYTES")

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", recovered_bytes)
        mock_response = io.BytesIO(zip_buf.getvalue())
        mock_response.status = 200

        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/recovery.zip",
                "MODEL_EXPECTED_SIZE": str(recovered_size),
                "MODEL_SHA256": recovered_sha256
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    ensure_model_available(target_path)
                    self.assertTrue(target_path.exists())
                    self.assertEqual(target_path.read_bytes(), recovered_bytes)
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_20_duplicate_onnx_entries_rejected(self):
        """Verifies rejection when ZIP contains ambiguous/duplicate model entries."""
        import io
        import zipfile
        from api.main import ensure_model_available

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", b"FIRST_ENTRY")
            zf.writestr("subfolder/annadata_production.onnx", b"SECOND_ENTRY")
        mock_response = io.BytesIO(zip_buf.getvalue())
        mock_response.status = 200

        target_path = REPO_B_DIR / "models" / "duplicate_test.onnx"
        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/duplicate.zip",
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    self.assertIn("multiple files named 'annadata_production.onnx'", str(ctx.exception))
                    self.assertFalse(target_path.exists())
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_21_zip_download_exceeding_size_limit_aborts(self):
        """Verifies that ZIP download exceeding MAX_ZIP_DOWNLOAD_BYTES (40 MB) aborts and cleans up."""
        from api.main import ensure_model_available, MAX_ZIP_DOWNLOAD_BYTES

        class StreamExceedingLimit:
            def __init__(self, total_mb=42):
                self.chunk = b"X" * (1024 * 1024)
                self.chunks_left = total_mb
                self.status = 200
            def read(self, n=65536):
                if self.chunks_left > 0:
                    self.chunks_left -= 1
                    return self.chunk
                return b""
            def __enter__(self):
                return self
            def __exit__(self, *args):
                pass

        target_path = REPO_B_DIR / "models" / "oversized_download_test.onnx"
        temp_zip = target_path.with_suffix(".zip.tmp")
        temp_onnx = target_path.with_suffix(".onnx.tmp")

        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/huge.zip",
            }):
                with mock.patch("urllib.request.urlopen", return_value=StreamExceedingLimit(42)):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    self.assertIn("exceeds maximum allowed limit", str(ctx.exception).lower())
                    self.assertFalse(target_path.exists())
                    self.assertFalse(temp_zip.exists())
                    self.assertFalse(temp_onnx.exists())
        finally:
            for p in (target_path, temp_zip, temp_onnx):
                if p.exists():
                    p.unlink()

    def test_22_oversized_uncompressed_model_rejected(self):
        """Verifies rejection when declared uncompressed size exceeds MAX_UNCOMPRESSED_MODEL_BYTES (30 MB)."""
        import io
        import struct
        import zipfile
        from api.main import ensure_model_available, MAX_UNCOMPRESSED_MODEL_BYTES

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", b"SMALL_COMPRESSED_PAYLOAD")
        
        # Patch the central directory header's uncompressed_size (offset 24 after PK\x01\x02) to 35 MB
        raw_zip = bytearray(zip_buf.getvalue())
        cd_idx = raw_zip.find(b"PK\x01\x02")
        self.assertNotEqual(cd_idx, -1, "Central directory header not found")
        struct.pack_into("<I", raw_zip, cd_idx + 24, MAX_UNCOMPRESSED_MODEL_BYTES + (5 * 1024 * 1024))

        mock_response = io.BytesIO(bytes(raw_zip))
        mock_response.status = 200

        target_path = REPO_B_DIR / "models" / "oversized_uncompressed_test.onnx"
        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/oversized.zip",
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    self.assertIn("exceeds maximum allowed limit", str(ctx.exception).lower())
                    self.assertFalse(target_path.exists())
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_23_symbolic_link_rejected(self):
        """Verifies that ZIP archives containing symbolic links are strictly rejected."""
        import io
        import zipfile
        from api.main import ensure_model_available

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            symlink_info = zipfile.ZipInfo("symlink_pointer")
            symlink_info.external_attr = 0o120777 << 16  # S_IFLNK
            zf.writestr(symlink_info, b"target_path")
            zf.writestr("annadata_production.onnx", b"MODEL_DATA")
        mock_response = io.BytesIO(zip_buf.getvalue())
        mock_response.status = 200

        target_path = REPO_B_DIR / "models" / "symlink_test.onnx"
        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/symlink.zip",
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    self.assertIn("symbolic link detected", str(ctx.exception).lower())
                    self.assertFalse(target_path.exists())
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_24_empty_env_vars_fallback_to_secure_defaults(self):
        """Verifies that empty or whitespace MODEL_SHA256 and MODEL_EXPECTED_SIZE fall back to defaults."""
        import io
        import zipfile
        from api.main import ensure_model_available, DEFAULT_MODEL_SIZE

        # Create ZIP with small payload (does not match DEFAULT_MODEL_SIZE 18748076)
        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("annadata_production.onnx", b"TINY_PAYLOAD_NOT_18MB")
        mock_response = io.BytesIO(zip_buf.getvalue())
        mock_response.status = 200

        target_path = REPO_B_DIR / "models" / "empty_env_test.onnx"
        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/empty_env.zip",
                "MODEL_EXPECTED_SIZE": "   ",
                "MODEL_SHA256": ""
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    # Must fail on DEFAULT_MODEL_SIZE check, proving it didn't bypass verification
                    self.assertIn(str(DEFAULT_MODEL_SIZE), str(ctx.exception))
                    self.assertFalse(target_path.exists())
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_25_valid_local_model_bypasses_network_download(self):
        """Verifies that a valid local model file matching expected size and checksum completely bypasses network calls."""
        import hashlib
        from api.main import ensure_model_available

        valid_bytes = b"EXACT_VALID_LOCAL_MODEL_DATA_FOR_BYPASS_TEST"
        valid_size = len(valid_bytes)
        valid_sha256 = hashlib.sha256(valid_bytes).hexdigest()

        target_path = REPO_B_DIR / "models" / "valid_local_test.onnx"
        target_path.write_bytes(valid_bytes)

        try:
            with mock.patch.dict("os.environ", {
                "MODEL_EXPECTED_SIZE": str(valid_size),
                "MODEL_SHA256": valid_sha256,
                "MODEL_DOWNLOAD_URL": ""  # No download URL; if network was attempted, it would raise FileNotFoundError
            }):
                with mock.patch("urllib.request.urlopen", side_effect=AssertionError("Network called unexpectedly!")):
                    ensure_model_available(target_path)
                    self.assertTrue(target_path.exists())
                    self.assertEqual(target_path.read_bytes(), valid_bytes)
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_26_failed_recovery_preserves_previous_file(self):
        """Verifies that if recovery fails, the pre-existing local file is not clobbered or deleted."""
        import io
        import zipfile
        from api.main import ensure_model_available

        initial_content = b"ORIGINAL_LOCAL_FILE_BEFORE_FAILED_RECOVERY"
        target_path = REPO_B_DIR / "models" / "preserve_test.onnx"
        target_path.write_bytes(initial_content)

        # Download returns invalid ZIP
        corrupted_zip_bytes = b"CORRUPTED_RECOVERY_ZIP"
        mock_response = io.BytesIO(corrupted_zip_bytes)
        mock_response.status = 200

        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/bad_recovery.zip",
                "MODEL_EXPECTED_SIZE": "100",
                "MODEL_SHA256": "0" * 64
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError):
                        ensure_model_available(target_path)
                    # Crucial check: the original file must remain intact!
                    self.assertTrue(target_path.exists())
                    self.assertEqual(target_path.read_bytes(), initial_content)
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_27_excessive_archive_entries_rejected(self):
        """Verifies that a ZIP archive containing more than 50 entries is rejected."""
        import io
        import zipfile
        from api.main import ensure_model_available, MAX_ARCHIVE_ENTRIES

        zip_buf = io.BytesIO()
        with zipfile.ZipFile(zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for i in range(MAX_ARCHIVE_ENTRIES + 5):
                zf.writestr(f"dummy_file_{i}.txt", b"x")
            zf.writestr("annadata_production.onnx", b"MODEL_DATA")
        mock_response = io.BytesIO(zip_buf.getvalue())
        mock_response.status = 200

        target_path = REPO_B_DIR / "models" / "excess_entries_test.onnx"
        try:
            with mock.patch.dict("os.environ", {
                "MODEL_DOWNLOAD_URL": "https://example.com/excess.zip",
            }):
                with mock.patch("urllib.request.urlopen", return_value=mock_response):
                    with self.assertRaises(RuntimeError) as ctx:
                        ensure_model_available(target_path)
                    self.assertIn("exceeding the maximum allowed limit", str(ctx.exception).lower())
                    self.assertFalse(target_path.exists())
        finally:
            if target_path.exists():
                target_path.unlink()

    def test_28_malformed_non_empty_env_vars_fail_clearly(self):
        """Verifies that non-empty malformed MODEL_EXPECTED_SIZE and MODEL_SHA256 fail with clear ValueError."""
        from api.main import ensure_model_available
        target_path = REPO_B_DIR / "models" / "malformed_env_test.onnx"

        # Case A: Non-numeric size
        with mock.patch.dict("os.environ", {"MODEL_EXPECTED_SIZE": "invalid_number"}):
            with self.assertRaises(ValueError) as ctx:
                ensure_model_available(target_path)
            self.assertIn("invalid model_expected_size configuration", str(ctx.exception).lower())

        # Case B: Non-positive size (e.g. -10 or 0)
        with mock.patch.dict("os.environ", {"MODEL_EXPECTED_SIZE": "-10"}):
            with self.assertRaises(ValueError) as ctx:
                ensure_model_available(target_path)
            self.assertIn("positive integer", str(ctx.exception).lower())

        # Case C: Invalid non-hex characters in SHA-256
        with mock.patch.dict("os.environ", {"MODEL_SHA256": "g" * 64}):
            with self.assertRaises(ValueError) as ctx:
                ensure_model_available(target_path)
            self.assertIn("invalid model_sha256 configuration", str(ctx.exception).lower())

        # Case D: Wrong length SHA-256 (e.g. 6 chars instead of 64)
        with mock.patch.dict("os.environ", {"MODEL_SHA256": "abcdef"}):
            with self.assertRaises(ValueError) as ctx:
                ensure_model_available(target_path)
            self.assertIn("64-character hexadecimal", str(ctx.exception).lower())

if __name__ == "__main__":
    unittest.main()
