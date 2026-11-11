"""
Annadata - Unit and Regression Test Suite
test_inference.py: Validates class metadata schema, ONNX graph properties, and predictor functionality.
"""

import os
import json
import unittest
import numpy as np
from pathlib import Path
from PIL import Image

class TestAnnadataPipeline(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.artifacts_dir = Path("artifacts")
        cls.models_dir = Path("models")
        cls.class_json_path = cls.artifacts_dir / "disease_classes.json"
        cls.onnx_path = cls.models_dir / "annadata_production.onnx"
        cls.test_image = Path("test_leaf.jpg")
        
        if not cls.test_image.exists():
            mock_arr = np.random.randint(0, 255, size=(224, 224, 3), dtype=np.uint8)
            img = Image.fromarray(mock_arr)
            img.save(cls.test_image)

    @classmethod
    def tearDownClass(cls):
        pass

    def test_class_map_schema(self):
        """Verify disease_classes.json contains 38 classes and complete metadata."""
        self.assertTrue(self.class_json_path.exists(), f"Missing {self.class_json_path}")
        with open(self.class_json_path, "r", encoding="utf-8") as f:
            class_map = json.load(f)
            
        self.assertEqual(len(class_map), 38, f"Expected 38 classes, found {len(class_map)}")
        for idx in range(38):
            key = str(idx)
            self.assertIn(key, class_map, f"Missing index {key} in class mapping")
            entry = class_map[key]
            self.assertIn("name", entry)
            self.assertIn("crop", entry)
            self.assertIn("disease", entry)
            self.assertTrue(len(entry["name"]) > 0)

    def test_onnx_model_io(self):
        """Verify ONNX model input and output tensor shapes."""
        self.assertTrue(self.onnx_path.exists(), f"Missing {self.onnx_path}")
        import onnxruntime as ort
        
        session = ort.InferenceSession(str(self.onnx_path), providers=["CPUExecutionProvider"])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        self.assertEqual(len(inputs), 1)
        self.assertEqual(len(outputs), 1)
        
        # Input shape: [None, 224, 224, 3] or [batch, 224, 224, 3]
        input_shape = inputs[0].shape
        self.assertEqual(input_shape[1:], [224, 224, 3])
        
        # Output shape: [None, 38] or [batch, 38]
        output_shape = outputs[0].shape
        self.assertEqual(output_shape[1], 38)

    def test_end_to_end_prediction(self):
        """Verify AnnadataPredictor runs inference, returns sorted top-k, and valid probabilities."""
        from predict import AnnadataPredictor
        
        predictor = AnnadataPredictor(
            onnx_path=self.onnx_path,
            class_path=self.class_json_path
        )
        
        results = predictor.predict(str(self.test_image), top_k=3)
        self.assertEqual(len(results), 3)
        
        # Verify rank order and confidence values
        prev_conf = 101.0
        for res in results:
            conf = res["confidence_percent"]
            self.assertGreaterEqual(conf, 0.0)
            self.assertLessEqual(conf, 100.0)
            self.assertLessEqual(conf, prev_conf)
            prev_conf = conf

if __name__ == "__main__":
    unittest.main()
