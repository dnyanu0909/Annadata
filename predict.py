"""
Annadata - Plant Disease Diagnostics Engine
predict.py: Production inference client using ONNX Runtime with zero TensorFlow dependencies.
"""

import os
import sys

# Windows console encoding fix
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import json
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Any
import numpy as np
import onnxruntime as ort
import cv2

DEFAULT_ONNX_PATH = Path("models/annadata_production.onnx")
DEFAULT_CLASS_PATH = Path("artifacts/disease_classes.json")

class AnnadataPredictor:
    """Production predictor executing ONNX Runtime inference."""
    
    def __init__(self, onnx_path: Path = DEFAULT_ONNX_PATH, class_path: Path = DEFAULT_CLASS_PATH):
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX model missing at '{onnx_path}'. Run training/smoke-test first.")
        if not class_path.exists():
            raise FileNotFoundError(f"Class mapping missing at '{class_path}'. Run training/smoke-test first.")
            
        # Initialize ONNX Runtime session
        self.session = ort.InferenceSession(
            str(onnx_path),
            providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        
        # Load locked class mapping
        with open(class_path, "r", encoding="utf-8") as f:
            self.class_map = json.load(f)

    def preprocess_image(self, image_path: str) -> np.ndarray:
        """Load and preprocess image: BGR->RGB, resize (224, 224), float32 [0..255]."""
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Input leaf image not found at: {image_path}")
            
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"OpenCV could not decode image at: {image_path}")
            
        # Standardize color space to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Resize to model input dimensions
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        
        # Note: Do NOT divide by 255.0 here because Rescaling(1./255) is embedded directly in the ONNX graph
        img = img.astype(np.float32)
        
        # Add batch dimension: (1, 224, 224, 3)
        return np.expand_dims(img, axis=0)

    def predict(self, image_path: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Run inference and return top-k diagnoses."""
        input_tensor = self.preprocess_image(image_path)
        
        # Run ONNX inference
        raw_outputs = self.session.run([self.output_name], {self.input_name: input_tensor})[0][0]
        
        # Sort top-k
        top_indices = np.argsort(raw_outputs)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            idx_str = str(idx)
            class_info = self.class_map.get(idx_str, {"name": f"Class_{idx}", "crop": "Unknown", "disease": "Unknown"})
            confidence = float(raw_outputs[idx]) * 100.0
            
            results.append({
                "rank": len(results) + 1,
                "class_index": int(idx),
                "name": class_info["name"],
                "crop": class_info.get("crop", "Unknown"),
                "disease": class_info.get("disease", "Unknown"),
                "confidence_percent": round(confidence, 2)
            })
            
        return results

def print_diagnosis_table(results: List[Dict[str, Any]], image_path: str):
    """Render diagnosis report in the terminal."""
    print("\n" + "="*56)
    print("          🌾 ANNADATA AI PLANT DIAGNOSTICS 🌾")
    print("="*56)
    print(f" Analyzed Image: {image_path}")
    print("-" * 56)
    print(f" {'Rank':<5} {'Diagnosis':<36} {'Confidence':>10}")
    print("-" * 56)
    for res in results:
        star = "⭐" if res["rank"] == 1 else "  "
        print(f" {star} #{res['rank']:<2} {res['name']:<36} {res['confidence_percent']:>8.2f}%")
    print("="*56 + "\n")

def main():
    parser = argparse.ArgumentParser(description="Annadata Plant Disease Prediction Engine")
    parser.add_argument(
        "--image",
        type=str,
        default="test_leaf.jpg",
        help="Path to leaf image for diagnosis (default: test_leaf.jpg)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of top predictions to display (default: 3)"
    )
    parser.add_argument(
        "--onnx-path",
        type=str,
        default=str(DEFAULT_ONNX_PATH),
        help="Path to the ONNX model file"
    )
    parser.add_argument(
        "--class-path",
        type=str,
        default=str(DEFAULT_CLASS_PATH),
        help="Path to disease_classes.json"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON for API integration"
    )
    args = parser.parse_args()
    
    predictor = AnnadataPredictor(
        onnx_path=Path(args.onnx_path),
        class_path=Path(args.class_path)
    )
    
    predictions = predictor.predict(args.image, top_k=args.top_k)
    
    if args.json:
        print(json.dumps(predictions, indent=2))
    else:
        print_diagnosis_table(predictions, args.image)

if __name__ == "__main__":
    main()
