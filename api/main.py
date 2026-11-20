"""
Annadata Plant Disease Diagnostics API
FastAPI REST endpoint wrapping the ONNX Runtime inference engine.

Endpoints:
  GET  /                 - Health check + model metadata
  GET  /classes          - List all 38 disease classes
  POST /predict          - Upload a leaf image, get top-K diagnoses
  POST /predict/batch    - Upload up to 10 images, get diagnoses for each
  GET  /docs             - Interactive Swagger UI (auto-generated)
"""

import sys
import json
import time
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
import cv2
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ── Path resolution (works from any working directory) ──────────────────────
ROOT = Path(__file__).resolve().parent.parent
ONNX_PATH  = ROOT / "models" / "annadata_production.onnx"
CLASS_PATH = ROOT / "artifacts" / "disease_classes.json"

# ── Pydantic response models ─────────────────────────────────────────────────
class Prediction(BaseModel):
    rank: int
    class_index: int
    name: str
    crop: str
    disease: str
    confidence_percent: float

class DiagnosisResponse(BaseModel):
    image_filename: str
    inference_time_ms: float
    top_predictions: List[Prediction]
    model_version: str = "EfficientNetB0-ONNX-Opset13"

class BatchDiagnosisResponse(BaseModel):
    results: List[DiagnosisResponse]
    total_images: int
    total_inference_time_ms: float

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    num_classes: int
    onnx_model_size_mb: float
    model_input_shape: List[int]
    model_version: str

class ClassInfo(BaseModel):
    class_index: int
    name: str
    crop: str
    disease: str

# ── ONNX Session (singleton, loaded once at startup) ─────────────────────────
class AnnadataONNXPredictor:
    def __init__(self):
        if not ONNX_PATH.exists():
            raise RuntimeError(
                f"ONNX model not found at '{ONNX_PATH}'. "
                "Download annadata_production.onnx from Colab and place it in models/."
            )
        if not CLASS_PATH.exists():
            raise RuntimeError(
                f"Class map not found at '{CLASS_PATH}'. "
                "Download disease_classes.json from Colab and place it in artifacts/."
            )

        # Use CUDA if available, fall back to CPU
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in ort.get_available_providers()
            else ["CPUExecutionProvider"]
        )
        self.session = ort.InferenceSession(str(ONNX_PATH), providers=providers)
        self.input_name  = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape   # [1, 224, 224, 3]

        with open(CLASS_PATH, "r", encoding="utf-8") as f:
            self.class_map: dict = json.load(f)

        self.model_size_mb = ONNX_PATH.stat().st_size / 1e6
        self.active_provider = self.session.get_providers()[0]

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """Decode bytes → RGB float32 (1, 224, 224, 3). No /255 needed (embedded in ONNX graph)."""
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Could not decode image. Ensure it is a valid JPEG or PNG.")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        return np.expand_dims(img.astype(np.float32), axis=0)

    def predict(self, image_bytes: bytes, top_k: int = 5) -> tuple[List[Prediction], float]:
        tensor = self.preprocess(image_bytes)
        t0 = time.perf_counter()
        raw = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        elapsed_ms = (time.perf_counter() - t0) * 1000

        top_indices = np.argsort(raw)[-top_k:][::-1]
        predictions = []
        for rank, idx in enumerate(top_indices, start=1):
            info = self.class_map.get(str(idx), {"name": f"Class_{idx}", "crop": "Unknown", "disease": "Unknown"})
            predictions.append(Prediction(
                rank=rank,
                class_index=int(idx),
                name=info.get("name", f"Class_{idx}"),
                crop=info.get("crop", "Unknown"),
                disease=info.get("disease", "Unknown"),
                confidence_percent=round(float(raw[idx]) * 100.0, 4),
            ))
        return predictions, elapsed_ms


# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Annadata Plant Disease Diagnostics API",
    description=(
        "AI-powered plant disease classification across 38 crop-disease categories. "
        "EfficientNetB0 backbone, 99.48% val accuracy, served via ONNX Runtime."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model at startup
predictor: Optional[AnnadataONNXPredictor] = None

@app.on_event("startup")
def load_model():
    global predictor
    try:
        predictor = AnnadataONNXPredictor()
        print(f"[Annadata API] Model loaded. Provider: {predictor.active_provider}  "
              f"Size: {predictor.model_size_mb:.2f} MB  Classes: {len(predictor.class_map)}")
    except RuntimeError as e:
        print(f"[Annadata API] WARNING: {e}")
        predictor = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["Health"])
def health_check():
    """Service health check and model metadata."""
    if predictor is None:
        return JSONResponse(
            status_code=503,
            content={"status": "unavailable", "detail": "Model not loaded. Check server logs."},
        )
    return HealthResponse(
        status="ok",
        model_loaded=True,
        num_classes=len(predictor.class_map),
        onnx_model_size_mb=round(predictor.model_size_mb, 2),
        model_input_shape=[1, 224, 224, 3],
        model_version="EfficientNetB0-ONNX-Opset13",
    )


@app.get("/classes", response_model=List[ClassInfo], tags=["Metadata"])
def list_classes():
    """Return all 38 disease class definitions in index order."""
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    return [
        ClassInfo(
            class_index=int(k),
            name=v.get("name", f"Class_{k}"),
            crop=v.get("crop", "Unknown"),
            disease=v.get("disease", "Unknown"),
        )
        for k, v in sorted(predictor.class_map.items(), key=lambda x: int(x[0]))
    ]


@app.post("/predict", response_model=DiagnosisResponse, tags=["Inference"])
async def predict(
    file: UploadFile = File(..., description="Leaf image (JPEG or PNG)"),
    top_k: int = Query(default=5, ge=1, le=38, description="Number of top diagnoses to return"),
):
    """
    Upload a single leaf image and receive a ranked diagnosis.

    - **file**: JPEG or PNG leaf photograph
    - **top_k**: how many candidate diagnoses to return (1–38, default 5)

    Returns ranked predictions with crop name, disease name, and confidence %.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check server logs.")

    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported media type '{file.content_type}'. Upload a JPEG or PNG image.",
        )

    image_bytes = await file.read()
    if len(image_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(image_bytes) > 20 * 1024 * 1024:  # 20 MB guard
        raise HTTPException(status_code=413, detail="Image too large. Max 20 MB.")

    try:
        predictions, elapsed_ms = predictor.predict(image_bytes, top_k=top_k)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return DiagnosisResponse(
        image_filename=file.filename or "uploaded_image",
        inference_time_ms=round(elapsed_ms, 2),
        top_predictions=predictions,
    )


@app.post("/predict/batch", response_model=BatchDiagnosisResponse, tags=["Inference"])
async def predict_batch(
    files: List[UploadFile] = File(..., description="Up to 10 leaf images (JPEG or PNG)"),
    top_k: int = Query(default=3, ge=1, le=38, description="Diagnoses per image"),
):
    """
    Upload up to 10 leaf images in a single request.
    Returns individual diagnosis results for each image.
    """
    if predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    if len(files) > 10:
        raise HTTPException(status_code=400, detail="Maximum 10 images per batch request.")

    results = []
    total_ms = 0.0

    for f in files:
        image_bytes = await f.read()
        try:
            preds, ms = predictor.predict(image_bytes, top_k=top_k)
        except ValueError as e:
            preds = []
            ms = 0.0

        total_ms += ms
        results.append(DiagnosisResponse(
            image_filename=f.filename or "uploaded_image",
            inference_time_ms=round(ms, 2),
            top_predictions=preds,
        ))

    return BatchDiagnosisResponse(
        results=results,
        total_images=len(results),
        total_inference_time_ms=round(total_ms, 2),
    )
