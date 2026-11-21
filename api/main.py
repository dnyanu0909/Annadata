"""
Annadata AI Plant Disease Diagnostics - FastAPI Inference Service
Production-grade REST API serving the EfficientNetB0 ONNX model.
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = BASE_DIR / "models" / "annadata_production.onnx"
CLASS_MAPPING_PATH = BASE_DIR / "artifacts" / "disease_classes.json"

# ── Pydantic Response Schemas (Matching Repository A Prediction Contract) ─────
class TopPrediction(BaseModel):
    class_: str = Field(..., alias="class", description="Raw class name e.g. Tomato___Early_blight")
    label: str = Field(..., description="Display label e.g. Tomato Early Blight")
    confidence: float = Field(..., description="Confidence score between 0.0 and 1.0")

    class Config:
        populate_by_name = True

class PredictionResponse(BaseModel):
    success: bool = True
    predicted_class: str = Field(..., description="Top predicted raw class e.g. Tomato___Early_blight")
    predicted_label: str = Field(..., description="Top predicted display label e.g. Tomato Early Blight")
    confidence: float = Field(..., description="Top confidence score between 0.0 and 1.0")
    top_predictions: List[TopPrediction] = Field(default_factory=list)
    processing_time_ms: float = Field(..., description="Inference latency in milliseconds")

class HealthResponse(BaseModel):
    status: str
    service: str = "Annadata AI Diagnostics"
    model_loaded: bool
    model_version: str = "EfficientNetB0-ONNX-Opset13"
    input_shape: List[Optional[int]] = [1, 224, 224, 3]
    num_classes: int = 38
    device: str = "CPU"

# ── ONNX Inference Engine (Singleton) ─────────────────────────────────────────
class AnnadataInferenceEngine:
    """Production ONNX Runtime inference manager loaded once at startup."""

    def __init__(self, model_path: Path = ONNX_MODEL_PATH, class_path: Path = CLASS_MAPPING_PATH):
        if not model_path.exists():
            raise FileNotFoundError(
                f"Production ONNX model not found at '{model_path}'. "
                "Ensure models/annadata_production.onnx is staged."
            )
        if not class_path.exists():
            raise FileNotFoundError(
                f"Disease class mapping not found at '{class_path}'. "
                "Ensure artifacts/disease_classes.json is staged."
            )

        # Provider selection: prioritize CUDA if available, fallback to CPU
        available_providers = ort.get_available_providers()
        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if "CUDAExecutionProvider" in available_providers
            else ["CPUExecutionProvider"]
        )

        # Initialize session once
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name
        self.active_provider = self.session.get_providers()[0]

        # Load locked 38-class mapping
        with open(class_path, "r", encoding="utf-8") as f:
            self.class_map: dict = json.load(f)

        if len(self.class_map) != 38:
            raise ValueError(f"Expected 38 classes in mapping, found {len(self.class_map)}")

    def preprocess(self, image_bytes: bytes) -> np.ndarray:
        """
        Decodes and preprocesses raw image bytes into the exact model input tensor.
        Steps:
          1. Decode raw bytes with OpenCV into BGR uint8.
          2. Convert BGR -> RGB.
          3. Resize to 224x224 using cv2.INTER_AREA (anti-aliased downsampling).
          4. Cast to float32 in range [0..255].
             NOTE: Do NOT divide by 255.0 here because layers.Rescaling(1./255)
             is embedded directly inside the model's computational graph.
          5. Expand batch dimension: (1, 224, 224, 3).
        """
        if not image_bytes:
            raise ValueError("Empty image bytes received.")

        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("The uploaded image could not be decoded. Ensure it is a valid, uncorrupted JPEG or PNG file.")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
        tensor = img.astype(np.float32)
        return np.expand_dims(tensor, axis=0)

    def predict(self, image_bytes: bytes, top_k: int = 3) -> PredictionResponse:
        """Runs inference and formats response according to Repository A's Prediction contract."""
        t_start = time.perf_counter()

        tensor = self.preprocess(image_bytes)

        # Run ONNX inference
        raw_outputs = self.session.run([self.output_name], {self.input_name: tensor})[0][0]
        elapsed_ms = (time.perf_counter() - t_start) * 1000

        # Sort top-K indices descending by probability
        top_indices = np.argsort(raw_outputs)[::-1][:top_k]

        top_predictions: List[TopPrediction] = []
        for idx in top_indices:
            idx_str = str(idx)
            info = self.class_map.get(idx_str, {})
            raw_class = info.get("raw_class", f"Class_{idx}")
            label = info.get("name", raw_class.replace("___", " "))
            conf = float(raw_outputs[idx])

            top_predictions.append(
                TopPrediction(
                    class_=raw_class,
                    label=label,
                    confidence=round(conf, 4)
                )
            )

        top_pred = top_predictions[0]

        return PredictionResponse(
            success=True,
            predicted_class=top_pred.class_,
            predicted_label=top_pred.label,
            confidence=top_pred.confidence,
            top_predictions=top_predictions,
            processing_time_ms=round(elapsed_ms, 2)
        )

from contextlib import asynccontextmanager

engine: Optional[AnnadataInferenceEngine] = None

def get_engine() -> Optional[AnnadataInferenceEngine]:
    global engine
    if engine is None:
        try:
            engine = AnnadataInferenceEngine()
            print(f"[Annadata Service] Model loaded successfully. Active provider: {engine.active_provider}")
        except Exception as exc:
            print(f"[Annadata Service] Initialization Error: {exc}", file=sys.stderr)
            engine = None
    return engine

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_engine()
    yield

# ── FastAPI App Setup ─────────────────────────────────────────────────────────
app = FastAPI(
    title="Annadata AI Diagnostics Service",
    description="Production plant disease classification API powered by EfficientNetB0 ONNX Opset 13.",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# CORS Configuration: allow frontend dev servers (Vite, Lovable, preview)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_model=HealthResponse, tags=["System"])
def root():
    """Health check endpoint providing model and service status."""
    eng = get_engine()
    return HealthResponse(
        status="ok" if eng is not None else "degraded",
        model_loaded=eng is not None,
        device=eng.active_provider if eng else "None"
    )

@app.get("/api/classes", tags=["Metadata"])
def get_classes():
    """Returns all 38 supported plant disease classes."""
    eng = get_engine()
    if eng is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Inference engine is not initialized."
        )
    return eng.class_map

@app.post("/api/predict", response_model=PredictionResponse, tags=["Inference"])
async def predict_leaf(
    image: Optional[UploadFile] = File(None, description="Leaf image file from Repository A"),
    file: Optional[UploadFile] = File(None, description="Alternative field name for curl/clients"),
    top_k: int = Query(default=3, ge=1, le=38, description="Number of top candidates to return")
):
    """
    Primary prediction endpoint conforming to the Annadata frontend specification.
    Accepts multipart/form-data with 'image' (or 'file') field containing a leaf image.
    """
    eng = get_engine()
    if eng is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The prediction model is unavailable or failed to initialize."
        )

    upload = image or file
    if upload is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No image uploaded. Expected multipart form field 'image'."
        )

    # Validate file extension / MIME type
    filename = upload.filename or "image.jpg"
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    valid_exts = {"jpg", "jpeg", "png", "webp"}
    valid_types = {"image/jpeg", "image/png", "image/webp", "application/octet-stream"}

    if ext not in valid_exts and upload.content_type not in valid_types:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail="Unsupported image format. Choose a JPG, JPEG, or PNG image."
        )

    try:
        content = await upload.read()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Failed to read uploaded image data."
        )

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded image file is empty."
        )

    if len(content) > 10 * 1024 * 1024:  # 10 MB guard matching Repository A
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="This image is larger than 10 MB. Choose a smaller file."
        )

    try:
        prediction = eng.predict(content, top_k=top_k)
        return prediction
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err)
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(exc)}"
        )
