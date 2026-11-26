"""
Annadata AI Plant Disease Diagnostics - FastAPI Inference Service
Production-grade REST API serving the EfficientNetB0 ONNX model.
"""

import os
import sys
import json
import time
import stat
import hashlib
import zipfile
import urllib.request
import urllib.error
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

# ── Paths & Model Artifact Specs ──────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
ONNX_MODEL_PATH = BASE_DIR / "models" / "annadata_production.onnx"
CLASS_MAPPING_PATH = BASE_DIR / "artifacts" / "disease_classes.json"

TARGET_MODEL_NAME = "annadata_production.onnx"
DEFAULT_MODEL_SIZE = 18748076
DEFAULT_MODEL_SHA256 = "d948d12e291d79daa226ed9385d7526e1ef294027831e35da7665afb0464790d"
MAX_ZIP_DOWNLOAD_BYTES = 40 * 1024 * 1024       # 40 MB max ZIP download limit
MAX_UNCOMPRESSED_MODEL_BYTES = 30 * 1024 * 1024 # 30 MB max uncompressed extraction limit
MAX_ARCHIVE_ENTRIES = 50                        # 50 max entries in ZIP archive

def is_safe_zip_path(path_str: str) -> bool:
    """
    Guards against path traversal, absolute paths, or directory escape ('..')
    within a ZIP archive entry.
    """
    normalized = path_str.replace("\\", "/")
    if normalized.startswith("/") or ".." in normalized.split("/"):
        return False
    p = Path(path_str)
    if p.is_absolute() or p.drive:
        return False
    return True

def is_symlink_zip_member(member: zipfile.ZipInfo) -> bool:
    """
    Checks if a ZIP archive member is a symbolic link.
    On Unix-like attributes stored in ZIP, mode 0o120000 indicates a symlink.
    """
    mode = member.external_attr >> 16
    return bool(mode and stat.S_ISLNK(mode))

def verify_file_integrity(file_path: Path, expected_size: int, expected_sha256: str) -> bool:
    """
    Verifies that a local file exists, is a regular file, matches expected_size,
    and matches expected_sha256 checksum.
    Returns True if valid, False otherwise.
    """
    if not file_path.is_file():
        return False
    try:
        if file_path.stat().st_size != expected_size:
            return False
        h = hashlib.sha256()
        with open(file_path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                h.update(chunk)
        return h.hexdigest().lower() == expected_sha256.lower()
    except Exception:
        return False

def ensure_model_available(model_path: Path = ONNX_MODEL_PATH) -> None:
    """
    Ensures that the production ONNX model is available and integrity-verified.
    1. If the model file already exists locally, verifies its exact size and SHA-256.
       If valid, returns immediately.
    2. If missing or corrupted, attempts recovery by downloading the release ZIP archive from MODEL_DOWNLOAD_URL.
    3. Enforces a 40 MB maximum download limit to prevent disk exhaustion.
    4. Validates that the downloaded file is a valid ZIP archive with at most 50 entries.
    5. Rejects symbolic links, path traversal, duplicate model files, and oversized entries (> 30 MB).
    6. Safely extracts ONLY annadata_production.onnx into a temporary file, verifying size and SHA-256.
    7. Atomically replaces final destination with the verified file.
    8. Preserves any existing file if recovery fails, and cleans up temporary files in a single finally block.
    """
    raw_size = os.getenv("MODEL_EXPECTED_SIZE", "").strip()
    if raw_size:
        try:
            expected_size = int(raw_size)
            if expected_size <= 0:
                raise ValueError(f"MODEL_EXPECTED_SIZE must be a positive integer, got '{raw_size}'")
        except ValueError as exc:
            raise ValueError(f"Invalid MODEL_EXPECTED_SIZE configuration: '{raw_size}'. Must be a positive integer.") from exc
    else:
        expected_size = DEFAULT_MODEL_SIZE

    raw_sha256 = os.getenv("MODEL_SHA256", "").strip().lower()
    if raw_sha256:
        if len(raw_sha256) != 64 or not all(c in "0123456789abcdef" for c in raw_sha256):
            raise ValueError(
                f"Invalid MODEL_SHA256 configuration: '{raw_sha256}'. Must be a 64-character hexadecimal SHA-256 hash."
            )
        expected_sha256 = raw_sha256
    else:
        expected_sha256 = DEFAULT_MODEL_SHA256

    if model_path.exists():
        if verify_file_integrity(model_path, expected_size, expected_sha256):
            return
        print(
            f"[Annadata Service] Warning: Local model at '{model_path}' failed integrity check "
            f"(corrupted, wrong size, or checksum mismatch). Attempting recovery via download..."
        )

    download_url = os.getenv("MODEL_DOWNLOAD_URL", "").strip()
    if not download_url:
        raise FileNotFoundError(
            f"Production ONNX model at '{model_path}' is missing or corrupted. "
            "Set the 'MODEL_DOWNLOAD_URL' environment variable to a valid GitHub Release ZIP asset URL "
            "or place the model file in 'models/' before starting."
        )

    print(f"[Annadata Service] Production model missing or invalid. Downloading ZIP from: {download_url}")
    model_path.parent.mkdir(parents=True, exist_ok=True)

    temp_zip_path = model_path.with_suffix(".zip.tmp")
    temp_onnx_path = model_path.with_suffix(".onnx.tmp")

    try:
        # Step 1: Download the ZIP archive into a temporary file (max 40 MB)
        req = urllib.request.Request(
            download_url,
            headers={"User-Agent": "Annadata-Inference-Service/2.0"}
        )
        with urllib.request.urlopen(req, timeout=120) as response, open(temp_zip_path, "wb") as out_zip:
            resp_code = getattr(response, "status", 200)
            if resp_code != 200:
                raise RuntimeError(f"HTTP {resp_code} received when downloading model ZIP from {download_url}")

            downloaded_bytes = 0
            while True:
                chunk = response.read(65536)
                if not chunk:
                    break
                downloaded_bytes += len(chunk)
                if downloaded_bytes > MAX_ZIP_DOWNLOAD_BYTES:
                    raise ValueError(
                        f"Downloaded ZIP archive exceeds maximum allowed limit of {MAX_ZIP_DOWNLOAD_BYTES} bytes."
                    )
                out_zip.write(chunk)

        # Step 2: Validate that downloaded file is a valid, uncorrupted ZIP archive
        if not zipfile.is_zipfile(temp_zip_path):
            raise ValueError(f"Downloaded file from {download_url} is not a valid ZIP archive.")

        # Step 3: Inspect archive entries and safely extract only annadata_production.onnx
        with zipfile.ZipFile(temp_zip_path, "r") as zf:
            infolist = zf.infolist()

            if len(infolist) > MAX_ARCHIVE_ENTRIES:
                raise ValueError(
                    f"ZIP archive contains {len(infolist)} entries, exceeding the maximum allowed limit of {MAX_ARCHIVE_ENTRIES}."
                )

            # Reject symlinks, path traversal, or malformed paths in any member
            for member in infolist:
                if is_symlink_zip_member(member):
                    raise ValueError(f"Symbolic link detected in ZIP member: '{member.filename}'. Symlinks are forbidden.")
                if not is_safe_zip_path(member.filename):
                    raise ValueError(f"Path traversal or invalid path detected in ZIP member: '{member.filename}'")

            # Locate the expected model member
            model_members = [
                m for m in infolist
                if not m.is_dir() and Path(m.filename).name == TARGET_MODEL_NAME
            ]

            if not model_members:
                found_names = [m.filename for m in infolist if not m.is_dir()]
                raise FileNotFoundError(
                    f"Required model file '{TARGET_MODEL_NAME}' not found inside the ZIP archive. "
                    f"Archive contents: {found_names}"
                )

            if len(model_members) > 1:
                raise ValueError(
                    f"Ambiguous ZIP archive: multiple files named '{TARGET_MODEL_NAME}' found: "
                    f"{[m.filename for m in model_members]}"
                )

            target_member = model_members[0]

            if target_member.file_size > MAX_UNCOMPRESSED_MODEL_BYTES:
                raise ValueError(
                    f"ZIP member '{target_member.filename}' declared size ({target_member.file_size} bytes) "
                    f"exceeds maximum allowed limit of {MAX_UNCOMPRESSED_MODEL_BYTES} bytes."
                )

            # Stream out to temporary ONNX file while calculating SHA-256 and byte size (max 30 MB)
            sha256 = hashlib.sha256()
            extracted_bytes = 0

            with zf.open(target_member) as src_stream, open(temp_onnx_path, "wb") as dst_stream:
                while True:
                    chunk = src_stream.read(65536)
                    if not chunk:
                        break
                    extracted_bytes += len(chunk)
                    if extracted_bytes > MAX_UNCOMPRESSED_MODEL_BYTES:
                        raise ValueError(
                            f"Extracted model data exceeds maximum allowed limit of {MAX_UNCOMPRESSED_MODEL_BYTES} bytes."
                        )
                    dst_stream.write(chunk)
                    sha256.update(chunk)

        # Step 4: Verify the extracted ONNX file's exact size
        if extracted_bytes != expected_size:
            raise ValueError(
                f"Extracted model size verification failed: expected {expected_size} bytes, extracted {extracted_bytes} bytes."
            )

        # Step 5: Verify SHA-256 checksum on extracted ONNX file
        computed_sha256 = sha256.hexdigest().lower()
        if computed_sha256 != expected_sha256:
            raise ValueError(
                f"Extracted model SHA-256 checksum verification failed!\n"
                f"  Expected: {expected_sha256}\n"
                f"  Computed: {computed_sha256}"
            )

        # Step 6: Atomic replacement of final model file
        temp_onnx_path.replace(model_path)
        print(
            f"[Annadata Service] Production ONNX model successfully extracted and verified: "
            f"{model_path} ({extracted_bytes} bytes, SHA-256: {computed_sha256})"
        )

    except Exception as exc:
        raise RuntimeError(f"Failed to retrieve and verify production model from '{download_url}': {exc}") from exc

    finally:
        # Always remove any temporary files
        for tmp_file in (temp_zip_path, temp_onnx_path):
            if tmp_file.exists():
                try:
                    tmp_file.unlink()
                except OSError:
                    pass

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
        # Ensure model exists (or is retrieved and integrity-checked via MODEL_DOWNLOAD_URL)
        ensure_model_available(model_path)

        if not model_path.exists():
            raise FileNotFoundError(
                f"Production ONNX model not found at '{model_path}'. "
                "Ensure models/annadata_production.onnx is staged or set MODEL_DOWNLOAD_URL."
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

# ── CORS Configuration ────────────────────────────────────────────────────────
def get_cors_origins() -> List[str]:
    dev_origins = [
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:8080",
        "http://127.0.0.1:8080",
    ]
    env_origins_str = os.getenv("CORS_ORIGINS", "").strip()
    if not env_origins_str:
        return dev_origins

    custom_origins = [
        origin.strip().rstrip("/")
        for origin in env_origins_str.split(",")
        if origin.strip()
    ]
    return list(dict.fromkeys(dev_origins + custom_origins))

cors_origins = get_cors_origins()
env_cors = os.getenv("CORS_ORIGINS", "").strip()

if env_cors == "*":
    # If explicitly set to wildcard in dev/testing, credentials must be False per browser standards
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
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

# ── Production Entry Point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    host = os.getenv("HOST", "0.0.0.0")
    print(f"[Annadata Service] Starting Uvicorn server on http://{host}:{port}")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)
