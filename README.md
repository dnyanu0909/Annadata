<div align="center">

# 🌾 Annadata
### Autonomous Plant Disease Diagnostics Engine

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21-FF6F00?style=flat&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![ONNX](https://img.shields.io/badge/ONNX-Opset%2013-005CED?style=flat&logo=onnx&logoColor=white)](https://onnx.ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688?style=flat&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat)](LICENSE)
[![Val Accuracy](https://img.shields.io/badge/Val%20Accuracy-99.48%25-brightgreen?style=flat)]()

**Production-grade AI pipeline for classifying 38 crop-disease categories from leaf images.**
EfficientNetB0 backbone · 99.48% validation accuracy · ONNX Runtime inference · FastAPI REST backend

[Overview](#overview) · [Architecture](#architecture) · [CV Pipeline](#computer-vision-pipeline) · [Training](#training-details) · [API](#api) · [Quickstart](#quickstart) · [Results](#results)

</div>

---

## Overview

Annadata is a full-stack computer vision system that diagnoses plant diseases from leaf photographs. It is designed around three engineering principles:

- **Determinism** — class indices are atomically locked from the training dataset's folder order at initialization, eliminating any index-to-label drift across training, export, and inference.
- **Portability** — the trained model is exported to ONNX Opset 13 immediately after training, enabling inference via `onnxruntime` on any CPU/GPU without TensorFlow installed in production.
- **Reproducibility** — a two-phase learning rate schedule with embedded preprocessing inside the model graph makes the entire pipeline environment-agnostic.

**Dataset:** [PlantVillage](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset) — 54,305 images, 38 classes, 14 crop species.

---

## Tech Stack

| Layer | Technology | Version |
|:---|:---|:---|
| **Training Framework** | TensorFlow + tf-keras (Legacy Keras 2) | 2.21.0 |
| **Backbone** | EfficientNetB0 (ImageNet pretrained) | via `tf_keras.applications` |
| **ONNX Conversion** | tf2onnx | 1.17.0 |
| **Production Inference** | ONNX Runtime | 1.31.1 |
| **REST API** | FastAPI + Uvicorn | 0.111+ |
| **Image Processing** | OpenCV | 4.x |
| **Numerical Compute** | NumPy | 1.26.x |
| **Data Ingestion** | Kaggle API | — |
| **API Validation** | Pydantic v2 | 2.x |
| **Training Platform** | Google Colab T4 GPU | — |
| **Language** | Python | 3.13 |

---

## Architecture

### Full System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     DATA INGESTION LAYER                        │
│                                                                 │
│  Kaggle API (automated)  ──or──  Local --data-dir flag          │
│       ↓                                                         │
│  data_prep.py                                                   │
│  • Download plantvillage-dataset (2.04 GB)                      │
│  • Discover 38 class folders                                    │
│  • Stratified 80/20 train/val split                             │
│  • 43,429 train  |  10,876 val images                           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    TRAINING GRAPH (Keras Functional API)         │
│                                                                 │
│  Input(shape=(224, 224, 3), dtype=float32)                      │
│       │                                                         │
│  layers.Rescaling(1./255)   ◄── embedded normalization          │
│       │                         (no client-side /255 needed)    │
│  EfficientNetB0(include_top=False, weights="imagenet")          │
│       │  237 layers, 4.05M parameters, frozen in Phase 1        │
│  GlobalAveragePooling2D()                                       │
│       │                                                         │
│  BatchNormalization()                                           │
│  Dropout(0.30)                                                  │
│  Dense(512, activation="relu")                                  │
│  BatchNormalization()                                           │
│  Dropout(0.15)                                                  │
│  Dense(38, activation="softmax")  ◄── 38-class output           │
│       │                                                         │
│  Total params: ~5.3M (4.05M backbone + 1.25M head)             │
└──────────────────────────┬──────────────────────────────────────┘
                           │  tf2onnx.convert.from_keras()
                           │  (in-memory, no intermediate save)
┌──────────────────────────▼──────────────────────────────────────┐
│                    PRODUCTION ARTIFACTS                          │
│                                                                 │
│  models/annadata_production.onnx  (17.88 MB, ONNX Opset 13)    │
│  models/best_model.keras          (23.96 MB, checkpoint)        │
│  artifacts/disease_classes.json   (6.6 KB, index → label map)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    INFERENCE LAYER                               │
│                                                                 │
│  predict.py                           FastAPI (api/main.py)     │
│  • CLI tool                           • POST /predict           │
│  • onnxruntime CPUExecutionProvider   • POST /predict/batch     │
│  • Top-K ranked output                • GET  /classes           │
│  • ~12ms per image                    • ngrok public tunnel     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Computer Vision Pipeline

### Input Processing

| Step | Operation | Detail |
|:---|:---|:---|
| **1. Load** | `cv2.imread()` | Reads JPEG/PNG from disk or upload bytes |
| **2. Color space** | `cv2.cvtColor(BGR→RGB)` | OpenCV reads BGR by default; model expects RGB |
| **3. Resize** | `cv2.resize((224, 224), INTER_AREA)` | INTER_AREA is optimal for downscaling (anti-aliasing) |
| **4. Type cast** | `astype(np.float32)` | Raw pixel range [0, 255] — no manual /255 |
| **5. Batch dim** | `np.expand_dims(axis=0)` | Shape: `(1, 224, 224, 3)` |
| **6. Normalize** | `Rescaling(1./255)` inside ONNX graph | Applied automatically during inference |

> **Why embed normalization in the graph?**
> Keeping `Rescaling(1./255)` inside the model means every inference path — CLI, API, mobile SDK — gets identical preprocessing with zero divergence risk.

### EfficientNetB0 Backbone

EfficientNetB0 is selected via **Neural Architecture Search (NAS)** using a compound coefficient that jointly scales depth, width, and input resolution. It achieves higher accuracy per FLOP than ResNet, VGG, or Inception families.

| Property | Value |
|:---|:---|
| Input resolution | 224 × 224 |
| Pretrained weights | ImageNet-1K (1,000 classes) |
| Extracted feature maps | 1,280-dim via GlobalAveragePooling2D |
| Total layers | 237 |
| Trainable in Phase 1 | 0 (frozen backbone) |
| Trainable in Phase 2 | Top 30 layers (excluding BatchNorm) |
| Key block type | MBConv (Mobile Inverted Bottleneck + Squeeze-Excitation) |

### Classification Head

```
GlobalAveragePooling2D       →  (batch, 1280)
BatchNormalization           →  stable activations across batches
Dropout(0.30)                →  regularization, prevents co-adaptation
Dense(512, relu)             →  learns crop-disease feature combinations
BatchNormalization           →  re-centers after dense transformation
Dropout(0.15)                →  light regularization before final layer
Dense(38, softmax)           →  probability distribution over 38 classes
```

---

## Training Details

### Dataset Split

| Split | Images | Percentage |
|:---|:---:|:---:|
| **Train** | 43,429 | 80% |
| **Validation** | 10,876 | 20% |
| **Total** | 54,305 | 100% |
| **Classes** | 38 | 14 crop species |

### Two-Phase Training Schedule

#### Phase 1 — Classification Head Warmup

| Parameter | Value |
|:---|:---|
| **Epochs** | 5 |
| **Learning Rate** | `1e-3` (Adam optimizer) |
| **Loss function** | Categorical cross-entropy |
| **Backbone** | Fully frozen (0 trainable layers) |
| **Steps/epoch** | 679 (43,429 images ÷ batch 64) |
| **Checkpoint** | `ModelCheckpoint(monitor="val_loss", save_best_only=True)` |
| **Batch size** | 64 |
| **Time/epoch** | ~113–128 seconds on T4 GPU |

**Phase 1 Convergence:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.2984 | 90.84% | 0.0951 | 96.91% |
| 2 | 0.1253 | 95.91% | 0.0729 | 97.64% |
| 3 | 0.0970 | 96.61% | 0.0637 | 98.02% |
| 4 | 0.0849 | 97.15% | 0.0577 | 98.30% |
| **5** | **0.0788** | **97.38%** | **0.0546** | **98.28%** |

#### Phase 2 — Backbone Fine-Tuning

| Parameter | Value |
|:---|:---|
| **Epochs** | 15 |
| **Learning Rate** | `3e-5` (Adam optimizer) — 33× smaller than Phase 1 |
| **Unfrozen layers** | Top 30 EfficientNetB0 layers |
| **BatchNorm layers** | Kept frozen throughout (prevents feature drift) |
| **EarlyStopping** | `patience=3`, `restore_best_weights=True` |
| **ReduceLROnPlateau** | `factor=0.2`, `patience=2`, `min_lr=1e-7` |
| **Time/epoch** | ~120–132 seconds on T4 GPU |

**Phase 2 Convergence:**

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0444 | 98.43% | 0.0360 | 98.90% |
| 3 | 0.0271 | 99.12% | 0.0271 | 99.13% |
| 5 | 0.0182 | 99.43% | 0.0235 | 99.25% |
| 10 | 0.0110 | 99.69% | 0.0214 | 99.35% |
| **15** | **0.0068** | **99.84%** | **0.0198** | **99.48%** |

**Total training time:** ~40 minutes on Google Colab T4 GPU

### Why These Hyperparameters?

| Choice | Rationale |
|:---|:---|
| LR `1e-3` for Phase 1 | High LR acceptable since backbone is frozen; head initializes randomly and needs aggressive gradient steps |
| LR `3e-5` for Phase 2 | Low LR prevents catastrophic forgetting of ImageNet features while allowing disease-specific fine-tuning |
| Dropout 0.30 → 0.15 | Heavier dropout near feature space, lighter near decision boundary — standard for transfer learning heads |
| Batch size 64 | Fits T4 GPU VRAM (16 GB); large enough for stable BatchNorm statistics |
| BatchNorm frozen in Phase 2 | Keeps running mean/variance from ImageNet pretraining; avoids gradient instability with small LR |
| `INTER_AREA` resize | Mathematically correct for downsampling (averages pixels), vs. INTER_LINEAR which interpolates |

---

## ONNX Export

The model is converted **in-memory** directly after training — no intermediate `.h5` save/reload cycle that could cause layer mismatch errors:

```python
model_proto, _ = tf2onnx.convert.from_keras(
    model,
    input_signature=[tf.TensorSpec((None, 224, 224, 3), tf.float32, name="input")],
    opset=13,
    output_path=str(output_path),
)
```

| Property | Value |
|:---|:---|
| Format | ONNX Opset 13 |
| File size | 17.88 MB |
| Input tensor | `(batch, 224, 224, 3)` float32 |
| Output tensor | `(batch, 38)` float32 softmax probabilities |
| Provider | `CPUExecutionProvider` (auto-upgrades to CUDA if available) |
| TF-ONNX agreement | **100%** (100/100 val images, argmax identical) |
| Inference latency | ~12 ms per image on CPU |

---

## API

The FastAPI backend wraps the ONNX Runtime session and exposes a REST interface for integration with any frontend, mobile app, or external service.

### Endpoints

```
GET  /                    Health check + model metadata
GET  /classes             All 38 disease class definitions (index, crop, disease)
POST /predict             Single image upload → ranked JSON diagnosis
POST /predict/batch       Up to 10 images in one request
GET  /docs                Interactive Swagger UI
GET  /redoc               ReDoc documentation
```

### Example — `POST /predict`

**Request:**
```bash
curl -X POST "https://your-api-url/predict?top_k=5" \
  -H "accept: application/json" \
  -F "file=@tomato_leaf.jpg;type=image/jpeg"
```

**Response:**
```json
{
  "image_filename": "tomato_leaf.jpg",
  "inference_time_ms": 12.4,
  "model_version": "EfficientNetB0-ONNX-Opset13",
  "top_predictions": [
    {
      "rank": 1,
      "class_index": 29,
      "name": "Tomato Early Blight",
      "crop": "Tomato",
      "disease": "Early Blight",
      "confidence_percent": 100.0
    },
    {
      "rank": 2,
      "name": "Tomato Late Blight",
      "crop": "Tomato",
      "confidence_percent": 0.0
    }
  ]
}
```

---

## Repository Structure

```
Annadata/
│
├── data_prep.py                # Dual-mode data ingestion & 80/20 split
├── train_and_export.py         # 2-phase training + atomic class lock + ONNX export
├── predict.py                  # CLI inference client (onnxruntime, zero TF deps)
│
├── api/
│   ├── main.py                 # FastAPI REST backend (4 endpoints, Pydantic v2)
│   └── requirements_api.txt    # API-specific deps (fastapi, uvicorn, onnxruntime)
│
├── tests/
│   └── test_inference.py       # Schema, ONNX graph shape, regression tests
│
├── models/                     # (git-ignored) generated at training time
│   ├── annadata_production.onnx    # 17.88 MB production ONNX model
│   └── best_model.keras            # 23.96 MB Keras checkpoint
│
├── artifacts/                  # (git-ignored) generated at training time
│   └── disease_classes.json    # 6.6 KB · 38-class index-to-label map
│
├── Annadata_Colab.ipynb        # End-to-end training notebook (Google Colab)
├── Annadata_API_Server.ipynb   # API launch notebook with ngrok tunnel
│
├── requirements.txt            # Full training deps (TF 2.21, tf2onnx, etc.)
├── requirements_colab.txt      # Colab extras only (TF is pre-installed)
└── .gitignore
```

---

## 38 Disease Classes

The model classifies across 14 crop species and 26 distinct disease conditions (plus healthy variants):

| Index | Crop | Condition |
|:---:|:---|:---|
| 0 | Apple | Apple Scab |
| 1 | Apple | Black Rot |
| 2 | Apple | Cedar Apple Rust |
| 3 | Apple | Healthy |
| 4 | Blueberry | Healthy |
| 5 | Cherry | Powdery Mildew |
| 6 | Cherry | Healthy |
| 7 | Corn (Maize) | Cercospora Leaf Spot / Gray Leaf Spot |
| 8 | Corn (Maize) | Common Rust |
| 9 | Corn (Maize) | Northern Leaf Blight |
| 10 | Corn (Maize) | Healthy |
| 11 | Grape | Black Rot |
| 12 | Grape | Esca (Black Measles) |
| 13 | Grape | Leaf Blight (Isariopsis Leaf Spot) |
| 14 | Grape | Healthy |
| 15 | Orange | Haunglongbing (Citrus Greening) |
| 16 | Peach | Bacterial Spot |
| 17 | Peach | Healthy |
| 18 | Bell Pepper | Bacterial Spot |
| 19 | Bell Pepper | Healthy |
| 20 | Potato | Early Blight |
| 21 | Potato | Late Blight |
| 22 | Potato | Healthy |
| 23 | Raspberry | Healthy |
| 24 | Soybean | Healthy |
| 25 | Squash | Powdery Mildew |
| 26 | Strawberry | Leaf Scorch |
| 27 | Strawberry | Healthy |
| 28 | Tomato | Bacterial Spot |
| 29 | Tomato | Early Blight |
| 30 | Tomato | Late Blight |
| 31 | Tomato | Leaf Mold |
| 32 | Tomato | Septoria Leaf Spot |
| 33 | Tomato | Spider Mites / Two-spotted Spider Mite |
| 34 | Tomato | Target Spot |
| 35 | Tomato | Tomato Yellow Leaf Curl Virus |
| 36 | Tomato | Tomato Mosaic Virus |
| 37 | Tomato | Healthy |

---

## Quickstart

### Local CPU Smoke Test (< 60 seconds)

Verifies the full pipeline with synthetic data before spending GPU time:

```bash
# Clone
git clone https://github.com/dnyanu0909/Annadata.git
cd Annadata

# Install (Python 3.11+ recommended)
pip install -r requirements.txt

# Smoke test — synthetic data, 1 step, in-memory ONNX export, predict.py
python train_and_export.py --smoke-test

# Diagnose any leaf image
python predict.py --image test_leaf.jpg --top-k 5

# Run test suite
python -m unittest discover tests
```

### Full GPU Training on Google Colab

1. Open **[`Annadata_Colab.ipynb`](Annadata_Colab.ipynb)** in Google Colab
2. Set runtime: **Runtime → Change runtime type → T4 GPU**
3. Run all cells — the notebook handles data download, 2-phase training, ONNX export, and verification automatically
4. Download `models/annadata_production.onnx` + `artifacts/disease_classes.json` via Cell 8

```python
# Colab fast install (TF is pre-installed — only extras needed)
!pip install -q tf2onnx "onnx<=1.16.2" onnxruntime scikit-learn opencv-python-headless kaggle tqdm pillow

# Full training
!python train_and_export.py --epochs-warmup 5 --epochs-finetune 15 --batch-size 64
```

### Launch the REST API

```bash
# Install API deps (no TensorFlow needed)
pip install -r api/requirements_api.txt

# Start the server
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Interactive docs at:
# http://localhost:8000/docs
```

**Or on Colab with a public ngrok tunnel:** open **[`Annadata_API_Server.ipynb`](Annadata_API_Server.ipynb)**

---

## Results

| Metric | Value |
|:---|:---|
| **Final Validation Accuracy** | **99.48%** |
| **Final Validation Loss** | 0.0198 |
| **Final Training Accuracy** | 99.84% |
| **TF ↔ ONNX Prediction Agreement** | 100% (100/100 val images) |
| **ONNX Inference Latency** | ~12 ms / image (CPU) |
| **Model Size (ONNX)** | 17.88 MB |
| **Model Size (Keras checkpoint)** | 23.96 MB |
| **Training Dataset** | 43,429 images, 38 classes |
| **Validation Dataset** | 10,876 images |
| **Total Training Time** | ~40 min on Colab T4 GPU |

---

## Engineering Decisions & Post-Mortem

| Failure | Root Cause | Fix Applied |
|:---|:---|:---|
| `pip install tensorflow` hung 15+ min | Python 3.13 has no TF wheel → pip built from source | Use Colab's pre-installed TF; install only extras |
| `ValueError: Layer count mismatch 133 vs 5` | Wrapping `base_model` as a callable broke layer graph | Single continuous Functional API graph |
| ONNX exported random weights | Converted before loading trained weights | Direct in-memory export immediately after `model.fit()` |
| 99.98% confidence on Index 15 for every image | Mode collapse from high LR + frozen backbone | 2-phase schedule: Phase 1 LR=1e-3 head-only, Phase 2 LR=3e-5 fine-tune |
| Index 15 predicted for Tomato images | Generator folder order ≠ manual JSON dict | Atomic lock: write `disease_classes.json` from `train_ds.class_names` at init |
| `NameError` after kernel reset | Ephemeral Colab state | All state encapsulated in CLI scripts, nothing held in notebook memory |
| Double normalization (blank output) | OpenCV BGR + manual /255 + embedded Rescaling | Embed `Rescaling(1./255)` in graph; pass raw [0–255] RGB everywhere |
| `tf2onnx` crash with Keras 3 | TF 2.16+ defaults to Keras 3 graph tracing | `os.environ["TF_USE_LEGACY_KERAS"] = "1"` + import from `tf_keras` |

---

## License

MIT License — see [LICENSE](LICENSE)

---

<div align="center">
Built with ❤️ for Indian agriculture · <a href="https://github.com/dnyanu0909/Annadata">github.com/dnyanu0909/Annadata</a>
</div>
