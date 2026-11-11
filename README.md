# Annadata 🌾: Autonomous Plant Disease Diagnostics Pipeline

Production-grade, end-to-end plant disease diagnostics pipeline built on **EfficientNetB0** (pretrained on ImageNet), locked class index mappings, embedded normalization, and zero-TensorFlow **ONNX Runtime** deployment for 38 agricultural disease categories.

---

## Architecture Overview

```text
Leaf Image (RGB) ──> [224x224x3]
  │
  ▼
layers.Rescaling(1./255)            <-- Embedded normalization inside model graph
  │
  ▼
EfficientNetB0 (ImageNet Backbone)
  │
  ▼
GlobalAveragePooling2D + BatchNorm + Dropout(0.30)
  │
  ▼
Dense(512, relu) + BatchNorm + Dropout(0.15)
  │
  ▼
Dense(38, Softmax)                  <-- Predictions mapped to artifacts/disease_classes.json
  │
  ▼
In-Memory Direct tf2onnx Export     <-- models/annadata_production.onnx (Opset 13)
  │
  ▼
predict.py (onnxruntime)            <-- Ultra-fast cross-platform inference
```

---

## Quickstart

### 1. Setup Virtual Environment (Python 3.12)

```bash
# Windows
py -3.12 -m venv .venv
.\.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify Pipeline in 60s (CPU Smoke Test)

Run the fast CPU dry-run to verify data generation, 1 warmup step, direct in-memory ONNX baking, and `predict.py` in under 60 seconds:

```bash
python train_and_export.py --smoke-test
```

Run test suite:
```bash
python -m unittest discover tests
```

### 3. Diagnose Any Leaf Image

```bash
python predict.py --image test_leaf.jpg --top-k 3
```

Example Output:
```text
========================================================
          🌾 ANNADATA AI PLANT DIAGNOSTICS 🌾
========================================================
 Analyzed Image: test_leaf.jpg
--------------------------------------------------------
 Rank  Diagnosis                            Confidence
--------------------------------------------------------
 ⭐ #1  Tomato Early Blight                       98.42%
    #2  Potato Early Blight                        1.15%
    #3  Tomato Target Spot                         0.28%
========================================================
```

---

## Full GPU Training (Google Colab / Cloud Instance)

For training on the 54,000+ PlantVillage images:

1. **Prepare Data**:
   - Dual-mode support (local path or Kaggle download):
   ```bash
   # From existing folder
   python data_prep.py --data-dir /path/to/plantvillage/color

   # Or automated Kaggle download (requires kaggle.json or credentials)
   export KAGGLE_USERNAME="your_username"
   export KAGGLE_KEY="your_api_key"
   python data_prep.py
   ```

2. **Execute 2-Phase Training**:
   ```bash
   python train_and_export.py --epochs-warmup 5 --epochs-finetune 15 --batch-size 64
   ```
   - **Phase 1 (Warmup)**: Trains classification head with LR = 1e-3 (5 epochs).
   - **Phase 2 (Fine-Tuning)**: Unfreezes top 30 layers with LR = 3e-5, EarlyStopping, and ReduceLROnPlateau.
   - Automatically saves `artifacts/disease_classes.json` and exports `models/annadata_production.onnx`.

---

## Solved Post-Mortem Failure Modes

1. **Python 3.12 Compatibility**: Resolved by pairing TensorFlow 2.16+ with `tf-keras~=2.16` and pinning `numpy<2.0.0`.
2. **Keras 3 / tf2onnx Export Crash**: Fixed with `os.environ["TF_USE_LEGACY_KERAS"] = "1"` and importing directly from `tf_keras`.
3. **Layer Count Mismatches**: Bypassed layer-by-layer flat weights loading by defining a single continuous Functional API graph.
4. **Uninitialized Weight Exports**: Prevented by directly exporting the trained in-memory model to ONNX.
5. **Single-Class Overfitting & Collapse**: Eliminated via controlled two-phase learning rate scheduling and EarlyStopping.
6. **Class-Index Desynchronization**: Guaranteed 100% alignment by atomically writing `disease_classes.json` from `train_ds.class_names`.
7. **Double Normalization**: Fixed by embedding `layers.Rescaling(1./255)` into the model graph and passing raw [0..255] RGB in `predict.py`.
