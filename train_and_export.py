"""
Annadata - Plant Disease Diagnostics Pipeline
train_and_export.py: Atomic class lock, embedded rescaling, two-phase training, and direct in-memory ONNX export.
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

# CRITICAL: Force TensorFlow to use Keras 2 tracing for tf2onnx compatibility
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import json
import argparse
from pathlib import Path
import numpy as np

import tensorflow as tf
import tf_keras as keras
from tf_keras import layers, Model
from tf_keras.applications import EfficientNetB0
import tf2onnx

# ----------------- CONSTANTS & CONFIGURATION -----------------
PROJECT_ROOT = Path("./")
DATA_DIR = PROJECT_ROOT / "data" / "plantvillage" / "splits"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = PROJECT_ROOT / "models"

IMG_SIZE = 224

CANONICAL_38_CLASSES = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry_(including_sour)___Powdery_mildew",
    "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
    "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight",
    "Corn_(maize)___healthy",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Raspberry___healthy",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___Leaf_scorch",
    "Strawberry___healthy",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
]

def parse_class_metadata(raw_name: str) -> tuple[str, str, str]:
    """Parse raw directory name into formatted (name, crop, disease)."""
    parts = raw_name.split("___")
    crop = parts[0].replace("_", " ").replace(",", "").strip()
    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "Healthy"
    name = f"{crop} {disease}".title()
    return name, crop, disease

def save_atomic_class_map(class_names: list[str], output_path: Path) -> dict:
    """Atomically persist the exact dataset class mapping to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    class_map = {}
    for idx, raw in enumerate(class_names):
        name, crop, disease = parse_class_metadata(raw)
        class_map[str(idx)] = {
            "name": name,
            "crop": crop,
            "disease": disease,
            "raw_class": raw
        }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(class_map, f, indent=4, ensure_ascii=False)
    print(f"🔒 Single Source-of-Truth class map ({len(class_map)} classes) saved to {output_path}")
    return class_map

def build_model(num_classes: int) -> tuple[Model, Model]:
    """Construct EfficientNetB0 architecture with embedded rescaling."""
    print("🧠 Constructing EfficientNetB0 architecture with embedded Rescaling(1./255)...")
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="input_image")
    
    # In-graph normalization: accepts raw [0-255] float32 pixels, normalizes to [0, 1]
    x = layers.Rescaling(1.0 / 255.0, name="embedded_rescaling")(inputs)
    
    base_model = EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=x
    )
    base_model.trainable = False  # Freeze for initial warmup
    
    x = layers.GlobalAveragePooling2D(name="gap")(base_model.output)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.30, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.15, name="dropout_2")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)
    
    full_model = Model(inputs=inputs, outputs=outputs, name="annadata_production")
    return full_model, base_model

def export_to_onnx(model: Model, output_path: Path):
    """Directly convert the in-memory trained Keras model to ONNX Opset 13."""
    print(f"\n⚡ Exporting trained in-memory model to ONNX at {output_path}...")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    spec = (tf.TensorSpec((None, IMG_SIZE, IMG_SIZE, 3), tf.float32, name="input_image"),)
    
    model_proto, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=spec,
        opset=13,
        output_path=str(output_path)
    )
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"🎉 Model successfully baked to ONNX: {output_path} ({file_size_mb:.2f} MB)")

def run_smoke_test():
    """Run a rapid end-to-end dry run on synthetic data in under 60 seconds on CPU."""
    print("\n" + "="*60)
    print("🧪 LAUNCHING RAPID CPU SMOKE TEST (< 60 SECONDS)")
    print("="*60)
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    num_classes = len(CANONICAL_38_CLASSES)
    json_path = ARTIFACTS_DIR / "disease_classes.json"
    save_atomic_class_map(CANONICAL_38_CLASSES, json_path)
    
    # Generate tiny synthetic batch
    print("📦 Synthesizing mock image batch (2 steps, 4 samples each)...")
    batch_size = 4
    dummy_images = np.random.uniform(0.0, 255.0, size=(batch_size * 2, IMG_SIZE, IMG_SIZE, 3)).astype(np.float32)
    dummy_labels = np.zeros((batch_size * 2, num_classes), dtype=np.float32)
    dummy_labels[:, 0] = 1.0  # Class 0
    
    train_ds = tf.data.Dataset.from_tensor_slices((dummy_images, dummy_labels)).batch(batch_size)
    val_ds = train_ds
    
    model, base_model = build_model(num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    print("🚀 Running 1 warmup step...")
    model.fit(train_ds, validation_data=val_ds, epochs=1, steps_per_epoch=2)
    
    onnx_path = MODELS_DIR / "annadata_production.onnx"
    export_to_onnx(model, onnx_path)
    
    # Save a mock test leaf image for verification
    sample_img_path = PROJECT_ROOT / "test_leaf.jpg"
    try:
        from PIL import Image
        mock_img = Image.fromarray(np.uint8(dummy_images[0]))
        mock_img.save(sample_img_path)
        print(f"📷 Generated test leaf image at {sample_img_path}")
    except Exception as e:
        print(f"Notice: Pillow save skipped: {e}")
        
    print("\n✅ Smoke test completed successfully!")
    print(f"   Artifacts generated:\n    - {json_path}\n    - {onnx_path}")

def run_training(args):
    """Execute full two-phase training pipeline on real dataset."""
    train_dir = Path(args.data_dir) / "train"
    val_dir = Path(args.data_dir) / "val"
    
    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset not found at {args.data_dir}. "
            "Please run `python data_prep.py` first to download/split data."
        )
        
    print(f"📂 Loading training dataset from {train_dir}...")
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=True,
        seed=42
    )
    
    print(f"📂 Loading validation dataset from {val_dir}...")
    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        label_mode="categorical",
        shuffle=False
    )
    
    # Atomic class locking from dataset ground truth
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print(f"✅ Found {num_classes} classes in dataset.")
    
    json_path = ARTIFACTS_DIR / "disease_classes.json"
    save_atomic_class_map(class_names, json_path)
    
    # Optimize data pipeline
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
    
    # Build Model
    model, base_model = build_model(num_classes)
    
    # Phase 1: Warmup
    print(f"\n🚀 Phase 1: Training Classification Head ({args.epochs_warmup} epochs, LR=1e-3)...")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_warmup
    )
    
    # Phase 2: Fine-Tuning
    print(f"\n🔓 Phase 2: Unfreezing Top 30 Layers ({args.epochs_finetune} epochs, LR=3e-5)...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
        
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=3e-5),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=2, min_lr=1e-7)
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs_finetune,
        callbacks=callbacks
    )
    
    # Direct In-Memory ONNX Export
    onnx_path = MODELS_DIR / "annadata_production.onnx"
    export_to_onnx(model, onnx_path)

def main():
    parser = argparse.ArgumentParser(description="Annadata Production Training & ONNX Export")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run rapid CPU smoke test to verify pipeline without waiting for full dataset."
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=str(DATA_DIR),
        help="Directory containing 'train' and 'val' subdirectories."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for training (default: 64)."
    )
    parser.add_argument(
        "--epochs-warmup",
        type=int,
        default=5,
        help="Warmup epochs for head training (default: 5)."
    )
    parser.add_argument(
        "--epochs-finetune",
        type=int,
        default=15,
        help="Fine-tuning epochs for top layers (default: 15)."
    )
    args = parser.parse_args()
    
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    if args.smoke_test:
        run_smoke_test()
    else:
        run_training(args)

if __name__ == "__main__":
    main()
