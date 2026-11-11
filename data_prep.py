"""
Annadata - Plant Disease Diagnostics Pipeline
data_prep.py: Dual-mode dataset preparation (Local folder or automated Kaggle download)
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
import shutil
import random
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp'}

def setup_kaggle_credentials() -> bool:
    """Ensure Kaggle credentials exist either in ~/.kaggle/kaggle.json or from env."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        return True
    
    username = os.environ.get("KAGGLE_USERNAME")
    key = os.environ.get("KAGGLE_KEY")
    if username and key:
        kaggle_dir.mkdir(parents=True, exist_ok=True)
        with open(kaggle_json, "w", encoding="utf-8") as f:
            json.dump({"username": username, "key": key}, f)
        print(f"✅ Kaggle credentials written to {kaggle_json}")
        return True
    return False

def download_plantvillage_kaggle(target_dir: Path) -> Path:
    """Download PlantVillage dataset from Kaggle if not present."""
    target_dir.mkdir(parents=True, exist_ok=True)
    
    if not setup_kaggle_credentials():
        raise RuntimeError(
            "Kaggle credentials not found. Either place 'kaggle.json' in '~/.kaggle/' "
            "or set KAGGLE_USERNAME and KAGGLE_KEY environment variables."
        )
    
    print("📥 Downloading PlantVillage dataset from Kaggle (abdallahalidev/plantvillage-dataset)...")
    import kaggle
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "abdallahalidev/plantvillage-dataset",
        path=str(target_dir),
        unzip=True,
        quiet=False
    )
    print("✅ Download and extraction complete.")
    return target_dir

def find_class_directories(root_dir: Path) -> Dict[str, List[Path]]:
    """Recursively search for class directories containing image files."""
    class_images: Dict[str, List[Path]] = {}
    
    # Check if there is a 'color' subdirectory which is standard for PlantVillage
    color_candidates = list(root_dir.glob("**/color"))
    search_root = color_candidates[0] if color_candidates else root_dir
    
    for item in search_root.iterdir():
        if item.is_dir() and not item.name.startswith("."):
            images = [p for p in item.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
            if images:
                class_images[item.name] = images
                
    # If no images found directly, search one level deeper
    if not class_images:
        for sub in search_root.iterdir():
            if sub.is_dir():
                for item in sub.iterdir():
                    if item.is_dir() and not item.name.startswith("."):
                        images = [p for p in item.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
                        if images:
                            class_images[item.name] = images
                            
    return class_images

def split_and_copy_dataset(
    class_images: Dict[str, List[Path]],
    output_dir: Path,
    val_ratio: float = 0.20,
    seed: int = 42
) -> Tuple[int, int]:
    """Deterministically split class images into train and val directories."""
    random.seed(seed)
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    total_train = 0
    total_val = 0
    
    print(f"🔄 Splitting {len(class_images)} classes into Train ({(1-val_ratio)*100:.0f}%) / Val ({val_ratio*100:.0f}%)...")
    
    for class_name, images in sorted(class_images.items()):
        shuffled = list(images)
        random.shuffle(shuffled)
        
        split_idx = int(len(shuffled) * (1.0 - val_ratio))
        train_files = shuffled[:split_idx]
        val_files = shuffled[split_idx:]
        
        target_train_cls = train_dir / class_name
        target_val_cls = val_dir / class_name
        target_train_cls.mkdir(parents=True, exist_ok=True)
        target_val_cls.mkdir(parents=True, exist_ok=True)
        
        for f in train_files:
            dst = target_train_cls / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                
        for f in val_files:
            dst = target_val_cls / f.name
            if not dst.exists():
                shutil.copy2(f, dst)
                
        total_train += len(train_files)
        total_val += len(val_files)
        
    print(f"✅ Dataset split ready at: {output_dir}")
    print(f"   Train images: {total_train:,}")
    print(f"   Val images:   {total_val:,}")
    print(f"   Total classes: {len(class_images)}")
    return total_train, total_val

def main():
    parser = argparse.ArgumentParser(description="Annadata PlantVillage Data Ingestion & Splitting")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Path to existing PlantVillage images directory (e.g. 'color' folder)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/plantvillage/splits",
        help="Output directory where 'train' and 'val' subfolders will be created."
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Validation split ratio (default: 0.20)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic train/val split."
    )
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    
    # Check if already prepared
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    if train_dir.exists() and val_dir.exists() and any(train_dir.iterdir()):
        existing_classes = [d.name for d in train_dir.iterdir() if d.is_dir()]
        print(f"ℹ️ Found existing prepared dataset in {output_dir} with {len(existing_classes)} classes.")
        if len(existing_classes) >= 38:
            print("✅ Dataset already split and verified.")
            return

    if args.data_dir:
        input_path = Path(args.data_dir)
        if not input_path.exists():
            raise FileNotFoundError(f"Provided --data-dir does not exist: {input_path}")
        print(f"📂 Searching for images in {input_path}...")
        class_images = find_class_directories(input_path)
    else:
        # Fallback to automated Kaggle download
        raw_dir = Path("data/plantvillage/raw")
        print("📂 No --data-dir specified. Checking for Kaggle automated ingestion...")
        download_plantvillage_kaggle(raw_dir)
        class_images = find_class_directories(raw_dir)
        
    if not class_images:
        raise RuntimeError("No class folders with valid leaf images found!")
        
    print(f"🔍 Discovered {len(class_images)} candidate class folders.")
    split_and_copy_dataset(class_images, output_dir, val_ratio=args.val_ratio, seed=args.seed)

if __name__ == "__main__":
    main()
