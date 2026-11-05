"""
dataset_loader.py: Dataset Management, Download, Leakage Verification, and Preprocessing.
Supports Roboflow Universe, Kaggle RDD2022, and local YOLO format validation.
"""

import os
import sys
import glob
import hashlib
import random
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None

def compute_file_hash(filepath):
    """Computes SHA-256 hash to detect duplicates across splits and avoid data leakage."""
    hasher = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()

def verify_dataset_integrity(dataset_dir):
    """
    Checks that:
    1. Every image in train/val/test has a corresponding label file.
    2. No image hash appears in multiple splits (Strict Zero Data Leakage).
    3. Annotations are normalized within [0.0, 1.0].
    """
    splits = ['train', 'val', 'test']
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    split_hashes = {s: {} for s in splits}
    split_counts = {s: 0 for s in splits}
    
    print("=" * 60)
    print("      DATASET INTEGRITY & LEAKAGE VERIFICATION")
    print("=" * 60)
    
    for split in splits:
        img_dir = os.path.join(dataset_dir, 'images', split)
        lbl_dir = os.path.join(dataset_dir, 'labels', split)
        
        if not os.path.exists(img_dir):
            print(f"[-] Split directory missing: {img_dir}")
            continue
            
        img_files = [f for f in os.listdir(img_dir) if Path(f).suffix.lower() in image_extensions]
        split_counts[split] = len(img_files)
        
        for img_name in img_files:
            img_path = os.path.join(img_dir, img_name)
            img_hash = compute_file_hash(img_path)
            split_hashes[split][img_hash] = img_name
            
            # Check corresponding label
            lbl_name = Path(img_name).stem + ".txt"
            lbl_path = os.path.join(lbl_dir, lbl_name)
            if not os.path.exists(lbl_path):
                print(f"[!] Warning: Missing label for {split}/{img_name}")
            else:
                # Validate coordinates
                with open(lbl_path, 'r') as lf:
                    for line_no, line in enumerate(lf):
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            cls_id = int(parts[0])
                            coords = [float(x) for x in parts[1:5]]
                            for c in coords:
                                if c < 0.0 or c > 1.0:
                                    print(f"[!] Out-of-bounds coord in {lbl_name}: line {line_no+1} -> {coords}")

    # Check for leakage across splits
    leakage_found = False
    train_hashes = set(split_hashes['train'].keys())
    val_hashes = set(split_hashes['val'].keys())
    test_hashes = set(split_hashes['test'].keys())
    
    train_val_leak = train_hashes.intersection(val_hashes)
    train_test_leak = train_hashes.intersection(test_hashes)
    val_test_leak = val_hashes.intersection(test_hashes)
    
    if train_val_leak:
        leakage_found = True
        print(f"[ERROR] Data Leakage between Train & Val: {len(train_val_leak)} duplicate images!")
    if train_test_leak:
        leakage_found = True
        print(f"[ERROR] Data Leakage between Train & Test: {len(train_test_leak)} duplicate images!")
    if val_test_leak:
        leakage_found = True
        print(f"[ERROR] Data Leakage between Val & Test: {len(val_test_leak)} duplicate images!")
        
    if not leakage_found:
        print("[SUCCESS] Zero Data Leakage verified across Train, Validation, and Test partitions.")
        
    total_imgs = sum(split_counts.values())
    if total_imgs > 0:
        print(f"Total Images: {total_imgs}")
        for s in splits:
            pct = (split_counts[s] / total_imgs) * 100
            print(f"  - {s.capitalize()}: {split_counts[s]} images ({pct:.1f}%)")
            
    print("=" * 60)
    return not leakage_found

def download_roboflow_dataset(api_key, workspace, project, version, target_dir):
    """
    Helper function to download pothole dataset via Roboflow API.
    Executable in Google Colab or local Python with network.
    """
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location=target_dir)
        print(f"[+] Roboflow dataset downloaded successfully to {target_dir}")
        return dataset
    except ImportError:
        print("[-] roboflow package not installed. Install with: pip install roboflow")
    except Exception as e:
        print(f"[-] Roboflow download failed: {e}")

if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent.parent
    ds_dir = base_dir / "dataset"
    verify_dataset_integrity(str(ds_dir))
