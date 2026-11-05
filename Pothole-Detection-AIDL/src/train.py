"""
train.py: Multi-YOLO Custom Object Detection Training Script
Supports YOLOv5, YOLOv8, YOLOv9, YOLOv10, YOLO11, and YOLO12 architectures
Integrated with Ultralytics engine with automatic checkpointing and resume support.
"""

import os
import sys
import argparse
from pathlib import Path

# Verification registry of supported genuine YOLO architectures
GENUINE_MODELS = {
    'yolov5s': 'yolov5s.pt',
    'yolov5m': 'yolov5m.pt',
    'yolov8n': 'yolov8n.pt',
    'yolov8s': 'yolov8s.pt',
    'yolov8m': 'yolov8m.pt',
    'yolov9t': 'yolov9t.pt',
    'yolov9c': 'yolov9c.pt',
    'yolov10n': 'yolov10n.pt',
    'yolov10s': 'yolov10s.pt',
    'yolov10m': 'yolov10m.pt',
    'yolo11n': 'yolo11n.pt',
    'yolo11s': 'yolo11s.pt',
    'yolo11m': 'yolo11m.pt',
    'yolo12s': 'yolo12s.pt'
}

UNAVAILABLE_MODELS = ['yolo21', 'yolo22', 'yolo23', 'yolo24', 'yolo25', 'yolo26']

def train_yolo(model_name='yolo11s', data_path='dataset/data.yaml', epochs=100, batch_size=16, imgsz=640, device='0', resume=False, save_dir='results/train'):
    """
    Executes training using Ultralytics PyTorch framework.
    Handles GPU allocation, checkpoints, learning rate warmup, and validation.
    """
    clean_name = model_name.lower().replace('-', '').replace('_', '')
    
    if any(un in clean_name for un in UNAVAILABLE_MODELS):
        raise ValueError(f"Error: {model_name} is a non-existent/mythical architecture. Official YOLO series ends at YOLO12.")
        
    print(f"\n=======================================================")
    print(f"  INITIALIZING YOLO TRAINING: {model_name.upper()}")
    print(f"=======================================================")
    print(f"Data config : {data_path}")
    print(f"Target Epochs: {epochs}")
    print(f"Batch Size  : {batch_size}")
    print(f"Resolution  : {imgsz}x{imgsz}")
    print(f"Device      : {device}")
    print(f"Output Dir  : {save_dir}")
    print(f"=======================================================\n")
    
    try:
        from ultralytics import YOLO
        
        # Load base model weights or checkpoint
        weights = GENUINE_MODELS.get(clean_name, f"{clean_name}.pt")
        print(f"[+] Loading base architecture: {weights}")
        model = YOLO(weights)
        
        # Train model with early stopping patience and AdamW
        results = model.train(
            data=data_path,
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            resume=resume,
            patience=20,
            save=True,
            save_period=10,
            project=save_dir,
            name=clean_name,
            optimizer='AdamW',
            lr0=0.001,
            lrf=0.01,
            plots=True,
            val=True
        )
        print(f"[SUCCESS] Training complete for {model_name}!")
        return results
        
    except ImportError:
        print("[!] Ultralytics package not installed in current environment.")
        print("    For Google Colab or GPU servers, run: pip install ultralytics")
        print(f"    Simulating training pipeline config for: {model_name}")
        return None

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-YOLO Pothole Detection Training")
    parser.add_argument('--model', type=str, default='yolo11s', help='YOLO model name (e.g. yolo11s, yolov8s, yolov10s)')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image resolution')
    parser.add_argument('--device', type=str, default='0', help='CUDA device ID or cpu')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--save-dir', type=str, default='results/train', help='Directory to save results')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    train_yolo(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        resume=args.resume,
        save_dir=args.save_dir
    )
