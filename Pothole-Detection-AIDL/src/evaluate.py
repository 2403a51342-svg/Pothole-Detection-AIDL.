"""
evaluate.py: Quantitative Model Evaluation & Benchmarking on Held-Out Test Set.
Calculates Precision, Recall, F1, mAP50, mAP50-95, latency (GPU/CPU), and confusion matrices.
"""

import os
import sys
import time
import argparse
import json
from pathlib import Path

def evaluate_model(weights_path, data_path='dataset/data.yaml', device='0', imgsz=640):
    """
    Evaluates a trained YOLO model checkpoint on test dataset partition.
    Returns metrics dictionary with precision, recall, mAP50, mAP50-95, latency.
    """
    print(f"\n=======================================================")
    print(f"       EVALUATING MODEL CHECKPOINT: {weights_path}")
    print(f"=======================================================")
    
    try:
        from ultralytics import YOLO
        model = YOLO(weights_path)
        
        # Benchmark validation metrics
        t0 = time.time()
        metrics = model.val(data=data_path, split='test', imgsz=imgsz, device=device, plots=True)
        eval_time = time.time() - t0
        
        # Extract metrics
        results = {
            'precision': float(metrics.box.mp),
            'recall': float(metrics.box.mr),
            'f1_score': float(2 * (metrics.box.mp * metrics.box.mr) / max(1e-6, (metrics.box.mp + metrics.box.mr))),
            'map50': float(metrics.box.map50),
            'map50_95': float(metrics.box.map),
            'inference_speed_ms': float(metrics.speed['inference']),
            'preprocess_speed_ms': float(metrics.speed['preprocess']),
            'postprocess_speed_ms': float(metrics.speed['postprocess'])
        }
        
        print("\n--- Quantitative Evaluation Results ---")
        for k, v in results.items():
            print(f"  * {k.replace('_', ' ').title():<22}: {v:.4f}")
        return results
        
    except ImportError:
        print("[!] Ultralytics not installed. Loading precomputed test benchmark metrics.")
        base_dir = Path(__file__).resolve().parent.parent
        metrics_file = base_dir / "models" / "best_model_info.json"
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                return data.get('evaluation_metrics', {})
        return {}

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate YOLO Pothole Detection Model")
    parser.add_argument('--weights', type=str, default='models/best_pothole_model.pt', help='Path to model weights')
    parser.add_argument('--data', type=str, default='dataset/data.yaml', help='Path to data.yaml')
    parser.add_argument('--device', type=str, default='cpu', help='Device (cuda/cpu)')
    parser.add_argument('--imgsz', type=int, default=640, help='Input resolution')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    evaluate_model(args.weights, args.data, args.device, args.imgsz)
