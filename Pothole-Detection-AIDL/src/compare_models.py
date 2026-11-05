"""
compare_models.py: Multi-YOLO Performance Benchmarking, Comparative Analysis,
and Automated Best Model Selection.
"""

import os
import csv
import json
import math
from pathlib import Path

def load_metrics(csv_path):
    models = []
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            models.append({
                'Model': row['Model'],
                'Scale': row['Scale'],
                'Parameters_M': float(row['Parameters_M']),
                'GFLOPs': float(row['GFLOPs']),
                'Model_Size_MB': float(row['Model_Size_MB']),
                'Precision': float(row['Precision']),
                'Recall': float(row['Recall']),
                'F1_Score': float(row['F1_Score']),
                'mAP50': float(row['mAP50']),
                'mAP50_95': float(row['mAP50_95']),
                'Inference_GPU_ms': float(row['Inference_GPU_ms']),
                'Inference_CPU_ms': float(row['Inference_CPU_ms']),
                'Training_Time_hrs': float(row['Training_Time_hrs']),
                'Composite_Score': float(row.get('Composite_Score', 0.0))
            })
    return models

def compute_composite_scores(models, weights=None):
    """
    Computes a Pareto composite score S considering:
    - mAP50 (weight: 0.35)
    - mAP50_95 (weight: 0.25)
    - Latency efficiency (weight: 0.20)
    - Size efficiency (weight: 0.20)
    Normalized via min-max scaling across candidates.
    """
    if weights is None:
        weights = {'map50': 0.35, 'map95': 0.25, 'gpu_speed': 0.20, 'compactness': 0.20}
        
    min_lat = min(m['Inference_GPU_ms'] for m in models)
    max_lat = max(m['Inference_GPU_ms'] for m in models)
    min_sz = min(m['Model_Size_MB'] for m in models)
    max_sz = max(m['Model_Size_MB'] for m in models)
    
    for m in models:
        norm_map50 = m['mAP50']
        norm_map95 = m['mAP50_95']
        # Lower latency is better -> invert
        norm_lat = 1.0 - (m['Inference_GPU_ms'] - min_lat) / max(1e-5, (max_lat - min_lat))
        # Lower size is better -> invert
        norm_sz = 1.0 - (m['Model_Size_MB'] - min_sz) / max(1e-5, (max_sz - min_sz))
        
        score = (
            weights['map50'] * norm_map50 +
            weights['map95'] * norm_map95 +
            weights['gpu_speed'] * norm_lat +
            weights['compactness'] * norm_sz
        )
        m['Composite_Score'] = round(score, 4)
        
    return models

def select_best_models(models):
    best_precision = max(models, key=lambda x: x['Precision'])
    best_recall = max(models, key=lambda x: x['Recall'])
    best_f1 = max(models, key=lambda x: x['F1_Score'])
    best_map50 = max(models, key=lambda x: x['mAP50'])
    best_map95 = max(models, key=lambda x: x['mAP50_95'])
    fastest_gpu = min(models, key=lambda x: x['Inference_GPU_ms'])
    fastest_cpu = min(models, key=lambda x: x['Inference_CPU_ms'])
    smallest = min(models, key=lambda x: x['Model_Size_MB'])
    best_overall = max(models, key=lambda x: x['Composite_Score'])
    
    results = {
        "highest_precision": {"model": best_precision['Model'], "value": best_precision['Precision']},
        "highest_recall": {"model": best_recall['Model'], "value": best_recall['Recall']},
        "highest_f1": {"model": best_f1['Model'], "value": best_f1['F1_Score']},
        "highest_map50": {"model": best_map50['Model'], "value": best_map50['mAP50']},
        "highest_map95": {"model": best_map95['Model'], "value": best_map95['mAP50_95']},
        "fastest_gpu": {"model": fastest_gpu['Model'], "value_ms": fastest_gpu['Inference_GPU_ms']},
        "fastest_cpu": {"model": fastest_cpu['Model'], "value_ms": fastest_cpu['Inference_CPU_ms']},
        "smallest_model": {"model": smallest['Model'], "size_mb": smallest['Model_Size_MB']},
        "best_overall": {"model": best_overall['Model'], "composite_score": best_overall['Composite_Score']}
    }
    return results

def generate_svg_bar_chart(models, metric_key, title, unit, output_path, lower_is_better=False):
    """Generates a clean, standalone modern SVG bar chart without requiring external graphics packages."""
    width = 750
    height = 420
    margin_left = 130
    margin_right = 40
    margin_top = 60
    margin_bottom = 40
    
    chart_w = width - margin_left - margin_right
    chart_h = height - margin_top - margin_bottom
    
    values = [m[metric_key] for m in models]
    max_val = max(values) * 1.15 if max(values) > 0 else 1.0
    
    n = len(models)
    bar_height = max(12, int(chart_h / (n * 1.35)))
    gap = int(bar_height * 0.35)
    
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" style="background:#0F172A; font-family:sans-serif; border-radius:12px;">']
    svg.append(f'<text x="{width//2}" y="32" text-anchor="middle" fill="#F8FAFC" font-size="18" font-weight="bold">{title}</text>')
    
    y = margin_top
    for m in models:
        val = m[metric_key]
        bar_w = int((val / max_val) * chart_w)
        
        # Color coding: Highlight YOLO11s in vibrant cyan/amber
        if m['Model'] == 'YOLO11s':
            fill_color = '#38BDF8' # Sky cyan
            text_color = '#38BDF8'
            font_weight = "bold"
        elif 'YOLO11' in m['Model']:
            fill_color = '#0284C7'
            text_color = '#E2E8F0'
            font_weight = "normal"
        elif 'v10' in m['Model']:
            fill_color = '#818CF8' # Indigo
            text_color = '#E2E8F0'
            font_weight = "normal"
        elif 'v8' in m['Model']:
            fill_color = '#34D399' # Emerald
            text_color = '#E2E8F0'
            font_weight = "normal"
        else:
            fill_color = '#64748B' # Slate
            text_color = '#94A3B8'
            font_weight = "normal"
            
        # Model Label
        svg.append(f'<text x="{margin_left - 10}" y="{y + bar_height - 3}" text-anchor="end" fill="{text_color}" font-size="12" font-weight="{font_weight}">{m["Model"]}</text>')
        # Bar
        svg.append(f'<rect x="{margin_left}" y="{y}" width="{bar_w}" height="{bar_height}" fill="{fill_color}" rx="4"/>')
        # Value Label
        disp_val = f"{val:.3f}" if isinstance(val, float) and val < 1.0 else (f"{val:.1f}" if isinstance(val, float) else str(val))
        svg.append(f'<text x="{margin_left + bar_w + 8}" y="{y + bar_height - 3}" fill="#F1F5F9" font-size="11" font-weight="{font_weight}">{disp_val} {unit}</text>')
        
        y += bar_height + gap
        
    svg.append('</svg>')
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(svg))
    print(f"[+] Saved chart: {output_path}")

def main():
    base_dir = Path(__file__).resolve().parent.parent
    csv_path = base_dir / "results" / "model_metrics.csv"
    graphs_dir = base_dir / "results" / "graphs"
    os.makedirs(graphs_dir, exist_ok=True)
    
    models = load_metrics(str(csv_path))
    models = compute_composite_scores(models)
    best_results = select_best_models(models)
    
    print("\n" + "=" * 65)
    print("      AUTOMATIC BEST MODEL SELECTION & SUPERLATIVES")
    print("=" * 65)
    for cat, data in best_results.items():
        print(f"  * {cat.replace('_', ' ').title():<22}: {data}")
    print("=" * 65)
    
    # Generate standalone SVG comparison graphs
    generate_svg_bar_chart(models, 'mAP50', 'Mean Average Precision @ IoU 0.50 (mAP50)', '', graphs_dir / 'map50_comparison.svg')
    generate_svg_bar_chart(models, 'mAP50_95', 'Mean Average Precision @ IoU 0.50:0.95', '', graphs_dir / 'map50_95_comparison.svg')
    generate_svg_bar_chart(models, 'F1_Score', 'F1-Score Comparison Across YOLO Generations', '', graphs_dir / 'f1_score_comparison.svg')
    generate_svg_bar_chart(models, 'Inference_GPU_ms', 'Inference Latency on NVIDIA T4 GPU (Lower is Faster)', 'ms', graphs_dir / 'inference_gpu_comparison.svg')
    generate_svg_bar_chart(models, 'Model_Size_MB', 'Model Checkpoint Size (Lower is More Compact)', 'MB', graphs_dir / 'model_size_comparison.svg')
    generate_svg_bar_chart(models, 'Composite_Score', 'Overall Composite Selection Score (Pareto-Optimal)', 'pts', graphs_dir / 'composite_score_comparison.svg')

if __name__ == '__main__':
    main()
