"""
Refined ablation series for v7_speed dataset.

Two-phase approach:
- Phase A: Quick sieve (12 epochs, no early stopping)
  - imgsz=896, batch=48
  - Grid: LR × WD × cls_gain (27 experiments)
  
- Phase B: Final ablation (50 epochs, no early stopping)
  - imgsz=1024, batch=48
  - Top 3 promoted from Phase A

Metrics tracked:
- Recall@0.4 (overall + worst bin)
- FP/image
- Small-panel recall

PROMOTION RULE (explicit):
--------------------------
Promote to Phase B only if:
1. worst-bin Recall@0.4 improves by ≥5 pp over baseline (median)
2. AND FP/image is ≤ baseline FP/image (median)

If none qualify, promote top-3 by worst-bin recall anyway.

This keeps us honest when results are messy.
"""

import os
import json
import random
import numpy as np
from pathlib import Path
from datetime import datetime
import pandas as pd
from ultralytics import YOLO
import torch
import sys

# Add root directory to sys.path to allow importing from utils
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Fix OpenMP duplicate runtime on Windows
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')
os.environ.setdefault('OMP_NUM_THREADS', '1')

# Reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed_all(RANDOM_SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Dataset frozen as v7_speed
DATASET_YAML = "datasets/rcp_bbox_v7_speed/dataset_tiled1536.yaml"
INIT_WEIGHTS = "yolov8n.pt"
PROJECT = "runs/train"
BASE_NAME = "v7_speed_ablation"

# Phase A: Quick sieve grid
PHASE_A_LR = [0.001, 0.0015, 0.002]
PHASE_A_WD = [0.0003, 0.0005, 0.001]
PHASE_A_CLS_GAIN = [1.0, 1.3, 1.6]  # Start at 1.0+, not 0.5

# Phase A config (fast grid): imgsz=896, batch=48
PHASE_A_CONFIG = {
    'imgsz': 896,
    'batch': 48,
    'device': '0',
    'optimizer': 'AdamW',
    'cos_lr': True,
    'workers': 0,  # Windows compatibility (0 = main process only)
    'cache': 'disk',
    'resume': False,
    # Augmentations
    'mosaic': 0.0,
    'auto_augment': None,
    'erasing': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.0,
    'hsv_v': 0.3,
    'translate': 0.05,
    'scale': 0.2,
    'degrees': 0.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mixup': 0.0,
    'copy_paste': 0.0,
    # Loss weights
    'box': 7.5,
    'dfl': 1.5,
    'kobj': 0.85,  # Objectness loss weight (kobj, not obj)
}

# Phase B config (final ablation): imgsz=1024, batch from Phase A
# Using batch=48 (same as Phase A) for consistency and speed
PHASE_B_CONFIG = {
    'imgsz': 1024,
    'batch': 48,  # Same as Phase A for consistency
    'device': '0',
    'optimizer': 'AdamW',
    'cos_lr': True,
    'workers': 0,  # Windows compatibility (0 = main process only)
    'cache': 'disk',
    'resume': False,
    # Augmentations
    'mosaic': 0.0,
    'auto_augment': None,
    'erasing': 0.0,
    'hsv_h': 0.015,
    'hsv_s': 0.0,
    'hsv_v': 0.3,
    'translate': 0.05,
    'scale': 0.2,
    'degrees': 0.0,
    'flipud': 0.5,
    'fliplr': 0.5,
    'mixup': 0.0,
    'copy_paste': 0.0,
    # Loss weights
    'box': 7.5,
    'dfl': 1.5,
    'kobj': 0.85,  # Objectness loss weight (kobj, not obj)
}

# Backward compatibility
FIXED_CONFIG = PHASE_A_CONFIG  # Default to Phase A for evaluation functions

# Evaluation config
IOU_THRESHOLD = 0.4  # Consistent everywhere
CONF_THRESHOLD = 0.4

RESULTS_FILE = "v7_ablation_results.json"

from utils.matching import match_and_count

def hungarian_matching(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD):
    """
    Deprecated local matcher retained for compatibility. Prefer utils.matching.match_and_count.
    """
    tp, fp, fn = match_and_count(pred_boxes, gt_boxes, iou_threshold)
    # Reconstruct lists to preserve existing call sites if needed
    # matched_pairs length equals tp; indices are not required by downstream when using counts.
    matched_pairs = [(i, i) for i in range(tp)]
    unmatched_preds = list(range(fp))
    unmatched_gts = list(range(fn))
    return matched_pairs, unmatched_preds, unmatched_gts


def calculate_iou(box1, box2):
    """Calculate IoU between two normalized boxes."""
    x1_1 = box1['x_center'] - box1['width'] / 2
    y1_1 = box1['y_center'] - box1['height'] / 2
    x2_1 = box1['x_center'] + box1['width'] / 2
    y2_1 = box1['y_center'] + box1['height'] / 2
    
    x1_2 = box2['x_center'] - box2['width'] / 2
    y1_2 = box2['y_center'] - box2['height'] / 2
    x2_2 = box2['x_center'] + box2['width'] / 2
    y2_2 = box2['y_center'] + box2['height'] / 2
    
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    area1 = box1['width'] * box1['height']
    area2 = box2['width'] * box2['height']
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def load_panel_areas(label_path):
    """Load panel areas from label file."""
    if not label_path.exists():
        return []
    
    areas = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                width_norm = float(parts[3])
                height_norm = float(parts[4])
                areas.append(width_norm * height_norm)
    
    return areas


def get_density_bin(panel_count):
    """Get density bin for an image."""
    if panel_count <= 2:
        return '<=2'
    elif panel_count <= 5:
        return '3-5'
    elif panel_count <= 10:
        return '6-10'
    elif panel_count <= 15:
        return '11-15'
    else:
        return '>15'


def evaluate_val_metrics(run_dir, conf_threshold=CONF_THRESHOLD, iou_threshold=IOU_THRESHOLD):
    """Evaluate Recall@0.4, FP/image, worst-bin recall, small-panel recall."""
    weights_path = Path(run_dir) / "weights" / "best.pt"
    
    if not weights_path.exists():
        return None
    
    model = YOLO(str(weights_path))
    
    val_img_dir = Path("datasets/rcp_bbox_v7_speed/images/val/tiled1536")
    val_lbl_dir = Path("datasets/rcp_bbox_v7_speed/labels/val/tiled1536")
    val_images = list(val_img_dir.glob("*.jpg"))
    
    # Load density info from Excel if available
    density_info = {}
    try:
        df = pd.read_excel("analysis_v6_reshuffled.xlsx", sheet_name='Images')
        for _, row in df.iterrows():
            img_name = str(row.get('filename', '')).replace('.jpg', '').replace('.png', '')
            panel_count = row.get('num_panels', 0)
            if pd.isna(panel_count):
                panel_count = 0
            density_info[img_name] = {
                'panel_count': int(panel_count),
                'density_bin': get_density_bin(int(panel_count))
            }
    except:
        pass
    
    # Per-image metrics
    all_tp = 0
    all_fp = 0
    all_fn = 0
    all_gt = 0
    total_predictions = 0
    
    # Density bin tracking
    bin_metrics = {
        '<=2': {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
        '3-5': {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
        '6-10': {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
        '11-15': {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
        '>15': {'tp': 0, 'fp': 0, 'fn': 0, 'gt': 0},
    }
    
    # Small-panel tracking
    small_panel_areas = []  # Will collect areas from smallest quartile
    small_panel_tp = 0
    small_panel_fn = 0
    
    # First pass: collect all panel areas to find quartile threshold
    all_panel_areas = []
    for img_path in val_images:
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        areas = load_panel_areas(lbl_path)
        all_panel_areas.extend(areas)
    
    quartile_threshold = np.percentile(all_panel_areas, 25) if all_panel_areas else 0.0
    
    # Second pass: evaluate
    for img_path in val_images:
        # Load GT
        lbl_path = val_lbl_dir / f"{img_path.stem}.txt"
        gt_boxes = []
        gt_areas = []
        if lbl_path.exists():
            with open(lbl_path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append({
                            'x_center': float(parts[1]),
                            'y_center': float(parts[2]),
                            'width': float(parts[3]),
                            'height': float(parts[4])
                        })
                        area = float(parts[3]) * float(parts[4])
                        gt_areas.append(area)
        
        # Get predictions (use Phase A imgsz for evaluation consistency)
        pred_results = model.predict(
            source=str(img_path),
            imgsz=PHASE_A_CONFIG['imgsz'],  # Use Phase A imgsz for evaluation
            conf=conf_threshold,
            iou=iou_threshold,
            verbose=False
        )
        
        pred_boxes = []
        if pred_results[0].boxes is not None:
            boxes = pred_results[0].boxes
            for i in range(len(boxes)):
                box = boxes.xywhn[i].cpu().numpy()
                pred_boxes.append({
                    'x_center': float(box[0]),
                    'y_center': float(box[1]),
                    'width': float(box[2]),
                    'height': float(box[3])
                })
        
        total_predictions += len(pred_boxes)
        
        # One-to-one matching (shared implementation)
        tp, fp, fn = match_and_count(pred_boxes, gt_boxes, iou_threshold)
        all_tp += tp
        all_fp += fp
        all_fn += fn
        all_gt += len(gt_boxes)
        
        # Update density bin metrics
        img_name = img_path.stem
        density_bin = density_info.get(img_name, {}).get('density_bin', '6-10')
        if density_bin in bin_metrics:
            bin_metrics[density_bin]['tp'] += len(matched_pairs)
            bin_metrics[density_bin]['fp'] += len(unmatched_preds)
            bin_metrics[density_bin]['fn'] += len(unmatched_gts)
            bin_metrics[density_bin]['gt'] += len(gt_boxes)
        
        # Track small panels
        for gt_idx, area in enumerate(gt_areas):
            if area <= quartile_threshold:
                if gt_idx not in unmatched_gts:
                    small_panel_tp += 1
                else:
                    small_panel_fn += 1
    
    # Calculate metrics
    recall_overall = all_tp / all_gt if all_gt > 0 else 0.0
    fp_per_image = all_fp / len(val_images) if len(val_images) > 0 else 0.0  # Fixed: use all_fp not total_predictions
    
    # Worst-bin recall
    worst_bin_recall = 1.0
    worst_bin_name = None
    for bin_name, metrics in bin_metrics.items():
        if metrics['gt'] > 0:
            bin_recall = metrics['tp'] / (metrics['tp'] + metrics['fn']) if (metrics['tp'] + metrics['fn']) > 0 else 0.0
            if bin_recall < worst_bin_recall:
                worst_bin_recall = bin_recall
                worst_bin_name = bin_name
    
    # Small-panel recall
    small_panel_total = small_panel_tp + small_panel_fn
    small_panel_recall = small_panel_tp / small_panel_total if small_panel_total > 0 else 0.0
    
    return {
        'recall_at_04': recall_overall,
        'worst_bin_recall': worst_bin_recall,
        'worst_bin_name': worst_bin_name,
        'fp_per_image': fp_per_image,
        'small_panel_recall': small_panel_recall,
        'total_tp': all_tp,
        'total_fp': all_fp,
        'total_fn': all_fn,
        'total_gt': all_gt,
        'bin_metrics': bin_metrics,
    }


def run_training(config, phase='A'):
    """Run training for specified phase.
    
    Phase A: imgsz=896, batch=48, 12 epochs (no early stopping)
    Phase B: imgsz=1024, batch=48, 50 epochs (no early stopping)
    """
    epochs = 12 if phase == 'A' else 50
    
    # Use phase-specific base config
    base_config = PHASE_A_CONFIG if phase == 'A' else PHASE_B_CONFIG
    
    print(f"\n{'='*80}")
    print(f"TRAINING: {config['name']} (Phase {phase})")
    print(f"{'='*80}")
    print(f"imgsz: {base_config['imgsz']} (Phase {'A' if phase == 'A' else 'B'})")
    print(f"batch: {base_config['batch']}")
    print(f"LR0: {config['lr0']}, WD: {config['weight_decay']}, cls: {config['cls']}, kobj: {config['kobj']}")
    print(f"Epochs: {epochs} (no early stopping)")
    print()
    
    model = YOLO(config['init_weights'])
    
    train_kwargs = {
        'data': config['dataset_yaml'],
        'imgsz': base_config['imgsz'],  # Phase A: 896, Phase B: 1024
        'batch': base_config['batch'],  # Phase A: 48, Phase B: 48
        'epochs': epochs,
        'device': base_config['device'],
        'patience': 1000,  # Effectively disable early stopping
        'optimizer': base_config['optimizer'],
        'lr0': config['lr0'],
        'weight_decay': config['weight_decay'],
        'project': config['project'],
        'name': config['name'],
        'cos_lr': base_config['cos_lr'],
        'workers': base_config['workers'],
        'cache': base_config['cache'],
        'resume': base_config['resume'],
        # Augmentations
        'mosaic': base_config['mosaic'],
        'auto_augment': base_config['auto_augment'],
        'erasing': base_config['erasing'],
        'hsv_h': base_config['hsv_h'],
        'hsv_s': base_config['hsv_s'],
        'hsv_v': base_config['hsv_v'],
        'translate': base_config['translate'],
        'scale': base_config['scale'],
        'degrees': base_config['degrees'],
        'flipud': base_config['flipud'],
        'fliplr': base_config['fliplr'],
        'mixup': base_config['mixup'],
        'copy_paste': base_config['copy_paste'],
        # Loss weights
        'box': base_config['box'],
        'cls': config['cls'],
        'dfl': base_config['dfl'],
        'kobj': base_config['kobj'],
    }
    
    try:
        results = model.train(**train_kwargs)
        return results
    except Exception as e:
        print(f"ERROR during training: {e}")
        return None


def run_phase_a_sieve():
    """Phase A: Quick 12-epoch sieve (no early stopping)."""
    print("="*80)
    print("PHASE A: QUICK SIEVE (12 epochs, no early stopping)")
    print("="*80)
    print(f"Grid: LR={PHASE_A_LR}, WD={PHASE_A_WD}, cls_gain={PHASE_A_CLS_GAIN}")
    print(f"Total experiments: {len(PHASE_A_LR) * len(PHASE_A_WD) * len(PHASE_A_CLS_GAIN)}")
    print()
    
    results = []
    exp_num = 0
    
    for lr in PHASE_A_LR:
        for wd in PHASE_A_WD:
            for cls_gain in PHASE_A_CLS_GAIN:
                exp_num += 1
                exp_id = f"lr{lr}_wd{wd}_cls{cls_gain}".replace('.', 'p')
                
                print(f"\n{'='*80}")
                print(f"EXPERIMENT {exp_num}/{len(PHASE_A_LR) * len(PHASE_A_WD) * len(PHASE_A_CLS_GAIN)}")
                print(f"{'='*80}")
                
                config = {
                    'dataset_yaml': DATASET_YAML,
                    'init_weights': INIT_WEIGHTS,
                    'lr0': lr,
                    'weight_decay': wd,
                    'cls': cls_gain,
                    'kobj': 0.85,
                    'project': PROJECT,
                    'name': f"{BASE_NAME}_A_{exp_id}",
                }
                
                # Train
                train_results = run_training(config, phase='A')
                
                if train_results is None:
                    continue
                
                run_dir = train_results.save_dir
                
                # Evaluate
                print("Evaluating on val set...")
                val_metrics = evaluate_val_metrics(run_dir)
                
                if val_metrics:
                    # Build full training config for this experiment
                    full_config = PHASE_A_CONFIG.copy()
                    full_config.update({
                        'lr0': lr,
                        'weight_decay': wd,
                        'cls': cls_gain,
                        'kobj': 0.85,
                        'epochs': 12,
                        'patience': 1000,  # No early stopping
                        'data': DATASET_YAML,
                        'init_weights': INIT_WEIGHTS,
                    })
                    
                    result = {
                        'experiment_id': exp_id,
                        'exp_num': exp_num,
                        'phase': 'A',
                        'lr0': lr,
                        'weight_decay': wd,
                        'cls_gain': cls_gain,
                        'kobj_gain': 0.85,
                        'full_training_config': full_config,  # Complete config
                        'run_dir': str(run_dir),
                        'recall_at_04': val_metrics['recall_at_04'],
                        'worst_bin_recall': val_metrics['worst_bin_recall'],
                        'worst_bin_name': val_metrics['worst_bin_name'],
                        'fp_per_image': val_metrics['fp_per_image'],
                        'small_panel_recall': val_metrics['small_panel_recall'],
                        'total_tp': val_metrics['total_tp'],
                        'total_fp': val_metrics['total_fp'],
                        'total_fn': val_metrics['total_fn'],
                        'timestamp': datetime.now().isoformat(),
                    }
                    results.append(result)
                    
                    # 2-line summary
                    print(f"\n  Recall@0.4: {val_metrics['recall_at_04']:.3f} | Worst-bin ({val_metrics['worst_bin_name']}): {val_metrics['worst_bin_recall']:.3f} | FP/img: {val_metrics['fp_per_image']:.2f} | Small-panel: {val_metrics['small_panel_recall']:.3f}")
                    print(f"  TP: {val_metrics['total_tp']}, FP: {val_metrics['total_fp']}, FN: {val_metrics['total_fn']}, GT: {val_metrics['total_gt']}")
                    
                    # Save intermediate
                    with open(RESULTS_FILE, 'w') as f:
                        json.dump(results, f, indent=2)
    
    return results


def promote_top3_to_phase_b(phase_a_results):
    """Promote top-3 experiments to Phase B (50 epochs).
    
    PROMOTION RULE (explicit):
    -------------------------
    Promote to Phase B only if:
    1. worst-bin Recall@0.4 improves by ≥5 pp over baseline (median)
    2. AND FP/image is ≤ baseline FP/image (median)
    
    If none qualify, promote top-3 by worst-bin recall anyway.
    
    This keeps us honest when results are messy.
    """
    if len(phase_a_results) < 3:
        print("Not enough Phase A results to promote!")
        return []
    
    df = pd.DataFrame(phase_a_results)
    
    # Calculate baseline (median)
    baseline_worst_bin = df['worst_bin_recall'].median()
    baseline_fp = df['fp_per_image'].median()
    
    print(f"\n{'='*80}")
    print("PROMOTION RULE EVALUATION")
    print(f"{'='*80}")
    print(f"Baseline (median): Worst-bin recall={baseline_worst_bin:.3f}, FP/img={baseline_fp:.2f}")
    print()
    print("Promotion criteria:")
    print("  1. worst-bin Recall@0.4 improvement ≥ 5 pp over baseline")
    print("  2. FP/image ≤ baseline FP/image")
    print()
    
    # Calculate improvements
    df['worst_bin_improvement'] = df['worst_bin_recall'] - baseline_worst_bin
    df['fp_ok'] = df['fp_per_image'] <= baseline_fp
    
    # Sort by worst-bin recall (primary), then FP/image (secondary)
    df = df.sort_values(['worst_bin_recall', 'fp_per_image'], ascending=[False, True])
    
    # Apply promotion rule: +≥5 pp improvement AND FP ≤ baseline
    promoted = df[(df['worst_bin_improvement'] >= 0.05) & (df['fp_ok'] == True)]
    
    if len(promoted) == 0:
        print("No experiments meet promotion rule (≥5pp worst-bin recall improvement AND FP ≤ baseline)")
        print("Promoting top-3 by worst-bin recall anyway...")
        top3 = df.head(3)
    else:
        print(f"{len(promoted)} experiment(s) meet promotion rule")
        # Take top-3 from those that qualify
        top3 = promoted.head(3)
    
    print("\n" + "="*80)
    print("PHASE B: PROMOTING TOP-3")
    print("="*80)
    print("\nTop 3 experiments:")
    for idx, row in top3.iterrows():
        print(f"  {row['experiment_id']}: Worst-bin recall={row['worst_bin_recall']:.3f}, FP/img={row['fp_per_image']:.2f}")
    print()
    
    results = []
    
    for idx, row in top3.iterrows():
        exp_id = row['experiment_id']
        config = {
            'dataset_yaml': DATASET_YAML,
            'init_weights': INIT_WEIGHTS,
            'lr0': row['lr0'],
            'weight_decay': row['weight_decay'],
            'cls': row['cls_gain'],
            'kobj': row['kobj_gain'],
            'project': PROJECT,
            'name': f"{BASE_NAME}_B_{exp_id}",
        }
        
        # Train Phase B
        train_results = run_training(config, phase='B')
        
        if train_results is None:
            continue
        
        run_dir = train_results.save_dir
        
        # Evaluate
        print("Evaluating on val set...")
        val_metrics = evaluate_val_metrics(run_dir)
        
        if val_metrics:
            # Build full training config for this experiment
            full_config = PHASE_B_CONFIG.copy()
            full_config.update({
                'lr0': row['lr0'],
                'weight_decay': row['weight_decay'],
                'cls': row['cls_gain'],
                'kobj': row['kobj_gain'],
                'epochs': 50,
                'patience': 1000,  # No early stopping
                'data': DATASET_YAML,
                'init_weights': INIT_WEIGHTS,
            })
            
            result = {
                'experiment_id': exp_id,
                'phase': 'B',
                'lr0': row['lr0'],
                'weight_decay': row['weight_decay'],
                'cls_gain': row['cls_gain'],
                'kobj_gain': row['kobj_gain'],
                'full_training_config': full_config,  # Complete config
                'run_dir': str(run_dir),
                'recall_at_04': val_metrics['recall_at_04'],
                'worst_bin_recall': val_metrics['worst_bin_recall'],
                'worst_bin_name': val_metrics['worst_bin_name'],
                'fp_per_image': val_metrics['fp_per_image'],
                'small_panel_recall': val_metrics['small_panel_recall'],
                'total_tp': val_metrics['total_tp'],
                'total_fp': val_metrics['total_fp'],
                'total_fn': val_metrics['total_fn'],
                'timestamp': datetime.now().isoformat(),
            }
            results.append(result)
            
            print(f"\n  Recall@0.4: {val_metrics['recall_at_04']:.3f} | Worst-bin ({val_metrics['worst_bin_name']}): {val_metrics['worst_bin_recall']:.3f} | FP/img: {val_metrics['fp_per_image']:.2f} | Small-panel: {val_metrics['small_panel_recall']:.3f}")
    
    return results


def main():
    """Run two-phase ablation series."""
    print("="*80)
    print("V7 SPEED ABLATION SERIES (REFINED)")
    print("="*80)
    print(f"Dataset: {DATASET_YAML}")
    print(f"Model: {INIT_WEIGHTS}")
    print(f"IoU threshold: {IOU_THRESHOLD} (consistent everywhere)")
    print(f"Conf threshold: {CONF_THRESHOLD}")
    print()
    
    # Phase A: Quick sieve
    phase_a_results = run_phase_a_sieve()
    
    if not phase_a_results:
        print("Phase A failed - no results!")
        return
    
    # Save Phase A results
    with open(RESULTS_FILE, 'w') as f:
        json.dump(phase_a_results, f, indent=2)
    
    # Phase B: Promote top-3
    phase_b_results = promote_top3_to_phase_b(phase_a_results)
    
    # Combine results
    all_results = phase_a_results + phase_b_results
    
    # Final summary
    print("\n" + "="*80)
    print("ABLATION SERIES COMPLETE")
    print("="*80)
    
    if all_results:
        df = pd.DataFrame(all_results)
        df_phase_b = df[df['phase'] == 'B'].copy()
        
        if len(df_phase_b) > 0:
            df_phase_b = df_phase_b.sort_values('worst_bin_recall', ascending=False)
            
            print("\nPhase B Results (sorted by worst-bin recall):")
            print(df_phase_b[['experiment_id', 'lr0', 'weight_decay', 'cls_gain', 'kobj_gain',
                             'recall_at_04', 'worst_bin_recall', 'worst_bin_name', 
                             'fp_per_image', 'small_panel_recall']].to_string(index=False))
            
            # Save summary
            summary_file = "v7_ablation_summary.csv"
            df.to_csv(summary_file, index=False)
            print(f"\nFull results saved to: {summary_file}")
            print(f"JSON results saved to: {RESULTS_FILE}")
            
            print("\n" + "="*80)
            print("WINNER ANALYSIS")
            print("="*80)
            
            best = df_phase_b.iloc[0]
            print(f"\nBest experiment (worst-bin recall):")
            print(f"  ID: {best['experiment_id']}")
            print(f"  LR: {best['lr0']}, WD: {best['weight_decay']}, cls: {best['cls_gain']}, kobj: {best['kobj_gain']}")
            print(f"  Recall@0.4: {best['recall_at_04']:.3f}")
            print(f"  Worst-bin ({best['worst_bin_name']}): {best['worst_bin_recall']:.3f}")
            print(f"  FP/image: {best['fp_per_image']:.2f}")
            print(f"  Small-panel recall: {best['small_panel_recall']:.3f}")
            
            print("\n" + "="*80)
            print("NEXT STEP: Evaluate winners on stress set")
            print("="*80)
            print(f"Run: python evaluate_stress_set_refined.py {best['experiment_id']}")
        else:
            print("Phase B failed - no results!")


if __name__ == "__main__":
    main()

