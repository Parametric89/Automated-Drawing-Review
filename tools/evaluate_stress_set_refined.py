"""
Evaluate winning experiment on stress set (only run after recipe clearly wins).
Uses same metrics and matching as val evaluation (IoU=0.4, Hungarian matching).
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import json
import numpy as np

from utils.matching import match_and_count

IOU_THRESHOLD = 0.4
CONF_THRESHOLD = 0.4
PROJECT = "runs/train"
BASE_NAME = "v7_speed_ablation"
IMGSZ = 1024

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


def hungarian_matching(pred_boxes, gt_boxes, iou_threshold=IOU_THRESHOLD):
    """One-to-one matching using Hungarian algorithm (greedy approximation)."""
    if len(pred_boxes) == 0 or len(gt_boxes) == 0:
        return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))
    
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred in enumerate(pred_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred, gt)
    
    matched_pairs = []
    matched_preds = set()
    matched_gts = set()
    
    matches = []
    for i in range(len(pred_boxes)):
        for j in range(len(gt_boxes)):
            if iou_matrix[i, j] >= iou_threshold:
                matches.append((iou_matrix[i, j], i, j))
    
    matches.sort(reverse=True)
    
    for iou, i, j in matches:
        if i not in matched_preds and j not in matched_gts:
            matched_pairs.append((i, j))
            matched_preds.add(i)
            matched_gts.add(j)
    
    unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
    unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]
    
    return matched_pairs, unmatched_preds, unmatched_gts


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


def evaluate_stress(experiment_id):
    """Evaluate experiment on stress set."""
    print("="*80)
    print(f"EVALUATING STRESS SET: {experiment_id}")
    print("="*80)
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Conf threshold: {CONF_THRESHOLD}")
    print()
    
    # Find experiment run directory (try Phase B first, then Phase A)
    run_dir_b = Path(PROJECT) / f"{BASE_NAME}_B_{experiment_id}"
    run_dir_a = Path(PROJECT) / f"{BASE_NAME}_A_{experiment_id}"
    
    if (run_dir_b / "weights" / "best.pt").exists():
        weights_path = run_dir_b / "weights" / "best.pt"
        print(f"Using Phase B weights: {weights_path}")
    elif (run_dir_a / "weights" / "best.pt").exists():
        weights_path = run_dir_a / "weights" / "best.pt"
        print(f"Using Phase A weights: {weights_path}")
    else:
        print(f"ERROR: Best weights not found")
        print(f"  Checked: {run_dir_b / 'weights' / 'best.pt'}")
        print(f"  Checked: {run_dir_a / 'weights' / 'best.pt'}")
        return
    
    model = YOLO(str(weights_path))
    
    # Load stress set explicitly (not from YAML)
    stress_img_dir = Path("datasets/rcp_bbox_v7_speed/images/stress/tiled1536")
    stress_lbl_dir = Path("datasets/rcp_bbox_v7_speed/labels/stress/tiled1536")
    stress_images = sorted(list(stress_img_dir.glob("*.jpg")))
    
    print(f"Found {len(stress_images)} stress images")
    
    # Collect all panel areas for quartile calculation
    all_panel_areas = []
    for img_path in stress_images:
        lbl_path = stress_lbl_dir / f"{img_path.stem}.txt"
        areas = load_panel_areas(lbl_path)
        all_panel_areas.extend(areas)
    
    quartile_threshold = np.percentile(all_panel_areas, 25) if all_panel_areas else 0.0
    
    # Evaluate
    total_gt = 0
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_predictions = 0
    
    small_panel_tp = 0
    small_panel_fn = 0
    
    for img_path in stress_images:
        # Load GT
        lbl_path = stress_lbl_dir / f"{img_path.stem}.txt"
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
        
        total_gt += len(gt_boxes)
        
        # Get predictions
        pred_results = model.predict(
            source=str(img_path),
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
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
        tp, fp, fn = match_and_count(pred_boxes, gt_boxes, IOU_THRESHOLD)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        
        # Track small panels
        for gt_idx, area in enumerate(gt_areas):
            if area <= quartile_threshold:
                if gt_idx not in unmatched_gts:
                    small_panel_tp += 1
                else:
                    small_panel_fn += 1
    
    # Calculate metrics
    recall_overall = total_tp / total_gt if total_gt > 0 else 0.0
    fp_per_image = total_predictions / len(stress_images) if len(stress_images) > 0 else 0.0
    precision = total_tp / total_predictions if total_predictions > 0 else 0.0
    small_panel_total = small_panel_tp + small_panel_fn
    small_panel_recall = small_panel_tp / small_panel_total if small_panel_total > 0 else 0.0
    
    print("\n" + "="*80)
    print("STRESS SET RESULTS")
    print("="*80)
    print(f"Recall@0.4: {recall_overall:.3f}")
    print(f"FP/image: {fp_per_image:.2f}")
    print(f"Precision: {precision:.3f}")
    print(f"Small-panel recall: {small_panel_recall:.3f}")
    print(f"TP: {total_tp}, FP: {total_fp}, FN: {total_fn}, GT: {total_gt}")
    
    # Save results
    result = {
        'experiment_id': experiment_id,
        'stress_recall_at_04': recall_overall,
        'stress_fp_per_image': fp_per_image,
        'stress_precision': precision,
        'stress_small_panel_recall': small_panel_recall,
        'total_gt': total_gt,
        'total_tp': total_tp,
        'total_fp': total_fp,
        'total_fn': total_fn,
    }
    
    results_file = f"stress_results_{experiment_id}.json"
    with open(results_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python evaluate_stress_set_refined.py <experiment_id>")
        print("Example: python evaluate_stress_set_refined.py lr0p0015_wd0p0005_cls1p3")
        sys.exit(1)
    
    experiment_id = sys.argv[1]
    evaluate_stress(experiment_id)

