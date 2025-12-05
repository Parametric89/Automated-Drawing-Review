"""
Train YOLOv8s on v7_speed dataset with winning recipe.

Winning recipe:
- lr0=0.0015
- weight_decay=0.001
- cls=1.0
- kobj=0.85
- imgsz=1024
- batch=24
- epochs=50 (no early stopping)
"""

import sys
from pathlib import Path
from ultralytics import YOLO
import json
from datetime import datetime

# Add root to sys.path to find tools
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Import evaluation function from ablation script
from tools.run_v7_ablation_refined import evaluate_val_metrics, IOU_THRESHOLD, CONF_THRESHOLD

# Dataset
DATASET_YAML = "datasets/rcp_bbox_v7_speed/dataset_tiled1536.yaml"
INIT_WEIGHTS = "yolov8s.pt"  # Small model (not nano)
PROJECT = "runs/train"
RUN_NAME = "v7_speed_yolov8s_winning"

# Winning recipe
WINNING_CONFIG = {
    'imgsz': 1024,
    'batch': 24,
    'epochs': 50,
    'device': '0',
    'optimizer': 'AdamW',
    'cos_lr': True,
    'workers': 0,  # Windows compatibility
    'cache': 'disk',
    'resume': False,
    # Hyperparameters (winning recipe)
    'lr0': 0.0015,
    'weight_decay': 0.001,
    'cls': 1.0,
    'kobj': 0.85,
    # Augmentations (same as ablation)
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
    'patience': 1000,  # No early stopping
    'verbose': True,
}


def main():
    print("=" * 80)
    print("TRAINING YOLOv8s WITH WINNING RECIPE")
    print("=" * 80)
    print(f"Model: {INIT_WEIGHTS}")
    print(f"Dataset: {DATASET_YAML}")
    print(f"Run name: {RUN_NAME}")
    print()
    print("Hyperparameters:")
    print(f"  lr0: {WINNING_CONFIG['lr0']}")
    print(f"  weight_decay: {WINNING_CONFIG['weight_decay']}")
    print(f"  cls: {WINNING_CONFIG['cls']}")
    print(f"  kobj: {WINNING_CONFIG['kobj']}")
    print(f"  imgsz: {WINNING_CONFIG['imgsz']}")
    print(f"  batch: {WINNING_CONFIG['batch']}")
    print(f"  epochs: {WINNING_CONFIG['epochs']} (no early stopping)")
    print()
    
    # Load model
    model = YOLO(INIT_WEIGHTS)
    
    # Prepare training config
    train_kwargs = {
        'data': DATASET_YAML,
        'project': PROJECT,
        'name': RUN_NAME,
        **WINNING_CONFIG
    }
    
    print("Starting training...")
    print("=" * 80)
    
    # Train
    results = model.train(**train_kwargs)
    
    run_dir = Path(results.save_dir)
    print(f"\nTraining complete!")
    print(f"Run directory: {run_dir}")
    
    # Evaluate on validation set
    print("\n" + "=" * 80)
    print("FULL VALIDATION EVALUATION")
    print("=" * 80)
    
    val_metrics = evaluate_val_metrics(
        run_dir=run_dir,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=IOU_THRESHOLD
    )
    
    if val_metrics is None:
        print("ERROR: Could not evaluate model. Check weights path.")
        return
    
    print("\nResults:")
    print(f"  Recall@0.4        : {val_metrics['recall_at_04']:.3f}")
    print(f"  Worst-bin ({val_metrics['worst_bin_name']}) recall: {val_metrics['worst_bin_recall']:.3f}")
    print(f"  Small-panel recall: {val_metrics['small_panel_recall']:.3f}")
    print(f"  FP/image          : {val_metrics['fp_per_image']:.2f}")
    print(f"  TP: {val_metrics['total_tp']}, FP: {val_metrics['total_fp']}, FN: {val_metrics['total_fn']}, GT: {val_metrics['total_gt']}")
    
    # Save evaluation results
    eval_results = {
        'model': 'yolov8s',
        'config': WINNING_CONFIG,
        'run_dir': str(run_dir),
        'evaluation': {
            'conf_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD,
            'recall_at_04': val_metrics['recall_at_04'],
            'worst_bin_recall': val_metrics['worst_bin_recall'],
            'worst_bin_name': val_metrics['worst_bin_name'],
            'small_panel_recall': val_metrics['small_panel_recall'],
            'fp_per_image': val_metrics['fp_per_image'],
            'total_tp': val_metrics['total_tp'],
            'total_fp': val_metrics['total_fp'],
            'total_fn': val_metrics['total_fn'],
            'total_gt': val_metrics['total_gt'],
        },
        'timestamp': datetime.now().isoformat(),
    }
    
    eval_file = run_dir / "val_evaluation.json"
    with open(eval_file, 'w') as f:
        json.dump(eval_results, f, indent=2)
    
    print(f"\nEvaluation results saved to: {eval_file}")
    
    # Run confidence sweep
    print("\n" + "=" * 80)
    print("CONFIDENCE SWEEP")
    print("=" * 80)
    print("Running confidence sweep...")
    print("(This will be saved to conf_sweep_val.json)")
    print()
    
    # Import and run confidence sweep
    from tools.sweep_conf_val import DEFAULT_CONF_SWEEP as CONF_SWEEP
    
    sweep_results = []
    for conf in CONF_SWEEP:
        print(f"Evaluating at conf = {conf:.2f}...", end=" ")
        metrics = evaluate_val_metrics(
            run_dir=run_dir,
            conf_threshold=conf,
            iou_threshold=IOU_THRESHOLD
        )
        
        if metrics:
            sweep_results.append({
                'conf': conf,
                'recall_at_04': metrics['recall_at_04'],
                'worst_bin_recall': metrics['worst_bin_recall'],
                'worst_bin_name': metrics['worst_bin_name'],
                'fp_per_image': metrics['fp_per_image'],
                'small_panel_recall': metrics['small_panel_recall'],
                'total_tp': metrics['total_tp'],
                'total_fp': metrics['total_fp'],
                'total_fn': metrics['total_fn'],
                'total_gt': metrics['total_gt'],
            })
            print(f"Recall@0.4={metrics['recall_at_04']:.3f}, FP/img={metrics['fp_per_image']:.2f}")
        else:
            print("FAILED")
    
    # Save confidence sweep results
    sweep_file = run_dir / "conf_sweep_val.json"
    with open(sweep_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)
    
    print(f"\nConfidence sweep saved to: {sweep_file}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Model: YOLOv8s")
    print(f"Run directory: {run_dir}")
    print(f"\nValidation metrics (conf={CONF_THRESHOLD}):")
    print(f"  Recall@0.4        : {val_metrics['recall_at_04']:.3f}")
    print(f"  Worst-bin recall  : {val_metrics['worst_bin_recall']:.3f} ({val_metrics['worst_bin_name']})")
    print(f"  Small-panel recall: {val_metrics['small_panel_recall']:.3f}")
    print(f"  FP/image          : {val_metrics['fp_per_image']:.2f}")
    print(f"\nConfidence sweep: {len(sweep_results)} thresholds evaluated")
    print(f"  See {sweep_file} for full results")
    print("\nDone.")


if __name__ == "__main__":
    main()

