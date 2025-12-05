# V7 Speed Ablation Series (Refined)

## Dataset
**Frozen as:** `datasets/rcp_bbox_v7_speed`
- Train: 1,404 tiles (27.7% hard negatives)
- Val: 317 tiles (20.5% hard negatives)
- Stress: 558 tiles (hard holdout, only check winners)

## Two-Phase Approach

### Phase A: Quick Sieve (8-12 epochs)
- Early stop by worst-bin recall (patience=5)
- Grid: LR × WD × cls_gain
  - **LR:** [0.001, 0.0015, 0.002]
  - **WD:** [0.0003, 0.0005, 0.001]
  - **cls_gain:** [1.0, 1.3, 1.6] (start at 1.0+, not 0.5)
- **Total:** 3 × 3 × 3 = 27 experiments

### Phase B: Promote Top-3 (30-50 epochs)
- Extend top-3 from Phase A to full training
- Then evaluate on stress set

## Fixed Config
- Model: YOLOv8n (fast iteration)
- imgsz: 1024px
- batch: 8
- optimizer: AdamW
- cos_lr: True
- **kobj:** 0.85 (objectness loss weight, parameter name is `kobj`)
- Augmentations: From v6 best practices

## Metrics Tracked (Consistent IoU=0.4)
- **Recall@0.4** (overall + worst density bin)
- **FP/image** (val)
- **Small-panel recall** (bottom-quartile area)
- TP/FP/FN counts (one-to-one Hungarian matching)

## Evaluation Details
- **IoU threshold:** 0.4 (consistent everywhere)
- **Conf threshold:** 0.4
- **Matching:** Hungarian (one-to-one, no double-matching)
- **Density bins:** <=2, 3-5, 6-10, 11-15, >15 panels

## Usage

### Run ablation series:
```bash
python run_v7_ablation_refined.py
```

This will:
1. Run Phase A (27 experiments, 8-12 epochs each)
2. Identify top-3 by worst-bin recall
3. Run Phase B (top-3, 30-50 epochs)
4. Save results to `v7_ablation_results.json` and `v7_ablation_summary.csv`

### Evaluate winners on stress set:
```bash
python evaluate_stress_set_refined.py <experiment_id>
```

Example:
```bash
python evaluate_stress_set_refined.py lr0p0015_wd0p0005_cls1p3
```

## Promotion Rule
Promote to Phase B if: **+≥5 pp in worst-bin recall at ≤ FP/image vs baseline**

## Reproducibility
- Fixed random seeds (data loader + augmentations)
- Deterministic CuDNN
- Frozen val/stress splits

## Notes
- Stress set only evaluated after Phase B completes
- All experiments use same dataset (frozen v7_speed)
- Results saved incrementally (can resume if interrupted)
- Per-epoch logging: 2-line summary (Recall@0.4, worst-bin, FP/img, small-panel)

