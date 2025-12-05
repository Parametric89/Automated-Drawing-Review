## Evaluation Summary

- **Run Directory**: `runs/train/v7_speed_yolov8s_winning2`
- **Timestamp**: 2025-11-13T13:22:49.549294
- **Model**: `yolov8s`
- **Configuration**:
  - `imgsz`: 1024
  - `batch`: 24
  - `epochs`: 50
  - `device`: 0
  - `optimizer`: AdamW
  - `cos_lr`: true
  - `workers`: 0
  - `cache`: disk
  - `resume`: false
  - `lr0`: 0.0015
  - `weight_decay`: 0.001
  - `cls`: 1.0
  - `kobj`: 0.85
  - `mosaic`: 0.0
  - `auto_augment`: null
  - `erasing`: 0.0
  - `hsv_h`: 0.015
  - `hsv_s`: 0.0
  - `hsv_v`: 0.3
  - `translate`: 0.05
  - `scale`: 0.2
  - `degrees`: 0.0
  - `flipud`: 0.5
  - `fliplr`: 0.5
  - `mixup`: 0.0
  - `copy_paste`: 0.0
  - `box`: 7.5
  - `dfl`: 1.5
  - `patience`: 1000
  - `verbose`: true

## Validation Metrics (conf=0.40, IoU=0.40)

- **Recall**: 0.854
- **Worst Bin Recall**: 0.798 (`6-10`)
- **Small Panel Recall**: 0.866
- **False Positives per Image**: 5.688
- **Totals**:
  - **True Positives**: 1422
  - **False Positives**: 381
  - **False Negatives**: 243
  - **Ground Truth**: 1665



