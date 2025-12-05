## Training Runs Log

Evaluation set: `datasets/rcp_dual_seg_v3/dataset_augmented1k.yaml` (val)

### Latest comparisons (same val set)

- Run: `v2_p2_yolov8s_seg` (best.pt evaluated on v3 val)
  - Box: mAP50 0.698, mAP50-95 0.437, P 0.813, R 0.595
  - Mask: mAP50 0.028, mAP50-95 0.00545, P 0.134, R 0.0582

- Run: `r3_yolov8s_seg` (best.pt evaluated on v3 val)
  - Box: mAP50 0.639, mAP50-95 0.300, P 0.729, R 0.567
  - Mask: mAP50 0.302, mAP50-95 0.0313, P 0.302, R 0.148

Notes:
- Box detection is higher in `v2_p2_yolov8s_seg`; mask segmentation is much higher in `r3_yolov8s_seg`.
- All metrics above are from the same evaluation set for fair comparison.

### Observations
- Mask AP improved substantially in r3, likely due to safer augmentation and polygon clipping.
- Box AP dropped versus v2_p2; may be due to data distribution shifts and negatives.

### Next actions
- Visual QA masks on hard cases (overlays on val): confirm tightness and edge behavior.
- Mine and fix label outliers (rank worst val; inspect top-10).
- Add/oversample small/edge panels; verify negatives don’t resemble tags.
- Short fine-tune from `r3` best (50–80 epochs, lr0 ~ 0.5×), same val.
- If box AP lags, lightly increase positive samples for tags/panels with balanced aug.

### Run entries template

- Run name: <name>
  - Data: <yaml>
  - Epochs/batch/imgsz: <e,b,s>
  - Init: <weights>
  - Val (same set):
    - Box: mAP50 <..>, mAP50-95 <..>, P <..>, R <..>
    - Mask: mAP50 <..>, mAP50-95 <..>, P <..>, R <..>
  - Key changes: <bullets>
  - Outcome: <bullets>
  - Next steps: <bullets>


