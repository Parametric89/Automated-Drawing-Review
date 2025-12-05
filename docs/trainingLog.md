# Training Log — Panel Detection System

## High-Level Evolution

- **v1–v5:** Segmentation → multiple tiling strategies → detection → ROI crops. Early explorations; inconsistent supervision.
- **v6:** First disciplined dataset shuffle with density-aware splitting. Defined the *panel-centric* evaluation metric and revealed the root cause: high FP rate from missing hard negatives + inconsistent labels.
- **v7 (current):** Purpose-built *speed dataset* + lightweight model to isolate real performance levers (geometry, label quality, augmentations).

---

## Evaluation Metric (Defined in v6)

**Core objective:** “Human-level panel finding” on CAD drawings.

- **Primary metric:**  
  - **Panel Recall@0.4 (PR@0.4):**  
    Fraction of GT panels that have **one matched prediction** with IoU ≥ 0.4 (Hungarian one-to-one matching, panels only).
- **Secondary constraints:**
  - **FP/image:** Average number of unmatched predictions per image after de-duplication.
- **Targets (val set, human benchmark ≈ 99.5% recall, ~0 FP/image):**
  - **PR@0.4 ≥ 99.2%**,  
  - **FP/image ≤ 0.10.**

These metrics are used consistently for v6, v7 and all upcoming runs.

---

# v6 — YOLOv8l (Historical Baseline)

## Dataset

- Source: `rcp_bbox_v6_reshuffled`
- Removed 8 obsolete drawings (non-hatched) → 124 train, 21 val, 5 test sheets  
- Shape-balanced: tall (28.6%), square (17.8%), wide (53.6%)
- Tiling: 1536px, 0.3 overlap  
- Train: 4,226 tiles  
- Val: 841 tiles  
- Classes: `nc=2` (panel, tag)

## Model & Training

- YOLOv8l (43.7M params)  
- imgsz=1536, batch=4  
- AdamW, lr0=1.5e-3, cosine LR  
- CAD-safe augmentations:  
  - scale=0.2, translate=0.05, flip=0.5  
  - hsv_h=0.015, hsv_v=0.3  
  - **No** mosaic/mixup/perspective/rotation

## Key Results

- **Run 1–2:** Low or unstable recall (≈0.33–0.36 as reported by YOLO’s all-class metrics; panel-only recall is roughly ×2 but still far from target).
- **Run 3:**  
  - Panel recall ≈ 0.68 at epoch 14  
  - Hungarian IoU=0.5 → density-weighted recall ≈ 92%  
  - Major issue: **FP explosion** (~17k predictions vs 2.1k GT)
- **Run 4:** Oversampling “normal” density bin hurt performance  
  → **Conclusion:** Need more and faster hyperparameter search *and* better data hygiene; FP issues driven by label inconsistency + missing hard negatives.

## Pivot Decision

Further YOLOv8l training looked inefficient until hyperparameters and data issues were addressed.  
Decision: shift to a compact, well-controlled v7 dataset for rapid iteration.

---

# v7_speed — Fast Iteration Dataset

## Purpose

A small, high-signal dataset enabling rapid ablations with **≤8-hour** end-to-end experiments.

## Dataset Construction (`rcp_bbox_v7_speed`)

- **Panel-only labels (tag class removed):**  
  All labels converted to pure detection (no masks) with `nc=1` (“panel”), with an eye towards adding a small utility in the hub to go *segmentation → detect* automatically when needed.
- **Curated hard negatives (25–30% of tiles):**  
  Reused all 212 “v6” negatives and mined ~300 additional tiles from fullsize sheets; some negatives are white/empty tiles, others contain panel-like clutter. 
- **Small-panel coverage:**  
  Panel area distribution monitored so the lowest-area quartile remains represented. During label overlays, some extremely tiny corner panels are clipped away by tiling; these are noted and will be addressed if they become a systematic failure mode.
- **Strict sheet separation:**  
  Train, val and stress splits have no sheet overlap.
- **Density-stratified sampling:**  
  Each split maintains a mix of ≤2, 3–5, 6–10, 11–15, and >15-panels per tile.

### Initial Counts (rcp_bbox_v7_speed)

- **Train:** 1,404 tiles (27.7% negatives)  
- **Val:** 317 tiles (20.5% negatives)  
- **Stress:** 558 tiles (difficult holdout sheets)

---

## Ablation Strategy

### Phase A — Quick Sieve

- imgsz=896, batch=48  
- 12 epochs, **no early stopping**  
- Grid:  
  - LR: {0.001, 0.0015, 0.002}  
  - WD: {3e-4, 5e-4, 1e-3}  
  - cls gain: {1.0, 1.3, 1.6}  
- Total: **27 runs**

### Phase B — Top-3 Promotion

- imgsz=1024, batch=48  
- 50 epochs, no early stop  
- Promotion rule (conceptual):  
  - ≥ +5 pp worst-bin Recall@0.4 vs Phase-A baseline  
  - AND FP/image ≤ baseline  
  - If no config meets this, promote top-3 by worst-bin recall.

### Fixed Hyperparameters (All Runs)

- kobj = 0.85  
- box = 7.5  
- dfl = 1.5  
- Hungarian matching @ IoU 0.4  
- Metrics: overall Recall@0.4, per-density-bin recall, FP/image, small-panel recall

---

## Tooling

- `create_speed_set.py` — dataset builder with quotas and sheet checks  
- `mine_hard_negatives*.py` — hard negative mining  
- `move_hard_negatives_to_val.py` — val rebalance  
- `run_v7_ablation_refined.py` — full two-phase grid search  
- `evaluate_stress_set_refined.py` — isolated stress-set evaluation  
- `sweep_conf_val.py` / `sweep_iou_v8s.py` — metric sweeps  
- `analyze_near_misses.py` — IoU band analysis + overlays

---

# 2025-11-14 — Cleanup & Re-evaluation

## 1. Dataset & Label Integrity Fixes

- Discovered one corrupt validation sheet with inconsistent labels (x17_s3.jpg).
- Removed all associated tiles (49) from all splits (no replacement, documented here for traceability).
- Cleaned mixed segmentation/detection labels.
- Regenerated unified panel-only detection labels.
- Val set adjusted:  
  - 317 → 268 images  
  - 1,665 → 1,358 ground-truth panels


### Updated Counts (rcp_bbox_v7_speed)

- **Train:** 1,404 tiles (27.7% negatives)  
- **Val:** 268 tiles 
- **Stress:** 558 tiles (difficult holdout sheets)

---

## 2. Updated Metric Tools

- All sweep scripts now support CLI inputs.  
- Added precision logging and JSON output.  
- Near-miss analysis added (IoU bands + overlay images).

---

## 3. Validation After Cleanup  

**Model:** `runs/train/v7_speed_yolov8s_winning2/weights/best.pt`

### Confidence Sweep (IoU=0.4)

- Recall ≥ **0.998** for conf ∈ [0.05, 0.40]  
- Precision: 0.963 → 0.985  
- FP/image: 0.19 → 0.075  
- Small-panel recall ≥ 0.988

### IoU Sweep (conf=0.40)

- Recall 0.997 from IoU=0.20 → 0.50  
- FP/image: 0.075 → 0.082  
- Approx. **4 true misses** in entire val set

### Interpretation

Label corruption and mixed formats caused the earlier geometry failures.  
Once fixed, YOLOv8s surpasses the human-level targets on the v7_speed validation split.

---

## 4. Near-Miss Analysis

**Before cleanup (old val):**

- Hits (IoU ≥ 0.4): 1,422  
- Near misses (0.2–0.4): 107  
- True misses (<0.2): 136  
- Near-misses dominated by sheet `x17_s3` (bad labels)

**After cleanup:**

- Near-miss cluster disappears.  
- Only ~4 true misses remain.  
- Geometry and GT alignment now consistent.

---

# Key Insights & Recent Ops

1. **Label quality was the primary bottleneck**, not model size or optimiser choice.  
2. After cleanup, **YOLOv8s meets the desired band** (≥0.99 recall, ≤0.1 FP/image) on the v7_speed val set.  
3. Hard negatives + consistent detection labels stabilise precision.  
4. Remaining errors are genuine coverage gaps, not alignment issues.

---

# Next Steps

- Optionally re-run Phase A/B ablations on the cleaned dataset (for confirmation, not rescue).  
- Evaluate the winning recipe on the stress set.  
- If needed, scale to YOLOv8s+ or YOLOv8m on the **full** dataset to test for additional gains under more variability.  
- Continue documenting sweeps, stress-set metrics, and near-miss overlays.

---

2025-11-15 — Fullsize Inference, Tiling Reproduction & New Direction
1. Objective

Bridge the gap between tile-level validation performance (≈99.8% recall, ≈0.08 FP/image) and fullsize sheet inference, where geometry drift, duplicates and inconsistent stitching were still occurring.

2. Work Completed (Past 48 Hours)
2.1 Reproduced Training-Time Tiling at Test-Time

Implemented --use-training-tiler inside predict_panels.py so fullsize inference uses exactly the same grid as training:

1536 px tiles

0.30 overlap

optional pre-scaling

Disabled the previous “Option 11” tiler, which used a different tiling geometry and caused misalignment.

2.2 Expanded Post-Merge Tools

Added multiple stitching strategies to analyse differences:

global NMS
IoU-cluster merging (avg)
min–max cluster merging
centre-distance merging
inner-window suppression (keep-inner-frac)

This enabled controlled sweeps across parameters to observe their effects.

2.3 Conducted a Large Parameter Sweep

54 configurations tested across:

merge-IoU

centre-merge-ratio

keep-inner-frac

mask filters

On one sheet (x11_s1), several settings achieved exact panel count (171) and visually clean geometry.

2.4 Key Finding

The “perfect” settings from the sweep were overfitted to that one sheet.
When reapplied to other drawings:

stitching inconsistencies reappeared

some boxes shrank

some duplicates persisted

FP behavior varied unpredictably

Conclusion:

The model is stable. The tiling + stitching pipeline is the true bottleneck.

---

## 16/11/2025

**Thoughts of today:** “Doing a resolution sweep with runs\train\v7_speed_yolov8s_winning2\weights\best.pt to see if the model can predict fullsize images with a lower res and still get close to my eval metrics. 

* created a curated set (datasets\rcp_bbox_v7_speed\images\curated) with 8 images: 5 from val and 3 from stress.
* Maybe set up a excel eval matrix to compare results.

- 8192, 7008 and 6528 px sweep OOMs (works on some images)
- 6144 px sweep works for the entire curated set at once
---
# results

gt_set	resolution	total_gt	matched_gt_0_4	total_fp	num_images
curated	4096	601	433	67	8
curated	6144	601	555	24	8
---

Matched (IoU≥0.4): 555 → Recall ≈ 92.3 %
FP total: 24 → FP/image = 3.0

That’s way off my target band (≥99.2 % recall, ≤0.1 FP/image).
Treating “6144 non-tiled” as a negative result.
---
## 18/11/2025

**Thoughts of today:**
1. Should I 100% align pre and post training tilers (without label-aware tiling)?
2. How does it work if i blend in fullsize images to the tiled training set?
3. ROI cropping again?(maybe 6144 is enough when zoomed in?).
4. train a model to zoom in on all panels prior to detecting each one (maybe 6144 is enough when zoomed in?).
5. Re-train on my entire dataset? Current best is using the v7 speed set. 


## 24/11/2025
**Thoughts of today:**
I've realized that the most beneficial path for me to take is to actually develop the project from start to end without further improving metric performance. 
Meaning I should move over to post inference tiling and OCR --> break down text --> categorize. After the pipeline is complete, I can further improve
performance, that is go back to 18/11/2025 thoughts and figure out next steps.  


1. Croppring out predicted bbox from original drawings, isolating the panels.

2. OCR one by one. Start with a single panel, set it up so that i can pick a random index amongst predicted indecies. 