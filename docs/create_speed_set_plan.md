# Plan: Creating Speed-Set Dataset (v7) - Panel-Only with Hard Negatives

## Objective
Create a lightweight panel-only dataset (~1,400 train tiles with 25-30% hard negatives, 300 val tiles) for fast iteration with YOLOv8n. Focus on FP/image reduction through hard negatives. Keep tiles at 1536px, train with imgsz=1024.

## Key Changes (High-Leverage)
- **Panel-only:** Drop tag class (nc:1, names: ['panel']) to de-entangle supervision
- **Hard negatives:** 25-30% of TRAIN = tiles with no panels but panel-like clutter (frames, hatching, dims, legends)
- **Small-panel quota:** ≥25% of masks from smallest-area quartile (guards recall on tiny panels)
- **Stress split:** Rename "test" → "stress" (sheet-level hard holdout, only checked at end of iteration)

## Requirements
- **Train:** ~1,400 tiles (25-30% hard negatives included)
- **Val:** 300 tiles (stratified by density, no sheet overlap with train)
- **Stress:** 10-15% sheet-level holdout (hard examples, separate from tuning)
- **Tile size:** Keep at 1536px (from existing dataset)
- **Model input:** 1024px (set during training)
- **Overlap:** 20-25% overlap, 64-96px context

## Data Source
- **Source dataset:** `datasets/rcp_bbox_v6_reshuffled/`
  - Train: 4,225 tiles @ 1536px
  - Val: 841 tiles @ 1536px
- **Analysis file:** `analysis_v6_reshuffled.xlsx` (contains panel counts, density bins)

## Strategy

### Step 1: Load and Analyze Existing Data
1. Read `analysis_v6_reshuffled.xlsx` to get:
   - Panel counts per image
   - Density bin assignments (<=2, 3-5, 6-10, 11-15, >15 panels)
   - Sheet names (extract from tile filenames: `x10_s1_r4300_c10750.jpg` → sheet `x10`)
2. Identify hard examples:
   - High density bins (>15 panels, 11-15 panels)
   - Images with many small panels
   - Images from sheets with low recall (if available)

### Step 2: Select Stress Set (Sheet-Level Holdout)
1. Extract unique sheet names from all tiles
2. Identify "hard" sheets (highest density, worst contrast, many panels)
3. Select 10-15% of hardest sheets for stress set
4. Extract ALL tiles from selected sheets → stress set
5. Ensure stress set has ~100-200 tiles (10-15% of total)
6. **Keep stress separate** - only evaluate at end of iteration, not during tuning

### Step 3: Sample Train/Val Sets (with Hard Negatives)
1. **Remove stress sheets** from consideration
2. **Identify hard negatives:**
   - Source tiles with 0 panels from sheets NOT used for positive train tiles
   - Look for panel-like clutter: frames, hatching, dimensions, legends
   - Target: 25-30% of TRAIN should be hard negatives (~350-420 tiles)
3. **Stratified sampling by density bin (for positives):**
   - Target distribution (based on v6 analysis):
     - <=2 panels: ~15% (very_sparse)
     - 3-5 panels: ~20% (sparse)
     - 6-10 panels: ~35% (normal - most common)
     - 11-15 panels: ~20% (dense)
     - >15 panels: ~10% (very_dense)
4. **Small-panel quota:**
   - Calculate panel areas, identify smallest-area quartile
   - Ensure ≥25% of masks come from smallest-area quartile
5. **Train set (~1,400 tiles):**
   - Sample ~1,000-1,050 positive tiles (stratified by density + small-panel quota)
   - Add ~350-420 hard negatives (25-30%)
   - Ensure diversity across sheets
   - Total: ~1,400 tiles
6. **Val set (300 tiles):**
   - Sample proportionally from each density bin
   - Ensure no sheet overlap with train
   - Target: 300 tiles exactly
   - **No hard negatives in val** (keep clean for tuning)

### Step 4: Create Dataset Structure
```
datasets/rcp_bbox_v7_speed/
├── images/
│   ├── train/tiled1536/     (~1,400 tiles; 25-30% hard negatives)
│   ├── val/tiled1536/       (300 tiles; no sheet overlap with train)
│   └── stress/tiled1536/    (sheet-level hard holdout; 10-15% sheets)
├── labels/
│   ├── train/tiled1536/
│   ├── val/tiled1536/
│   └── stress/tiled1536/
└── dataset_tiled1536.yaml
```

### Step 5: Copy Selected Files
1. Copy selected images and labels from `rcp_bbox_v6_reshuffled/`
2. For hard negatives: Create empty label files (0 panels)
3. Maintain original filenames
4. Create dataset YAML file (panel-only, stress NOT in YAML):
   ```yaml
   path: datasets/rcp_bbox_v7_speed
   train: images/train/tiled1536
   val: images/val/tiled1536
   # stress kept separate, not in YAML
   nc: 1
   names: ['panel']
   ```

### Step 6: Validation & Reporting
1. Verify file counts match targets
2. Check density distribution matches targets
3. Verify hard negatives ratio (25-30% of train)
4. Verify small-panel quota (≥25% from smallest quartile)
5. Ensure no train/val/stress overlap (sheet-level)
6. Verify all images have corresponding labels (empty for hard negatives)
7. **Report metrics:**
   - Recall@0.4 (on val every epoch, stress only at end)
   - FP/image (on val every epoch, stress only at end)
   - Worst-bin recall (on val every epoch, stress only at end)

## Implementation Script: `create_speed_set.py`

### Functions Needed:
1. `load_excel_analysis()` - Read panel counts and density bins
2. `extract_sheet_names()` - Extract sheet names from tile filenames
3. `identify_hard_sheets()` - Find sheets with high density/low recall
4. `select_test_sheets()` - Select 10-15% of sheets for test
5. `stratified_sample()` - Sample tiles by density bin
6. `create_dataset_structure()` - Create directories
7. `copy_files()` - Copy selected images/labels
8. `create_yaml()` - Generate dataset config
9. `validate_dataset()` - Check counts and distributions

## Expected Outputs
- **Dataset:** `datasets/rcp_bbox_v7_speed/`
- **Summary report:** `speed_set_summary.txt` with:
  - Total tiles per split
  - Density distribution per split
  - Sheet-level breakdown
  - File paths

## Notes
- **Panel-only:** Drop tag class to isolate "when not to fire" problem (main failure mode)
- **Hard negatives:** Biggest lever on FP/image - tiles with no panels but panel-like clutter
- **Stress split:** Keep separate from tuning - only check at end of iteration to avoid overfitting to tail cases
- Keep tiles at 1536px (no re-tiling needed)
- Model will resize to 1024px during training
- Ensure diversity in train/val sets
- **Historical analogy:** Clean test range for tuning, rough-weather range only after design "wins"

