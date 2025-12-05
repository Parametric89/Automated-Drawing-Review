# Test Set Prediction Report
## Dataset: rcp_bbox_v6 Test Set

---

## Executive Summary

**Model:** `fine-tune_v4.4_yolov8m_v5_2`  
**Test Images:** 5 fullsize technical drawings  
**Overall Detection Rate:** **85.5%** (638/746 panels)  
**Status:** ⚠️ Under-detection with 108 panels missed

---

## Model Configuration

### Training Details
- **Base Model:** YOLOv8m (medium)
- **Task:** Detection (bounding box only)
- **Training Dataset:** `rcp_bbox_v5_non-ROI` (augmented1280)
- **Classes:** 1 (panel only - NO TAGS)
- **Image Size:** 1280px
- **Batch Size:** 2
- **Epochs:** 200
- **Patience:** 50
- **Optimizer:** AdamW (lr=0.002, wd=0.0005)

### Prediction Settings
- **Tiling:** 1280px with 30% overlap
- **Confidence Threshold:** 0.25
- **IoU Threshold:** 0.60
- **Merge IoU:** 0.70 (for combining overlapping tile predictions)
- **Max Detections:** 3000 per image

---

## Detection Results

### Per-Image Breakdown

| Image   | Ground Truth Panels | Predicted Panels | Difference | Detection Rate |
|---------|--------------------:|-----------------:|-----------:|---------------:|
| x11_s1  | 342                 | 185              | -157       | 54.1%          |
| x12_s1  | 306                 | 329              | +23        | 107.5%         |
| x16_s2  | 40                  | 15               | -25        | 37.5%          |
| x32_s2  | 40                  | 88               | +48        | 220.0%         |
| x37_s1  | 18                  | 21               | +3         | 116.7%         |
| **TOTAL** | **746**           | **638**          | **-108**   | **85.5%**      |

### Performance Analysis

#### ✅ **Good Performance:**
- **x12_s1:** Near-perfect with slight over-detection (+23)
- **x37_s1:** Good match (+3 difference)

#### ⚠️ **Under-Detection Issues:**
- **x11_s1:** Significant miss rate (54.1%) - 157 panels missed
- **x16_s2:** Poor detection (37.5%) - 25 panels missed

#### ⚠️ **Over-Detection Issues:**
- **x32_s2:** Severe over-detection (220%) - 48 false positives
- Likely detecting panel sub-components as separate panels

---

## Critical Findings

### 🚨 Issue #1: No Tag Detection
- **Expected:** Model should detect panel tags (class 1)
- **Reality:** 0 tags detected across all 5 images
- **Root Cause:** Model was trained on **single-class dataset** (panels only)
- **Impact:** Cannot identify panel labels/tags for matching workflow

### ⚠️ Issue #2: Dataset Mismatch
- **Training Data:** `rcp_bbox_v5_non-ROI` - detection only, 1 class
- **Test Data:** `rcp_bbox_v6` - has segmentation ground truth, 2 classes
- **Impact:** Incomplete evaluation (tags not being tested)

### ⚠️ Issue #3: Inconsistent Performance
- Detection rate varies wildly: **37.5% to 220%**
- Suggests sensitivity to:
  - Drawing complexity/density
  - Panel size variations
  - Image quality/resolution differences

### ⚠️ Issue #4: Tile Merging Issues
- Large images (x11_s1, x12_s1) processed with 150 tiles each
- Over-detection in x32_s2 suggests duplicate detections not properly merged
- Under-detection in x11_s1 suggests panels split across tile boundaries

---

## Output Files Generated

### Predictions Directory
`datasets/rcp_bbox_v6/images/test/pred_fine-tune_v4.4_yolov8m_v5_2/pred/`

**Contents:**
- `x11_s1.txt` - YOLO label file (185 detections)
- `x11_s1.jpg` - Visualization with bounding boxes
- `x12_s1.txt` - YOLO label file (329 detections)
- `x12_s1.jpg` - Visualization with bounding boxes
- `x16_s2.txt` - YOLO label file (15 detections)
- `x16_s2.jpg` - Visualization with bounding boxes
- `x32_s2.txt` - YOLO label file (88 detections)
- `x32_s2.jpg` - Visualization with bounding boxes
- `x37_s1.txt` - YOLO label file (21 detections)
- `x37_s1.jpg` - Visualization with bounding boxes

---

## Recommendations

### Immediate Actions

1. **Train Dual-Class Model** ⚠️ **HIGH PRIORITY**
   - Use `rcp_bbox_v6` or `rcp_dual_seg_v3` for training
   - Include both panel (class 0) AND tag (class 1) detection
   - This is critical for your panel matching workflow

2. **Investigate x11_s1 Under-Detection**
   - 157 panels missed (46% recall)
   - Review tile boundaries and merge strategy
   - Consider larger tile overlap or different tiling approach

3. **Fix x32_s2 Over-Detection**
   - 48 false positives (220% detection rate)
   - Review confidence threshold (may be too low at 0.25)
   - Check if panel sub-components are being detected

4. **Optimize Tiling Strategy**
   - Test different overlap percentages (current: 30%)
   - Consider adaptive tiling based on image size
   - Evaluate `keep-inner-frac` parameter for edge handling

### Training Improvements

5. **Use Latest Dataset (v6)**
   - 129 training images with 5,495 tiles
   - Already split and tiled (ready to use)
   - Larger and more diverse than v5

6. **Consider Two-Stage Approach**
   - Stage 1: Train on panel detection only (high recall)
   - Stage 2: Fine-tune on dual-class (panels + tags)

7. **Address Data Quality**
   - Review ground truth for x32_s2 (only 40 panels labeled?)
   - Ensure consistent labeling across all images
   - Validate tile merging doesn't create duplicate labels

### Evaluation Improvements

8. **Run Full Validation**
   - Test on v6 validation set (23 images with 1,307 tiles)
   - Generate precision/recall curves
   - Calculate per-class metrics

9. **Visual Inspection**
   - Review generated overlay images in pred/ folder
   - Identify common failure patterns
   - Check for systematic errors in specific panel types

---

## Next Steps

Based on your memories and project plan, you should:

1. ✅ **Complete pending data approval** (157 images ready)
2. 🔄 **Train v7 model on rcp_bbox_v6** with proper dual-class support
3. 📊 **Re-evaluate with tag detection enabled**
4. 🎯 **Aim for >95% detection rate** as per project goals
5. 🚀 **Move to Phase 2** (Viewport+Tag detector) once Phase 1 is solid

---

## Technical Notes

### Model Architecture
- **Type:** YOLOv8m-detect (detection only, not segmentation)
- **Input Size:** 1280x1280px
- **Outputs:** Bounding boxes only (cx, cy, w, h)
- **Classes:** 1 (panel) - expandable to 2 with retraining

### Tiling Strategy Used
- **Purpose:** Handle large technical drawings (4000x6000+ pixels)
- **Tile Size:** 1280x1280px
- **Overlap:** 30% (384px)
- **Merge Method:** IoU-based with threshold 0.70
- **Mode:** minmax (keeps detections with min/max overlap)

### Ground Truth Format (v6)
- **Task:** Segmentation (polygons)
- **Classes:** 2 (panel=0, tag=1)
- **Format:** YOLO-seg with polygon coordinates
- **Note:** Model outputs don't match GT format (bbox vs polygon)

---

**Report Generated:** 2025-11-10  
**Model:** fine-tune_v4.4_yolov8m_v5_2  
**Dataset:** rcp_bbox_v6 test set (5 images, 746 panels)

