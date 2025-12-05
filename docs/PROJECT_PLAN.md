# Automated Drawing Review System

## 🎯 Project Overview
The goal of this project is to automate the quality control of shop drawings against production data. The system identifies panels in drawings, extracts their IDs via OCR, compares them with a ground truth (data sheet), and visualizes any discrepancies.

---

## 🔄 Pipeline & Workflow

The project is divided into 4 distinct phases:

### ✅ Phase 1: Panel Detection (Current Focus)
Identify the location of all panels on a drawing.
*   **Input:** PDF or images of drawings.
*   **Technology:** YOLOv8 (Object Detection).
*   **Model:** `v7_speed_yolov8s_winning` (Panel-only, no tags/segmentation).
*   **Status:**
    *   Dataset `rcp_bbox_v7_speed` established (Detection only).
    *   High Recall@0.4 achieved (>99%).
    *   Tiled inference implemented for high-resolution drawings.
    *   Remaining: Fine-tune inference on full-scale drawings to eliminate residual stitching errors.

### ⏳ Phase 2: OCR (Optical Character Recognition)
Extract text (e.g., "P-105") from the detected panel bounding boxes.
*   **Input:** Cropped images of panels (from Phase 1).
*   **Strategy:** 
    1.  Run OCR (Tesseract/PaddleOCR/EasyOCR) on each panel crop.
    2.  Regex filtering to identify panel ID patterns.
    3.  Handle rotation (text may be rotated 90/180/270 degrees).

### ⏳ Phase 3: Data Reconciliation
Match extracted IDs and geometry against production data (Excel/Database).
*   **Input:** JSON/List of {Panel_ID, Bbox_W, Bbox_H} from Phase 2 + Production Data.
*   **Logic:**
    *   Does the panel exist in the production list?
    *   Do the dimensions match (Bbox aspect ratio vs. data)?
    *   Is the panel positioned correctly (if position data exists)?

### ⏳ Phase 4: Deviation Handling & Reporting
Visualize results for the user.
*   **Output:**
    *   Overlay on original drawing:
        *   🟢 Green box: Panel found and matched OK.
        *   🔴 Red box: Panel found, but data discrepancy or unknown ID.
        *   ⚠️ Yellow box: Panel in data, but not found on drawing.
    *   Excel/PDF report listing deviations.

---

## 🛠️ Technical Stack

### Core
*   **Language:** Python 3.10+
*   **ML Framework:** PyTorch, Ultralytics YOLOv8
*   **Image Processing:** OpenCV, Shapely (geometry)

### Folder Structure
*   `src/`: Source code for training and inference.
*   `tools/`: Helper scripts for dataset processing and analysis.
*   `config/`: YAML configurations for training and tiling.
*   `docs/`: Documentation and logs.

---

## 📈 Status and History

See `docs/trainingLog.md` for a detailed technical log of training experiments.

*   **Historical (v1-v6):** Attempted simultaneous segmentation and tag detection. Proved unstable with high false positive rates.
*   **Current (v7+):** "Panel-only" detection. We currently ignore tags and segmentation to achieve 100% recall on panel locations. OCR (Phase 2) will handle information extraction.
