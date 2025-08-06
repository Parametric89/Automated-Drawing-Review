# Panel Detection & Validation System - Multi-Model Architecture

## 🚀 **Project Overview**
Advanced multi-model system to validate shop drawings against production drawings using specialized YOLO models for different detection tasks.

## 📊 **New Model Architecture**

| Model | Purpose | Classes | Status | Dataset |
|-------|---------|---------|--------|---------|
| **RCP-Dual-Seg** | Panel + Tag Detection | 2 (panel, panel_tag) | ✅ **ACTIVE** | 37 toProduction pages |
| **Viewport+Tag** | Viewport + Tag Detection | 2 (viewport, panel_tag) | ⏳ **PENDING** | 50 fromProduction pages |
| **Dim-Text** | Dimension Text Detection | 1 (dim_text) | ⏳ **PENDING** | Panel crops |
| **Arrow-Dir-CNN** | Arrow Direction Classification | 8 (directions) | ⏳ **PENDING** | Arrow patches |
| **Siamese Match** | Match/No-Match Classification | 2 (match, mismatch) | ⏳ **PENDING** | Paired crops |

---

## 🤖 **Model 1: RCP-Dual-Seg** ✅ **ACTIVE**

### **Purpose**
Detect both panel polygons and panel tags in RCP/elevation drawings.

### **Classes**
- **Class 0:** Panel polygon (mask)
- **Class 1:** Panel tag bounding box

### **Training Data & Strategy**
- **Source:** 37 toProduction sheets
- **Data Prep:** Panel-centric crops (1024px) → Smart Augmentation (3x data)
- **Training Strategy:** Two-Stage Transfer Learning
    1.  **Stage 1:** Fine-tune YOLOv8s-seg on `cropped1k` dataset to learn panel features.
    2.  **Stage 2:** Fine-tune the result on `augmented1k` dataset for robustness.
- **Workflow:** Managed via `workflow.py` script.

### **Current Status**
- ✅ **Data preparation pipeline:** Panel cropping and smart augmentation scripts are complete.
- ✅ **Training workflow:** Implemented a two-stage transfer learning process in `workflow.py`.
- 🔄 **Training:** Actively training and iterating on the model using the new workflow.
- ⏳ **Evaluation:** Pending final model training.

### **Technical Implementation**
```python
# Label format for RCP-Dual-Seg
# Class 0 (panel): polygon mask
"0 <cx> <cy> <width> <height> <x1> <y1> <x2> <y2> ..."

# Class 1 (panel_tag): bounding box
"1 <cx> <cy> <width> <height>"
```

---

## 🖼️ **Model 2: Viewport + Tag Detector** ⏳ **PENDING**

### **Purpose**
Detect viewport (main drawing window) and panel tags in fromProduction pages.

### **Classes**
- **Class 0:** Viewport bounding box
- **Class 1:** Panel tag bounding box

### **Training Data**
- **Source:** 50 fromProduction PDFs (single-panel sheets)
- **Configuration:** imgsz=960, epochs<40
- **Balance:** 1:1 viewport:tag ratio

### **Expected Performance**
- **Fast convergence:** <40 epochs
- **Balanced dataset:** Equal viewport and tag instances
- **High precision:** >90% for both classes

---

## 📏 **Model 3: Dim-Text Detector** ⏳ **PENDING**

### **Purpose**
Detect dimension text bounding boxes inside panel crops.

### **Classes**
- **Class 0:** Dimension text bounding box

### **Training Data**
- **Source:** Panel crops from Model 1
- **Quantity:** 150-300 dim-text boxes
- **Coverage:** ≈10% of panel crops
- **Optional:** Improves numeric mismatch recall

---

## 🧭 **Model 4: Arrow-Dir-CNN** ⏳ **PENDING**

### **Purpose**
Classify textile/arrow glyphs into 8 directions.

### **Classes**
- **8 directions:** 0°, 45°, 90°, 135°, 180°, 225°, 270°, 315°

### **Training Data**
- **Source:** Arrow patches cropped from panel crops
- **Quantity:** 1,000 patches (augmented with rotations)
- **Architecture:** 5-layer Keras CNN
- **Training time:** Minutes

---

## 🤝 **Model 5: Siamese Match/No-Match** ⏳ **PENDING**

### **Purpose**
Decide if panel crop and viewport crop agree on dimensions + direction + tag.

### **Classes**
- **Class 0:** Mismatch
- **Class 1:** Match

### **Training Data**
- **Source:** Paired crops from Models 1 & 2
- **Quantity:** 500 labeled pairs
- **Balance:** 50/50 via oversampling mismatches
- **Features:** Dimensions, direction, tag comparison

---

## 🔄 **Revised Data Flow**
toProduction page
└─ content-crop → tile
└─ RCP-Dual-Seg (panel + tag)
├─ panel mask ────────────┐
└─ tag box → OCR "P-105" ─┘
│
fromProduction page
└─ Viewport+Tag detector
├─ viewport crop ──────────┐
└─ tag box → OCR "P-105" ─┘
│
(panels paired by identical tag)
│
▼
Siamese Match-CNN → 0 / 1 (review flag)


---

## 📋 **Label Creation Checklist**

| Task | New Labels | Re-use Existing? |
|------|------------|------------------|
| Add tag boxes on 37 toProduction sheets | ≈1,200 (one per panel) | ✅ Extend existing files |
| Label viewport + tag on fromProduction pages | 50×2 boxes = 100 | ❌ New |
| Label dim_text on 10% of panel crops | 150-300 boxes | ❌ New |
| Extract & label arrow patches | 1,000 (augmented) | ✅ Auto-crop + QC |
| Curate pair labels (OK/mismatch) | 500 pairs | ❌ Simulate mismatches |

---

## 🏅 **Training Order (Minimal Friction)**

### **Phase 1: Foundational Model**
1.  **RCP-Dual-Seg (Model 1):**
    -   **Action:** Use `workflow.py` to execute the two-stage transfer learning.
    -   **Status:** ✅ **ACTIVE**
2.  **Viewport + Tag detector (Model 2):**
    -   **Action:** Label 50 pages and train.
    -   **Status:** ⏳ **PENDING**

### **Phase 2: Specialized Models**
3. **Generate panel crops** → label dim_text + extract arrows
4. **Train Models 3 & 4** (Dim-Text, Arrow-Dir)

### **Phase 3: Integration**
5. **Build 500 pair dataset** → train Siamese classifier (Model 5)

---

## 📁 **Updated File Structure**
ML review/
├── datasets/
│ ├── rcp_dual_seg/ # Model 1: Panel + Tag
│ │ ├── images/
│ │ ├── labels/
│ │ └── dataset.yaml
│ ├── viewport_tag/ # Model 2: Viewport + Tag
│ │ ├── images/
│ │ ├── labels/
│ │ └── dataset.yaml
│ ├── dim_text/ # Model 3: Dimension Text
│ │ ├── images/
│ │ ├── labels/
│ │ └── dataset.yaml
│ └── arrow_direction/ # Model 4: Arrow Direction
│ ├── images/
│ ├── labels/
│ └── dataset.yaml
├── models/
│ ├── rcp_dual_seg/ # Model 1 weights
│ ├── viewport_tag/ # Model 2 weights
│ ├── dim_text/ # Model 3 weights
│ ├── arrow_direction/ # Model 4 weights
│ └── siamese_match/ # Model 5 weights
├── scripts/
│ ├── train_rcp_dual_seg.py
│ ├── train_viewport_tag.py
│ ├── train_dim_text.py
│ ├── train_arrow_direction.py
│ ├── train_siamese_match.py
│ └── inference_pipeline.py
└── PROJECT_PLAN.md


---

## 🎯 **Success Metrics**

### **Model Performance Targets**
- **RCP-Dual-Seg:** >95% mAP for both classes
- **Viewport+Tag:** >95% precision for viewport detection
- **Dim-Text:** >95% recall for dimension text
- **Arrow-Dir:** >100% accuracy for direction classification
- **Siamese Match:** >95% precision on mismatch class

### **System Performance**
- **End-to-end processing:** <30 seconds per page
- **False positive rate:** <5% for review flags
- **Tag matching accuracy:** >95% OCR success rate

---

*Last Updated: 2025-07-31*
*Project Status: Multi-Model Architecture - Phase 1 (RCP-Dual-Seg)*
