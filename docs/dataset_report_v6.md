# Dataset Insights Report

**Generated:** 2025-11-10 20:23:49

**Source:** `analysis_v6_reshuffled.xlsx`

---

## [+] Positive Characteristics

### Panel Density

- **Average panels per image: Train 6.9, Val 6.1**
  - Median: Train 5, Val 5

### Panel Coverage

- **Average panel coverage: Train 50.2%, Val 52.7%**
  - Panels occupy substantial portion of images

### Size Distribution

- **99.2% of panels are large (≥128px)**
  - Large panels are easier to detect

### Aspect Ratios

- **Balanced aspect ratio distribution**
  - No single category dominates

### Train-Val Comparison

- **Train-Val panel density similar (11.2% difference)**
  - Train: 6.9, Val: 6.1 panels/image

- **Train-Val panel sizes similar (18.1% difference)**

---

## [!] Challenging Characteristics

### Panel Density

- **High variance in panels per image (std: 5.7)**
  - Range: 1 to 35 panels

### Panel Coverage

- **High variance in coverage (std: 31.7%)**
  - Range: 0.2% to 183.9%

### Sparsity/Crowding

- **High proportion of very sparse images (26.1%)**
  - Model may struggle with sparse cases during training

- **Bimodal distribution: both sparse and dense cases common**
  - Dataset contains 26.1% very sparse AND 21.3% very dense images

### Variance

- **High coefficient of variation in panel counts (83%)**
  - Std dev: 5.7, Mean: 6.9

---

## [#] Key Statistics

### Dataset Size

- Training set: 4225 images, Validation set: 841 images
  - Train/Val ratio: 5.0:1

- Total panels: 34,288 (29,136 train, 5,152 val)

### Size Distribution

- Panel sizes: 0.0% tiny, 0.0% small, 0.8% medium, 99.2% large
  - Thresholds: <32px (tiny), 32-64px (small), 64-128px (medium), ≥128px (large)

- Average panel area: 171,862 px² (415×415 px equivalent)

### Aspect Ratios

- Aspect ratios: 32.0% tall, 29.0% square, 39.0% wide
  - Thresholds: <0.67 (tall), 0.67-1.5 (square), >1.5 (wide)

### Sparsity/Crowding

- 26.1% very sparse (≤2 panels), 21.3% dense (>10 panels)
  - 52.4% of images have ≤5 panels

### Edge Cases

- 26.2% of panels are at image edges
  - Moderate proportion of edge panels

### Variance

- 201 outlier images with extreme panel counts
  - Maximum: 35 panels in a single image

---

*End of Report*
