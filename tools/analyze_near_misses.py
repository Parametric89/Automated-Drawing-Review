"""
Quantify near misses for YOLOv8s run and generate visual diagnostics.

Steps:
1. Match predictions to GT with Hungarian algorithm at IoU >= 0.20.
2. Bin each matched GT by IoU:
     - [0.40, 1.00] -> hit
     - [0.20, 0.40) -> near miss
     - [0.00, 0.20) -> poor match (should be empty with 0.20 threshold)
   Unmatched GTs are counted as true misses (<0.20).
3. Aggregate counts and percentages for the entire validation set.
4. Visualize near-miss examples (GT in green, prediction in red) for manual inspection.
"""

from pathlib import Path
import json
from typing import Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from run_v7_ablation_refined import (
    hungarian_matching,
    calculate_iou,
    PHASE_A_CONFIG,
)

# ---------------------------------------------------------------------------
# Configuration

RUN_DIR = Path(r"runs/train/v7_speed_yolov8s_winning2")
CONF_THRESHOLD = 0.40          # fixed confidence threshold
MATCH_IOU_THRESHOLD = 0.20     # relaxed threshold for matching (near miss detection)
HIT_IOU_THRESHOLD = 0.40       # production recall threshold

DATASET_ROOT = Path("datasets/rcp_bbox_v7_speed")
VAL_IMG_DIR = DATASET_ROOT / "images/val/tiled1536"
VAL_LBL_DIR = DATASET_ROOT / "labels/val/tiled1536"

VIS_COUNT = 8  # number of images with near misses to visualize
VIS_OUTPUT_DIR = RUN_DIR / "near_miss_visuals"

# ---------------------------------------------------------------------------
# Helpers


def load_gt_boxes(label_path: Path) -> List[Dict[str, float]]:
    """Load YOLO boxes from label file."""
    boxes = []
    if label_path.exists():
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    boxes.append(
                        {
                            "x_center": float(parts[1]),
                            "y_center": float(parts[2]),
                            "width": float(parts[3]),
                            "height": float(parts[4]),
                        }
                    )
    return boxes


def predict_boxes(model: YOLO, image_path: Path) -> List[Dict[str, float]]:
    """Run YOLO prediction and return normalized boxes."""
    results = model.predict(
        source=str(image_path),
        imgsz=PHASE_A_CONFIG["imgsz"],  # keep consistent with evaluator
        conf=CONF_THRESHOLD,
        iou=HIT_IOU_THRESHOLD,  # NMS threshold (does not affect recall bins)
        verbose=False,
    )
    boxes_out = []
    if results and results[0].boxes is not None:
        boxes = results[0].boxes.xywhn.cpu().numpy()
        for box in boxes:
            boxes_out.append(
                {
                    "x_center": float(box[0]),
                    "y_center": float(box[1]),
                    "width": float(box[2]),
                    "height": float(box[3]),
                }
            )
    return boxes_out


def to_pixel_box(box: Dict[str, float], width: int, height: int) -> Tuple[int, int, int, int]:
    """Convert normalized YOLO box to pixel coordinates."""
    cx = box["x_center"] * width
    cy = box["y_center"] * height
    bw = box["width"] * width
    bh = box["height"] * height

    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    return x1, y1, x2, y2


def draw_near_miss(image: np.ndarray, gt_box: Dict[str, float], pred_box: Dict[str, float], iou: float) -> np.ndarray:
    """Overlay GT (green) and prediction (red) boxes with IoU annotation."""
    h, w = image.shape[:2]
    x1_gt, y1_gt, x2_gt, y2_gt = to_pixel_box(gt_box, w, h)
    x1_pr, y1_pr, x2_pr, y2_pr = to_pixel_box(pred_box, w, h)

    annotated = image.copy()
    cv2.rectangle(annotated, (x1_gt, y1_gt), (x2_gt, y2_gt), (0, 255, 0), 3)   # GT in green
    cv2.rectangle(annotated, (x1_pr, y1_pr), (x2_pr, y2_pr), (0, 0, 255), 3)   # Prediction in red

    label = f"IoU={iou:.2f}"
    text_origin = (min(x1_gt, x1_pr), max(0, min(y1_gt, y1_pr) - 10))
    cv2.putText(annotated, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
    return annotated


# ---------------------------------------------------------------------------
# Main analysis


def main() -> None:
    if not RUN_DIR.exists():
        raise FileNotFoundError(f"Run directory does not exist: {RUN_DIR}")

    model_path = RUN_DIR / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"Model weights not found: {model_path}")

    model = YOLO(str(model_path))

    val_images = sorted(VAL_IMG_DIR.glob("*.jpg"))
    if not val_images:
        raise FileNotFoundError(f"No validation images found in {VAL_IMG_DIR}")

    VIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total_gt = 0
    hit_count = 0
    near_miss_count = 0
    miss_count = 0

    per_image_stats = {}
    near_miss_examples = []

    print("=" * 80)
    print("NEAR MISS ANALYSIS @ conf=0.40, match IoU=0.20")
    print("=" * 80)
    print()

    for img_path in val_images:
        gt_boxes = load_gt_boxes(VAL_LBL_DIR / f"{img_path.stem}.txt")
        pred_boxes = predict_boxes(model, img_path)

        total_gt += len(gt_boxes)

        if not gt_boxes:
            continue

        matched_pairs, _, unmatched_gts = hungarian_matching(
            pred_boxes, gt_boxes, MATCH_IOU_THRESHOLD
        )

        image_hits = 0
        image_near_misses = []

        for pred_idx, gt_idx in matched_pairs:
            pred_box = pred_boxes[pred_idx]
            gt_box = gt_boxes[gt_idx]
            iou = calculate_iou(pred_box, gt_box)

            if iou >= HIT_IOU_THRESHOLD:
                hit_count += 1
                image_hits += 1
            elif MATCH_IOU_THRESHOLD <= iou < HIT_IOU_THRESHOLD:
                near_miss_count += 1
                image_near_misses.append(
                    {
                        "image": img_path,
                        "iou": iou,
                        "pred_box": pred_box,
                        "gt_box": gt_box,
                    }
                )
            else:
                # Should rarely happen because match threshold is 0.20
                miss_count += 1

        # Unmatched GTs are misses
        miss_count += len(unmatched_gts)

        per_image_stats[img_path] = {
            "gt": len(gt_boxes),
            "hits": image_hits,
            "near_misses": len(image_near_misses),
            "misses": len(unmatched_gts),
        }

        if image_near_misses:
            near_miss_examples.append((img_path, image_near_misses))

    # Aggregate percentages
    hit_pct = (hit_count / total_gt * 100) if total_gt else 0.0
    near_miss_pct = (near_miss_count / total_gt * 100) if total_gt else 0.0
    miss_pct = (miss_count / total_gt * 100) if total_gt else 0.0

    summary = {
        "total_gt": total_gt,
        "hit_count": hit_count,
        "near_miss_count": near_miss_count,
        "miss_count": miss_count,
        "hit_pct": hit_pct,
        "near_miss_pct": near_miss_pct,
        "miss_pct": miss_pct,
        "per_image_stats": {
            str(path): stats for path, stats in per_image_stats.items()
        },
    }

    out_json = RUN_DIR / "near_miss_analysis.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Summary:")
    print(f"  Total GT panels     : {total_gt}")
    print(f"  Hits (IoU >= 0.40)  : {hit_count} ({hit_pct:.1f}%)")
    print(f"  Near misses [0.2,0.4): {near_miss_count} ({near_miss_pct:.1f}%)")
    print(f"  Misses  (< 0.20)    : {miss_count} ({miss_pct:.1f}%)")
    print()
    print(f"Detailed stats saved to: {out_json}")

    # Visualize near misses
    if near_miss_examples:
        print()
        print("Generating near-miss visualizations...")
        near_miss_examples.sort(key=lambda item: len(item[1]), reverse=True)
        selected = near_miss_examples[:VIS_COUNT]

        for img_path, mismatches in selected:
            image = cv2.imread(str(img_path))
            if image is None:
                continue

            annotated = image.copy()
            for example in mismatches:
                annotated = draw_near_miss(annotated, example["gt_box"], example["pred_box"], example["iou"])

            out_path = VIS_OUTPUT_DIR / f"{img_path.stem}_near_miss.jpg"
            cv2.imwrite(str(out_path), annotated)
            print(f"  Saved: {out_path.name} ({len(mismatches)} near misses)")

        print("Visualization complete.")
    else:
        print("No near misses found for visualization.")

    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("Inspect the saved images in 'near_miss_visuals/' to spot systematic biases:")
    print("  - Are predictions consistently too short/tall/shifted?")
    print("  - Are GT boxes inconsistent in tightness across sheets?")
    print("Use these clues to decide whether to tackle label geometry or model coverage first.")


if __name__ == "__main__":
    main()

