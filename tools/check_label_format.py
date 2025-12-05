"""
Check if YOLO labels have width/height swapped or coordinate issues.

Compares how Ultralytics visualizes labels vs predictions to identify format issues.
"""

from pathlib import Path
from ultralytics import YOLO
import cv2
import numpy as np

# Paths
RUN_DIR = Path(r"runs/train/v7_speed_yolov8s_winning2")
DATASET_YAML = "datasets/rcp_bbox_v7_speed/dataset_tiled1536.yaml"
VAL_IMG_DIR = Path("datasets/rcp_bbox_v7_speed/images/val/tiled1536")
VAL_LBL_DIR = Path("datasets/rcp_bbox_v7_speed/labels/val/tiled1536")

def parse_yolo_label(label_path):
    """Parse YOLO label file."""
    boxes = []
    if not label_path.exists():
        return boxes
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls = int(parts[0])
                cx = float(parts[1])
                cy = float(parts[2])
                w = float(parts[3])
                h = float(parts[4])
                boxes.append({
                    'class': cls,
                    'cx': cx,
                    'cy': cy,
                    'w': w,
                    'h': h
                })
    return boxes

def draw_box_on_image(img, box, color=(0, 255, 0), thickness=2):
    """Draw normalized YOLO box on image."""
    h, w = img.shape[:2]
    cx = box['cx'] * w
    cy = box['cy'] * h
    width = box['w'] * w
    height = box['h'] * h
    
    x1 = int(cx - width / 2)
    y1 = int(cy - height / 2)
    x2 = int(cx + width / 2)
    y2 = int(cy + height / 2)
    
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return img

def check_label_file(img_name):
    """Check a specific label file and visualize it."""
    img_path = VAL_IMG_DIR / f"{img_name}.jpg"
    lbl_path = VAL_LBL_DIR / f"{img_name}.txt"
    
    if not img_path.exists() or not lbl_path.exists():
        print(f"Files not found: {img_name}")
        return None
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not load image: {img_path}")
        return None
    
    h, w = img.shape[:2]
    print(f"\nImage: {img_name}")
    print(f"  Image size: {w}x{h}")
    
    # Parse labels
    boxes = parse_yolo_label(lbl_path)
    print(f"  Found {len(boxes)} labels")
    
    # Show first few boxes
    for i, box in enumerate(boxes[:5]):
        print(f"  Box {i+1}: cx={box['cx']:.4f}, cy={box['cy']:.4f}, w={box['w']:.4f}, h={box['h']:.4f}")
        # Calculate pixel coordinates
        px_cx = box['cx'] * w
        px_cy = box['cy'] * h
        px_w = box['w'] * w
        px_h = box['h'] * h
        print(f"    -> Pixel: cx={px_cx:.1f}, cy={px_cy:.1f}, w={px_w:.1f}, h={px_h:.1f}")
        
        # Check if box seems reasonable
        if px_w > px_h:
            aspect = px_w / px_h
            print(f"    -> Horizontal box (w/h={aspect:.2f})")
        else:
            aspect = px_h / px_w
            print(f"    -> Vertical box (h/w={aspect:.2f})")
    
    # Draw boxes on image
    img_with_boxes = img.copy()
    for box in boxes:
        img_with_boxes = draw_box_on_image(img_with_boxes, box, color=(0, 255, 0), thickness=2)
    
    # Save visualization
    output_path = Path(f"label_check_{img_name}.jpg")
    cv2.imwrite(str(output_path), img_with_boxes)
    print(f"\n  Saved visualization to: {output_path}")
    
    return img_with_boxes

def main():
    print("=" * 80)
    print("CHECKING LABEL FORMAT")
    print("=" * 80)
    
    # Check a few sample images
    sample_images = [
        "x17_s1_r4300_c10750",  # From the sample we saw
        "x17_s3_r7525_c6450",    # From the image filenames
        "k17_s3_r6450_c8600",    # From the image filenames
    ]
    
    for img_name in sample_images:
        result = check_label_file(img_name)
        if result is None:
            continue
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    print("Compare the saved visualizations with val_batch2_labels.jpg")
    print("If boxes appear swapped (horizontal vs vertical), labels may have w/h swapped.")
    print("\nTo check predictions, load the model and visualize predictions:")
    print("  model = YOLO('runs/train/v7_speed_yolov8s_winning2/weights/best.pt')")
    print("  results = model.predict(source='datasets/.../images/val/tiled1536/x17_s1_r4300_c10750.jpg', save=True)")

if __name__ == "__main__":
    main()

