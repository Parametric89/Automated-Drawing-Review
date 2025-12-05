"""
Visualize ground truth labels directly (without predictions)
to check if GT labels themselves are imprecise/seeping.
"""
import cv2
import numpy as np
from pathlib import Path

def draw_yolo_boxes(img_path, label_path, output_path):
    """Draw YOLO format bounding boxes on image."""
    
    # Read image
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Failed to load image: {img_path}")
        return
    
    h, w = img.shape[:2]
    
    # Read YOLO format labels (class x_center y_center width height - all normalized 0-1)
    if not label_path.exists():
        print(f"No label file: {label_path}")
        return
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    panel_count = 0
    
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        
        cls_id = int(parts[0])
        
        # Only draw panels (class 0), skip tags (class 1)
        if cls_id != 0:
            continue
        
        panel_count += 1
        
        x_center = float(parts[1]) * w
        y_center = float(parts[2]) * h
        box_w = float(parts[3]) * w
        box_h = float(parts[4]) * h
        
        # Convert center format to corner format
        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)
        
        # Draw bbox - GREEN for GT
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        
        # Add label
        cv2.putText(img, f"GT_{panel_count}", (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save
    cv2.imwrite(str(output_path), img)
    print(f"[OK] Saved GT visualization: {output_path.name} ({panel_count} panels)")

def main():
    """Check GT labels for the problematic images identified by user."""
    
    # Images user identified as "poor" with seepage issues
    problematic_images = [
        'x10_s1_r4300_c10750.jpg',  # Image 3: false negatives, seeping
        'x17_s1_r1075_c7525.jpg',   # Image 4: very bad, FP/FN, poor alignment
        'x34_s1_r4226_c2538.jpg',   # Image 5: poor, FN, large negative offset
        'x10_s1_r5375_c3225.jpg',   # Image 7: poor, seeping over many panels
        'x39_s1_r2150_c6450.jpg',   # Image 10: bad, FN, detected panel twice
    ]
    
    # Also check "perfect" images to compare
    perfect_images = [
        'x62_s3_r0_c4300.jpg',      # Image 1: perfect
        'x31_s3_r4300_c4300.jpg',   # Image 6: perfect
        'x52_s3_r2150_c6450.jpg',   # Image 8: perfect
    ]
    
    base_dir = Path('datasets/rcp_bbox_v6_reshuffled')
    img_dir = base_dir / 'images' / 'val' / 'tiled1536'
    label_dir = base_dir / 'labels' / 'val' / 'tiled1536'
    output_dir = Path('runs/gt_check')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("GROUND TRUTH LABEL QUALITY CHECK")
    print("="*80)
    print()
    print("Drawing GT labels (GREEN boxes) for visual inspection...")
    print()
    
    print("PROBLEMATIC IMAGES (user reported seepage/poor quality):")
    print("-" * 80)
    for img_name in problematic_images:
        img_path = img_dir / img_name
        label_path = label_dir / img_name.replace('.jpg', '.txt')
        output_path = output_dir / f"GT_{img_name}"
        
        if img_path.exists():
            draw_yolo_boxes(img_path, label_path, output_path)
        else:
            print(f"[X] Image not found: {img_name}")
    
    print()
    print("PERFECT IMAGES (user reported good quality):")
    print("-" * 80)
    for img_name in perfect_images:
        img_path = img_dir / img_name
        label_path = label_dir / img_name.replace('.jpg', '.txt')
        output_path = output_dir / f"GT_{img_name}"
        
        if img_path.exists():
            draw_yolo_boxes(img_path, label_path, output_path)
        else:
            print(f"[X] Image not found: {img_name}")
    
    print()
    print("="*80)
    print("GT VISUALIZATIONS SAVED")
    print("="*80)
    print()
    print(f"Location: {output_dir}/")
    print()
    print("WHAT TO LOOK FOR:")
    print("  1. Do GT boxes seep into adjacent panels?")
    print("  2. Are GT box edges precise or loose?")
    print("  3. Do 'perfect' images have tighter GT labels than 'poor' images?")
    print()
    print("If GT labels are also imprecise → Root cause is LABELING, not model")
    print("If GT labels are tight → Root cause is MODEL bbox regression")
    print()

if __name__ == "__main__":
    main()

