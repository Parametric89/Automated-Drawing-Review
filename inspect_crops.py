#!/usr/bin/env python3
"""
inspect_crops.py
----------------
Visual inspection tool for cropped images and their labels.

Features:
- Randomly selects a cropped image and its label
- Displays the image with bounding boxes and segmentation masks
- Shows class information and coordinate details
- Helps verify coordinate transformations are correct
"""

import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse


def parse_yolo_seg_label(label_path):
    """Parse YOLO-Seg labels and extract panel and tag information"""
    panels = []
    if not os.path.exists(label_path):
        return panels
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            if cls in {0, 1}:  # Panel (0) and tag (1) classes
                # Extract bbox and polygon points
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                pts = [float(v) for v in parts[5:]]
                panels.append({
                    'class_id': cls,
                    'bbox': (cx, cy, w, h),
                    'polygon': pts
                })
    return panels


def draw_polygon_mask(img, polygon_points, color=(0, 255, 0), alpha=0.3):
    """Draw polygon mask on image"""
    h, w = img.shape[:2]
    
    # Convert normalized points to pixel coordinates
    points = []
    for i in range(0, len(polygon_points), 2):
        x = int(polygon_points[i] * w)
        y = int(polygon_points[i+1] * h)
        points.append([x, y])
    
    if len(points) < 3:
        return img
    
    # Create mask
    mask = np.zeros((h, w), dtype=np.uint8)
    points = np.array(points, dtype=np.int32)
    cv2.fillPoly(mask, [points], 255)
    
    # Apply mask with transparency
    overlay = img.copy()
    cv2.fillPoly(overlay, [points], color)
    img = cv2.addWeighted(img, 1-alpha, overlay, alpha, 0)
    
    # Draw polygon outline
    cv2.polylines(img, [points], True, color, 2)
    
    return img


def draw_bbox(img, bbox, label, color=(255, 0, 0)):
    """Draw bounding box on image"""
    h, w = img.shape[:2]
    cx, cy, bw, bh = bbox
    
    # Convert normalized bbox to pixel coordinates
    x1 = int((cx - bw/2) * w)
    y1 = int((cy - bh/2) * h)
    x2 = int((cx + bw/2) * w)
    y2 = int((cy + bh/2) * h)
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
    
    # Draw label
    cv2.putText(img, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    return img


def inspect_random_crop(base_dir="datasets/rcp_dual_seg", split="train", data_type="cropped1k"):
    """Inspect a random cropped image and its label"""
    
    # Get list of cropped images
    img_dir = f"{base_dir}/images/{split}/{data_type}"
    lbl_dir = f"{base_dir}/labels/{split}/{data_type}"
    
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    if not os.path.exists(lbl_dir):
        print(f"❌ Label directory not found: {lbl_dir}")
        return
    
    # Get list of image files
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not img_files:
        print(f"❌ No image files found in {img_dir}")
        return
    
    # Randomly select an image
    selected_img = random.choice(img_files)
    img_path = os.path.join(img_dir, selected_img)
    lbl_path = os.path.join(lbl_dir, Path(selected_img).stem + '.txt')
    
    print(f"🔍 Inspecting: {selected_img}")
    print(f"📁 Image: {img_path}")
    print(f"📁 Label: {lbl_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return
    
    h, w = img.shape[:2]
    print(f"📏 Image size: {w}x{h}")
    
    # Load and parse labels
    panels = parse_yolo_seg_label(lbl_path)
    print(f"🏷️  Found {len(panels)} objects in label file")
    
    # Count by class
    class_counts = {}
    for panel in panels:
        cls = panel['class_id']
        class_counts[cls] = class_counts.get(cls, 0) + 1
    
    print(f"📊 Class breakdown:")
    for cls, count in sorted(class_counts.items()):
        class_name = "Panel" if cls == 0 else "Tag" if cls == 1 else f"Class {cls}"
        print(f"   - {class_name} (class {cls}): {count} objects")
    
    if not panels:
        print("⚠️  No panels found in label file")
        # Show just the image
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        plt.title(f"No panels found in {selected_img}")
        plt.axis('off')
        plt.show()
        return
    
    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"Original Image: {selected_img}")
    axes[0].axis('off')
    
    # Image with annotations
    annotated_img = img.copy()
    
    for i, panel in enumerate(panels):
        class_id = panel['class_id']
        bbox = panel['bbox']
        polygon = panel['polygon']
        
        # Get class name and color
        if class_id == 0:
            class_name = "Panel"
            color = (0, 255, 0)  # Green for panels
        elif class_id == 1:
            class_name = "Tag"
            color = (255, 0, 0)  # Red for tags
        else:
            class_name = f"Class {class_id}"
            color = (0, 0, 255)  # Blue for unknown classes
        
        print(f"\n📦 {class_name} {i+1}:")
        print(f"   Class ID: {class_id}")
        print(f"   Bbox (cx, cy, w, h): {bbox}")
        print(f"   Polygon points: {len(polygon)} coordinates")
        
        # Draw segmentation mask
        annotated_img = draw_polygon_mask(annotated_img, polygon, color, alpha=0.3)
        
        # Draw bounding box
        label = f"{class_name} {class_id}"
        annotated_img = draw_bbox(annotated_img, bbox, label, color)
        
        # Print coordinate details
        print(f"   Polygon coordinates (normalized):")
        for j in range(0, len(polygon), 2):
            x, y = polygon[j], polygon[j+1]
            print(f"     Point {j//2}: ({x:.6f}, {y:.6f})")
    
    axes[1].imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Annotated Image: {selected_img}")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # Print summary
    print(f"\n✅ Inspection complete!")
    print(f"📊 Summary:")
    print(f"   - Image: {selected_img}")
    print(f"   - Size: {w}x{h}")
    print(f"   - Total objects: {len(panels)}")
    print(f"   - Classes found: {sorted(set(p['class_id'] for p in panels))}")
    
    # Show class breakdown in summary
    if class_counts:
        print(f"   - Class breakdown:")
        for cls, count in sorted(class_counts.items()):
            class_name = "Panel" if cls == 0 else "Tag" if cls == 1 else f"Class {cls}"
            print(f"     • {class_name}: {count}")


def inspect_multiple_crops(base_dir="datasets/rcp_dual_seg", split="train", num_samples=5, data_type="cropped1k"):
    """Inspect multiple random crops"""
    print(f"🔍 Inspecting {num_samples} random crops from {split}/{data_type}...")
    
    for i in range(num_samples):
        print(f"\n{'='*50}")
        print(f"Sample {i+1}/{num_samples}")
        print(f"{'='*50}")
        inspect_random_crop(base_dir, split, data_type)
        
        
        # if i < num_samples - 1:
        #     input("\nPress Enter to continue to next sample...")


def main():
    parser = argparse.ArgumentParser(description="Inspect cropped images and labels")
    parser.add_argument("--split", default="train", choices=["train", "val", "test"], 
                       help="Dataset split to inspect")
    parser.add_argument("--samples", type=int, default=1, 
                       help="Number of random samples to inspect")
    parser.add_argument("--base-dir", default="datasets/rcp_dual_seg",
                       help="Base directory for dataset")
    
    parser.add_argument("--data-type", default="cropped1k", choices=["cropped1k", "augmented1k"],
                       help="Data type to inspect")
    
    args = parser.parse_args()
    
    if args.samples == 1:
        inspect_random_crop(args.base_dir, args.split, args.data_type)
    else:
        inspect_multiple_crops(args.base_dir, args.split, args.samples, args.data_type)


if __name__ == "__main__":
    main() 