#!/usr/bin/env python3
"""
Test version of panel_cropper.py to identify hanging issues
"""
import os
import cv2
import numpy as np
from pathlib import Path


def parse_yolo_seg_label(label_path):
    """Parse YOLO-Seg labels and extract panel information"""
    panels = []
    if not os.path.exists(label_path):
        return panels
    
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 7:
                continue
            cls = int(parts[0])
            if cls == 0:  # Panel class
                # Extract bbox and polygon points
                cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                pts = [float(v) for v in parts[5:]]
                panels.append({
                    'class_id': cls,
                    'bbox': (cx, cy, w, h),
                    'polygon': pts
                })
    return panels


def polygon_to_bbox(points, img_w, img_h):
    """Convert polygon points to absolute bbox coordinates"""
    # Convert normalized points to absolute
    abs_pts = []
    for i in range(0, len(points), 2):
        x_px = points[i] * img_w
        y_px = points[i+1] * img_h
        abs_pts.append((x_px, y_px))
    
    # Calculate bbox
    xs, ys = zip(*abs_pts)
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    return xmin, ymin, xmax, ymax


def test_single_image():
    """Test processing a single image"""
    print("Testing single image processing...")
    
    # Test with first available image
    img_dir = "datasets/rcp_dual_seg/images/train/fullsize"
    lbl_dir = "datasets/rcp_dual_seg/labels/train/fullsize"
    
    if not os.path.exists(img_dir):
        print(f"❌ Image directory not found: {img_dir}")
        return
    
    # Get first image
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not img_files:
        print(f"❌ No image files found in {img_dir}")
        return
    
    test_img = img_files[0]
    img_path = os.path.join(img_dir, test_img)
    lbl_path = os.path.join(lbl_dir, Path(test_img).stem + '.txt')
    
    print(f"Testing with: {test_img}")
    print(f"Image: {img_path}")
    print(f"Label: {lbl_path}")
    
    # Load image
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Failed to load image: {img_path}")
        return
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Parse panels
    panels = parse_yolo_seg_label(lbl_path)
    print(f"Found {len(panels)} panels")
    
    if not panels:
        print("❌ No panels found")
        return
    
    # Test first panel
    panel = panels[0]
    print(f"Testing panel 0: {len(panel['polygon'])} polygon points")
    
    # Get panel bbox
    xmin, ymin, xmax, ymax = polygon_to_bbox(panel['polygon'], w, h)
    print(f"Panel bbox: ({xmin:.1f}, {ymin:.1f}) to ({xmax:.1f}, {ymax:.1f})")
    
    # Calculate crop bounds
    panel_w = xmax - xmin
    panel_h = ymax - ymin
    panel_cx = (xmin + xmax) / 2
    panel_cy = (ymin + ymax) / 2
    
    padding_ratio = 0.15
    pad_w = panel_w * padding_ratio
    pad_h = panel_h * padding_ratio
    
    crop_x1 = max(0, int(panel_cx - (panel_w + pad_w) / 2))
    crop_y1 = max(0, int(panel_cy - (panel_h + pad_h) / 2))
    crop_x2 = min(w, int(panel_cx + (panel_w + pad_w) / 2))
    crop_y2 = min(h, int(panel_cy + (panel_h + pad_h) / 2))
    
    print(f"Crop bounds: ({crop_x1}, {crop_y1}) to ({crop_x2}, {crop_y2})")
    
    # Extract crop
    crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
    if crop.size == 0:
        print("❌ Empty crop")
        return
    
    print(f"Crop size: {crop.shape}")
    
    # Test coordinate transformation
    print("Testing coordinate transformation...")
    polygon = panel['polygon']
    transformed_points = []
    
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    target_size = 1024
    scale = target_size / max(crop_w, crop_h)
    
    print(f"Scale factor: {scale}")
    
    for i in range(0, len(polygon), 2):
        # Convert normalized to absolute coordinates
        x_abs = polygon[i] * w
        y_abs = polygon[i+1] * h
        
        # Transform to crop coordinates
        x_crop = x_abs - crop_x1
        y_crop = y_abs - crop_y1
        
        print(f"Point {i//2}: original=({polygon[i]:.3f}, {polygon[i+1]:.3f}), "
              f"absolute=({x_abs:.1f}, {y_abs:.1f}), crop=({x_crop:.1f}, {y_crop:.1f})")
        
        # Check if point is within crop bounds
        if 0 <= x_crop <= crop_w and 0 <= y_crop <= crop_h:
            # Scale to target size
            x_scaled = x_crop * scale
            y_scaled = y_crop * scale
            
            # Normalize to final 1024x1024 dimensions
            x_norm = x_scaled / target_size
            y_norm = y_scaled / target_size
            
            # Clamp to [0, 1] range
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            transformed_points.extend([x_norm, y_norm])
            print(f"  -> normalized=({x_norm:.3f}, {y_norm:.3f})")
        else:
            print(f"  -> outside crop bounds, skipping")
    
    print(f"Transformed {len(transformed_points)//2} points")
    print("✅ Test completed successfully!")


if __name__ == "__main__":
    test_single_image() 