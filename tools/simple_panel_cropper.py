#!/usr/bin/env python3
"""
Simplified panel cropper - processes just one image to test
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime


def cleanup_old_crops(crop_dir):
    """Clean up old crop directory"""
    if os.path.exists(crop_dir):
        for file in os.listdir(crop_dir):
            os.remove(os.path.join(crop_dir, file))
    else:
        os.makedirs(crop_dir, exist_ok=True)


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


def transform_labels_for_crop(panel, crop_x, crop_y, crop_w, crop_h, 
                             x_offset, y_offset, target_size, img_w, img_h):
    """Transform panel labels for the crop with proper padding and scaling"""
    labels = []
    
    # Transform polygon points
    polygon = panel['polygon']
    transformed_polygon = []
    
    # Calculate scale factor
    scale = target_size / max(crop_w, crop_h)
    
    for i in range(0, len(polygon), 2):
        # Convert normalized to absolute coordinates (using original image dimensions)
        x_abs = polygon[i] * img_w
        y_abs = polygon[i+1] * img_h
        
        # Transform to crop coordinates
        x_crop = x_abs - crop_x
        y_crop = y_abs - crop_y
        
        # Check if point is within crop bounds
        if 0 <= x_crop <= crop_w and 0 <= y_crop <= crop_h:
            # Scale to target size
            x_scaled = x_crop * scale
            y_scaled = y_crop * scale
            
            # Add padding offset
            x_final = x_scaled + x_offset
            y_final = y_scaled + y_offset
            
            # Normalize to final 1024x1024 dimensions
            x_norm = x_final / target_size
            y_norm = y_final / target_size
            
            # Clamp to [0, 1] range to prevent out-of-bounds errors
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            transformed_polygon.extend([x_norm, y_norm])
    
    # Only keep if we have enough points
    if len(transformed_polygon) >= 6:  # At least 3 points
        # Calculate new bbox
        xs = [transformed_polygon[i] for i in range(0, len(transformed_polygon), 2)]
        ys = [transformed_polygon[i+1] for i in range(0, len(transformed_polygon), 2)]
        
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        
        cx = (xmin + xmax) / 2
        cy = (ymin + ymax) / 2
        w = xmax - xmin
        h = ymax - ymin
        
        # Create label line
        label_line = f"{panel['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} " + " ".join(f"{p:.6f}" for p in transformed_polygon)
        labels.append(label_line)
    
    return labels


def create_panel_crop(image_path, label_path, target_size=1024, padding_ratio=0.15):
    """Create a crop centered on a panel with padding"""
    img = cv2.imread(image_path)
    if img is None:
        return None, None
    
    h, w = img.shape[:2]
    
    # Parse panels from label file
    panels = parse_yolo_seg_label(label_path)
    if not panels:
        return None, None
    
    crops = []
    
    for i, panel in enumerate(panels):
        # Get panel bbox
        xmin, ymin, xmax, ymax = polygon_to_bbox(panel['polygon'], w, h)
        
        # Calculate panel center and size
        panel_w = xmax - xmin
        panel_h = ymax - ymin
        panel_cx = (xmin + xmax) / 2
        panel_cy = (ymin + ymax) / 2
        
        # Add padding
        pad_w = panel_w * padding_ratio
        pad_h = panel_h * padding_ratio
        
        # Calculate crop bounds with padding
        crop_x1 = max(0, int(panel_cx - (panel_w + pad_w) / 2))
        crop_y1 = max(0, int(panel_cy - (panel_h + pad_h) / 2))
        crop_x2 = min(w, int(panel_cx + (panel_w + pad_w) / 2))
        crop_y2 = min(h, int(panel_cy + (panel_h + pad_h) / 2))
        
        # Extract crop
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue
        
        # Calculate crop dimensions
        crop_w = crop_x2 - crop_x1
        crop_h = crop_y2 - crop_y1
        
        # Resize to target size (maintain aspect ratio)
        crop_h_resized, crop_w_resized = crop.shape[:2]
        scale = target_size / max(crop_w_resized, crop_h_resized)
        new_w = int(crop_w_resized * scale)
        new_h = int(crop_h_resized * scale)
        
        # Resize crop
        crop_resized = cv2.resize(crop, (new_w, new_h))
        
        # Initialize padding offsets
        x_offset = y_offset = 0
        
        # Pad to target size if needed
        if new_w != target_size or new_h != target_size:
            padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
            y_offset = (target_size - new_h) // 2
            x_offset = (target_size - new_w) // 2
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
            crop_resized = padded
        
        # Transform labels for this crop
        crop_labels = transform_labels_for_crop(panel, crop_x1, crop_y1, crop_w, crop_h, 
                                              x_offset, y_offset, target_size, w, h)
        
        crops.append({
            'image': crop_resized,
            'labels': crop_labels,
            'panel_index': i,
            'original_bounds': (crop_x1, crop_y1, crop_x2, crop_y2)
        })
    
    return crops


def process_single_split(split="train", base_dir="datasets/rcp_dual_seg", target_size=1024):
    """Process a single split with just one image"""
    img_dir = f"{base_dir}/images/{split}/fullsize"
    lbl_dir = f"{base_dir}/labels/{split}/fullsize"
    out_img = f"{base_dir}/images/{split}/cropped1k"
    out_lbl = f"{base_dir}/labels/{split}/cropped1k"
    
    # Check if fullsize directories exist
    if not os.path.exists(img_dir):
        print(f"Warning: Image directory not found: {img_dir}")
        return 0
    
    if not os.path.exists(lbl_dir):
        print(f"Warning: Label directory not found: {lbl_dir}")
        return 0
    
    cleanup_old_crops(out_img)
    cleanup_old_crops(out_lbl)
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    # Get first image only
    img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png'))]
    if not img_files:
        print(f"No image files found in {img_dir}")
        return 0
    
    # Process just the first image
    fn = img_files[0]
    img_path = f"{img_dir}/{fn}"
    lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
    
    print(f"Processing {fn}...")
    
    # Create panel-centric crops
    crops = create_panel_crop(img_path, lbl_path, target_size)
    
    if crops:
        for i, crop in enumerate(crops):
            # Save crop image
            crop_name = f"{Path(fn).stem}_panel{i:03d}"
            cv2.imwrite(f"{out_img}/{crop_name}.jpg", crop['image'])
            
            # Save crop labels
            with open(f"{out_lbl}/{crop_name}.txt", 'w') as f:
                f.write('\n'.join(crop['labels']))
            
            print(f"  Created crop {i+1}: {crop_name}")
        
        print(f"  Created {len(crops)} panel crops")
    else:
        print(f"  No panels found")
    
    return len(crops)


def main():
    """Main function for panel-centric cropping"""
    print("=== Simplified Panel-Centric Cropping ===")
    print("Processing just one image to test")
    print(f"Target size: 1024px, Padding: 15%")
    
    # Process just train split
    print(f"\nProcessing train split...")
    count = process_single_split("train")
    
    print(f"\nCropping Summary: {count} crops created")
    print("Simplified panel cropping complete!")


if __name__ == "__main__":
    main() 