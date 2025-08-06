#!/usr/bin/env python3
"""
panel_cropper.py
----------------
Panel-centric cropping: Use existing panel coordinates to create focused training data.

Strategy:
1. Extract panel bbox from polygon coordinates
2. Pad by 15% for context
3. Clamp to image bounds
4. Resize to 1024px (longest side)
5. Rewrite labels for the crop

This creates much more efficient training data than random tiling.
"""
import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import argparse
from tqdm import tqdm
from shapely.geometry import Polygon, box

def cleanup_old_crops(crop_dir):
    """Clean up old crop directory"""
    if os.path.exists(crop_dir):
        for file in os.listdir(crop_dir):
            os.remove(os.path.join(crop_dir, file))
    else:
        os.makedirs(crop_dir, exist_ok=True)

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
            if cls in {0, 1}:  # Panel and tag classes
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
    abs_pts = [(points[i] * img_w, points[i+1] * img_h) for i in range(0, len(points), 2)]
    if not abs_pts:
        return 0, 0, 0, 0
    xs, ys = zip(*abs_pts)
    return min(xs), min(ys), max(xs), max(ys)

def transform_labels_for_crop(panel, crop_x, crop_y, crop_w, crop_h, scale, x_offset, y_offset, target_size, img_w, img_h):
    """Transform panel labels for the crop using Shapely for true clipping."""
    cls = panel['class_id']
    polygon_points = panel['polygon']

    abs_pts = [(polygon_points[i] * img_w, polygon_points[i+1] * img_h) for i in range(0, len(polygon_points), 2)]
    
    if len(abs_pts) < 3:
        return []

    shapely_poly = Polygon(abs_pts)
    crop_rect = box(crop_x, crop_y, crop_x + crop_w, crop_y + crop_h)
    clipped_geom = shapely_poly.intersection(crop_rect)

    if clipped_geom.is_empty:
        return []

    # Handle MultiPolygon result from intersection
    if clipped_geom.geom_type == 'MultiPolygon':
        # Naive: just take the largest polygon if there are multiple.
        clipped_geom = max(clipped_geom.geoms, key=lambda p: p.area)
    
    if clipped_geom.geom_type != 'Polygon' or len(clipped_geom.exterior.coords) < 3:
        return []

    clipped_coords = list(clipped_geom.exterior.coords)

    transformed = []
    for x_abs_clipped, y_abs_clipped in clipped_coords:
        x_t = (x_abs_clipped - crop_x) * scale + x_offset
        y_t = (y_abs_clipped - crop_y) * scale + y_offset
        transformed.extend([x_t / target_size, y_t / target_size])

    if len(transformed) < 6:
        return []
    
    xs, ys = transformed[0::2], transformed[1::2]
    cx, cy = (min(xs)+max(xs))/2, (min(ys)+max(ys))/2
    bw, bh = max(xs)-min(xs), max(ys)-min(ys)
    
    if (bw < 0.01 and bw * target_size < 10) or \
       (bh < 0.01 and bh * target_size < 10):
        return []
        
    label = f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f} " + " ".join(f"{p:.6f}" for p in transformed)
    return [label]

def create_panel_crop(image_path, label_path, target_size=1024, padding_ratio=0.15):
    """Create a crop centered on a panel with padding"""
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        return None
    
    h, w = img.shape[:2]
    num_channels = img.shape[2] if len(img.shape) > 2 else 1

    panels = parse_yolo_seg_label(label_path)
    if not panels:
        return []
    
    crops = []
    panel_objects = [p for p in panels if p['class_id'] == 0]
    
    for i, panel in enumerate(panel_objects):
        xmin, ymin, xmax, ymax = polygon_to_bbox(panel['polygon'], w, h)
        
        panel_w, panel_h = xmax - xmin, ymax - ymin
        panel_cx, panel_cy = (xmin + xmax) / 2, (ymin + ymax) / 2
        
        pad_w, pad_h = panel_w * padding_ratio, panel_h * padding_ratio
        
        crop_x1 = max(0, int(panel_cx - (panel_w + pad_w) / 2))
        crop_y1 = max(0, int(panel_cy - (panel_h + pad_h) / 2))
        crop_x2 = min(w, int(panel_cx + (panel_w + pad_w) / 2))
        crop_y2 = min(h, int(panel_cy + (panel_h + pad_h) / 2))
        
        crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
        if crop.size == 0:
            continue
        
        crop_w, crop_h = crop_x2 - crop_x1, crop_y2 - crop_y1
        scale = target_size / max(crop_w, crop_h)
        new_w, new_h = int(crop_w * scale), int(crop_h * scale)
        
        crop_resized = cv2.resize(crop, (new_w, new_h))
        
        x_offset, y_offset = (target_size - new_w) // 2, (target_size - new_h) // 2
        
        if new_w != target_size or new_h != target_size:
            padded_shape = (target_size, target_size, num_channels) if num_channels > 1 else (target_size, target_size)
            padded = np.zeros(padded_shape, dtype=np.uint8)
            padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
            crop_resized = padded
        
        crop_labels = []
        
        # Add all objects (panels and tags) that fall within the crop
        for obj in panels:
            labels = transform_labels_for_crop(obj, crop_x1, crop_y1, crop_w, crop_h, scale, x_offset, y_offset, target_size, w, h)
            crop_labels.extend(labels)
        
        if not crop_labels:
            continue

        crops.append({
            'image': crop_resized,
            'labels': crop_labels,
            'panel_index': i
        })
    
    return crops

def crop_split_images(split, base_dir="datasets/rcp_dual_seg", target_size=1024, force_cleanup=False):
    """Crop images in a split using panel-centric approach"""
    img_dir = f"{base_dir}/images/{split}/fullsize"
    lbl_dir = f"{base_dir}/labels/{split}/fullsize"
    out_img = f"{base_dir}/images/{split}/cropped1k"
    out_lbl = f"{base_dir}/labels/{split}/cropped1k"
    
    if not os.path.exists(img_dir) or not os.path.exists(lbl_dir):
        print(f"Warning: fullsize directories not found for {split} split. Skipping.")
        return 0
    
    if force_cleanup:
        print(f"Cleaning up old crops in {split} split...")
        cleanup_old_crops(out_img)
        cleanup_old_crops(out_lbl)
    
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lbl, exist_ok=True)
    
    total_crops = 0
    image_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('jpg', 'png'))]

    for fn in tqdm(image_files, desc=f"Cropping {split}"):
        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
        
        crops = create_panel_crop(img_path, lbl_path, target_size)
        
        if crops:
            for crop in crops:
                ext = Path(fn).suffix.lower()
                # If image has alpha, save as png
                if crop['image'].shape[2] == 4 and ext != ".png":
                    ext = '.png'
                
                crop_name = f"{Path(fn).stem}_panel{crop['panel_index']:03d}"
                cv2.imwrite(f"{out_img}/{crop_name}{ext}", crop['image'])
                
                with open(f"{out_lbl}/{crop_name}.txt", 'w') as f:
                    f.write('\n'.join(crop['labels']))
                
                total_crops += 1
    
    print(f"  Created {total_crops} panel crops for {split} split")
    return total_crops

def main():
    """Main function for panel-centric cropping"""
    parser = argparse.ArgumentParser(description="Create panel-centric crops from full-size images.")
    parser.add_argument("--force", action="store_true", help="Force cleanup of existing crops before running.")
    args = parser.parse_args()

    print("=== Panel-Centric Cropping ===")
    print("Creating focused training data using panel coordinates")
    print(f"Target size: 1024px, Padding: 15%")
    if args.force:
        print("FORCE mode: Cleaning up old crops.")

    splits = ['train', 'val', 'test']
    info = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        info[split] = crop_split_images(split, force_cleanup=args.force)
    
    Path("datasets/rcp_dual_seg").mkdir(exist_ok=True)
    with open(f"datasets/rcp_dual_seg/cropping_info.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'counts': info,
            'target_size': 1024,
            'padding_ratio': 0.15
        }, f, indent=2)
    
    print(f"\nCropping Summary: {info}")
    print("Panel-centric cropping complete!")

if __name__ == "__main__":
    main()
