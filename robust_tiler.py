#!/usr/bin/env python3
"""
robust_tiler.py
---------------
Robust image tiling with proper coordinate transformation.

Based on the tiling recipe:
- T: Tile size (1024px)
- P: Padding (128px ≈ 12% of T)
- S: Stride (T - 2P = 768px)
- O: Overlap (2P = 256px)

This ensures objects near borders appear whole in at least one tile.
"""

import os
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime


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
            if cls in {0, 1}:  # Panel and tag classes
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


def transform_polygon_for_tile(polygon, x1, y1, cw, ch, new_w, new_h, target_size, img_w, img_h):
    """Transform polygon coordinates for a tile with proper scaling and padding"""
    transformed_polygon = []
    
    # Calculate scaling and padding
    scale_x = scale_y = target_size / max(cw, ch)
    
    # Calculate padding offsets (centered padding)
    pad_left = (target_size - new_w) / 2
    pad_top = (target_size - new_h) / 2
    
    for i in range(0, len(polygon), 2):
        # Convert normalized to absolute coordinates
        x_abs = polygon[i] * img_w
        y_abs = polygon[i+1] * img_h
        
        # Transform to crop coordinates
        x_crop = x_abs - x1
        y_crop = y_abs - y1
        
        # Check if point is within crop bounds
        if 0 <= x_crop <= cw and 0 <= y_crop <= ch:
            # Scale to target size
            x_scaled = x_crop * scale_x
            y_scaled = y_crop * scale_y
            
            # Add padding offset
            x_final = x_scaled + pad_left
            y_final = y_scaled + pad_top
            
            # Normalize to final target_size dimensions
            x_norm = x_final / target_size
            y_norm = y_final / target_size
            
            # Clamp to [0, 1] range to prevent out-of-bounds errors
            x_norm = max(0.0, min(1.0, x_norm))
            y_norm = max(0.0, min(1.0, y_norm))
            
            transformed_polygon.extend([x_norm, y_norm])
    
    return transformed_polygon


def create_tiles_from_image(image_path, label_path, target_size=1024, padding=128):
    """Create tiles from an image with proper coordinate transformation"""
    img = cv2.imread(image_path)
    if img is None:
        return []
    
    h, w = img.shape[:2]
    
    # Parse panels from label file
    panels = parse_yolo_seg_label(label_path)
    
    # Calculate tiling parameters
    T = target_size  # Tile size
    P = padding      # Padding
    S = T - 2 * P   # Stride
    O = 2 * P       # Overlap
    
    print(f"Tiling parameters: T={T}, P={P}, S={S}, O={O}")
    print(f"Image size: {w}x{h}")
    
    tiles = []
    tile_count = 0
    
    # Generate tile grid
    for y0 in range(0, h, S):
        for x0 in range(0, w, S):
            # Top-left of logical tile window
            # Expand by padding, clamped to image edges
            x1 = max(0, x0 - P)
            y1 = max(0, y0 - P)
            x2 = min(w, x0 + T + P)
            y2 = min(h, y0 + T + P)
            
            # Extract crop
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            
            cw = x2 - x1  # Crop width
            ch = y2 - y1  # Crop height
            
            # Resize crop to target size (maintain aspect ratio)
            scale = target_size / max(cw, ch)
            new_w = int(cw * scale)
            new_h = int(ch * scale)
            
            # Resize crop
            crop_resized = cv2.resize(crop, (new_w, new_h))
            
            # Pad to target size if needed
            if new_w != target_size or new_h != target_size:
                padded = np.zeros((target_size, target_size, 3), dtype=np.uint8)
                y_offset = (target_size - new_h) // 2
                x_offset = (target_size - new_w) // 2
                padded[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = crop_resized
                crop_resized = padded
            
            # Transform labels for this tile
            tile_labels = []
            
            for panel in panels:
                # Get panel bbox in absolute coordinates
                xmin, ymin, xmax, ymax = polygon_to_bbox(panel['polygon'], w, h)
                
                # Check intersection with tile
                ix1 = max(xmin, x1)
                iy1 = max(ymin, y1)
                ix2 = min(xmax, x2)
                iy2 = min(ymax, y2)
                
                if ix2 <= ix1 or iy2 <= iy1:
                    continue  # No overlap
                
                # Transform polygon coordinates
                transformed_polygon = transform_polygon_for_tile(
                    panel['polygon'], x1, y1, cw, ch, new_w, new_h, target_size, w, h
                )
                
                # Only keep if we have enough points
                if len(transformed_polygon) >= 6:  # At least 3 points
                    # Calculate new bbox
                    xs = [transformed_polygon[i] for i in range(0, len(transformed_polygon), 2)]
                    ys = [transformed_polygon[i+1] for i in range(0, len(transformed_polygon), 2)]
                    
                    xmin_norm, xmax_norm = min(xs), max(xs)
                    ymin_norm, ymax_norm = min(ys), max(ys)
                    
                    cx = (xmin_norm + xmax_norm) / 2
                    cy = (ymin_norm + ymax_norm) / 2
                    w_norm = xmax_norm - xmin_norm
                    h_norm = ymax_norm - ymin_norm
                    
                    # Filter out tiny polygons (less than 1% of tile area)
                    area_threshold = 0.01 * target_size * target_size
                    if w_norm * h_norm * target_size * target_size >= area_threshold:
                        # Create label line
                        label_line = f"{panel['class_id']} {cx:.6f} {cy:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(f"{p:.6f}" for p in transformed_polygon)
                        tile_labels.append(label_line)
            
            # Only keep tiles with labels
            if tile_labels:
                tiles.append({
                    'image': crop_resized,
                    'labels': tile_labels,
                    'tile_id': f"r{y0}_c{x0}",
                    'bounds': (x1, y1, x2, y2)
                })
                tile_count += 1
    
    print(f"Created {tile_count} tiles with labels")
    return tiles


def tile_split_images(split, base_dir="datasets/rcp_dual_seg", target_size=1024, padding=128):
    """Tile images in a split using robust tiling approach"""
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
    
    # Clean up old tiles
    if os.path.exists(out_img):
        for file in os.listdir(out_img):
            os.remove(os.path.join(out_img, file))
    else:
        os.makedirs(out_img, exist_ok=True)
    
    if os.path.exists(out_lbl):
        for file in os.listdir(out_lbl):
            os.remove(os.path.join(out_lbl, file))
    else:
        os.makedirs(out_lbl, exist_ok=True)
    
    total_tiles = 0
    
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
        
        print(f"Processing {fn}...")
        
        # Create tiles
        tiles = create_tiles_from_image(img_path, lbl_path, target_size, padding)
        
        if tiles:
            for tile in tiles:
                # Save tile image
                tile_name = f"{Path(fn).stem}_{tile['tile_id']}"
                cv2.imwrite(f"{out_img}/{tile_name}.jpg", tile['image'])
                
                # Save tile labels
                with open(f"{out_lbl}/{tile_name}.txt", 'w') as f:
                    f.write('\n'.join(tile['labels']))
                
                total_tiles += 1
            
            print(f"  Created {len(tiles)} tiles")
        else:
            print(f"  No tiles created")
    
    return total_tiles


def main():
    """Main function for robust tiling"""
    print("=== Robust Tiling ===")
    print("Creating tiles with proper coordinate transformation")
    print(f"Target size: 1024px, Padding: 128px")
    
    splits = ['train', 'val', 'test']
    info = {}
    
    for split in splits:
        print(f"\nProcessing {split} split...")
        info[split] = tile_split_images(split)
    
    # Save tiling info
    Path("datasets/rcp_dual_seg").mkdir(exist_ok=True)
    with open(f"datasets/rcp_dual_seg/tiling_info.json", 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'counts': info,
            'target_size': 1024,
            'padding': 128,
            'stride': 1024 - 2 * 128,
            'overlap': 2 * 128
        }, f, indent=2)
    
    print(f"\nTiling Summary: {info}")
    print("Robust tiling complete!")


if __name__ == "__main__":
    main() 