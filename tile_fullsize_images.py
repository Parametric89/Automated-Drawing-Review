#!/usr/bin/env python3
"""
tile_fullsize_images.py
-----------------------
Tile full-size images with proper YOLO-Seg label transformation.

Follows the exact tiling recipe:
- Tile size: 1024x1024
- Overlap: 10% (0.10)
- Proper polygon clipping and coordinate transformation
"""

import os
import cv2
import numpy as np
from pathlib import Path
import argparse
import glob


# --- helpers ----------------------------------------------------------
def parse_yolo_seg_line(line: str):
    """Parse a YOLO-Seg label line into components"""
    parts = line.strip().split()
    cls = int(parts[0])
    cx, cy, w, h = map(float, parts[1:5])
    coords = list(map(float, parts[5:]))
    poly = [(coords[i], coords[i+1]) for i in range(0, len(coords), 2)]
    return cls, cx, cy, w, h, poly


def clip_poly_to_tile(poly, tx, ty, ts):
    """Clip polygon to tile bounds"""
    tx2, ty2 = tx + ts, ty + ts
    return [(x, y) for x, y in poly if tx <= x < tx2 and ty <= y < ty2]


def bbox_from_poly(poly):
    """Calculate bounding box from polygon points"""
    if not poly:
        return 0, 0, 0, 0
    xs, ys = zip(*poly)
    return min(xs), min(ys), max(xs), max(ys)


def transform_annotation_for_tile(line, tile_x, tile_y, tile_size, img_w, img_h):
    """Transform a YOLO-Seg annotation for a specific tile"""
    # Parse the original line
    cls, cx, cy, w, h, poly = parse_yolo_seg_line(line)
    
    # Convert normalized polygon to absolute coordinates
    abs_poly = [(x * img_w, y * img_h) for x, y in poly]
    
    # Clip polygon to tile bounds
    clipped_poly = clip_poly_to_tile(abs_poly, tile_x, tile_y, tile_size)
    
    # Discard if too few points
    if len(clipped_poly) < 3:
        return None
    
    # Shift and normalize coordinates
    norm_poly = [((x - tile_x) / tile_size, (y - tile_y) / tile_size) for x, y in clipped_poly]
    
    # Flatten polygon coordinates
    flat_poly = []
    for x, y in norm_poly:
        flat_poly.extend([x, y])
    
    # Re-compute bbox from clipped polygon
    xmin, ymin, xmax, ymax = bbox_from_poly(clipped_poly)
    
    # Convert bbox to normalized coordinates
    cx_norm = (xmin + xmax) / (2 * tile_size)
    cy_norm = (ymin + ymax) / (2 * tile_size)
    w_norm = (xmax - xmin) / tile_size
    h_norm = (ymax - ymin) / tile_size
    
    # Discard tiny masks (less than 1% of tile area)
    if w_norm < 0.01 or h_norm < 0.01:
        return None
    
    # Create new YOLO-Seg line
    label_line = f"{cls} {cx_norm:.6f} {cy_norm:.6f} {w_norm:.6f} {h_norm:.6f} " + " ".join(f"{p:.6f}" for p in flat_poly)
    
    return label_line


def create_tiles_from_image(image_path, label_path, tile_size=1024, overlap=0.10):
    """Create tiles from an image with proper label transformation"""
    # Read image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return []
    
    h, w = img.shape[:2]
    print(f"Image size: {w}x{h}")
    
    # Read labels
    if not os.path.exists(label_path):
        print(f"Label file not found: {label_path}")
        return []
    
    with open(label_path, 'r') as f:
        lines = f.readlines()
    
    print(f"Found {len(lines)} annotations")
    
    # Calculate stride
    stride = int(tile_size * (1 - overlap))
    
    tiles = []
    tile_count = 0
    
    # Generate tiles
    for y in range(0, h, stride):
        for x in range(0, w, stride):
            # Extract tile
            tile_x = x
            tile_y = y
            tile_w = min(tile_size, w - x)
            tile_h = min(tile_size, h - y)
            
            # Extract image tile
            tile_img = img[tile_y:tile_y + tile_h, tile_x:tile_x + tile_w]
            
            # Pad to tile_size if needed
            if tile_w < tile_size or tile_h < tile_size:
                padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                padded[:tile_h, :tile_w] = tile_img
                tile_img = padded
            
            # Transform annotations for this tile
            tile_labels = []
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                transformed_line = transform_annotation_for_tile(
                    line, tile_x, tile_y, tile_size, w, h
                )
                
                if transformed_line:
                    tile_labels.append(transformed_line)
            
            # Only keep tiles with annotations
            if tile_labels:
                tiles.append({
                    'image': tile_img,
                    'labels': tile_labels,
                    'tile_id': f"r{y}_c{x}",
                    'bounds': (tile_x, tile_y, tile_x + tile_w, tile_y + tile_h)
                })
                tile_count += 1
    
    print(f"Created {tile_count} tiles with annotations")
    return tiles


def tile_split_images(split, base_dir="datasets/rcp_dual_seg", tile_size=1024, overlap=0.10):
    """Tile images in a split"""
    img_dir = f"{base_dir}/images/{split}/fullsize"
    lbl_dir = f"{base_dir}/labels/{split}/fullsize"
    out_img = f"{base_dir}/images/{split}/tiled1k"
    out_lbl = f"{base_dir}/labels/{split}/tiled1k"
    
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
    
    # Process each image
    for fn in os.listdir(img_dir):
        if not fn.lower().endswith(('.jpg', '.png')):
            continue
        
        img_path = f"{img_dir}/{fn}"
        lbl_path = f"{lbl_dir}/{Path(fn).stem}.txt"
        
        print(f"Processing {fn}...")
        
        # Create tiles
        tiles = create_tiles_from_image(img_path, lbl_path, tile_size, overlap)
        
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
    """Main function"""
    parser = argparse.ArgumentParser(description="Tile full-size images with proper YOLO-Seg label transformation")
    parser.add_argument("--base_dir", default="datasets/rcp_dual_seg", help="Base directory for dataset")
    parser.add_argument("--tile_size", type=int, default=1024, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.10, help="Overlap ratio (0.0-1.0)")
    
    args = parser.parse_args()
    
    print("=== Full-Size Image Tiling ===")
    print(f"Base directory: {args.base_dir}")
    print(f"Tile size: {args.tile_size}px")
    print(f"Overlap: {args.overlap:.1%}")
    print()
    
    splits = ['train', 'val', 'test']
    info = {}
    
    for split in splits:
        print(f"Processing {split} split...")
        info[split] = tile_split_images(split, args.base_dir, args.tile_size, args.overlap)
    
    print(f"\nTiling Summary: {info}")
    print("Tiling complete!")


if __name__ == "__main__":
    main()