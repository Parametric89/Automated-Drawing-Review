"""
Mine hard negatives by cropping tiles from fullsize images in regions with no panels.
Target: 200 additional hard negatives from datasets/rcp_bbox_v6_reshuffled/images/train/fullsize
"""

import cv2
import numpy as np
from pathlib import Path
import random
import shutil

SOURCE_IMAGES = Path("datasets/rcp_bbox_v6_reshuffled/images/train/fullsize")
SOURCE_LABELS = Path("datasets/rcp_bbox_v6_reshuffled/labels/train/fullsize")
TARGET_DATASET = Path("datasets/rcp_bbox_v7_speed")
TILE_SIZE = 1536
TARGET_COUNT = 200
OVERLAP = 0.2  # 20% overlap

def load_yolo_labels(label_path):
    """Load YOLO labels and return list of bboxes (normalized)."""
    if not label_path.exists():
        return []
    
    bboxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                try:
                    cls_id = int(parts[0])
                    if cls_id == 0:  # Panel class only
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        bboxes.append({
                            'x_center': x_center,
                            'y_center': y_center,
                            'width': width,
                            'height': height
                        })
                except (ValueError, IndexError):
                    continue
    
    return bboxes

def bbox_overlaps_tile(bbox, tile_x, tile_y, tile_size, img_width, img_height):
    """Check if bbox overlaps with tile region."""
    # Convert normalized bbox to pixel coordinates
    bbox_x_center = bbox['x_center'] * img_width
    bbox_y_center = bbox['y_center'] * img_height
    bbox_width = bbox['width'] * img_width
    bbox_height = bbox['height'] * img_height
    
    bbox_x1 = bbox_x_center - bbox_width / 2
    bbox_y1 = bbox_y_center - bbox_height / 2
    bbox_x2 = bbox_x_center + bbox_width / 2
    bbox_y2 = bbox_y_center + bbox_height / 2
    
    tile_x2 = tile_x + tile_size
    tile_y2 = tile_y + tile_size
    
    # Check for overlap
    return not (bbox_x2 < tile_x or bbox_x1 > tile_x2 or bbox_y2 < tile_y or bbox_y1 > tile_y2)

def has_panels_in_tile(bboxes, tile_x, tile_y, tile_size, img_width, img_height):
    """Check if tile region contains any panels."""
    for bbox in bboxes:
        if bbox_overlaps_tile(bbox, tile_x, tile_y, tile_size, img_width, img_height):
            return True
    return False

def extract_tiles_from_image(img_path, label_path, tile_size=TILE_SIZE, overlap=OVERLAP):
    """Extract tiles with no panels from fullsize image."""
    if not img_path.exists():
        return []
    
    # Load image
    img = cv2.imread(str(img_path))
    if img is None:
        return []
    
    img_height, img_width = img.shape[:2]
    
    # Load labels
    bboxes = load_yolo_labels(label_path)
    
    # Calculate step size with overlap
    step = int(tile_size * (1 - overlap))
    
    tiles = []
    
    # Generate all possible tile positions
    for y in range(0, img_height - tile_size + 1, step):
        for x in range(0, img_width - tile_size + 1, step):
            # Check if this tile region has no panels
            if not has_panels_in_tile(bboxes, x, y, tile_size, img_width, img_height):
                # Extract tile
                tile = img[y:y+tile_size, x:x+tile_size]
                if tile.shape[0] == tile_size and tile.shape[1] == tile_size:
                    tiles.append({
                        'image': tile,
                        'x': x,
                        'y': y,
                        'source': img_path.stem
                    })
    
    return tiles

def mine_hard_negatives():
    """Mine hard negatives from fullsize images."""
    print("="*80)
    print("MINING HARD NEGATIVES")
    print("="*80)
    print(f"Target: {TARGET_COUNT} hard negatives")
    print(f"Source: {SOURCE_IMAGES}")
    print(f"Tile size: {TILE_SIZE}px")
    print()
    
    # Get all fullsize images
    image_files = list(SOURCE_IMAGES.glob("*.jpg")) + list(SOURCE_IMAGES.glob("*.png"))
    print(f"Found {len(image_files)} fullsize images")
    
    if not image_files:
        print("ERROR: No images found!")
        return
    
    # Shuffle for randomness
    random.seed(42)
    random.shuffle(image_files)
    
    # Target directories
    target_img_dir = TARGET_DATASET / "images" / "train" / "tiled1536"
    target_lbl_dir = TARGET_DATASET / "labels" / "train" / "tiled1536"
    
    all_tiles = []
    processed_images = 0
    
    print("\nScanning images for empty regions...")
    
    # Process images until we have enough tiles
    for img_path in image_files:
        label_path = SOURCE_LABELS / f"{img_path.stem}.txt"
        
        # Extract tiles with no panels
        tiles = extract_tiles_from_image(img_path, label_path)
        
        if tiles:
            all_tiles.extend(tiles)
            processed_images += 1
            print(f"  {img_path.name}: Found {len(tiles)} empty tiles (total: {len(all_tiles)})")
        
        # Stop if we have enough
        if len(all_tiles) >= TARGET_COUNT:
            break
    
    print(f"\nFound {len(all_tiles)} potential hard negatives from {processed_images} images")
    
    # Sample exactly TARGET_COUNT tiles
    if len(all_tiles) >= TARGET_COUNT:
        selected_tiles = random.sample(all_tiles, TARGET_COUNT)
    else:
        print(f"WARNING: Only found {len(all_tiles)} tiles, using all")
        selected_tiles = all_tiles
    
    print(f"\nSaving {len(selected_tiles)} hard negatives...")
    
    # Get existing tile names to avoid duplicates
    existing_tiles = {f.stem for f in target_img_dir.glob("*.jpg")}
    
    saved_count = 0
    for i, tile_data in enumerate(selected_tiles):
        # Generate unique name
        base_name = f"{tile_data['source']}_hn_{tile_data['x']}_{tile_data['y']}"
        tile_name = base_name
        counter = 0
        while tile_name in existing_tiles:
            counter += 1
            tile_name = f"{base_name}_{counter}"
        
        existing_tiles.add(tile_name)
        
        # Save image
        img_path = target_img_dir / f"{tile_name}.jpg"
        cv2.imwrite(str(img_path), tile_data['image'])
        
        # Save empty label
        lbl_path = target_lbl_dir / f"{tile_name}.txt"
        lbl_path.write_text('')
        
        saved_count += 1
        
        if (i + 1) % 50 == 0:
            print(f"  Saved {i + 1}/{len(selected_tiles)}...")
    
    print(f"\nSaved {saved_count} hard negatives")
    
    # Verify final counts
    total_train_labels = len(list(target_lbl_dir.glob("*.txt")))
    empty_labels = [f for f in target_lbl_dir.glob("*.txt") if f.stat().st_size == 0]
    non_empty_labels = [f for f in target_lbl_dir.glob("*.txt") if f.stat().st_size > 0]
    
    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)
    print(f"Total train labels: {total_train_labels}")
    print(f"  - With panels: {len(non_empty_labels)}")
    print(f"  - Hard negatives: {len(empty_labels)}")
    print(f"  - Hard negative ratio: {len(empty_labels)/total_train_labels:.1%}")
    print()
    print(f"Hard negatives saved to: {target_img_dir}")
    print(f"Empty labels saved to: {target_lbl_dir}")

if __name__ == "__main__":
    mine_hard_negatives()

